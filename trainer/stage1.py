from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_factory.triplet_dataset import Stage1AdjacentPairDataset


__all__ = [
    '_prepare_stage1_pair_batch',
    '_relational_mode_weights',
    '_use_relational_batch_negative',
    '_inject_stage1_negative_batch',
    '_extract_label_window',
    '_build_stage1_real_anomaly_index_pool',
    '_normal_only_warmup_epoch',
    '_build_stage1_loader',
    'run_stage1_epoch',
]


def _prepare_stage1_pair_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("Stage 1 triplet warmup expects batches shaped as (anchor, positive).")
    anchor, positive = batch
    return (
        anchor.float().to(self.device, non_blocking=self._pin_memory()),
        positive.float().to(self.device, non_blocking=self._pin_memory()),
    )


def _relational_mode_weights(self):
    return (
        float(getattr(self.config, "relational_time_shift_weight", 0.45)),
        float(getattr(self.config, "relational_channel_replace_weight", 0.40)),
        float(getattr(self.config, "relational_channel_shuffle_weight", 0.15)),
    )


def _use_relational_batch_negative(self, stage: str) -> bool:
    stage_key = str(stage or "stage1").strip().lower()
    profile = str(getattr(self.config, "negative_injection_profile", "default")).strip().lower()
    if profile not in {"relational", "relational_smap", "smap_relational"}:
        return False
    p_name = "stage2_relational_negative_p" if stage_key == "stage2" else "stage1_relational_negative_p"
    return float(getattr(self.config, p_name, 0.0)) > 0.0


def _inject_stage1_negative_batch(self, x: torch.Tensor, stage: str = "stage1") -> torch.Tensor:
    stage_key = str(stage or "stage1").strip().lower()
    if self._use_relational_batch_negative(stage_key):
        prefix = "stage2" if stage_key == "stage2" else "stage1"
        return self.injector.inject_relational_batch(
            x.detach(),
            p=float(getattr(self.config, f"{prefix}_relational_negative_p", 1.0)),
            max_shift_ratio=float(getattr(self.config, f"{prefix}_relational_max_shift_ratio", 0.15)),
            max_channels=int(getattr(self.config, f"{prefix}_relational_max_channels", 0)),
            mode_weights=self._relational_mode_weights(),
            return_mask=False,
        ).to(device=x.device, dtype=x.dtype)

    negatives = [self.injector(sample) for sample in x.detach()]
    return torch.stack(negatives, dim=0).to(device=x.device, dtype=x.dtype)


@staticmethod
def _extract_label_window(sample):
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[1]
    return None


def _build_stage1_real_anomaly_index_pool(self) -> List[int]:
    if not getattr(self.config, "stage1_log_real_anomaly_distance", False):
        return []

    dataset = self.test_loader.dataset
    min_fraction = max(0.0, float(getattr(self.config, "stage1_real_anomaly_min_fraction", 0.0)))
    anomaly_indices: List[int] = []

    for idx in range(len(dataset)):
        labels = self._extract_label_window(dataset[idx])
        if labels is None:
            continue

        label_array = np.asarray(labels, dtype=np.float32).reshape(-1)
        if label_array.size == 0:
            continue

        anomaly_fraction = float((label_array > 0).mean())
        if anomaly_fraction > 0.0 and anomaly_fraction >= min_fraction:
            anomaly_indices.append(idx)

    return anomaly_indices


def _normal_only_warmup_epoch(self, loader: DataLoader, epoch: int, total_epoch: int, stage_name: str):
    """
    Stage 1 latent warmup.

    This stage uses the unlabeled training split only. When the injected
    triplet option is enabled, labels are still ignored: the negative is a
    synthetic anomaly generated from the anchor window.
    """
    self.model.train()
    total_loss = 0.0
    total_rec = 0.0
    total_triplet = 0.0
    total_cv = 0.0
    total_rec_v1 = 0.0
    total_rec_v2 = 0.0
    total_triplet_v1 = 0.0
    total_triplet_v2 = 0.0
    total_z1_norm = 0.0
    total_z2_norm = 0.0
    total_z1_std = 0.0
    total_z2_std = 0.0
    total_z_norm = 0.0
    total_z_std = 0.0
    use_masked = bool(getattr(self.config, "stage1_use_masked_reconstruction", False))
    use_triplet = bool(getattr(self.config, "stage1_use_injected_triplet", False))

    progress = tqdm(
        loader,
        desc=f"{stage_name} {epoch}",
        leave=False,
        disable=not self._show_batch_progress(),
    )
    for batch in progress:
        if use_triplet:
            x, x_positive = self._prepare_stage1_pair_batch(batch)
        else:
            x = self._prepare_batch(batch)
            x_positive = None

        if use_masked:
            x_input, recon_mask = self._build_stage1_masked_input(x)
        else:
            x_input, recon_mask = x, None

        outputs = self.model(x_input, stage="stage1")
        z = outputs["z"]
        rec_v1, rec_v2 = self._dual_reconstruction_losses_from_outputs(outputs, x, recon_mask)
        loss_rec = 0.5 * (rec_v1 + rec_v2) if self._is_dual_view_model() else rec_v1
        loss_cv = self._cross_view_consistency_loss(outputs, x)
        loss_triplet = torch.zeros((), device=self.device, dtype=loss_rec.dtype)
        triplet_v1 = torch.zeros((), device=self.device, dtype=loss_rec.dtype)
        triplet_v2 = torch.zeros((), device=self.device, dtype=loss_rec.dtype)
        if use_triplet and x_positive is not None:
            x_negative = self._inject_stage1_negative_batch(x)
            if self._is_dual_view_model():
                if use_masked:
                    z_anchor1, z_anchor2 = self.model.encode_views(x)
                else:
                    z_anchor1, z_anchor2 = outputs["z1"], outputs["z2"]
                z_positive1, z_positive2 = self.model.encode_views(x_positive)
                z_negative1, z_negative2 = self.model.encode_views(x_negative)
                triplet_v1 = self._stage1_triplet_embedding_loss(z_anchor1, z_positive1, z_negative1)
                triplet_v2 = self._stage1_triplet_embedding_loss(z_anchor2, z_positive2, z_negative2)
                loss_triplet = 0.5 * (triplet_v1 + triplet_v2)
            else:
                z_anchor = self.model.encode(x) if use_masked else z
                z_positive = self.model.encode(x_positive)
                z_negative = self.model.encode(x_negative)
                loss_triplet = self._stage1_triplet_embedding_loss(z_anchor, z_positive, z_negative)
                triplet_v1 = loss_triplet
                triplet_v2 = loss_triplet

        loss = (
            float(getattr(self.config, "lambda_rec", 1.0)) * loss_rec
            + float(getattr(self.config, "lambda_stage1_triplet", 1.0)) * loss_triplet
            + float(getattr(self.config, "lambda_cv_stage1", 0.0)) * loss_cv
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()
        total_rec += loss_rec.item()
        total_triplet += loss_triplet.item()
        total_cv += loss_cv.item()
        total_rec_v1 += rec_v1.item()
        total_rec_v2 += rec_v2.item()
        total_triplet_v1 += triplet_v1.item()
        total_triplet_v2 += triplet_v2.item()
        total_z_norm += torch.norm(z, dim=1).mean().item()
        total_z_std += z.std(dim=0, unbiased=False).mean().item()
        if self._is_dual_view_model():
            total_z1_norm += torch.norm(outputs["z1"], dim=1).mean().item()
            total_z2_norm += torch.norm(outputs["z2"], dim=1).mean().item()
            total_z1_std += outputs["z1"].std(dim=0, unbiased=False).mean().item()
            total_z2_std += outputs["z2"].std(dim=0, unbiased=False).mean().item()
        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "r1": f"{rec_v1.item():.4f}",
                "r2": f"{rec_v2.item():.4f}",
                "t1": f"{triplet_v1.item():.4f}",
                "t2": f"{triplet_v2.item():.4f}",
                "cv": f"{loss_cv.item():.4f}",
            }
        )

    denom = max(1, len(loader))
    logs = {
        "loss": total_loss / denom,
        "loss_rec": total_rec / denom,
        "z_norm": total_z_norm / denom,
        "z_std": total_z_std / denom,
    }
    if self._is_dual_view_model():
        logs.update(
            {
                "rec_v1": total_rec_v1 / denom,
                "rec_v2": total_rec_v2 / denom,
                "z1_norm": total_z1_norm / denom,
                "z2_norm": total_z2_norm / denom,
                "z1_std": total_z1_std / denom,
                "z2_std": total_z2_std / denom,
            }
        )
    if use_triplet:
        logs["loss_triplet"] = total_triplet / denom
        if self._is_dual_view_model():
            logs["triplet_v1"] = total_triplet_v1 / denom
            logs["triplet_v2"] = total_triplet_v2 / denom
    if float(getattr(self.config, "lambda_cv_stage1", 0.0)) != 0.0:
        logs["loss_cv"] = total_cv / denom
    self._log_epoch(stage_name, epoch, total_epoch, logs)


def _build_stage1_loader(self) -> DataLoader:
    if bool(getattr(self.config, "stage1_use_injected_triplet", False)):
        stage1_dataset = Stage1AdjacentPairDataset(
            base_dataset=getattr(self, "full_train_dataset", self.train_loader.dataset),
            positive_offset=int(getattr(self.config, "stage1_positive_offset", 1)),
            positive_direction=str(getattr(self.config, "stage1_positive_direction", "past")),
            active_mask=self._current_active_train_mask()
            if hasattr(self, "_current_active_train_mask")
            else None,
        )
    else:
        stage1_dataset = self.train_loader.dataset
    return DataLoader(
        dataset=stage1_dataset,
        batch_size=self.config.batch_size,
        shuffle=True,
        num_workers=self._effective_num_workers(),
        drop_last=False,
        pin_memory=self._pin_memory(),
    )


def run_stage1_epoch(solver, loader, epoch):
    return solver._normal_only_warmup_epoch(
        loader,
        epoch=epoch,
        total_epoch=solver.config.epoch_stage1,
        stage_name="StageA1",
    )
