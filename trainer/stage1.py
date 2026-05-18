from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_factory.triplet_dataset import Stage1AdjacentPairDataset


__all__ = [
    "_inject_context_len",
    "_inject_last_context",
    "_prepare_stage1_apn_batch",
    "_normal_only_warmup_epoch",
    "_build_stage1_loader",
    "run_stage1_epoch",
]


def _inject_context_len(self, stage: str = "stage1") -> int:
    stage_key = str(stage or "stage1").strip().lower()
    configured = int(getattr(self.config, f"{stage_key}_inject_context_len", 0))
    # New route: the dataloader window itself is the local L context. By
    # default anomaly injection acts on the whole current L-window, not on the
    # tail of a larger T=100 window.
    if configured <= 0:
        configured = int(getattr(self.config, "seq_len", 20))
    return max(1, configured)


def _inject_last_context(
    self,
    x: torch.Tensor,
    *,
    context_len: int = None,
    stage: str = "stage1",
) -> torch.Tensor:
    """
    Inject anomalies into the local L-window.

    The name is kept for compatibility with the previous code path, but the
    default behavior has changed: because x is already X_t=[x_{t-L+1},...,x_t],
    the configured/default context length is L, so the whole sample is injected.
    """
    if x.dim() != 3:
        raise ValueError(f"inject_last_context expects [B, M, L], got {tuple(x.shape)}.")

    length = int(x.size(-1))
    inject_len = int(context_len) if context_len is not None else self._inject_context_len(stage)
    inject_len = max(1, min(inject_len, length))

    x_negative = x.detach().clone()
    context = x_negative[:, :, -inject_len:]
    x_negative[:, :, -inject_len:] = self.injector(context)
    return x_negative


def _prepare_stage1_apn_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare Stage1 A/P samples.

    The Stage1 loader yields (X_t, X_{t-1}) where both tensors are already
    channel-first [B, M, L]. The fallback path keeps ordinary-window loaders
    usable, but the full A/P/N route should normally provide P explicitly.
    """
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        left, right = batch[0], batch[1]
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and left.dim() == 3 and right.dim() == 3:
            anchor = left.float().to(self.device, non_blocking=self._pin_memory())
            positive = right.float().to(self.device, non_blocking=self._pin_memory())
            return anchor, positive
    anchor = self._prepare_batch(batch)
    return anchor, None


def _normal_only_warmup_epoch(self, loader: DataLoader, epoch: int, total_epoch: int, stage_name: str):
    """
    Stage1 local-window A/P/N training.

    For each sample X_t: [B, M, L], the encoder produces only A=H_t: [B,d].
    P is produced by a separate previous sample X_{t-1}; it is never taken
    from H[:, -2, :] because H has no time-token axis in the new encoder.
    N is obtained by injecting anomalies into the current local L-window.

    Geometry follows the proto-no-view-align route in raw latent space:

        L_ap   = max(||A-P|| - ap_margin, 0)^2
        L_away = max(margin - ||A-N||, 0)^2

    A/P smoothness is intra-view only. The cross-view consistency term is
    retained as a diagnostic, but it does not contribute to the objective.
    """
    self.model.train()
    total_loss = 0.0
    total_rec = 0.0
    total_triplet = 0.0
    total_cv_diag = 0.0
    total_ap = 0.0
    total_away = 0.0
    total_rec_v1 = 0.0
    total_rec_v2 = 0.0
    total_triplet_v1 = 0.0
    total_triplet_v2 = 0.0
    total_ap_v1 = 0.0
    total_ap_v2 = 0.0
    total_away_v1 = 0.0
    total_away_v2 = 0.0
    total_ap_dist_v1 = 0.0
    total_ap_dist_v2 = 0.0
    total_ap_active_ratio_v1 = 0.0
    total_ap_active_ratio_v2 = 0.0
    total_an_dist_v1 = 0.0
    total_an_dist_v2 = 0.0
    total_z1_norm = 0.0
    total_z2_norm = 0.0
    total_z1_std = 0.0
    total_z2_std = 0.0
    total_z_norm = 0.0
    total_z_std = 0.0

    lambda_rec = float(getattr(self.config, "lambda_rec", 1.0))
    lambda_triplet = float(getattr(self.config, "lambda_stage1_triplet", 1.0))
    ap_margin = float(getattr(self.config, "stage1_ap_margin", 0.1))
    margin = float(getattr(self.config, "stage1_triplet_margin", 0.3))

    progress = tqdm(
        loader,
        desc=f"{stage_name} {epoch}",
        leave=False,
        disable=not self._show_batch_progress(),
    )
    for batch in progress:
        x, x_prev = self._prepare_stage1_apn_batch(batch)
        outputs = self.model(x, stage="stage1")
        z = outputs["z"]
        rec_v1, rec_v2 = self._dual_reconstruction_losses_from_outputs(outputs, x)
        loss_rec = 0.5 * (rec_v1 + rec_v2) if self._is_dual_view_model() else rec_v1
        loss_cv_diag = self._cross_view_consistency_loss(outputs, x)

        loss_triplet = self._zero_stage2_loss() if hasattr(self, "_zero_stage2_loss") else torch.zeros((), device=x.device)
        triplet_v1 = loss_triplet
        triplet_v2 = loss_triplet

        if self._is_dual_view_model():
            a1 = outputs["z1"]
            a2 = outputs["z2"]
            if x_prev is None:
                loss_ap = self._zero_stage2_loss() if hasattr(self, "_zero_stage2_loss") else torch.zeros((), device=x.device)
                ap_v1 = loss_ap
                ap_v2 = loss_ap
                ap_dist_v1 = loss_ap
                ap_dist_v2 = loss_ap
                ap_active_ratio_v1 = loss_ap
                ap_active_ratio_v2 = loss_ap
                p1 = None
                p2 = None
            else:
                outputs_prev = self.model(x_prev, stage="stage1")
                p1 = outputs_prev["z1"]
                p2 = outputs_prev["z2"]

            x_negative = self._inject_last_context(x, stage="stage1")
            outputs_negative = self.model(x_negative, stage="stage1")
            n1 = outputs_negative["z1"]
            n2 = outputs_negative["z2"]
            if p1 is not None and p2 is not None:
                ap_dist_each_v1 = torch.norm(a1 - p1, dim=-1)
                ap_dist_each_v2 = torch.norm(a2 - p2, dim=-1)
                ap_dist_v1 = ap_dist_each_v1.mean()
                ap_dist_v2 = ap_dist_each_v2.mean()
                ap_v1 = F.relu(ap_dist_each_v1 - ap_margin).pow(2).mean()
                ap_v2 = F.relu(ap_dist_each_v2 - ap_margin).pow(2).mean()
                ap_active_ratio_v1 = (ap_dist_each_v1 > ap_margin).float().mean()
                ap_active_ratio_v2 = (ap_dist_each_v2 > ap_margin).float().mean()
                loss_ap = 0.5 * (ap_v1 + ap_v2)
            an_norm_v1 = torch.norm(a1 - n1, dim=-1)
            an_norm_v2 = torch.norm(a2 - n2, dim=-1)
            an_dist_v1 = an_norm_v1.mean()
            an_dist_v2 = an_norm_v2.mean()
            away_v1 = F.relu(margin - an_norm_v1).pow(2).mean()
            away_v2 = F.relu(margin - an_norm_v2).pow(2).mean()
            loss_away = 0.5 * (away_v1 + away_v2)
            if p1 is not None and p2 is not None:
                triplet_v1 = self._stage1_triplet_embedding_loss(a1, p1, n1)
                triplet_v2 = self._stage1_triplet_embedding_loss(a2, p2, n2)
                loss_triplet = 0.5 * (triplet_v1 + triplet_v2)
        else:
            if x_prev is None:
                loss_ap = self._zero_stage2_loss() if hasattr(self, "_zero_stage2_loss") else torch.zeros((), device=x.device)
                z_prev = None
                ap_dist_v1 = loss_ap
                ap_v1 = loss_ap
                ap_v2 = loss_ap
                ap_dist_v2 = ap_dist_v1
                ap_active_ratio_v1 = loss_ap
                ap_active_ratio_v2 = loss_ap
            else:
                z_prev = self.model.encode(x_prev)
            x_negative = self._inject_last_context(x, stage="stage1")
            n = self.model.encode(x_negative)
            if z_prev is not None:
                ap_dist_each = torch.norm(z - z_prev, dim=-1)
                ap_dist_v1 = ap_dist_each.mean()
                loss_ap = F.relu(ap_dist_each - ap_margin).pow(2).mean()
                ap_v1 = loss_ap
                ap_v2 = loss_ap
                ap_dist_v2 = ap_dist_v1
                ap_active_ratio_v1 = (ap_dist_each > ap_margin).float().mean()
                ap_active_ratio_v2 = ap_active_ratio_v1
            an_norm = torch.norm(z - n, dim=-1)
            an_dist_v1 = an_norm.mean()
            away_v1 = F.relu(margin - an_norm).pow(2).mean()
            away_v2 = away_v1
            an_dist_v2 = an_dist_v1
            loss_away = away_v1
            if z_prev is not None:
                loss_triplet = self._stage1_triplet_embedding_loss(z, z_prev, n)
                triplet_v1 = loss_triplet
                triplet_v2 = loss_triplet

        loss = lambda_rec * loss_rec + lambda_triplet * loss_triplet

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += float(loss.item())
        total_rec += float(loss_rec.item())
        total_triplet += float(loss_triplet.item())
        total_cv_diag += float(loss_cv_diag.item())
        total_ap += float(loss_ap.item())
        total_away += float(loss_away.item())
        total_rec_v1 += float(rec_v1.item())
        total_rec_v2 += float(rec_v2.item())
        total_triplet_v1 += float(triplet_v1.item())
        total_triplet_v2 += float(triplet_v2.item())
        total_ap_v1 += float(ap_v1.item())
        total_ap_v2 += float(ap_v2.item())
        total_away_v1 += float(away_v1.item())
        total_away_v2 += float(away_v2.item())
        total_ap_dist_v1 += float(ap_dist_v1.item())
        total_ap_dist_v2 += float(ap_dist_v2.item())
        total_ap_active_ratio_v1 += float(ap_active_ratio_v1.item())
        total_ap_active_ratio_v2 += float(ap_active_ratio_v2.item())
        total_an_dist_v1 += float(an_dist_v1.item())
        total_an_dist_v2 += float(an_dist_v2.item())
        total_z_norm += float(torch.norm(z, dim=1).mean().item())
        total_z_std += float(z.std(dim=0, unbiased=False).mean().item())
        if self._is_dual_view_model():
            total_z1_norm += float(torch.norm(outputs["z1"], dim=1).mean().item())
            total_z2_norm += float(torch.norm(outputs["z2"], dim=1).mean().item())
            total_z1_std += float(outputs["z1"].std(dim=0, unbiased=False).mean().item())
            total_z2_std += float(outputs["z2"].std(dim=0, unbiased=False).mean().item())
        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "r1": f"{rec_v1.item():.4f}",
                "r2": f"{rec_v2.item():.4f}",
                "t1": f"{triplet_v1.item():.4f}",
                "t2": f"{triplet_v2.item():.4f}",
                "ap": f"{loss_ap.item():.4f}",
            }
        )

    denom = max(1, len(loader))
    logs = {
        "loss": total_loss / denom,
        "loss_rec": total_rec / denom,
        "loss_triplet": total_triplet / denom,
        "loss_ap": total_ap / denom,
        "loss_cv_diag": total_cv_diag / denom,
        "loss_away": total_away / denom,
        "z_norm": total_z_norm / denom,
        "z_std": total_z_std / denom,
        "stage1_ap_margin": ap_margin,
    }
    if self._is_dual_view_model():
        logs.update(
            {
                "rec_v1": total_rec_v1 / denom,
                "rec_v2": total_rec_v2 / denom,
                "triplet_v1": total_triplet_v1 / denom,
                "triplet_v2": total_triplet_v2 / denom,
                "ap_v1": total_ap_v1 / denom,
                "ap_v2": total_ap_v2 / denom,
                "away_v1": total_away_v1 / denom,
                "away_v2": total_away_v2 / denom,
                "ap_dist_v1_mean": total_ap_dist_v1 / denom,
                "ap_dist_v2_mean": total_ap_dist_v2 / denom,
                "ap_active_ratio_v1": total_ap_active_ratio_v1 / denom,
                "ap_active_ratio_v2": total_ap_active_ratio_v2 / denom,
                "an_dist_v1": total_an_dist_v1 / denom,
                "an_dist_v2": total_an_dist_v2 / denom,
                "z1_norm": total_z1_norm / denom,
                "z2_norm": total_z2_norm / denom,
                "z1_std": total_z1_std / denom,
                "z2_std": total_z2_std / denom,
            }
        )
    self._log_epoch(stage_name, epoch, total_epoch, logs)


def _build_stage1_loader(self) -> DataLoader:
    # Complete local-window A/P/N: A comes from X_t and P comes from the
    # previous real-time local window X_{t-1}. This intentionally avoids
    # H[:, -2, :] because the encoder only emits H_t: [B, d].
    base_dataset = getattr(self, "full_train_dataset", self.train_loader.dataset)
    active_mask = None
    if hasattr(self, "_current_active_train_mask"):
        mask = self._current_active_train_mask()
        if np.asarray(mask).reshape(-1).shape[0] == len(base_dataset):
            active_mask = mask

    step = int(getattr(base_dataset, "step", 1))
    if step != 1:
        print(
            "[StageA1][Warning] Stage1 P is intended to be X_{t-1}. "
            f"The base training dataset step is {step}, so adjacent dataset "
            "indices may be separated by more than one real timestamp."
        )

    pair_dataset = Stage1AdjacentPairDataset(
        base_dataset=base_dataset,
        positive_offset=int(getattr(self.config, "stage1_positive_offset", 1)),
        positive_direction="past",
        active_mask=active_mask,
        in_channels=int(getattr(self.config, "in_channels", 0)),
        seq_len=int(getattr(self.config, "seq_len", 0)),
    )
    return DataLoader(
        dataset=pair_dataset,
        batch_size=self.config.batch_size,
        shuffle=True,
        num_workers=self._effective_num_workers(),
        drop_last=False,
        pin_memory=self._pin_memory(),
        generator=self._make_loader_generator(201),
    )


def run_stage1_epoch(solver, loader, epoch):
    return solver._normal_only_warmup_epoch(
        loader,
        epoch=epoch,
        total_epoch=solver.config.epoch_stage1,
        stage_name="StageA1",
    )
