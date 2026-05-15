from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_factory.triplet_dataset import Stage1AdjacentPairDataset


__all__ = [
    '_prepare_stage1_pair_batch',
    '_relational_mode_weights',
    '_use_relational_batch_negative',
    '_stage1_context_shift',
    '_stage1_aligned_point_tokens',
    '_stage1_context_consistency_loss',
    '_normal_only_warmup_epoch',
    '_build_stage1_loader',
    'run_stage1_epoch',
]


def _prepare_stage1_pair_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("Stage 1 expects batches shaped as (anchor, positive).")
    anchor, positive = batch
    return (
        anchor.float().to(self.device, non_blocking=self._pin_memory()),
        positive.float().to(self.device, non_blocking=self._pin_memory()),
    )


def _relational_mode_weights(self):
    # Kept for Stage2 relational negative generation.
    return (
        float(getattr(self.config, "relational_time_shift_weight", 0.45)),
        float(getattr(self.config, "relational_channel_replace_weight", 0.40)),
        float(getattr(self.config, "relational_channel_shuffle_weight", 0.15)),
    )


def _use_relational_batch_negative(self, stage: str) -> bool:
    # Stage1 no longer uses injected negatives. This helper remains only
    # because Stage2 still supports relational negative generation.
    stage_key = str(stage or "").strip().lower()
    if stage_key != "stage2":
        return False
    profile = str(getattr(self.config, "negative_injection_profile", "default")).strip().lower()
    if profile not in {"relational", "relational_smap", "smap_relational"}:
        return False
    return float(getattr(self.config, "stage2_relational_negative_p", 0.0)) > 0.0

def _stage1_context_shift(self) -> int:
    positive_offset = max(1, int(getattr(self.config, "stage1_positive_offset", 1)))
    dataset_step = max(
        1,
        int(
            getattr(
                getattr(self, "full_train_dataset", self.train_loader.dataset),
                "step",
                getattr(self.config, "train_step", getattr(self.config, "step", 1)),
            )
        ),
    )
    return positive_offset * dataset_step


def _stage1_aligned_point_tokens(
    self,
    anchor_tokens: torch.Tensor,
    positive_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if anchor_tokens.dim() != 3 or positive_tokens.dim() != 3:
        raise ValueError(
            "Stage1 point-token alignment expects [B, T, D] tensors, "
            f"got {tuple(anchor_tokens.shape)} and {tuple(positive_tokens.shape)}."
        )
    if tuple(anchor_tokens.shape) != tuple(positive_tokens.shape):
        raise ValueError(
            "Stage1 anchor/positive point-token tensors should have the same shape, "
            f"got {tuple(anchor_tokens.shape)} vs {tuple(positive_tokens.shape)}."
        )

    shift = self._stage1_context_shift()
    length = int(anchor_tokens.size(1))
    if shift >= length:
        raise ValueError(
            "Stage1 context shift should be smaller than the window length: "
            f"shift={shift}, length={length}."
        )

    direction = str(getattr(self.config, "stage1_positive_direction", "past")).strip().lower()
    if direction in {"past", "previous", "prev", "behind"}:
        return anchor_tokens[:, :-shift, :], positive_tokens[:, shift:, :]
    if direction in {"future", "next"}:
        return anchor_tokens[:, shift:, :], positive_tokens[:, :-shift, :]
    raise ValueError(
        "stage1_positive_direction should be one of: past, previous, prev, behind, future, next."
    )


def _stage1_context_consistency_loss(
    self,
    outputs_anchor,
    outputs_positive,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reference = outputs_anchor.get("z")
    if reference is None:
        raise RuntimeError("Stage1 outputs should contain z for dtype/device reference.")
    zero = torch.zeros((), device=reference.device, dtype=reference.dtype)
    if not all(key in outputs_anchor and key in outputs_positive for key in ("H1", "H2")):
        return zero, zero, zero

    anchor_v1, positive_v1 = self._stage1_aligned_point_tokens(
        outputs_anchor["H1"],
        outputs_positive["H1"],
    )
    anchor_v2, positive_v2 = self._stage1_aligned_point_tokens(
        outputs_anchor["H2"],
        outputs_positive["H2"],
    )
    loss_ctx_v1 = torch.mean((anchor_v1 - positive_v1) ** 2)
    loss_ctx_v2 = torch.mean((anchor_v2 - positive_v2) ** 2)
    return 0.5 * (loss_ctx_v1 + loss_ctx_v2), loss_ctx_v1, loss_ctx_v2


def _normal_only_warmup_epoch(self, loader: DataLoader, epoch: int, total_epoch: int, stage_name: str):
    """
    Stage 1 point-level context warmup.

    This stage uses adjacent anchor-positive windows only:
        1. reconstruct the full anchor window;
        2. align the same real time points across neighboring windows and keep
           their point-level representations close.
    """
    self.model.train()
    total_loss = 0.0
    total_rec = 0.0
    total_ctx = 0.0
    total_rec_v1 = 0.0
    total_rec_v2 = 0.0
    total_ctx_v1 = 0.0
    total_ctx_v2 = 0.0
    total_z1_norm = 0.0
    total_z2_norm = 0.0
    total_z1_std = 0.0
    total_z2_std = 0.0
    total_z_norm = 0.0
    total_z_std = 0.0
    lambda_rec = float(getattr(self.config, "lambda_rec", 1.0))
    lambda_ctx = float(getattr(self.config, "lambda_ctx_stage1", 0.05))

    progress = tqdm(
        loader,
        desc=f"{stage_name} {epoch}",
        leave=False,
        disable=not self._show_batch_progress(),
    )
    for batch in progress:
        x_anchor, x_positive = self._prepare_stage1_pair_batch(batch)

        outputs_anchor = self.model(x_anchor, stage="stage1")
        outputs_positive = self.model(x_positive, stage="stage1")
        z = outputs_anchor["z"]
        rec_v1, rec_v2 = self._dual_reconstruction_losses_from_outputs(outputs_anchor, x_anchor)
        loss_rec = 0.5 * (rec_v1 + rec_v2) if self._is_dual_view_model() else rec_v1
        loss_ctx, ctx_v1, ctx_v2 = self._stage1_context_consistency_loss(
            outputs_anchor,
            outputs_positive,
        )
        loss = lambda_rec * loss_rec + lambda_ctx * loss_ctx

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()
        total_rec += loss_rec.item()
        total_ctx += loss_ctx.item()
        total_rec_v1 += rec_v1.item()
        total_rec_v2 += rec_v2.item()
        total_ctx_v1 += ctx_v1.item()
        total_ctx_v2 += ctx_v2.item()
        total_z_norm += torch.norm(z, dim=1).mean().item()
        total_z_std += z.std(dim=0, unbiased=False).mean().item()
        if self._is_dual_view_model():
            total_z1_norm += torch.norm(outputs_anchor["z1"], dim=1).mean().item()
            total_z2_norm += torch.norm(outputs_anchor["z2"], dim=1).mean().item()
            total_z1_std += outputs_anchor["z1"].std(dim=0, unbiased=False).mean().item()
            total_z2_std += outputs_anchor["z2"].std(dim=0, unbiased=False).mean().item()
        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "r1": f"{rec_v1.item():.4f}",
                "r2": f"{rec_v2.item():.4f}",
                "c1": f"{ctx_v1.item():.4f}",
                "c2": f"{ctx_v2.item():.4f}",
            }
        )

    denom = max(1, len(loader))
    logs = {
        "loss": total_loss / denom,
        "loss_rec": total_rec / denom,
        "loss_ctx": total_ctx / denom,
        "z_norm": total_z_norm / denom,
        "z_std": total_z_std / denom,
    }
    if self._is_dual_view_model():
        logs.update(
            {
                "rec_v1": total_rec_v1 / denom,
                "rec_v2": total_rec_v2 / denom,
                "ctx_v1": total_ctx_v1 / denom,
                "ctx_v2": total_ctx_v2 / denom,
                "z1_norm": total_z1_norm / denom,
                "z2_norm": total_z2_norm / denom,
                "z1_std": total_z1_std / denom,
                "z2_std": total_z2_std / denom,
            }
        )
    self._log_epoch(stage_name, epoch, total_epoch, logs)


def _build_stage1_loader(self) -> DataLoader:
    stage1_dataset = Stage1AdjacentPairDataset(
        base_dataset=getattr(self, "full_train_dataset", self.train_loader.dataset),
        positive_offset=int(getattr(self.config, "stage1_positive_offset", 1)),
        positive_direction=str(getattr(self.config, "stage1_positive_direction", "past")),
        active_mask=self._current_active_train_mask()
        if hasattr(self, "_current_active_train_mask")
        else None,
    )
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
