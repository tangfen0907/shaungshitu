from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


__all__ = [
    "_inject_context_len",
    "_inject_last_context",
    "_stage1_last_point_away_loss",
    "_normal_only_warmup_epoch",
    "_build_stage1_loader",
    "run_stage1_epoch",
]


def _inject_context_len(self, stage: str = "stage1") -> int:
    stage_key = str(stage or "stage1").strip().lower()
    if stage_key == "stage1":
        configured = int(getattr(self.config, "stage1_inject_context_len", 0))
    elif stage_key == "stage2":
        configured = int(getattr(self.config, "stage2_inject_context_len", 0))
    else:
        configured = 0
    if configured <= 0:
        configured = int(getattr(self.config, "dual_history_len", 20))
    return max(1, configured)


def _inject_last_context(
    self,
    x: torch.Tensor,
    *,
    context_len: int = None,
    stage: str = "stage1",
) -> torch.Tensor:
    """Inject anomalies only into the final context segment of each window."""
    if x.dim() != 3:
        raise ValueError(f"inject_last_context expects [B, C, T], got {tuple(x.shape)}.")

    length = int(x.size(-1))
    inject_len = int(context_len) if context_len is not None else self._inject_context_len(stage)
    inject_len = max(1, min(inject_len, length))

    x_negative = x.detach().clone()
    context = x_negative[:, :, -inject_len:]
    x_negative[:, :, -inject_len:] = self.injector(context)
    return x_negative


def _stage1_last_point_away_loss(
    self,
    outputs,
    outputs_negative,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not all(key in outputs and key in outputs_negative for key in ("H1", "H2")):
        raise RuntimeError("Stage1 dual-view outputs should contain point-level H1/H2 tensors.")

    z1 = outputs["H1"][:, -1, :]
    z2 = outputs["H2"][:, -1, :]
    z1_negative = outputs_negative["H1"][:, -1, :]
    z2_negative = outputs_negative["H2"][:, -1, :]
    margin = float(getattr(self.config, "margin_stage1", 1.0))

    away_v1 = F.relu(margin - torch.norm(z1 - z1_negative, dim=1)).mean()
    away_v2 = F.relu(margin - torch.norm(z2 - z2_negative, dim=1)).mean()
    return 0.5 * (away_v1 + away_v2), away_v1, away_v2


def _normal_only_warmup_epoch(self, loader: DataLoader, epoch: int, total_epoch: int, stage_name: str):
    """
    Stage 1 last-context anomaly separation.

    This stage:
        1. reconstructs the full normal window;
        2. injects anomalies only into the final L-step context segment;
        3. pushes the final normal point representation away from the injected
           final-point representation in each view.
    """
    self.model.train()
    total_loss = 0.0
    total_rec = 0.0
    total_away = 0.0
    total_rec_v1 = 0.0
    total_rec_v2 = 0.0
    total_away_v1 = 0.0
    total_away_v2 = 0.0
    total_z1_norm = 0.0
    total_z2_norm = 0.0
    total_z1_std = 0.0
    total_z2_std = 0.0
    total_z_norm = 0.0
    total_z_std = 0.0
    lambda_rec = float(getattr(self.config, "lambda_rec", 1.0))
    lambda_away = float(getattr(self.config, "lambda_away_stage1", 0.05))

    progress = tqdm(
        loader,
        desc=f"{stage_name} {epoch}",
        leave=False,
        disable=not self._show_batch_progress(),
    )
    for batch in progress:
        x = self._prepare_batch(batch)
        x_negative = self._inject_last_context(x, stage="stage1")

        outputs = self.model(x, stage="stage1")
        outputs_negative = self.model(x_negative, stage="stage1")
        z = outputs["z"]
        rec_v1, rec_v2 = self._dual_reconstruction_losses_from_outputs(outputs, x)
        loss_rec = 0.5 * (rec_v1 + rec_v2) if self._is_dual_view_model() else rec_v1
        loss_away, away_v1, away_v2 = self._stage1_last_point_away_loss(outputs, outputs_negative)
        loss = lambda_rec * loss_rec + lambda_away * loss_away

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()
        total_rec += loss_rec.item()
        total_away += loss_away.item()
        total_rec_v1 += rec_v1.item()
        total_rec_v2 += rec_v2.item()
        total_away_v1 += away_v1.item()
        total_away_v2 += away_v2.item()
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
                "a1": f"{away_v1.item():.4f}",
                "a2": f"{away_v2.item():.4f}",
            }
        )

    denom = max(1, len(loader))
    logs = {
        "loss": total_loss / denom,
        "loss_rec": total_rec / denom,
        "loss_away": total_away / denom,
        "z_norm": total_z_norm / denom,
        "z_std": total_z_std / denom,
    }
    if self._is_dual_view_model():
        logs.update(
            {
                "rec_v1": total_rec_v1 / denom,
                "rec_v2": total_rec_v2 / denom,
                "away_v1": total_away_v1 / denom,
                "away_v2": total_away_v2 / denom,
                "z1_norm": total_z1_norm / denom,
                "z2_norm": total_z2_norm / denom,
                "z1_std": total_z1_std / denom,
                "z2_std": total_z2_std / denom,
            }
        )
    self._log_epoch(stage_name, epoch, total_epoch, logs)


def _build_stage1_loader(self) -> DataLoader:
    # Stage1 now uses ordinary active training windows, not adjacent pairs.
    return self.train_loader


def run_stage1_epoch(solver, loader, epoch):
    return solver._normal_only_warmup_epoch(
        loader,
        epoch=epoch,
        total_epoch=solver.config.epoch_stage1,
        stage_name="StageA1",
    )
