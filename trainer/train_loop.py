from typing import Dict

import numpy as np


__all__ = ['_log_epoch', '_show_batch_progress', 'run_train_loop']


def _log_epoch(self, stage_name: str, epoch: int, total_epoch: int, logs: Dict[str, float]):
    message = f"[{stage_name}] Epoch {epoch}/{total_epoch}"
    for key, value in logs.items():
        if isinstance(value, (int, np.integer)):
            message += f" | {key}: {int(value)}"
        else:
            message += f" | {key}: {float(value):.6f}"
    print(message)


def _show_batch_progress(self) -> bool:
    return bool(getattr(self.config, "show_batch_progress", False))


def run_train_loop(solver):
    self = solver
    print("========== Stage A0: Unlabeled-train Reconstruction Warmup ==========")
    for epoch in range(1, self.config.epoch_stage0 + 1):
        self._stage0_epoch(epoch)
    self._save_dual_truth_visualizations("stage0")
    self._trim_active_training_pool("stage0")

    print("========== Stage A1: Last-context Anomaly Separation ==========")
    print(
        "[StageA1] setup | "
        "reconstruction=full_window | "
        "negative=inject_last_context | "
        f"context_len={self._inject_context_len('stage1')} | "
        f"lambda_away={float(getattr(self.config, 'lambda_away_stage1', 0.05)):.3f} | "
        f"margin={float(getattr(self.config, 'margin_stage1', 1.0)):.3f}"
    )
    if self._is_dual_view_model():
        channels = int(getattr(self.config, "in_channels", 0))
        history_len = int(getattr(self.config, "dual_history_len", 20))
        current_out = int(getattr(self.config, "dual_current_out", 8))
        short_out = int(getattr(self.config, "dual_short_out", 16))
        long_out = int(getattr(self.config, "dual_long_out", 16))
        print(
            "[Encoder] setup | "
            "pointwise_dual | "
            f"L={history_len} | "
            f"F1_dim={3 * channels + history_len} | "
            f"F2_dim={current_out + short_out + long_out} "
            f"({current_out}+{short_out}+{long_out}) | "
            f"d_model={int(getattr(self.config, 'latent_dim', 0))} | "
            "reconstruction=full_window"
        )
    else:
        print(
            "[Encoder] setup | "
            f"AttentivePooling={bool(getattr(self.config, 'use_attentive_pooling', False))}"
        )
    stage1_loader = self._build_stage1_loader()
    for epoch in range(1, self.config.epoch_stage1 + 1):
        self._normal_only_warmup_epoch(
            stage1_loader,
            epoch=epoch,
            total_epoch=self.config.epoch_stage1,
            stage_name="StageA1",
        )
    self._save_dual_truth_visualizations("stage1")
    self._trim_active_training_pool("stage1")

    print("========== Stage 2: Prototype A/B Refinement ==========")
    num_stage2_rounds, stage2_a_epochs, stage2_b_epochs = self._resolve_stage2_schedule()
    total_stage2_epochs = self._stage2_total_epochs()
    print(
        "[Stage2] setup | "
        f"method={self._stage2_method()} | "
        f"rounds={num_stage2_rounds} | "
        f"A_epochs={stage2_a_epochs} | "
        f"B_epochs={stage2_b_epochs} | "
        "schedule=Init->A->B | "
        f"num_prototypes={int(getattr(self.config, 'num_prototypes', 0))} | "
        f"state_dim={int(getattr(self.config, 'state_dim', 0))} | "
        f"active_pool={self._active_pool_summary_text()} | "
        f"A=(core={float(getattr(self.config, 'core_ratio_A', 0.5)):.3f}, "
        f"pull={float(getattr(self.config, 'lambda_pull_A', 1.0)):.3f}, "
        f"sep={float(getattr(self.config, 'lambda_sep_A', 0.1)):.3f}, "
        f"pair={float(getattr(self.config, 'lambda_pair_A', 0.1)):.3f}) | "
        f"B=(core={float(getattr(self.config, 'core_ratio_B', 0.5)):.3f}, "
        f"rec={float(getattr(self.config, 'lambda_rec_B', 1.0)):.3f}, "
        f"pull={float(getattr(self.config, 'lambda_pull_B', 0.5)):.3f}, "
        f"align={float(getattr(self.config, 'lambda_align_B', 0.05)):.3f}, "
        f"delta={float(getattr(self.config, 'lambda_delta_B', 0.05)):.3f}, "
        f"anom={float(getattr(self.config, 'lambda_anom_B', 0.05)):.3f})"
    )
    title = "Dual-View Aligned Prototype Learning"
    print(f"========== Stage 2 {title} ==========")
    self._run_stage2_ab_refinement(
        num_stage2_rounds=num_stage2_rounds,
        stage2_a_epochs=stage2_a_epochs,
        stage2_b_epochs=stage2_b_epochs,
        total_stage2_epochs=total_stage2_epochs,
    )

    self._save_dual_truth_visualizations("stage2")
    if self._is_dual_view_model():
        self._save_stage2_component_score_visualizations("stage2")
    self._save_checkpoint()
    return self.test()
