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
    self._trim_active_training_pool("stage0")
    self._save_dual_truth_visualizations("stage0")
    self._save_stage_checkpoint("stage0")

    print("========== Stage A1: Local-window Anomaly Separation ==========")
    print(
        "[StageA1] setup | "
        "reconstruction=current_point | "
        "negative=inject_local_L_window | "
        "positive=X_t_minus_1 | "
        "geometry=small_margin_AP_hinge+negative_boundary_hinge | "
        "away=absolute_margin | "
        f"context_len={self._inject_context_len('stage1')} | "
        f"lambda_triplet={float(getattr(self.config, 'lambda_stage1_triplet', 1.0)):.3f} | "
        f"ap_margin={float(getattr(self.config, 'stage1_ap_margin', 0.1)):.3f} | "
        "lambda_cv=off | "
        f"triplet_margin={float(getattr(self.config, 'stage1_triplet_margin', 0.3)):.3f}"
    )
    if self._is_dual_view_model():
        channels = int(getattr(self.config, "in_channels", 0))
        history_len = int(getattr(self.config, "seq_len", 20))
        short_len = max(1, history_len // 2)
        flat_len = channels * history_len
        flat_short_len = max(1, flat_len // 2)
        view1_dim = channels * (history_len - short_len + 2) + history_len
        view2_dim = flat_len - flat_short_len + 2
        print(
            "[Encoder] setup | "
            "local_window_dual | "
            "input=[B,M,L] | output H1/H2=[B,d_model] | "
            f"L={history_len} | "
            f"M={channels} | "
            f"F1_dim={view1_dim} | "
            f"F2_dim={view2_dim} | "
            f"d_model={int(getattr(self.config, 'latent_dim', 0))} | "
            "reconstruction=current_point"
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
    self._trim_active_training_pool("stage1")
    self._save_dual_truth_visualizations("stage1")
    self._save_stage_checkpoint("stage1")

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
        f"A=(pull={float(getattr(self.config, 'lambda_pull_A', 1.0)):.3f}, "
        f"sep={float(getattr(self.config, 'lambda_sep_A', 0.1)):.3f}) | "
        f"B=(core={float(getattr(self.config, 'core_ratio_B', 0.5)):.3f}, "
        f"rec={float(getattr(self.config, 'lambda_rec_B', 1.0)):.3f}, "
        f"ap={float(getattr(self.config, 'lambda_ap_B', 0.2)):.3f}, "
        f"core_pull={float(getattr(self.config, 'lambda_core_B', 0.5)):.3f}, "
        f"neg={float(getattr(self.config, 'lambda_neg_B', 0.05)):.3f}, "
        f"ap_margin={float(getattr(self.config, 'stage2_ap_margin', 0.1)):.3f})"
    )
    title = "Dual-View Independent Prototype Learning"
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
