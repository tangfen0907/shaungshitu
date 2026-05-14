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

    print("========== Stage A1: Unlabeled-train Latent Warmup ==========")
    print(
        "[StageA1] setup | "
        "reconstruction=enabled | "
        f"masked_reconstruction={bool(getattr(self.config, 'stage1_use_masked_reconstruction', False))} | "
        f"mask_ratio_time={float(getattr(self.config, 'stage1_mask_ratio_time', 0.0)):.3f} | "
        f"mask_num_channels={int(getattr(self.config, 'stage1_mask_num_channels', 0))} | "
        f"loss_on_mask_only={bool(getattr(self.config, 'stage1_recon_loss_on_mask_only', True))} | "
        f"injected_triplet={bool(getattr(self.config, 'stage1_use_injected_triplet', False))} | "
        f"positive_direction={str(getattr(self.config, 'stage1_positive_direction', 'past'))} | "
        f"positive_offset={int(getattr(self.config, 'stage1_positive_offset', 1))} | "
        f"triplet_margin={float(getattr(self.config, 'stage1_triplet_margin', 1.0)):.3f} | "
        f"lambda_triplet={float(getattr(self.config, 'lambda_stage1_triplet', 1.0)):.3f} | "
        f"negative_profile={str(getattr(self.config, 'negative_injection_profile', 'default'))} | "
        f"rel_stage1_p={float(getattr(self.config, 'stage1_relational_negative_p', 0.0)):.2f} | "
        f"rel_stage2_p={float(getattr(self.config, 'stage2_relational_negative_p', 0.0)):.2f}"
    )
    print(
        "[Encoder] setup | "
        f"AttentivePooling={bool(getattr(self.config, 'use_attentive_pooling', False))}"
    )
    if getattr(self.config, "stage1_log_real_anomaly_distance", False):
        pool_size = len(self.stage1_real_anomaly_indices)
        min_fraction = float(getattr(self.config, "stage1_real_anomaly_min_fraction", 0.0))
        print(
            "[StageA1] Real anomaly monitor pool: "
            f"{pool_size} windows | min_anomaly_fraction={min_fraction:.3f}"
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

    print("========== Stage 2: Prototype Refinement ==========")
    num_stage2_rounds, epochs_per_round = self._resolve_stage2_schedule()
    total_stage2_epochs = self._stage2_total_epochs()
    print(
        "[Stage2] setup | "
        f"method={self._stage2_method()} | "
        f"rounds={num_stage2_rounds} | "
        f"epochs_per_round={epochs_per_round} | "
        "refresh_unit=epoch | "
        f"num_prototypes={int(getattr(self.config, 'num_prototypes', 0))} | "
        f"state_dim={int(getattr(self.config, 'state_dim', 0))} | "
        f"tau_conf={float(getattr(self.config, 'tau_conf', 0.7)):.3f} | "
        f"joint_core={str(getattr(self.config, 'joint_core_mode', 'minimal'))} | "
        f"active_pool={self._active_pool_summary_text()} | "
        f"lambda=(state={float(getattr(self.config, 'lambda_state_consistency', 1.0)):.3f}, "
        f"pull={float(getattr(self.config, 'lambda_proto_pull', 1.0)):.3f}, "
        f"repulse={float(getattr(self.config, 'lambda_proto_repulsion', 1.0)):.3f}, "
        f"rec={float(getattr(self.config, 'stage2_lambda_rec', 0.0)):.3f}) | "
        f"proto_force=(lambda_sep={float(getattr(self.config, 'lambda_proto_separation', 0.3)):.3f}, "
        f"margin={float(getattr(self.config, 'proto_separation_margin', 1.0)):.3f}, "
        f"force={float(getattr(self.config, 'proto_separation_force_weight', 0.1)):.3f}) | "
        f"anti_collapse=(usage={float(getattr(self.config, 'lambda_proto_usage_balance', 0.0)):.3f}, "
        f"balanced_core={bool(getattr(self.config, 'stage2_balanced_core', False))}, "
        f"max_core_frac={float(getattr(self.config, 'stage2_balanced_core_max_fraction', 1.0)):.3f}) | "
        f"core_label_diag={bool(getattr(self.config, 'enable_joint_core_label_diagnostics', False))}"
    )
    title = "Dual-View Separate Prototype Learning"
    print(f"========== Stage 2 {title} ==========")
    self._run_stage2_separate_proto_refinement(
        num_stage2_rounds=num_stage2_rounds,
        epochs_per_round=epochs_per_round,
        total_stage2_epochs=total_stage2_epochs,
    )

    self._save_dual_truth_visualizations("stage2")
    if self._is_dual_view_model():
        self._save_stage2_component_score_visualizations("stage2")
    self._save_checkpoint()
    return self.test()
