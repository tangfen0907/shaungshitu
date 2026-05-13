import os

from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = "smap_experiment"
LOCAL_CONFIG_OVERRIDES = {
    # Edit this block directly when you want to tune the SMAP experiment.

    # Dataset and encoder
    "dataset": "SMAP",
    "data_path": os.path.join("dataset", "SMAP"),
    "seq_len": 100,
    # Use a sparser stride so highly overlapping windows do not dominate
    # k-means/prototype core selection or the window-level evaluation.
    "step": 5,
    "train_step": 10,
    "test_step": 10,
    "in_channels": 25,
    # SMAP has many long anomaly segments, so use a deeper TCN that can
    # cover the 100-step window instead of only short local context.
    "tcn_layers": (64, 128, 128),
    "latent_dim": 64,
    "tcn_kernel_size": 3,
    "v2_first_kernel_size": 3,
    "tcn_dropout": 0.1,
    "tcn_activation": "relu",
    "use_attentive_pooling": True,
    "active_view": "dual",
    "dual_view_feature_mode": "avg",

    "lambda_cv_stage0": 0.0,
    "lambda_cv_stage1": 0.05,
    "lambda_cv_stage2": 0.0,
    "stage2_method": "separate_proto",
    "state_dim": 64,
    "num_prototypes": 10,
    # Softer than the original 0.2 setting, but not as flat as 1.0.
    "proto_temperature": 0.5,
    "q_cons_sharpen_temperature": 1.0,
    "lambda_state_consistency": 0.05,
    "lambda_proto_pull": 0.2,
    "lambda_proto_repulsion": 1.0,
    "proto_repulsion_margin": 1.0,
    "lambda_proto_separation": 0.3,
    "proto_separation_margin": 1.0,
    "proto_separation_force_weight": 0.1,
    "lambda_proto_usage_balance": 0.05,
    "stage2_balanced_core": True,
    "stage2_balanced_core_max_fraction": 0.50,
    "stage2_balanced_core_min_per_proto": 16,
    # Start V2 with a permissive core gate; the robust gate filtered nearly
    # all SMAP samples out, leaving Stage2 with no prototype supervision.
    "tau_conf": 0.10,
    "joint_core_mode": "minimal",
    "joint_core_dist_quantile": 0.80,
    "joint_core_recon_quantile": 0.80,
    "lambda_js_score": 1.0,
    "prototype_recon_weight": 0.5,

    # Training
    "epoch_stage0": 10,
    "epoch_stage1": 10,
    "epoch_stage2": 20,
    # Stage1 triplet uses anchor/positive/negative dual-view encodings.
    # Keep this conservative for 8GB GPUs.
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "seed": 42,
    "num_workers": 8,
    "cache_windows": True,

    # Stage 1 injected triplet warmup
    # SMAP negatives need relation-aware corruption instead of only value
    # perturbations. Keep the triplet margin moderate to avoid latent blow-up.
    "stage1_use_injected_triplet": True,
    "stage1_positive_direction": "past",
    "stage1_positive_offset": 1,
    "stage1_triplet_margin": 0.5,
    "lambda_stage1_triplet": 1.0,

    # SMAP-specific relational negatives: break synchronization and channel
    # relationships while keeping individual channel values plausible.
    "negative_injection_profile": "relational_smap",
    "stage1_relational_negative_p": 1.0,
    "stage1_relational_max_shift_ratio": 0.15,
    "stage1_relational_max_channels": 5,
    "stage2_relational_negative_p": 1.0,
    "stage2_relational_max_shift_ratio": 0.10,
    "stage2_relational_max_channels": 4,
    "relational_time_shift_weight": 0.45,
    "relational_channel_replace_weight": 0.40,
    "relational_channel_shuffle_weight": 0.15,

    # Stage 2 prototype schedule
    "num_stage2_rounds": 4,
    "epochs_per_stage2_round": 5,

    # Stage 2 loss and anomaly score
    "lambda_rec": 1.0,
    "stage2_lambda_rec": 1.0,
    # Use a high training-score quantile for the unsupervised threshold.
    # Test-label anomaly ratios are useful for analysis, but should not set
    # the deployment threshold directly.
    "decision_quantile": 0.995,

    # Visualization and debugging
    "enable_stage_visualization": True,
    "visualization_max_points": 3000,
    "visualization_method": "pca",
    "visualization_tsne_perplexity": 35.0,
}
def main():
    args = parse_common_preset_args("Run the SMAP preset experiment.")
    run_dataset_preset("SMAP", RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
