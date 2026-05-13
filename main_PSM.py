import os

from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = "psm_experiment"
LOCAL_CONFIG_OVERRIDES = {
    # PSM train split is treated as clean normal data, so no active-pool trim.

    # Dataset and encoder
    "dataset": "PSM",
    "data_path": os.path.join("dataset", "PSM"),
    "seq_len": 100,
    "step": 1,
    "in_channels": 25,
    "tcn_layers": (64, 128, 128),
    "latent_dim": 64,
    "tcn_kernel_size": 3,
    "v2_first_kernel_size": 3,
    "tcn_dropout": 0.1,
    "tcn_activation": "relu",
    "use_attentive_pooling": True,
    "active_view": "dual",
    "dual_view_feature_mode": "avg",

    # Dual-view separate prototype Stage 2
    "lambda_cv_stage0": 0.0,
    "lambda_cv_stage1": 0.10,
    "lambda_cv_stage2": 0.0,
    "stage2_method": "separate_proto",
    "state_dim": 64,
    "num_prototypes": 10,
    "proto_temperature": 0.2,
    "q_cons_sharpen_temperature": 1.0,
    "lambda_state_consistency": 0.3,
    "lambda_proto_pull": 0.1,
    "lambda_proto_repulsion": 0.2,
    "proto_repulsion_margin": 1.0,
    "lambda_proto_separation": 0.3,
    "proto_separation_margin": 1.0,
    "proto_separation_force_weight": 0.1,
    "lambda_proto_usage_balance": 0.2,
    "lambda_proto_relation_consistency": 0.05,
    "stage2_balanced_core": True,
    "stage2_balanced_core_max_fraction": 0.15,
    "stage2_balanced_core_min_per_proto": 128,
    "tau_conf": 0.60,
    "joint_core_mode": "minimal",
    "joint_core_dist_quantile": 0.80,
    "joint_core_recon_quantile": 0.80,
    "lambda_js_score": 1.0,
    "prototype_recon_weight": 0.5,
    "active_pool_trim_enabled": False,

    # Training
    "epoch_stage0": 10,
    "epoch_stage1": 10,
    "epoch_stage2": 20,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "seed": 42,
    "num_workers": 4,
    "cache_windows": True,

    # Stage 1 injected triplet warmup. PSM has 25 channels, so relation-aware
    # negatives are useful, but less aggressive than SMAP's widest setting.
    "stage1_use_injected_triplet": True,
    "stage1_positive_direction": "past",
    "stage1_positive_offset": 1,
    "stage1_triplet_margin": 0.5,
    "lambda_stage1_triplet": 1.0,
    "negative_injection_profile": "relational",
    "stage1_relational_negative_p": 1.0,
    "stage1_relational_max_shift_ratio": 0.15,
    "stage1_relational_max_channels": 4,
    "stage2_relational_negative_p": 1.0,
    "stage2_relational_max_shift_ratio": 0.10,
    "stage2_relational_max_channels": 4,
    "relational_time_shift_weight": 0.45,
    "relational_channel_replace_weight": 0.40,
    "relational_channel_shuffle_weight": 0.15,

    # Stage 2 prototype schedule
    "num_stage2_rounds": 4,
    "epochs_per_stage2_round": 5,

    # Stage 2 loss and anomaly score. Since PSM train is normal-only, keep a
    # high training-score quantile instead of matching the test anomaly ratio.
    "lambda_rec": 1.0,
    "stage2_lambda_rec": 1.0,
    "decision_quantile": 0.995,

    # Visualization and debugging
    "enable_stage_visualization": True,
    "enable_stage1_recon_scoring": False,
    "visualization_max_points": 3000,
    "visualization_method": "pca",
    "visualization_tsne_perplexity": 35.0,
}


def main():
    args = parse_common_preset_args("Run the PSM preset experiment.")
    run_dataset_preset("PSM", RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
