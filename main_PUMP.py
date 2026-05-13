import os

from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = "pump_experiment"
LOCAL_CONFIG_OVERRIDES = {
    # Edit this block directly when you want to tune the PUMP experiment.

    # Dataset and encoder
    "dataset": "PUMP",
    "data_path": os.path.join("dataset", "PUMP"),
    "seq_len": 100,
    "step": 1,
    "in_channels": 51,
    "tcn_layers": (64, 128, 128),
    "latent_dim": 64,
    "tcn_kernel_size": 3,
    "v2_first_kernel_size": 3,
    "tcn_dropout": 0.1,
    "tcn_activation": "relu",
    "use_attentive_pooling": True,
    "active_view": "dual",
    "dual_view_feature_mode": "avg",

    # Dual-view paired prototype Stage 2
    "lambda_cv_stage0": 0.0,
    "lambda_cv_stage1": 0.20,
    "lambda_cv_stage2": 0.0,
    "stage2_method": "paired_proto",
    "state_dim": 64,
    "num_prototypes": 10,
    "proto_temperature": 0.2,
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
    "tau_conf": 0.70,
    "joint_core_mode": "minimal",
    "joint_core_dist_quantile": 0.80,
    "joint_core_recon_quantile": 0.80,
    "lambda_js_score": 1.0,
    "prototype_recon_weight": 0.5,

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

    # Stage 1 injected triplet warmup
    "stage1_use_injected_triplet": True,
    "stage1_positive_direction": "past",
    "stage1_positive_offset": 1,
    "stage1_triplet_margin": 0.3,
    "lambda_stage1_triplet": 1.0,

    # Stage 2 prototype schedule
    "num_stage2_rounds": 4,
    "epochs_per_stage2_round": 5,

    # Stage 2 loss and anomaly score
    "lambda_rec": 1.0,
    "stage2_lambda_rec": 1.0,
    # PUMP test split anomaly ratio:
    # raw points = 4357 / 88128 = 4.9439%
    # window flags (seq_len=100, step=1) = 4555 / 88029 = 5.1744%.
    # Thresholding happens on window scores, so match the window-level normal proportion.
    "decision_quantile": 0.948255688466,

    # Visualization and debugging
    "enable_stage_visualization": True,
    "visualization_max_points": 3000,
    "visualization_method": "pca",
    "visualization_tsne_perplexity": 35.0,
}


def main():
    args = parse_common_preset_args("Run the PUMP preset experiment.")
    run_dataset_preset("PUMP", RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
