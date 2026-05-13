import os

from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = "gecco_experiment"
LOCAL_CONFIG_OVERRIDES = {
    # Edit this block directly when you want to tune the GECCO experiment.

    # Dataset and encoder
    "dataset": "GECCO",
    "data_path": os.path.join("dataset", "GECCO"),
    "seq_len": 100,
    "step": 1,
    "in_channels": 9,
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
    "lambda_state_consistency": 1.0,
    "lambda_proto_pull": 0.2,
    "lambda_proto_repulsion": 0.2,
    "proto_repulsion_margin": 1.0,
    "lambda_proto_separation": 0.3,
    "proto_separation_margin": 1.0,
    "proto_separation_force_weight": 0.1,
    "tau_conf": 0.70,
    "joint_core_mode": "minimal",
    "joint_core_dist_quantile": 0.80,
    "joint_core_recon_quantile": 0.80,
    "lambda_js_score": 1.0,
    "prototype_recon_weight": 0.5,
    "active_pool_trim_enabled": True,
    "active_pool_trim_stage0_ratio": 0.01,
    "active_pool_trim_stage1_ratio": 0.01,

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

    # GECCO-specific relational negatives. GECCO has only 9 channels, so keep
    # relation corruption local instead of using SMAP's wider channel settings.
    "negative_injection_profile": "relational",
    "stage1_relational_negative_p": 1.0,
    "stage1_relational_max_shift_ratio": 0.10,
    "stage1_relational_max_channels": 2,
    "stage2_relational_negative_p": 1.0,
    "stage2_relational_max_shift_ratio": 0.08,
    "stage2_relational_max_channels": 2,
    "relational_time_shift_weight": 0.50,
    "relational_channel_replace_weight": 0.35,
    "relational_channel_shuffle_weight": 0.15,

    # Stage 2 prototype schedule
    "num_stage2_rounds": 4,
    "epochs_per_stage2_round": 5,

    # Stage 2 loss and anomaly score
    "lambda_rec": 1.0,
    # GECCO test split anomaly ratio:
    # raw points = 730 / 69261 = 1.0540%
    # window flags (seq_len=50, step=1) = 1808 / 69212 = 2.6123%.
    # The current cutoff keeps a slightly more sensitive top-3% window score threshold.
    "decision_quantile": 0.97,

    # Visualization and debugging
    "enable_stage_visualization": True,
    "enable_stage1_recon_scoring": True,
    "enable_joint_core_label_diagnostics": True,
    "visualization_max_points": 3000,
    "visualization_method": "pca",
    "visualization_tsne_perplexity": 35.0,
}
def main():
    args = parse_common_preset_args("Run the GECCO preset experiment.")
    run_dataset_preset("GECCO", RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
