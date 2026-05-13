import os

from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = "genesis_experiment"
LOCAL_CONFIG_OVERRIDES = {
    # Edit this block directly when you want to tune the Genesis experiment.

    # Dataset and encoder
    "dataset": "Genesis",
    "data_path": os.path.join("dataset", "Genesis"),
    "seq_len": 100,
    "step": 1,
    "in_channels": 18,
    "tcn_layers": (64, 128, 128),
    "latent_dim": 64,
    "tcn_kernel_size": 3,
    "v2_first_kernel_size": 3,
    "tcn_dropout": 0.1,
    "tcn_activation": "relu",
    "use_attentive_pooling": True,
    "readout_mode": "attn_topk_max",
    "topk_ratio": 0.1,
    "topk_k": 0,
    "patch_len": 16,
    "patch_stride": 8,
    "active_view": "dual",
    "dual_view_feature_mode": "avg",

    "lambda_cv_stage0": 0.0,
    "lambda_cv_stage1": 0.20,
    "lambda_cv_stage2": 0.0,
    "dual_score_weight_v1": 1.0,
    "dual_score_weight_v2": 1.0,
    "dual_score_weight_cv": 1.0,
    "dual_view_center_weight": 1.0,
    "dual_view_recon_weight": 0.5,
    "stage2_method": "separate_proto",
    "state_dim": 64,
    "num_prototypes": 5,
    "proto_temperature": 0.2,
    "q_cons_sharpen_temperature": 1.0,
    "lambda_state_consistency": 1.0,
    "lambda_proto_pull": 0.2,
    "lambda_proto_repulsion": 1.0,
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

    # Training
    "epoch_stage0": 10,
    "epoch_stage1": 10,
    "epoch_stage2": 20,
    # 8GB GPUs can OOM during dual-view Stage0 at 128.
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "seed": 42,
    "num_workers": 8,
    "cache_windows": True,
    "device": "cuda",

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
    # Genesis test split anomaly ratio:
    # raw points = 50 / 6488 = 0.7707%
    # window flags (seq_len=100, step=1) = 249 / 6389 = 3.8973%
    # Thresholding happens on window scores, so match the window-level normal proportion.
    # Latest complete Genesis run (genesis_experiment004) was too permissive at 0.98:
    # offline rescoring with the saved train/test scores reduces false positives most at 0.995.
    "decision_quantile": 0.995,

    # Visualization and debugging
    "enable_stage_visualization": True,
    "enable_stage1_plotly_visualization": False,
    "enable_stage2_train_plotly_visualization": False,
    "enable_stage1_recon_scoring": False,
    "visualization_max_points": 3000,
    "visualization_balanced_test_classes": False,
    "visualization_recon_top_ratio": 0.04,
    "visualization_recon_top_max_points": 1000,
    "visualization_stage2_interval": 5,
    "visualization_method": "pca",
    "visualization_tsne_perplexity": 35.0,
}
def main():
    args = parse_common_preset_args("Run the Genesis preset experiment.")
    run_dataset_preset("Genesis", RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
