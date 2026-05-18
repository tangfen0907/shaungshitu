RUN_NAME = 'skab_experiment'

# SKAB preset for the new local-window dual-view route:
#   X_t: [B, M, L] -> H1_t/H2_t: [B, d_model]
# Reconstruction targets only the current point x_t; early windows are left
# padded by repeating the first observed point.
LOCAL_CONFIG_OVERRIDES = {
    # Dataset and local windowing
    'dataset': 'SKAB',
    'data_path': 'dataset/SKAB',
    'seq_len': 20,
    'step': 1,
    'train_step': 1,
    'test_step': 1,
    'left_pad_windows': True,
    'in_channels': 8,

    # Dual-view encoder
    'active_view': 'dual',
    'dual_view_feature_mode': 'avg',
    'latent_dim': 64,
    'dual_history_len': 20,
    'dual_current_out': 4,
    'dual_short_out': 8,
    'dual_long_out': 8,

    # Training schedule
    'epoch_stage0': 10,
    'epoch_stage1': 10,
    'epoch_stage2': 6,
    'num_stage2_rounds': 3,
    'stage2_a_epochs': 1,
    'stage2_b_epochs': 1,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-5,

    # Stage1 local-window A/P/N.
    # P comes from X_{t-1}; geometry follows the teacher-version raw latent
    # objective with an absolute injected margin.
    'stage1_inject_context_len': 20,
    'stage1_negative_chunk_size': 1024,
    'lambda_rec': 1.0,
    'lambda_stage1_triplet': 1.0,
    'lambda_cv_stage1': 0.0,
    'stage1_ap_margin': 0.1,
    'stage1_triplet_margin': 0.3,
    'active_pool_trim_enabled': False,
    'active_pool_trim_stage0_ratio': 0.0,
    'active_pool_trim_stage1_ratio': 0.0,

    # Stage2 prototype refresh/calibration + encoder refinement
    'stage2_method': 'separate_proto',
    'prototype_mode': 'separate',
    'num_prototypes': 5,
    'proto_temperature': 0.2,
    'stage2_inject_context_len': 20,
    'core_ratio_A': 0.3,
    'alpha_A': 1.0,
    'beta_A': 1.0,
    'gamma_A': 0.5,
    'proto_momentum': 0.8,
    'pair_align_strength': 0.2,
    'min_core_per_proto': 1,
    'lambda_pull_A': 1.0,
    'lambda_sep_A': 0.1,
    'lambda_pair_A': 0.0,
    'core_ratio_B': 0.5,
    'alpha_B': 1.0,
    'beta_B': 1.0,
    'gamma_B': 0.0,
    'lambda_rec_B': 1.0,
    'lambda_ap_B': 0.2,
    'lambda_core_B': 0.5,
    'lambda_neg_B': 0.05,
    'lambda_pull_B': 0.5,
    'lambda_align_B': 0.0,
    'lambda_delta_B': 0.0,
    'lambda_anom_B': 0.05,
    'stage2_ap_margin': 0.1,
    'boundary_quantile': 0.95,
    'negative_boundary_margin': 0.1,
    'use_negative_boundary_radius': True,
    'margin_anom': 1.0,
    'topk_ratio': 0.1,
    'topk_k': 0,

    # Scoring and visualization
    'lambda_js_score': 1.0,
    'prototype_recon_weight': 0.5,
    'decision_quantile': 0.8,
    'enable_stage_visualization': False,
    'enable_stage1_recon_scoring': False,
    'visualization_max_points': 3000,
    'visualization_method': 'pca',
    'visualization_tsne_perplexity': 30.0,

    # Runtime
    'seed': 42,
    'num_workers': 8,
    'cache_windows': True,
    'pin_memory': True,
    'enable_tf32': True,
    'cudnn_benchmark': True,
    'device': 'cuda',
}


def main():
    from train import parse_common_preset_args, run_dataset_preset

    args = parse_common_preset_args("Run the SKAB preset experiment.")
    run_dataset_preset('SKAB', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
