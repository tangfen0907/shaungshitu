RUN_NAME = 'genesis_experiment'

# Edit this block when you want to run a different Genesis setting.
LOCAL_CONFIG_OVERRIDES = {
    # Dataset and windowing
    'dataset': 'Genesis',
    'data_path': 'dataset/Genesis',
    'seq_len': 100,
    'step': 1,
    'train_step': -1,
    'test_step': -1,
    'in_channels': 18,

    # Dual-view encoder
    'active_view': 'dual',
    'dual_view_feature_mode': 'avg',
    'latent_dim': 96,
    'dual_history_len': 20,
    'dual_current_out': 8,
    'dual_short_out': 16,
    'dual_long_out': 16,
    'tcn_dropout': 0.1,
    'tcn_activation': 'relu',

    # Training schedule
    'epoch_stage0': 10,
    'epoch_stage1': 5,
    'epoch_stage2': 6,
    'num_stage2_rounds': 3,
    'stage2_a_epochs': 1,
    'stage2_b_epochs': 1,
    'batch_size': 64,
    'lr': 1e-3,
    'weight_decay': 1e-5,

    # Stage1 last-context anomaly separation
    'stage1_inject_context_len': 20,
    'lambda_away_stage1': 0.05,
    'margin_stage1': 1.0,
    'active_pool_trim_enabled': False,
    'active_pool_trim_stage0_ratio': 0.0,
    'active_pool_trim_stage1_ratio': 0.0,

    # Stage2 aligned A/B prototype refinement
    'stage2_method': 'separate_proto',
    'prototype_mode': 'separate',
    'num_prototypes': 5,
    'proto_temperature': 0.2,
    'proto_separation_margin': 1.0,
    'proto_separation_force_weight': 0.1,
    'stage2_inject_context_len': 20,
    'core_ratio_A': 0.5,
    'lambda_pull_A': 1.0,
    'lambda_sep_A': 0.1,
    'lambda_pair_A': 0.1,
    'core_ratio_B': 0.5,
    'alpha_B': 1.0,
    'beta_B': 1.0,
    'gamma_B': 1.0,
    'lambda_rec_B': 1.0,
    'lambda_pull_B': 0.5,
    'lambda_align_B': 0.05,
    'lambda_delta_B': 0.05,
    'lambda_anom_B': 0.05,
    'margin_anom': 1.0,
    'topk_ratio': 0.1,
    'topk_k': 0,

    # Scoring and visualization
    'lambda_rec': 1.0,
    'lambda_js_score': 1.0,
    'prototype_recon_weight': 0.5,
    'decision_quantile': 0.995,
    'enable_stage_visualization': True,
    'enable_stage1_recon_scoring': False,
    'visualization_max_points': 3000,
    'visualization_method': 'pca',
    'visualization_tsne_perplexity': 35.0,

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

    args = parse_common_preset_args("Run the Genesis preset experiment.")
    run_dataset_preset('Genesis', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
