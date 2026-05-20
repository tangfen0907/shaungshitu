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

    # Training schedule
    'epoch_stage0': 10,
    'epoch_stage1': 10,
    'num_stage2_rounds': 3,
    'stage2_a_epochs': 1,
    'stage2_b_epochs': 1,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-5,

    # Stage1 local-window raw-latent objective.
    # A reconstructs the current point; A/P are kept close and injected N is
    # pushed away. No detach, no normalization, and prototypes stay off.
    'stage1_inject_context_len': 20,
    'lambda_rec': 0.1,
    'lambda_stage1_local': 1.0,
    'stage1_pos_margin': 0.1,
    'stage1_neg_margin': 0.3,
    'active_pool_trim_enabled': False,
    'active_pool_trim_stage0_ratio': 0.0,
    'active_pool_trim_stage1_ratio': 0.0,

    # Stage2 prototype A/B refinement
    'stage2_method': 'separate_proto',
    'num_prototypes': 5,
    'proto_temperature': 0.2,
    'stage2_inject_context_len': 20,
    'core_ratio_A': 0.3,
    'min_core_per_proto': 1,
    'lambda_pull_A': 1.0,
    'lambda_sep_A': 0.1,
    'core_ratio_B': 0.5,
    'lambda_rec_B': 0.1,
    'lambda_ap_B': 0.2,
    'lambda_core_B': 0.5,
    'lambda_neg_B': 0.05,
    'stage2_ap_margin': 0.1,
    'boundary_quantile': 0.95,
    'stage2_relative_neg_margin': 0.1,

    # Scoring and visualization
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
