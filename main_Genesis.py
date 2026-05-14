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
    'tcn_layers': (128, 128, 128),
    'latent_dim': 128,
    'state_dim': 128,
    'tcn_kernel_size': 3,
    'v2_first_kernel_size': 3,
    'tcn_dropout': 0.1,
    'tcn_activation': 'relu',
    'use_attentive_pooling': True,
    'readout_mode': 'attn_topk_max',
    'topk_ratio': 0.1,
    'topk_k': 0,
    'patch_len': 16,
    'patch_stride': 8,

    # Training schedule
    'epoch_stage0': 10,
    'epoch_stage1':0,
    'epoch_stage2': 20,
    'num_stage2_rounds': 4,
    'epochs_per_stage2_round': 5,
    'batch_size': 64,
    'lr': 1e-3,
    'weight_decay': 1e-5,

    # Stage1 representation stabilization
    'lambda_cv_stage0': 0.0,
    'lambda_cv_stage1': 0.20,
    'lambda_cv_stage2': 0.0,
    'stage1_use_masked_reconstruction': False,
    'stage1_mask_ratio_time': 0.15,
    'stage1_mask_num_channels': 0,
    'stage1_recon_loss_on_mask_only': True,
    'stage1_use_injected_triplet': True,
    'stage1_triplet_margin': 0.3,
    'lambda_stage1_triplet': 1.0,
    'stage1_positive_offset': 1,
    'stage1_positive_direction': 'past',
    'negative_injection_profile': 'default',
    'stage1_relational_negative_p': 0.0,
    'stage1_relational_max_shift_ratio': 0.15,
    'stage1_relational_max_channels': 5,
    'relational_time_shift_weight': 0.45,
    'relational_channel_replace_weight': 0.40,
    'relational_channel_shuffle_weight': 0.15,
    'active_pool_trim_enabled': False,
    'active_pool_trim_stage0_ratio': 0.0,
    'active_pool_trim_stage1_ratio': 0.0,

    # Stage2 separate prototypes
    'stage2_method': 'separate_proto',
    'prototype_mode': 'separate',
    'num_prototypes': 5,
    'proto_temperature': 0.2,
    'q_joint_sharpen_temperature': 1.0,
    'lambda_state_consistency': 1.0,
    'lambda_proto_pull': 0.2,
    'lambda_proto_repulsion': 1.0,
    'proto_repulsion_margin': 1.0,
    'lambda_proto_separation': 0.3,
    'proto_separation_margin': 1.0,
    'proto_separation_force_weight': 0.1,
    'lambda_proto_relation_consistency': 0.0,
    'lambda_proto_usage_balance': 0.05,
    'stage2_lambda_rec': 1.0,
    'stage2_token_kmeans_max_tokens': 200000,
    'lambda_injected_push': 0.1,
    'stage2_injected_margin': 1.0,
    'stage2_relational_negative_p': 0.0,
    'stage2_relational_max_shift_ratio': 0.10,
    'stage2_relational_max_channels': 4,
    'tau_conf': 0.7,
    'joint_core_mode': 'minimal',
    'joint_core_dist_quantile': 0.8,
    'joint_core_recon_quantile': 0.8,
    'stage2_balanced_core': True,
    'stage2_balanced_core_max_fraction': 0.35,
    'stage2_balanced_core_min_per_proto': 16,
    'enable_joint_core_label_diagnostics': False,

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
