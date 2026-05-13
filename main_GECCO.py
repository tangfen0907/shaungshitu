from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = 'gecco_experiment'
LOCAL_CONFIG_OVERRIDES = {
    'dataset': 'GECCO',
    'data_path': 'dataset/GECCO',
    'v2_first_kernel_size': 3,
    'active_view': 'dual',
    'state_dim': 64,
    'num_prototypes': 10,
    'lambda_state_consistency': 1.0,
    'lambda_proto_repulsion': 0.2,
    'active_pool_trim_enabled': True,
    'active_pool_trim_stage0_ratio': 0.01,
    'active_pool_trim_stage1_ratio': 0.01,
    'cache_windows': True,
    'stage1_use_injected_triplet': True,
    'stage1_triplet_margin': 0.3,
    'negative_injection_profile': 'relational',
    'stage1_relational_negative_p': 1.0,
    'stage1_relational_max_shift_ratio': 0.1,
    'stage1_relational_max_channels': 2,
    'stage2_relational_negative_p': 1.0,
    'stage2_relational_max_shift_ratio': 0.08,
    'stage2_relational_max_channels': 2,
    'relational_time_shift_weight': 0.5,
    'relational_channel_replace_weight': 0.35,
    'decision_quantile': 0.97,
    'enable_stage1_recon_scoring': True,
    'enable_joint_core_label_diagnostics': True,
    'visualization_method': 'pca',
}


def main():
    args = parse_common_preset_args("Run the GECCO preset experiment.")
    run_dataset_preset('GECCO', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
