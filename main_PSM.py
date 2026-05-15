from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = 'psm_experiment'
LOCAL_CONFIG_OVERRIDES = {
    'dataset': 'PSM',
    'data_path': 'dataset/PSM',
    'active_view': 'dual',
    'latent_dim': 96,
    'dual_history_len': 20,
    'dual_current_out': 8,
    'dual_short_out': 24,
    'dual_long_out': 24,
    'lambda_ctx_stage1': 0.1,
    'lambda_state_consistency': 0.3,
    'lambda_proto_pull': 0.1,
    'lambda_proto_repulsion': 0.2,
    'lambda_proto_usage_balance': 0.2,
    'lambda_proto_relation_consistency': 0.05,
    'stage2_balanced_core_max_fraction': 0.15,
    'stage2_balanced_core_min_per_proto': 128,
    'tau_conf': 0.6,
    'cache_windows': True,
    'negative_injection_profile': 'relational',
    'stage2_relational_negative_p': 1.0,
    'visualization_method': 'pca',
}


def main():
    args = parse_common_preset_args("Run the PSM preset experiment.")
    run_dataset_preset('PSM', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
