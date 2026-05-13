from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = 'skab_experiment'
LOCAL_CONFIG_OVERRIDES = {
    'dataset': 'SKAB',
    'data_path': 'dataset/SKAB',
    'v2_first_kernel_size': 3,
    'active_view': 'dual',
    'state_dim': 64,
    'num_prototypes': 5,
    'lambda_state_consistency': 1.0,
    'lambda_proto_relation_consistency': 0.05,
    'stage2_balanced_core_max_fraction': 0.5,
    'cache_windows': True,
    'device': 'cuda',
    'stage1_use_injected_triplet': True,
    'stage1_triplet_margin': 0.3,
    'decision_quantile': 0.8,
    'visualization_method': 'pca',
}


def main():
    args = parse_common_preset_args("Run the SKAB preset experiment.")
    run_dataset_preset('SKAB', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
