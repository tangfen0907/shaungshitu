from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = 'pump_experiment'
LOCAL_CONFIG_OVERRIDES = {
    'dataset': 'PUMP',
    'data_path': 'dataset/PUMP',
    'v2_first_kernel_size': 3,
    'active_view': 'dual',
    'state_dim': 64,
    'stage2_balanced_core_max_fraction': 0.5,
    'cache_windows': True,
    'stage1_use_injected_triplet': True,
    'stage1_triplet_margin': 0.3,
    'decision_quantile': 0.948255688466,
    'visualization_method': 'pca',
}


def main():
    args = parse_common_preset_args("Run the PUMP preset experiment.")
    run_dataset_preset('PUMP', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
