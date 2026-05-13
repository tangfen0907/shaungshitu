from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = 'smap_experiment'
LOCAL_CONFIG_OVERRIDES = {
    'dataset': 'SMAP',
    'data_path': 'dataset/SMAP',
    'step': 5,
    'train_step': 10,
    'test_step': 10,
    'tcn_layers': (64, 128, 128),
    'v2_first_kernel_size': 3,
    'active_view': 'dual',
    'lambda_cv_stage1': 0.05,
    'state_dim': 64,
    'num_prototypes': 10,
    'proto_temperature': 0.5,
    'stage2_balanced_core_max_fraction': 0.5,
    'tau_conf': 0.1,
    'batch_size': 64,
    'num_workers': 8,
    'cache_windows': True,
    'stage1_use_injected_triplet': True,
    'stage1_triplet_margin': 0.5,
    'negative_injection_profile': 'relational_smap',
    'stage1_relational_negative_p': 1.0,
    'stage2_relational_negative_p': 1.0,
    'visualization_method': 'pca',
}


def main():
    args = parse_common_preset_args("Run the SMAP preset experiment.")
    run_dataset_preset('SMAP', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
