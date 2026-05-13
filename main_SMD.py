from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = 'smd_experiment'
LOCAL_CONFIG_OVERRIDES = {
    'dataset': 'SMD',
    'data_path': 'dataset/SMD',
    'v2_first_kernel_size': 3,
    'active_view': 'dual',
    'state_dim': 64,
    'num_prototypes': 10,
    'stage2_balanced_core_max_fraction': 0.5,
    'stage1_use_injected_triplet': True,
    'stage1_triplet_margin': 0.3,
    'decision_quantile': 0.913198959229,
}


def main():
    args = parse_common_preset_args("Run the SMD preset experiment.")
    run_dataset_preset('SMD', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
