from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = 'smd_experiment'
LOCAL_CONFIG_OVERRIDES = {
    'dataset': 'SMD',
    'data_path': 'dataset/SMD',
    'active_view': 'dual',
    'latent_dim': 160,
    'dual_history_len': 20,
    'dual_current_out': 16,
    'dual_short_out': 24,
    'dual_long_out': 24,
    'num_prototypes': 10,
    'stage2_balanced_core_max_fraction': 0.5,
    'lambda_ctx_stage1': 0.05,
    'decision_quantile': 0.913198959229,
}


def main():
    args = parse_common_preset_args("Run the SMD preset experiment.")
    run_dataset_preset('SMD', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
