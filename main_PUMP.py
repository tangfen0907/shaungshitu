from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = 'pump_experiment'
LOCAL_CONFIG_OVERRIDES = {
    'dataset': 'PUMP',
    'data_path': 'dataset/PUMP',
    'active_view': 'dual',
    'latent_dim': 192,
    'dual_history_len': 20,
    'dual_current_out': 16,
    'dual_short_out': 32,
    'dual_long_out': 32,
    'cache_windows': True,
    'lambda_away_stage1': 0.05,
    'decision_quantile': 0.948255688466,
    'visualization_method': 'pca',
}


def main():
    args = parse_common_preset_args("Run the PUMP preset experiment.")
    run_dataset_preset('PUMP', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
