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
    'lambda_away_stage1': 0.1,
    'cache_windows': True,
    'visualization_method': 'pca',
}


def main():
    args = parse_common_preset_args("Run the PSM preset experiment.")
    run_dataset_preset('PSM', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
