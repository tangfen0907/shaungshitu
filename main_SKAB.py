from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = 'skab_experiment'
LOCAL_CONFIG_OVERRIDES = {
    'dataset': 'SKAB',
    'data_path': 'dataset/SKAB',
    'active_view': 'dual',
    'latent_dim': 64,
    'dual_history_len': 20,
    'dual_current_out': 4,
    'dual_short_out': 8,
    'dual_long_out': 8,
    'num_prototypes': 5,
    'cache_windows': True,
    'device': 'cuda',
    'lambda_away_stage1': 0.05,
    'decision_quantile': 0.8,
    'visualization_method': 'pca',
}


def main():
    args = parse_common_preset_args("Run the SKAB preset experiment.")
    run_dataset_preset('SKAB', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
