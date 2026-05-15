from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = 'gecco_experiment'
LOCAL_CONFIG_OVERRIDES = {
    'dataset': 'GECCO',
    'data_path': 'dataset/GECCO',
    'active_view': 'dual',
    'latent_dim': 64,
    'dual_history_len': 20,
    'dual_current_out': 4,
    'dual_short_out': 8,
    'dual_long_out': 8,
    'num_prototypes': 10,
    'active_pool_trim_enabled': True,
    'active_pool_trim_stage0_ratio': 0.01,
    'active_pool_trim_stage1_ratio': 0.01,
    'cache_windows': True,
    'lambda_away_stage1': 0.05,
    'decision_quantile': 0.97,
    'enable_stage1_recon_scoring': True,
    'visualization_method': 'pca',
}


def main():
    args = parse_common_preset_args("Run the GECCO preset experiment.")
    run_dataset_preset('GECCO', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
