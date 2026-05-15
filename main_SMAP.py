from train import parse_common_preset_args, run_dataset_preset


RUN_NAME = 'smap_experiment'
LOCAL_CONFIG_OVERRIDES = {
    'dataset': 'SMAP',
    'data_path': 'dataset/SMAP',
    'step': 1,
    'train_step': 5,
    'test_step': 1,
    'active_view': 'dual',
    'latent_dim': 96,
    'dual_history_len': 20,
    'dual_current_out': 8,
    'dual_short_out': 24,
    'dual_long_out': 24,
    'lambda_away_stage1': 0.05,
    'num_prototypes': 10,
    'proto_temperature': 0.5,
    'batch_size': 64,
    'num_workers': 8,
    'cache_windows': True,
    'visualization_method': 'pca',
}


def main():
    args = parse_common_preset_args("Run the SMAP preset experiment.")
    run_dataset_preset('SMAP', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
