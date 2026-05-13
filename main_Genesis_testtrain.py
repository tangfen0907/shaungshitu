from train import parse_common_preset_args, run_dataset_preset
from main_Genesis import LOCAL_CONFIG_OVERRIDES as GENESIS_CONFIG_OVERRIDES


RUN_NAME = "genesis_testtrain_experiment"
LOCAL_CONFIG_OVERRIDES = {
    # Diagnostic run:
    # Train on Genesis test windows as unlabeled data, and use labels only for
    # visualization/evaluation. No supervised loss is enabled by this script.
    **GENESIS_CONFIG_OVERRIDES,
    "train_split_mode": "test",
    "test_split_mode": "test",
    "scaler_fit_mode": "test",
    "stage1_log_real_anomaly_distance": False,
}


def main():
    args = parse_common_preset_args("Run the Genesis test-train diagnostic preset.")
    run_dataset_preset("Genesis", RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
