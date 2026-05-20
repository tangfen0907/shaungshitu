RUN_NAME = "genesis_tcn_experiment"

# Genesis TCN-backbone experiment.
# Keep the 076-style training/scoring recipe and only swap the dual encoder:
#   view1: TCN over [B, M, L]
#   view2: TCN over flattened [B, 1, L*M]

from main_Genesis import LOCAL_CONFIG_OVERRIDES as GENESIS_BASELINE_OVERRIDES


LOCAL_CONFIG_OVERRIDES = {
    **GENESIS_BASELINE_OVERRIDES,
    "dual_encoder_type": "tcn",
    "seq_len": 20,
    "stage1_inject_context_len": 20,
    "stage2_inject_context_len": 20,
    # Multi-run test: metrics and score arrays are enough; HTML can be generated after.
    "enable_stage_visualization": False,
}


def main():
    from train import parse_common_preset_args, run_dataset_preset

    args = parse_common_preset_args("Run the Genesis TCN dual-encoder experiment.")
    run_dataset_preset("Genesis", RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
