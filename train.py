import argparse
from typing import Dict, Tuple

from utils.config import available_datasets, build_dataset_config, dataset_entry_meta
from utils.run_entry import load_json_overrides, merge_configs, run_training


def _parse_tcn_layers(value: str) -> Tuple[int, ...]:
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("tcn_layers should be a comma-separated list like 64,128,128")
    try:
        return tuple(int(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tcn_layers should contain integers only") from exc


def _collect_cli_overrides(args: argparse.Namespace, keys) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return overrides


def parse_common_preset_args(description: str) -> argparse.Namespace:
    return build_parser(include_dataset=False, description=description).parse_args()


def build_common_preset_overrides(
    local_overrides: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    overrides = dict(local_overrides)
    json_overrides = load_json_overrides(getattr(args, "config_json", ""))
    if json_overrides:
        overrides = merge_configs(overrides, json_overrides)

    cli_overrides = _collect_cli_overrides(args, TRAIN_OVERRIDE_KEYS)
    for key in (
        "active_view",
        "dual_view_feature_mode",
        "joint_core_mode",
        "scaler_fit_mode",
        "stage1_positive_direction",
        "stage2_method",
        "tcn_activation",
        "test_split_mode",
        "train_split_mode",
        "visualization_method",
    ):
        if key in cli_overrides and isinstance(cli_overrides[key], str):
            cli_overrides[key] = cli_overrides[key].strip().lower()
    if cli_overrides:
        overrides = merge_configs(overrides, cli_overrides)
    return overrides


def run_dataset_preset(
    dataset: str,
    run_name: str,
    local_overrides: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    meta = dataset_entry_meta(dataset)
    config = build_dataset_config(
        dataset,
        overrides=build_common_preset_overrides(local_overrides, args),
    )
    show_visualization_artifacts = (
        bool(meta["print_visualization_artifacts"])
        if getattr(args, "print_visualization_artifacts", None) is None
        else bool(args.print_visualization_artifacts)
    )
    run_training(
        config=config,
        run_name=getattr(args, "run_name", "") or run_name,
        results_root=getattr(args, "results_root", "") or str(meta["results_root"]),
        experiment_name=str(dataset),
        show_visualization_artifacts=show_visualization_artifacts,
    )


TRAIN_OVERRIDE_KEYS = [
    "data_path",
    "train_split_mode",
    "test_split_mode",
    "scaler_fit_mode",
    "seq_len",
    "step",
    "train_step",
    "test_step",
    "in_channels",
    "tcn_layers",
    "latent_dim",
    "tcn_kernel_size",
    "v2_first_kernel_size",
    "tcn_dropout",
    "tcn_activation",
    "use_attentive_pooling",
    "topk_ratio",
    "topk_k",
    "dual_history_len",
    "dual_current_out",
    "dual_short_out",
    "dual_long_out",
    "active_view",
    "dual_view_feature_mode",
    "dual_score_weight_v1",
    "dual_score_weight_v2",
    "dual_score_weight_cv",
    "dual_view_center_weight",
    "dual_view_recon_weight",
    "stage2_method",
    "prototype_mode",
    "state_dim",
    "num_prototypes",
    "proto_temperature",
    "q_joint_sharpen_temperature",
    "lambda_state_consistency",
    "lambda_proto_pull",
    "lambda_proto_repulsion",
    "proto_repulsion_margin",
    "lambda_proto_separation",
    "proto_separation_margin",
    "proto_separation_force_weight",
    "tau_conf",
    "joint_core_mode",
    "joint_core_dist_quantile",
    "joint_core_recon_quantile",
    "stage2_balanced_core",
    "stage2_balanced_core_max_fraction",
    "stage2_balanced_core_min_per_proto",
    "lambda_proto_usage_balance",
    "stage2_time_kmeans_max_tokens",
    "stage2_time_kmeans_mode",
    "stage2_time_core_dist_quantile",
    "stage2_proto_ema_update",
    "stage2_proto_ema_decay",
    "stage2_proto_ema_min_tokens",
    "lambda_injected_push",
    "stage2_injected_margin",
    "lambda_js_score",
    "prototype_recon_weight",
    "active_pool_trim_enabled",
    "active_pool_trim_stage0_ratio",
    "active_pool_trim_stage1_ratio",
    "epoch_stage0",
    "epoch_stage1",
    "epoch_stage2",
    "batch_size",
    "lr",
    "weight_decay",
    "seed",
    "num_workers",
    "device",
    "cache_windows",
    "pin_memory",
    "enable_tf32",
    "cudnn_benchmark",
    "lambda_ctx_stage1",
    "stage1_positive_offset",
    "stage1_positive_direction",
    "num_stage2_rounds",
    "epochs_per_stage2_round",
    "stage2_lambda_rec",
    "decision_quantile",
    "enable_stage_visualization",
    "enable_stage1_recon_scoring",
    "visualization_max_points",
    "visualization_method",
    "visualization_tsne_perplexity",
]


def build_parser(
    include_dataset: bool = True,
    description: str = "Shared training entry for dataset presets.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    if include_dataset:
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help=f"Dataset preset name. Available: {', '.join(available_datasets())}",
        )
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--results_root", type=str, default="")
    parser.add_argument("--config_json", type=str, default="")
    parser.add_argument("--print_visualization_artifacts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--train_split_mode", type=str, default=None)
    parser.add_argument("--test_split_mode", type=str, default=None)
    parser.add_argument("--scaler_fit_mode", type=str, default=None)

    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--train_step", type=int, default=None)
    parser.add_argument("--test_step", type=int, default=None)
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--tcn_layers", type=_parse_tcn_layers, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--tcn_kernel_size", type=int, default=None)
    parser.add_argument("--v2_first_kernel_size", type=int, default=None)
    parser.add_argument("--tcn_dropout", type=float, default=None)
    parser.add_argument("--tcn_activation", type=str, default=None)
    parser.add_argument("--use_attentive_pooling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--topk_ratio", type=float, default=None)
    parser.add_argument("--topk_k", type=int, default=None)
    parser.add_argument("--dual_history_len", type=int, default=None)
    parser.add_argument("--dual_current_out", type=int, default=None)
    parser.add_argument("--dual_short_out", type=int, default=None)
    parser.add_argument("--dual_long_out", type=int, default=None)
    parser.add_argument(
        "--active_view",
        type=str,
        default=None,
        choices=["v1", "v2_flatten", "dual"],
    )
    parser.add_argument("--dual_view_feature_mode", type=str, default=None, choices=["avg", "v1", "v2"])
    parser.add_argument("--dual_score_weight_v1", type=float, default=None)
    parser.add_argument("--dual_score_weight_v2", type=float, default=None)
    parser.add_argument("--dual_score_weight_cv", type=float, default=None)
    parser.add_argument("--dual_view_center_weight", type=float, default=None)
    parser.add_argument("--dual_view_recon_weight", type=float, default=None)
    parser.add_argument(
        "--stage2_method",
        type=str,
        default=None,
        choices=["separate_proto"],
    )
    parser.add_argument("--state_dim", type=int, default=None)
    parser.add_argument("--num_prototypes", type=int, default=None)
    parser.add_argument("--proto_temperature", type=float, default=None)
    parser.add_argument("--q_joint_sharpen_temperature", type=float, default=None)
    parser.add_argument("--lambda_state_consistency", type=float, default=None)
    parser.add_argument("--lambda_proto_pull", type=float, default=None)
    parser.add_argument("--lambda_proto_repulsion", type=float, default=None)
    parser.add_argument("--proto_repulsion_margin", type=float, default=None)
    parser.add_argument("--lambda_proto_separation", type=float, default=None)
    parser.add_argument("--proto_separation_margin", type=float, default=None)
    parser.add_argument("--proto_separation_force_weight", type=float, default=None)
    parser.add_argument("--tau_conf", type=float, default=None)
    parser.add_argument("--joint_core_mode", type=str, default=None, choices=["minimal", "robust"])
    parser.add_argument("--joint_core_dist_quantile", type=float, default=None)
    parser.add_argument("--joint_core_recon_quantile", type=float, default=None)
    parser.add_argument("--stage2_balanced_core", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stage2_balanced_core_max_fraction", type=float, default=None)
    parser.add_argument("--stage2_balanced_core_min_per_proto", type=int, default=None)
    parser.add_argument("--lambda_proto_usage_balance", type=float, default=None)
    parser.add_argument("--stage2_time_kmeans_max_tokens", type=int, default=None)
    parser.add_argument("--stage2_time_kmeans_mode", type=str, default=None, choices=["last", "all"])
    parser.add_argument("--stage2_time_core_dist_quantile", type=float, default=None)
    parser.add_argument("--stage2_proto_ema_update", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stage2_proto_ema_decay", type=float, default=None)
    parser.add_argument("--stage2_proto_ema_min_tokens", type=int, default=None)
    parser.add_argument("--lambda_injected_push", type=float, default=None)
    parser.add_argument("--stage2_injected_margin", type=float, default=None)
    parser.add_argument("--lambda_js_score", type=float, default=None)
    parser.add_argument("--prototype_recon_weight", type=float, default=None)
    parser.add_argument("--active_pool_trim_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--active_pool_trim_stage0_ratio", type=float, default=None)
    parser.add_argument("--active_pool_trim_stage1_ratio", type=float, default=None)

    parser.add_argument("--epoch_stage0", type=int, default=None)
    parser.add_argument("--epoch_stage1", type=int, default=None)
    parser.add_argument("--epoch_stage2", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache_windows", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cudnn_benchmark", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--lambda_ctx_stage1", type=float, default=None)
    parser.add_argument("--stage1_positive_offset", type=int, default=None)
    parser.add_argument("--stage1_positive_direction", type=str, default=None)

    parser.add_argument("--num_stage2_rounds", type=int, default=None)
    parser.add_argument("--epochs_per_stage2_round", type=int, default=None)
    parser.add_argument("--stage2_lambda_rec", type=float, default=None)
    parser.add_argument("--decision_quantile", type=float, default=None)

    parser.add_argument("--enable_stage_visualization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_stage1_recon_scoring", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visualization_max_points", type=int, default=None)
    parser.add_argument("--visualization_method", type=str, default=None)
    parser.add_argument("--visualization_tsne_perplexity", type=float, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    meta = dataset_entry_meta(args.dataset)
    base_config = build_dataset_config(args.dataset)
    overrides = load_json_overrides(args.config_json)
    config = merge_configs(base_config, overrides)
    config = merge_configs(config, _collect_cli_overrides(args, TRAIN_OVERRIDE_KEYS))

    run_training(
        config=config,
        run_name=args.run_name or str(meta["run_name"]),
        results_root=args.results_root or str(meta["results_root"]),
        experiment_name=str(config.dataset),
        show_visualization_artifacts=(
            bool(meta["print_visualization_artifacts"])
            if args.print_visualization_artifacts is None
            else bool(args.print_visualization_artifacts)
        ),
    )


if __name__ == "__main__":
    main()
