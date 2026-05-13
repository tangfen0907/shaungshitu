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
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--visualization_method", choices=["pca", "tsne", "umap"], default=None)
    parser.add_argument("--enable_stage1_recon_scoring", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tcn_layers", type=_parse_tcn_layers, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--v2_first_kernel_size", type=int, default=None)
    parser.add_argument("--readout_mode", choices=["attn_topk_max", "attention", "attn", "last", "topk_max"], default=None)
    parser.add_argument("--topk_ratio", type=float, default=None)
    parser.add_argument("--topk_k", type=int, default=None)
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--patch_stride", type=int, default=None)
    parser.add_argument(
        "--active_view",
        choices=["v1", "v2_flatten", "dual"],
        default=None,
    )
    parser.add_argument("--dual_view_feature_mode", choices=["avg", "v1", "v2"], default=None)
    parser.add_argument("--lambda_cv_stage0", type=float, default=None)
    parser.add_argument("--lambda_cv_stage1", type=float, default=None)
    parser.add_argument("--lambda_cv_stage2", type=float, default=None)
    parser.add_argument("--dual_score_weight_v1", type=float, default=None)
    parser.add_argument("--dual_score_weight_v2", type=float, default=None)
    parser.add_argument("--dual_score_weight_cv", type=float, default=None)
    parser.add_argument("--dual_view_center_weight", type=float, default=None)
    parser.add_argument("--dual_view_recon_weight", type=float, default=None)
    parser.add_argument(
        "--stage2_method",
        choices=["separate_proto", "paired_proto", "consensus_proto", "consensus_proto_v2"],
        default=None,
    )
    parser.add_argument("--state_dim", type=int, default=None)
    parser.add_argument("--num_prototypes", type=int, default=None)
    parser.add_argument("--proto_temperature", type=float, default=None)
    parser.add_argument("--q_cons_sharpen_temperature", type=float, default=None)
    parser.add_argument("--lambda_state_consistency", type=float, default=None)
    parser.add_argument("--lambda_proto_pull", type=float, default=None)
    parser.add_argument("--lambda_proto_repulsion", type=float, default=None)
    parser.add_argument("--proto_repulsion_margin", type=float, default=None)
    parser.add_argument("--lambda_proto_separation", type=float, default=None)
    parser.add_argument("--proto_separation_margin", type=float, default=None)
    parser.add_argument("--proto_separation_force_weight", type=float, default=None)
    parser.add_argument("--tau_conf", type=float, default=None)
    parser.add_argument("--joint_core_mode", choices=["minimal", "robust"], default=None)
    parser.add_argument("--joint_core_dist_quantile", type=float, default=None)
    parser.add_argument("--joint_core_recon_quantile", type=float, default=None)
    parser.add_argument("--stage2_balanced_core", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stage2_balanced_core_max_fraction", type=float, default=None)
    parser.add_argument("--stage2_balanced_core_min_per_proto", type=int, default=None)
    parser.add_argument("--lambda_proto_usage_balance", type=float, default=None)
    parser.add_argument("--lambda_js_score", type=float, default=None)
    parser.add_argument("--prototype_recon_weight", type=float, default=None)
    parser.add_argument("--active_pool_trim_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--active_pool_trim_stage0_ratio", type=float, default=None)
    parser.add_argument("--active_pool_trim_stage1_ratio", type=float, default=None)
    parser.add_argument("--stage2_lambda_rec", type=float, default=None)
    parser.add_argument("--stage1_triplet_margin", type=float, default=None)
    parser.add_argument("--decision_quantile", type=float, default=None)
    parser.add_argument("--train_step", type=int, default=None)
    parser.add_argument("--test_step", type=int, default=None)
    parser.add_argument("--cache_windows", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cudnn_benchmark", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def build_common_preset_overrides(
    local_overrides: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    overrides = dict(local_overrides)
    if getattr(args, "visualization_method", None) is not None:
        overrides["visualization_method"] = str(args.visualization_method).strip().lower()
    if getattr(args, "enable_stage1_recon_scoring", None) is not None:
        overrides["enable_stage1_recon_scoring"] = bool(args.enable_stage1_recon_scoring)
    if getattr(args, "tcn_layers", None) is not None:
        overrides["tcn_layers"] = tuple(int(v) for v in args.tcn_layers)
    if getattr(args, "latent_dim", None) is not None:
        overrides["latent_dim"] = int(args.latent_dim)
    if getattr(args, "v2_first_kernel_size", None) is not None:
        overrides["v2_first_kernel_size"] = int(args.v2_first_kernel_size)
    if getattr(args, "readout_mode", None) is not None:
        overrides["readout_mode"] = str(args.readout_mode).strip().lower()
    if getattr(args, "topk_ratio", None) is not None:
        overrides["topk_ratio"] = float(args.topk_ratio)
    if getattr(args, "topk_k", None) is not None:
        overrides["topk_k"] = int(args.topk_k)
    if getattr(args, "patch_len", None) is not None:
        overrides["patch_len"] = int(args.patch_len)
    if getattr(args, "patch_stride", None) is not None:
        overrides["patch_stride"] = int(args.patch_stride)
    if getattr(args, "active_view", None) is not None:
        overrides["active_view"] = str(args.active_view).strip().lower()
    if getattr(args, "dual_view_feature_mode", None) is not None:
        overrides["dual_view_feature_mode"] = str(args.dual_view_feature_mode).strip().lower()
    if getattr(args, "lambda_cv_stage0", None) is not None:
        overrides["lambda_cv_stage0"] = float(args.lambda_cv_stage0)
    if getattr(args, "lambda_cv_stage1", None) is not None:
        overrides["lambda_cv_stage1"] = float(args.lambda_cv_stage1)
    if getattr(args, "lambda_cv_stage2", None) is not None:
        overrides["lambda_cv_stage2"] = float(args.lambda_cv_stage2)
    if getattr(args, "dual_score_weight_v1", None) is not None:
        overrides["dual_score_weight_v1"] = float(args.dual_score_weight_v1)
    if getattr(args, "dual_score_weight_v2", None) is not None:
        overrides["dual_score_weight_v2"] = float(args.dual_score_weight_v2)
    if getattr(args, "dual_score_weight_cv", None) is not None:
        overrides["dual_score_weight_cv"] = float(args.dual_score_weight_cv)
    if getattr(args, "dual_view_center_weight", None) is not None:
        overrides["dual_view_center_weight"] = float(args.dual_view_center_weight)
    if getattr(args, "dual_view_recon_weight", None) is not None:
        overrides["dual_view_recon_weight"] = float(args.dual_view_recon_weight)
    if getattr(args, "stage2_method", None) is not None:
        overrides["stage2_method"] = str(args.stage2_method).strip().lower()
    if getattr(args, "state_dim", None) is not None:
        overrides["state_dim"] = int(args.state_dim)
    if getattr(args, "num_prototypes", None) is not None:
        overrides["num_prototypes"] = int(args.num_prototypes)
    if getattr(args, "proto_temperature", None) is not None:
        overrides["proto_temperature"] = float(args.proto_temperature)
    if getattr(args, "q_cons_sharpen_temperature", None) is not None:
        overrides["q_cons_sharpen_temperature"] = float(args.q_cons_sharpen_temperature)
    if getattr(args, "lambda_state_consistency", None) is not None:
        overrides["lambda_state_consistency"] = float(args.lambda_state_consistency)
    if getattr(args, "lambda_proto_pull", None) is not None:
        overrides["lambda_proto_pull"] = float(args.lambda_proto_pull)
    if getattr(args, "lambda_proto_repulsion", None) is not None:
        overrides["lambda_proto_repulsion"] = float(args.lambda_proto_repulsion)
    if getattr(args, "proto_repulsion_margin", None) is not None:
        overrides["proto_repulsion_margin"] = float(args.proto_repulsion_margin)
    if getattr(args, "lambda_proto_separation", None) is not None:
        overrides["lambda_proto_separation"] = float(args.lambda_proto_separation)
    if getattr(args, "proto_separation_margin", None) is not None:
        overrides["proto_separation_margin"] = float(args.proto_separation_margin)
    if getattr(args, "proto_separation_force_weight", None) is not None:
        overrides["proto_separation_force_weight"] = float(args.proto_separation_force_weight)
    if getattr(args, "tau_conf", None) is not None:
        overrides["tau_conf"] = float(args.tau_conf)
    if getattr(args, "joint_core_mode", None) is not None:
        overrides["joint_core_mode"] = str(args.joint_core_mode).strip().lower()
    if getattr(args, "joint_core_dist_quantile", None) is not None:
        overrides["joint_core_dist_quantile"] = float(args.joint_core_dist_quantile)
    if getattr(args, "joint_core_recon_quantile", None) is not None:
        overrides["joint_core_recon_quantile"] = float(args.joint_core_recon_quantile)
    if getattr(args, "stage2_balanced_core", None) is not None:
        overrides["stage2_balanced_core"] = bool(args.stage2_balanced_core)
    if getattr(args, "stage2_balanced_core_max_fraction", None) is not None:
        overrides["stage2_balanced_core_max_fraction"] = float(args.stage2_balanced_core_max_fraction)
    if getattr(args, "stage2_balanced_core_min_per_proto", None) is not None:
        overrides["stage2_balanced_core_min_per_proto"] = int(args.stage2_balanced_core_min_per_proto)
    if getattr(args, "lambda_proto_usage_balance", None) is not None:
        overrides["lambda_proto_usage_balance"] = float(args.lambda_proto_usage_balance)
    if getattr(args, "lambda_js_score", None) is not None:
        overrides["lambda_js_score"] = float(args.lambda_js_score)
    if getattr(args, "prototype_recon_weight", None) is not None:
        overrides["prototype_recon_weight"] = float(args.prototype_recon_weight)
    if getattr(args, "active_pool_trim_enabled", None) is not None:
        overrides["active_pool_trim_enabled"] = bool(args.active_pool_trim_enabled)
    if getattr(args, "active_pool_trim_stage0_ratio", None) is not None:
        overrides["active_pool_trim_stage0_ratio"] = float(args.active_pool_trim_stage0_ratio)
    if getattr(args, "active_pool_trim_stage1_ratio", None) is not None:
        overrides["active_pool_trim_stage1_ratio"] = float(args.active_pool_trim_stage1_ratio)
    if getattr(args, "stage2_lambda_rec", None) is not None:
        overrides["stage2_lambda_rec"] = float(args.stage2_lambda_rec)
    if getattr(args, "stage1_triplet_margin", None) is not None:
        overrides["stage1_triplet_margin"] = float(args.stage1_triplet_margin)
    if getattr(args, "decision_quantile", None) is not None:
        overrides["decision_quantile"] = float(args.decision_quantile)
    if getattr(args, "train_step", None) is not None:
        overrides["train_step"] = int(args.train_step)
    if getattr(args, "test_step", None) is not None:
        overrides["test_step"] = int(args.test_step)
    if getattr(args, "cache_windows", None) is not None:
        overrides["cache_windows"] = bool(args.cache_windows)
    if getattr(args, "pin_memory", None) is not None:
        overrides["pin_memory"] = bool(args.pin_memory)
    if getattr(args, "enable_tf32", None) is not None:
        overrides["enable_tf32"] = bool(args.enable_tf32)
    if getattr(args, "cudnn_benchmark", None) is not None:
        overrides["cudnn_benchmark"] = bool(args.cudnn_benchmark)
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
    run_training(
        config=config,
        run_name=run_name,
        results_root=str(meta["results_root"]),
        experiment_name=str(dataset),
        show_visualization_artifacts=bool(meta["print_visualization_artifacts"]),
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
    "active_view",
    "dual_view_feature_mode",
    "lambda_cv_stage0",
    "lambda_cv_stage1",
    "lambda_cv_stage2",
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
    "q_cons_sharpen_temperature",
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
    "stage1_use_masked_reconstruction",
    "stage1_mask_ratio_time",
    "stage1_mask_num_channels",
    "stage1_recon_loss_on_mask_only",
    "stage1_use_injected_triplet",
    "stage1_triplet_margin",
    "lambda_stage1_triplet",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared training entry for dataset presets.")
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
    parser.add_argument(
        "--active_view",
        type=str,
        default=None,
        choices=["v1", "v2_flatten", "dual"],
    )
    parser.add_argument("--dual_view_feature_mode", type=str, default=None, choices=["avg", "v1", "v2"])
    parser.add_argument("--lambda_cv_stage0", type=float, default=None)
    parser.add_argument("--lambda_cv_stage1", type=float, default=None)
    parser.add_argument("--lambda_cv_stage2", type=float, default=None)
    parser.add_argument("--dual_score_weight_v1", type=float, default=None)
    parser.add_argument("--dual_score_weight_v2", type=float, default=None)
    parser.add_argument("--dual_score_weight_cv", type=float, default=None)
    parser.add_argument("--dual_view_center_weight", type=float, default=None)
    parser.add_argument("--dual_view_recon_weight", type=float, default=None)
    parser.add_argument(
        "--stage2_method",
        type=str,
        default=None,
        choices=["separate_proto", "paired_proto", "consensus_proto", "consensus_proto_v2"],
    )
    parser.add_argument("--state_dim", type=int, default=None)
    parser.add_argument("--num_prototypes", type=int, default=None)
    parser.add_argument("--proto_temperature", type=float, default=None)
    parser.add_argument("--q_cons_sharpen_temperature", type=float, default=None)
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

    parser.add_argument("--stage1_use_masked_reconstruction", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stage1_mask_ratio_time", type=float, default=None)
    parser.add_argument("--stage1_mask_num_channels", type=int, default=None)
    parser.add_argument("--stage1_recon_loss_on_mask_only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stage1_use_injected_triplet", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stage1_triplet_margin", type=float, default=None)
    parser.add_argument("--lambda_stage1_triplet", type=float, default=None)
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
