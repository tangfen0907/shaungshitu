import argparse
from typing import Dict

from utils.config import available_datasets, build_dataset_config, dataset_entry_meta
from utils.run_entry import load_json_overrides, merge_configs, run_evaluation


def _collect_cli_overrides(args: argparse.Namespace, keys) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return overrides


EVAL_OVERRIDE_KEYS = [
    # Keep this list aligned with utils.config.Config. Legacy channel-pattern
    # fields are intentionally not exposed here because they are not wired into
    # the current AnomalyDetector path.
    "data_path",
    "batch_size",
    "num_workers",
    "device",
    "decision_quantile",
    "stage2_score_mode",
    "alpha",
    "beta",
    "gamma",
    "stage2_score_topk_inner",
    "stage2_score_topk_temperature",
    "enable_stage_visualization",
    "visualization_method",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared evaluation entry for saved checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help=f"Optional dataset preset name. Available: {', '.join(available_datasets())}",
    )
    parser.add_argument("--run_name", type=str, default="eval")
    parser.add_argument("--results_root", type=str, default="")
    parser.add_argument("--config_json", type=str, default="")
    parser.add_argument("--print_visualization_artifacts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--decision_quantile", type=float, default=None)
    parser.add_argument(
        "--stage2_score_mode",
        type=str,
        default=None,
        choices=["simple_inner", "inner_axis_outer"],
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--stage2_score_topk_inner", type=int, default=None)
    parser.add_argument("--stage2_score_topk_temperature", type=float, default=None)
    parser.add_argument("--enable_stage_visualization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visualization_method", type=str, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    config = None
    show_visualization_artifacts = False
    experiment_name = "Evaluation"

    if args.dataset:
        meta = dataset_entry_meta(args.dataset)
        config = build_dataset_config(args.dataset)
        show_visualization_artifacts = bool(meta["print_visualization_artifacts"])
        experiment_name = str(config.dataset)
        if args.config_json:
            config = merge_configs(config, load_json_overrides(args.config_json))
        config = merge_configs(config, _collect_cli_overrides(args, EVAL_OVERRIDE_KEYS))
    elif args.config_json:
        raise ValueError("--config_json requires --dataset so overrides have a base config.")
    elif _collect_cli_overrides(args, EVAL_OVERRIDE_KEYS):
        raise ValueError("CLI override args require --dataset so overrides have a base config.")

    run_evaluation(
        checkpoint_path=args.checkpoint,
        config=config,
        run_name=args.run_name,
        results_root=args.results_root or None,
        experiment_name=experiment_name,
        show_visualization_artifacts=(
            show_visualization_artifacts
            if args.print_visualization_artifacts is None
            else bool(args.print_visualization_artifacts)
        ),
    )


if __name__ == "__main__":
    main()
