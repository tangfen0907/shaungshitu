import argparse
import os
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
    parser.add_argument("checkpoint_arg", nargs="?", help="Optional checkpoint path, kept for old eval_checkpoint.py usage.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help=f"Optional dataset preset name. Available: {', '.join(available_datasets())}",
    )
    parser.add_argument("--source_run_name", type=str, default="")
    parser.add_argument("--checkpoint_name", type=str, default="model_last.pt")
    parser.add_argument("--latest_run_prefix", type=str, default="")
    parser.add_argument("--run_name", type=str, default="eval")
    parser.add_argument("--results_root", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="")
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


def _latest_run_name(results_root: str, prefix: str, checkpoint_name: str) -> str:
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results root not found: {os.path.abspath(results_root)}")
    candidates = []
    for entry in os.listdir(results_root):
        full_path = os.path.join(results_root, entry)
        if not os.path.isdir(full_path) or not entry.startswith(prefix):
            continue
        checkpoint_path = os.path.join(full_path, checkpoint_name)
        if not os.path.isfile(checkpoint_path):
            continue
        suffix = entry[len(prefix):]
        if suffix.isdigit():
            candidates.append((int(suffix), entry))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint named {checkpoint_name!r} found under {os.path.abspath(results_root)} "
            f"with prefix {prefix!r}."
        )
    candidates.sort()
    return candidates[-1][1]


def _resolve_checkpoint_from_args(args: argparse.Namespace, results_root: str) -> str:
    explicit = str(args.checkpoint or args.checkpoint_arg or "").strip()
    if explicit:
        return os.path.abspath(explicit)
    if not args.dataset:
        raise ValueError("Either provide --checkpoint/path or provide --dataset with --source_run_name/latest lookup.")

    meta = dataset_entry_meta(args.dataset)
    root = results_root or str(meta["results_root"])
    source_run_name = str(args.source_run_name).strip()
    if not source_run_name:
        prefix = str(args.latest_run_prefix).strip() or str(meta["run_name"])
        source_run_name = _latest_run_name(root, prefix=prefix, checkpoint_name=args.checkpoint_name)
    return os.path.abspath(os.path.join(root, source_run_name, args.checkpoint_name))


def main():
    args = build_parser().parse_args()
    config = None
    show_visualization_artifacts = False
    experiment_name = "Evaluation"
    results_root = args.results_root.strip()

    if args.dataset:
        meta = dataset_entry_meta(args.dataset)
        config = build_dataset_config(args.dataset)
        show_visualization_artifacts = bool(meta["print_visualization_artifacts"])
        experiment_name = str(config.dataset)
        if not results_root:
            results_root = str(meta["results_root"])
        if args.config_json:
            config = merge_configs(config, load_json_overrides(args.config_json))
        config = merge_configs(config, _collect_cli_overrides(args, EVAL_OVERRIDE_KEYS))
    elif args.config_json:
        config = load_json_overrides(args.config_json)
    else:
        cli_overrides = _collect_cli_overrides(args, EVAL_OVERRIDE_KEYS)
        if cli_overrides:
            config = cli_overrides
    if args.experiment_name:
        experiment_name = str(args.experiment_name)

    checkpoint_path = _resolve_checkpoint_from_args(args, results_root=results_root)

    run_evaluation(
        checkpoint_path=checkpoint_path,
        config=config,
        run_name=args.run_name,
        results_root=results_root or None,
        experiment_name=experiment_name,
        show_visualization_artifacts=(
            show_visualization_artifacts
            if args.print_visualization_artifacts is None
            else bool(args.print_visualization_artifacts)
        ),
    )


if __name__ == "__main__":
    main()
