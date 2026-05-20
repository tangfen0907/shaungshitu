"""Run Genesis ablation experiments from the current 076-style baseline.

The script keeps ``main_Genesis.py`` as the single source of the baseline
configuration and applies small, named overrides for each ablation.  It runs
experiments sequentially and writes a compact summary focused on the prototype
and local component scores we currently care about.

Examples
--------
Dry-run the default core matrix:

    python scripts/run_genesis_ablation.py --dry_run

Run the core matrix:

    python scripts/run_genesis_ablation.py

Run only the ABBBB radius-refresh comparison:

    python scripts/run_genesis_ablation.py --only abbbb_radius_b_epoch,abbbb_radius_round

Run the next targeted sweep for AP margin, A/B schedule, and Stage2-B
reconstruction strength:

    python scripts/run_genesis_ablation.py --suite targeted
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_Genesis import LOCAL_CONFIG_OVERRIDES as GENESIS_076_OVERRIDES  # noqa: E402
from utils.config import build_dataset_config  # noqa: E402


FOCUS_SCORES = (
    "score_proto_v2",
    "score_local_v2",
    "score_local_sum",
    "score_proto_ap_gap_sum",
)

FOCUS_METRICS = (
    "pa_f_score",
    "pa_precision",
    "pa_recall",
    "R_AUC_PR",
    "R_AUC_ROC",
    "VUS_PR",
    "MCC_score",
)


@dataclass(frozen=True)
class Ablation:
    name: str
    description: str
    overrides: Dict[str, object]


CORE_ABLATIONS: List[Ablation] = [
    Ablation(
        name="baseline_076",
        description="Current Genesis 076/086 baseline: ABAB, B radius refreshed every B epoch.",
        overrides={},
    ),
    Ablation(
        name="ap_margin_0",
        description="Set Stage2 A/P margin to 0 to test stronger temporal smoothing.",
        overrides={"stage2_ap_margin": 0.0},
    ),
    Ablation(
        name="neg_boundary_margin_0",
        description="Set injected-negative boundary margin to 0.",
        overrides={"negative_boundary_margin": 0.0},
    ),
    Ablation(
        name="abbbb_radius_b_epoch",
        description=(
            "Fair-length ABBBB schedule: 8 rounds * (A1+B4) = 40 Stage2 epochs; "
            "radius refreshes before every B epoch."
        ),
        overrides={
            "num_stage2_rounds": 8,
            "stage2_a_epochs": 1,
            "stage2_b_epochs": 4,
            "stage2_radius_refresh_mode": "b_epoch",
        },
    ),
    Ablation(
        name="abbbb_radius_round",
        description=(
            "Same fair-length ABBBB schedule, but radius is fixed for the whole B block "
            "after each A block."
        ),
        overrides={
            "num_stage2_rounds": 8,
            "stage2_a_epochs": 1,
            "stage2_b_epochs": 4,
            "stage2_radius_refresh_mode": "round",
        },
    ),
]


AP_MARGIN_ABLATIONS: List[Ablation] = [
    Ablation(
        name="baseline_076",
        description="Current Genesis 076/086 baseline.",
        overrides={},
    ),
    Ablation(
        name="ap_margin_0",
        description="Stage2 A/P margin = 0.0.",
        overrides={"stage2_ap_margin": 0.0},
    ),
    Ablation(
        name="ap_margin_0p025",
        description="Stage2 A/P margin = 0.025.",
        overrides={"stage2_ap_margin": 0.025},
    ),
    Ablation(
        name="ap_margin_0p05",
        description="Stage2 A/P margin = 0.05.",
        overrides={"stage2_ap_margin": 0.05},
    ),
    Ablation(
        name="ap_margin_0p075",
        description="Stage2 A/P margin = 0.075.",
        overrides={"stage2_ap_margin": 0.075},
    ),
    Ablation(
        name="ap_margin_0p15",
        description="Stage2 A/P margin = 0.15.",
        overrides={"stage2_ap_margin": 0.15},
    ),
    Ablation(
        name="ap_margin_0p2",
        description="Stage2 A/P margin = 0.20.",
        overrides={"stage2_ap_margin": 0.20},
    ),
]


SCHEDULE_ABLATIONS: List[Ablation] = [
    Ablation(
        name="baseline_076",
        description="AB schedule: 20 rounds * (A1+B1) = 40 Stage2 epochs.",
        overrides={},
    ),
    Ablation(
        name="schedule_abb_13r",
        description="ABB schedule: 13 rounds * (A1+B2) = 39 Stage2 epochs.",
        overrides={
            "num_stage2_rounds": 13,
            "stage2_a_epochs": 1,
            "stage2_b_epochs": 2,
            "stage2_radius_refresh_mode": "b_epoch",
        },
    ),
    Ablation(
        name="schedule_abbb_10r",
        description="ABBB schedule: 10 rounds * (A1+B3) = 40 Stage2 epochs.",
        overrides={
            "num_stage2_rounds": 10,
            "stage2_a_epochs": 1,
            "stage2_b_epochs": 3,
            "stage2_radius_refresh_mode": "b_epoch",
        },
    ),
    Ablation(
        name="schedule_abbbb_8r",
        description="ABBBB schedule: 8 rounds * (A1+B4) = 40 Stage2 epochs.",
        overrides={
            "num_stage2_rounds": 8,
            "stage2_a_epochs": 1,
            "stage2_b_epochs": 4,
            "stage2_radius_refresh_mode": "b_epoch",
        },
    ),
    Ablation(
        name="schedule_aab_13r",
        description="AAB schedule: 13 rounds * (A2+B1) = 39 Stage2 epochs.",
        overrides={
            "num_stage2_rounds": 13,
            "stage2_a_epochs": 2,
            "stage2_b_epochs": 1,
            "stage2_radius_refresh_mode": "b_epoch",
        },
    ),
]


REC_ABLATIONS: List[Ablation] = [
    Ablation(
        name="baseline_076",
        description="Stage2-B reconstruction weight baseline: lambda_rec_B=1.0.",
        overrides={},
    ),
    Ablation(
        name="lambda_rec_B_0",
        description="Remove Stage2-B reconstruction stabilizer.",
        overrides={"lambda_rec_B": 0.0},
    ),
    Ablation(
        name="lambda_rec_B_0p25",
        description="Stage2-B reconstruction weight = 0.25.",
        overrides={"lambda_rec_B": 0.25},
    ),
    Ablation(
        name="lambda_rec_B_0p5",
        description="Stage2-B reconstruction weight = 0.5.",
        overrides={"lambda_rec_B": 0.5},
    ),
    Ablation(
        name="lambda_rec_B_0p75",
        description="Stage2-B reconstruction weight = 0.75.",
        overrides={"lambda_rec_B": 0.75},
    ),
    Ablation(
        name="lambda_rec_B_1p5",
        description="Stage2-B reconstruction weight = 1.5.",
        overrides={"lambda_rec_B": 1.5},
    ),
    Ablation(
        name="lambda_rec_B_2",
        description="Stage2-B reconstruction weight = 2.0.",
        overrides={"lambda_rec_B": 2.0},
    ),
]


TARGETED_ABLATIONS: List[Ablation] = [
    Ablation(
        name="baseline_076",
        description="Current Genesis 076/086 baseline.",
        overrides={},
    ),
    Ablation(
        name="ap_margin_0p025",
        description="Stage2 A/P margin = 0.025.",
        overrides={"stage2_ap_margin": 0.025},
    ),
    Ablation(
        name="ap_margin_0p05",
        description="Stage2 A/P margin = 0.05.",
        overrides={"stage2_ap_margin": 0.05},
    ),
    Ablation(
        name="ap_margin_0p075",
        description="Stage2 A/P margin = 0.075.",
        overrides={"stage2_ap_margin": 0.075},
    ),
    Ablation(
        name="ap_margin_0p15",
        description="Stage2 A/P margin = 0.15.",
        overrides={"stage2_ap_margin": 0.15},
    ),
    Ablation(
        name="schedule_abb_13r",
        description="ABB schedule: 13 rounds * (A1+B2) = 39 Stage2 epochs.",
        overrides={
            "num_stage2_rounds": 13,
            "stage2_a_epochs": 1,
            "stage2_b_epochs": 2,
            "stage2_radius_refresh_mode": "b_epoch",
        },
    ),
    Ablation(
        name="schedule_abbb_10r",
        description="ABBB schedule: 10 rounds * (A1+B3) = 40 Stage2 epochs.",
        overrides={
            "num_stage2_rounds": 10,
            "stage2_a_epochs": 1,
            "stage2_b_epochs": 3,
            "stage2_radius_refresh_mode": "b_epoch",
        },
    ),
    Ablation(
        name="schedule_aab_13r",
        description="AAB schedule: 13 rounds * (A2+B1) = 39 Stage2 epochs.",
        overrides={
            "num_stage2_rounds": 13,
            "stage2_a_epochs": 2,
            "stage2_b_epochs": 1,
            "stage2_radius_refresh_mode": "b_epoch",
        },
    ),
    Ablation(
        name="lambda_rec_B_0p25",
        description="Stage2-B reconstruction weight = 0.25.",
        overrides={"lambda_rec_B": 0.25},
    ),
    Ablation(
        name="lambda_rec_B_0p5",
        description="Stage2-B reconstruction weight = 0.5.",
        overrides={"lambda_rec_B": 0.5},
    ),
    Ablation(
        name="lambda_rec_B_0p75",
        description="Stage2-B reconstruction weight = 0.75.",
        overrides={"lambda_rec_B": 0.75},
    ),
    Ablation(
        name="lambda_rec_B_1p5",
        description="Stage2-B reconstruction weight = 1.5.",
        overrides={"lambda_rec_B": 1.5},
    ),
]


EXTENDED_ABLATIONS: List[Ablation] = TARGETED_ABLATIONS + [
    Ablation(
        name="boundary_q_090",
        description="Use a tighter normal prototype boundary radius.",
        overrides={"boundary_quantile": 0.90},
    ),
    Ablation(
        name="boundary_q_099",
        description="Use a looser normal prototype boundary radius.",
        overrides={"boundary_quantile": 0.99},
    ),
    Ablation(
        name="lambda_neg_0",
        description="Remove injected-negative prototype-boundary loss in Stage2-B.",
        overrides={"lambda_neg_B": 0.0},
    ),
    Ablation(
        name="lambda_neg_0p10",
        description="Double injected-negative prototype-boundary loss weight in Stage2-B.",
        overrides={"lambda_neg_B": 0.10},
    ),
    Ablation(
        name="core_ratio_030",
        description="Use fewer per-prototype core samples in Stage2-B.",
        overrides={"core_ratio_B": 0.30},
    ),
    Ablation(
        name="core_ratio_070",
        description="Use more per-prototype core samples in Stage2-B.",
        overrides={"core_ratio_B": 0.70},
    ),
]


SMOKE_ABLATIONS: List[Ablation] = [
    Ablation(
        name="smoke_baseline",
        description="Short baseline run for checking that the ablation runner works.",
        overrides={
            "epoch_stage0": 1,
            "epoch_stage1": 1,
            "num_stage2_rounds": 1,
            "stage2_a_epochs": 1,
            "stage2_b_epochs": 1,
        },
    ),
    Ablation(
        name="smoke_round_radius",
        description="Short run that exercises stage2_radius_refresh_mode='round'.",
        overrides={
            "epoch_stage0": 1,
            "epoch_stage1": 1,
            "num_stage2_rounds": 1,
            "stage2_a_epochs": 1,
            "stage2_b_epochs": 2,
            "stage2_radius_refresh_mode": "round",
        },
    ),
]


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _parse_seeds(value: str) -> List[int]:
    seeds = [int(item) for item in _parse_csv(value)]
    return seeds or [42]


def _suite_ablation_map(suite: str) -> Dict[str, Ablation]:
    if suite == "core":
        ablations = CORE_ABLATIONS
    elif suite == "ap":
        ablations = AP_MARGIN_ABLATIONS
    elif suite == "schedule":
        ablations = SCHEDULE_ABLATIONS
    elif suite == "rec":
        ablations = REC_ABLATIONS
    elif suite == "targeted":
        ablations = TARGETED_ABLATIONS
    elif suite == "extended":
        ablations = EXTENDED_ABLATIONS
    elif suite == "smoke":
        ablations = SMOKE_ABLATIONS
    else:
        raise ValueError(f"Unknown suite: {suite}")
    return {item.name: item for item in ablations}


def _select_ablations(suite: str, only: str) -> List[Ablation]:
    candidates = _suite_ablation_map(suite)
    if not only:
        return list(candidates.values())

    selected = []
    missing = []
    for name in _parse_csv(only):
        if name not in candidates:
            missing.append(name)
        else:
            selected.append(candidates[name])
    if missing:
        known = ", ".join(candidates)
        raise SystemExit(f"Unknown ablation(s): {', '.join(missing)}\nKnown for suite={suite}: {known}")
    return selected


def _plain_metric_value(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _extract_focus_summary(results: dict) -> Dict[str, object]:
    families = results.get("component_families", {}) or {}
    summary: Dict[str, object] = {}
    for score in FOCUS_SCORES:
        family = families.get(score, {}) or {}
        metrics = family.get("metrics", {}) or {}
        summary[f"{score}.threshold"] = _plain_metric_value(family.get("threshold"))
        for metric in FOCUS_METRICS:
            summary[f"{score}.{metric}"] = _plain_metric_value(metrics.get(metric))
    return summary


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = ["variant", "seed", "run_dir", "description"]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _score_for_sort(row: Dict[str, object], rank_by: str = "") -> float:
    if rank_by:
        value = row.get(rank_by)
        if isinstance(value, (int, float)):
            return float(value)
    preferred = [
        "score_proto_v2.MCC_score",
        "score_local_sum.pa_f_score",
        "score_local_v2.pa_f_score",
        "score_proto_ap_gap_sum.pa_f_score",
        "score_proto_v2.pa_f_score",
    ]
    for key in preferred:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float("-inf")


def _print_final_ranking(rows: List[Dict[str, object]], rank_by: str = "") -> None:
    if not rows:
        return
    ranked = sorted(rows, key=lambda row: _score_for_sort(row, rank_by), reverse=True)
    headers = [
        "rank",
        "variant",
        "seed",
        "rank_metric",
        "proto_v2_f1",
        "proto_v2_mcc",
        "proto_v2_vus_pr",
        "local_sum_f1",
        "local_sum_mcc",
        "local_sum_vus_pr",
        "local_v2_f1",
        "proto_gap_sum_f1",
        "run_dir",
    ]
    table = []
    for idx, row in enumerate(ranked, 1):
        table.append(
            {
                "rank": idx,
                "variant": row.get("variant", ""),
                "seed": row.get("seed", ""),
                "rank_metric": _score_for_sort(row, rank_by),
                "proto_v2_f1": row.get("score_proto_v2.pa_f_score"),
                "proto_v2_mcc": row.get("score_proto_v2.MCC_score"),
                "proto_v2_vus_pr": row.get("score_proto_v2.VUS_PR"),
                "local_sum_f1": row.get("score_local_sum.pa_f_score"),
                "local_sum_mcc": row.get("score_local_sum.MCC_score"),
                "local_sum_vus_pr": row.get("score_local_sum.VUS_PR"),
                "local_v2_f1": row.get("score_local_v2.pa_f_score"),
                "proto_gap_sum_f1": row.get("score_proto_ap_gap_sum.pa_f_score"),
                "run_dir": row.get("run_dir", ""),
            }
        )

    def fmt(value):
        if isinstance(value, float):
            return f"{value:.6f}"
        if value is None:
            return ""
        return str(value)

    widths = {
        header: max(len(header), *(len(fmt(row.get(header))) for row in table))
        for header in headers
    }
    print("\n========== Genesis Ablation Final Ranking ==========")
    if rank_by:
        print(f"Ranked by: {rank_by}")
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in table:
        print(" | ".join(fmt(row.get(header)).ljust(widths[header]) for header in headers))


def _build_effective_overrides(
    ablation: Ablation,
    *,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    overrides = dict(GENESIS_076_OVERRIDES)
    overrides.update(ablation.overrides)
    overrides["seed"] = int(seed)

    # Multi-run ablations are much faster without visual artifacts. This does
    # not affect model training or metrics.
    if not args.with_visuals:
        overrides["enable_stage_visualization"] = False

    if args.device:
        overrides["device"] = args.device
    if args.num_workers is not None:
        overrides["num_workers"] = int(args.num_workers)
    if args.batch_size is not None:
        overrides["batch_size"] = int(args.batch_size)
    if args.max_stage2_rounds is not None:
        overrides["num_stage2_rounds"] = min(
            int(overrides.get("num_stage2_rounds", 1)),
            int(args.max_stage2_rounds),
        )
    return overrides


def _print_matrix(ablations: Iterable[Ablation], seeds: Iterable[int], args: argparse.Namespace) -> None:
    print("Planned Genesis ablations:")
    for ablation in ablations:
        for seed in seeds:
            overrides = _build_effective_overrides(ablation, seed=seed, args=args)
            changed = {
                key: value
                for key, value in overrides.items()
                if GENESIS_076_OVERRIDES.get(key) != value
                and key not in {"enable_stage_visualization", "seed"}
            }
            print(f"- {ablation.name} | seed={seed} | {ablation.description}")
            print(f"  overrides={changed or '{}'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Genesis 076-based ablation experiments.")
    parser.add_argument(
        "--suite",
        choices=["core", "ap", "schedule", "rec", "targeted", "extended", "smoke"],
        default="core",
    )
    parser.add_argument("--only", type=str, default="", help="Comma-separated ablation names to run.")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated random seeds.")
    parser.add_argument("--results_root", type=str, default="results/genesis_ablation")
    parser.add_argument("--run_prefix", type=str, default="genesis_ablation")
    parser.add_argument(
        "--rank_by",
        type=str,
        default="score_proto_v2.MCC_score",
        help="Metric key used for the final ranking table.",
    )
    parser.add_argument("--device", type=str, default="", help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_stage2_rounds", type=int, default=None, help="Optional cap for quick debugging.")
    parser.add_argument("--with_visuals", action="store_true", help="Keep stage visualizations enabled.")
    parser.add_argument("--dry_run", action="store_true", help="Print the matrix without training.")
    parser.add_argument("--list", action="store_true", help="List ablation names and exit.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ablations = _select_ablations(args.suite, args.only)
    seeds = _parse_seeds(args.seeds)

    if args.list:
        for ablation in ablations:
            print(f"{ablation.name}: {ablation.description}")
        return

    _print_matrix(ablations, seeds, args)
    if args.dry_run:
        return

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {
        "started_at": started_at,
        "suite": args.suite,
        "only": args.only,
        "seeds": seeds,
        "results_root": str(results_root),
        "focus_scores": list(FOCUS_SCORES),
        "focus_metrics": list(FOCUS_METRICS),
        "ablations": [
            {
                "name": item.name,
                "description": item.description,
                "overrides": item.overrides,
            }
            for item in ablations
        ],
    }
    _write_json(results_root / f"ablation_manifest_{started_at}.json", manifest)

    rows: List[Dict[str, object]] = []
    for ablation in ablations:
        for seed in seeds:
            print("\n" + "=" * 88)
            print(f"Running ablation: {ablation.name} | seed={seed}")
            print("=" * 88)
            overrides = _build_effective_overrides(ablation, seed=seed, args=args)
            config = build_dataset_config("Genesis", overrides=overrides)
            run_name = f"{args.run_prefix}_{ablation.name}_"
            from utils.run_entry import run_training

            output = run_training(
                config=config,
                run_name=run_name,
                results_root=str(results_root),
                experiment_name="Genesis",
                show_visualization_artifacts=bool(args.with_visuals),
            )
            row = {
                "variant": ablation.name,
                "seed": int(seed),
                "run_dir": output["run_dir"],
                "description": ablation.description,
            }
            row.update(_extract_focus_summary(output["results"]))
            rows.append(row)

            _write_csv(results_root / f"ablation_summary_{started_at}.csv", rows)
            _write_json(results_root / f"ablation_summary_{started_at}.json", rows)

    print("\nAblation finished.")
    _print_final_ranking(rows, rank_by=args.rank_by)
    print(f"Summary CSV:  {results_root / f'ablation_summary_{started_at}.csv'}")
    print(f"Summary JSON: {results_root / f'ablation_summary_{started_at}.json'}")


if __name__ == "__main__":
    main()
