import json
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, Optional

import torch

from trainer.solver import Solver
from utils.config import Config, apply_model_defaults, apply_stage2_method_defaults
from utils.io_utils import save_config_artifact, save_result_artifacts


def _as_config(config) -> Config:
    return config if isinstance(config, Config) else Config.from_dict(config)


def _as_override_dict(config) -> Dict[str, object]:
    if config is None:
        return {}
    if isinstance(config, Config):
        return config.to_dict()
    return dict(config)


class TeeStream:
    """Mirror output to console and log, but keep progress-bar refreshes out of log files."""

    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream

    def write(self, data):
        self.console_stream.write(data)
        self.console_stream.flush()

        if "\r" in data:
            return

        self.log_stream.write(data)
        self.log_stream.flush()

    def flush(self):
        self.console_stream.flush()
        self.log_stream.flush()

    def isatty(self):
        return bool(getattr(self.console_stream, "isatty", lambda: False)())


_RUN_NAME_PATTERN = re.compile(r"^(?P<prefix>.*?)(?P<index>\d+)?$")


def _split_run_name(run_name: str):
    run_name = str(run_name).strip()
    if not run_name:
        raise ValueError("RUN_NAME cannot be empty.")

    match = _RUN_NAME_PATTERN.fullmatch(run_name)
    if match is None:
        raise ValueError(f"Invalid RUN_NAME: {run_name}")

    prefix = match.group("prefix") or ""
    index = match.group("index")
    return prefix, index


def _list_matching_run_indices(results_root: str, prefix: str):
    if not os.path.exists(results_root):
        return []

    indices = []
    for entry in os.listdir(results_root):
        full_path = os.path.join(results_root, entry)
        if not os.path.isdir(full_path):
            continue

        if entry == prefix:
            indices.append(0)
            continue

        if not entry.startswith(prefix):
            continue

        suffix = entry[len(prefix):]
        if suffix.isdigit():
            indices.append(int(suffix))

    return sorted(indices)


def build_run_dir(run_name: str, results_root: str = "results", digits: int = 3) -> str:
    prefix, explicit_index = _split_run_name(run_name)

    if explicit_index is not None:
        final_name = run_name
    else:
        existing_indices = _list_matching_run_indices(results_root, prefix)
        next_index = max(existing_indices, default=0) + 1
        final_name = f"{prefix}{next_index:0{digits}d}"

    run_dir = os.path.join(results_root, final_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def merge_configs(base_config: Config, overrides: Optional[Dict[str, object]] = None) -> Config:
    merged = base_config.to_dict()
    if overrides:
        merged.update(dict(overrides))
    merged = apply_stage2_method_defaults(merged, explicit_overrides=overrides)
    merged = apply_model_defaults(merged)
    return Config.from_dict(merged)


def load_json_overrides(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Config override JSON should be an object.")
    return payload


def print_score_family_results(results: dict):
    separated_labels = [
        ("view1_proto_dist", "View1 prototype-distance score"),
        ("view1_recon", "View1 reconstruction score"),
        ("view2_proto_dist", "View2 prototype-distance score"),
        ("view2_recon", "View2 reconstruction score"),
        ("proto_dist_gap", "Cross-view prototype-distance gap score"),
        ("js_conflict", "Cross-view JS/conflict score"),
    ]
    component_families = results.get("component_families", {})
    for key, label in separated_labels:
        family = component_families.get(key, {})
        if not family:
            continue
        threshold = family.get("threshold")
        if threshold is None:
            print(f"\n{label}:")
        else:
            print(f"\n{label} (threshold={float(threshold):.6f}):")
        for metric_key, value in family.get("metrics", {}).items():
            print(f"{metric_key}: {float(value):.6f}")

    composite_labels = [
        ("center_distance", "View1 combined evidence score (prototype distance + reconstruction)"),
        ("reconstruction", "View2 combined evidence score (prototype distance + reconstruction)"),
        ("cross_view", "Cross-view evidence disagreement score"),
    ]
    for key, label in composite_labels:
        family = results.get(key, {})
        if not family:
            continue
        threshold = family.get("threshold")
        if threshold is None:
            print(f"\n{label}:")
        else:
            print(f"\n{label} (threshold={float(threshold):.6f}):")
        for metric_key, value in family.get("metrics", {}).items():
            print(f"{metric_key}: {float(value):.6f}")


def print_visualization_artifacts(run_dir: str, experiment_name: str):
    vis_dir = os.path.join(run_dir, "visualizations")
    print(f"\n========== {experiment_name} Visualization Artifacts ==========")
    if not os.path.isdir(vis_dir):
        print(f"No visualization directory found: {os.path.abspath(vis_dir)}")
        return

    image_names = sorted(
        name
        for name in os.listdir(vis_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg"))
    )
    if not image_names:
        print(f"No visualization images found under: {os.path.abspath(vis_dir)}")
        return

    print(
        f"[Visualize] Saved {len(image_names)} artifact(s) under: "
        f"{os.path.abspath(vis_dir)}"
    )


def _resolved_window_steps(config: Config):
    base_step = max(1, int(getattr(config, "step", 1)))
    train_step = int(getattr(config, "train_step", -1))
    test_step = int(getattr(config, "test_step", -1))
    return (
        train_step if train_step > 0 else base_step,
        test_step if test_step > 0 else base_step,
    )


def _encoder_config_summary(config: Config) -> str:
    active_view = str(getattr(config, "active_view", "v1")).strip().lower()
    if active_view == "dual":
        channels = int(getattr(config, "in_channels", 0))
        history_len = int(getattr(config, "dual_history_len", 20))
        current_out = int(getattr(config, "dual_current_out", 8))
        short_out = int(getattr(config, "dual_short_out", 16))
        long_out = int(getattr(config, "dual_long_out", 16))
        view1_dim = 3 * channels + history_len
        view2_dim = current_out + short_out + long_out
        return (
            "pointwise_dual | "
            f"L={history_len} | "
            f"view1_dim={view1_dim} | "
            f"view2_out=({current_out},{short_out},{long_out})"
            f"->{view2_dim} | "
            f"view2_kernels=({channels},{(history_len // 2) * channels},{history_len * channels}) | "
            "reconstruction=full_window"
        )
    return (
        f"tcn_kernel={getattr(config, 'tcn_kernel_size', 3)} | "
        f"v2_first_kernel={getattr(config, 'v2_first_kernel_size', 0)}"
    )


def _encoder_backbone_summary(config: Config) -> str:
    active_view = str(getattr(config, "active_view", "v1")).strip().lower()
    if active_view == "dual":
        return f"d_model={getattr(config, 'latent_dim', 0)}"
    return (
        f"tcn_layers={config.tcn_layers} | "
        f"latent_dim={config.latent_dim}"
    )


def print_train_summary(config: Config, run_dir: str, run_name: str, experiment_name: str):
    train_step, test_step = _resolved_window_steps(config)
    stage1_shift = int(getattr(config, "stage1_positive_offset", 1)) * int(train_step)
    print(f"Results directory: {os.path.abspath(run_dir)}")
    print(f"Run name: {run_name}")
    print(f"Dataset: {config.dataset}")
    print(f"Experiment: {experiment_name}")
    print(f"Data path: {os.path.abspath(config.data_path)}")
    print(
        "Data split: "
        f"train_mode={getattr(config, 'train_split_mode', 'train')} | "
        f"test_mode={getattr(config, 'test_split_mode', 'test')} | "
        f"scaler_fit_mode={getattr(config, 'scaler_fit_mode', 'train')}"
    )
    print(f"Device: {config.device}")
    print(
        "Data/Model: "
        f"seq_len={config.seq_len} | "
        f"step={config.step} | "
        f"train_step={train_step} | "
        f"test_step={test_step} | "
        f"in_channels={config.in_channels} | "
        f"active_view={getattr(config, 'active_view', 'v1')} | "
        f"dual_feature={getattr(config, 'dual_view_feature_mode', 'avg')} | "
        f"dual_view_evidence=(center={getattr(config, 'dual_view_center_weight', 1.0)}, recon={getattr(config, 'dual_view_recon_weight', 0.5)}) | "
        f"{_encoder_config_summary(config)} | "
        f"{_encoder_backbone_summary(config)}"
    )
    print(
        "Training: "
        f"epochs=({config.epoch_stage0}, {config.epoch_stage1}, {config.epoch_stage2}) | "
        f"batch_size={config.batch_size} | "
        f"lr={config.lr} | "
        f"seed={config.seed} | "
        f"num_workers={config.num_workers} | "
        f"cache_windows={getattr(config, 'cache_windows', False)} | "
        f"pin_memory={getattr(config, 'pin_memory', True)} | "
        f"tf32={getattr(config, 'enable_tf32', True)}"
    )
    print(
        "Stage1: "
        "mode=anchor_recon+neighbor_point_context | "
        f"positive_direction={getattr(config, 'stage1_positive_direction', 'past')} | "
        f"positive_offset={getattr(config, 'stage1_positive_offset', 1)} | "
        f"raw_shift={stage1_shift} | "
        f"lambda_ctx={getattr(config, 'lambda_ctx_stage1', 0.05)}"
    )
    print(
        "Stage2: "
        f"method={getattr(config, 'stage2_method', 'separate_proto')} | "
        f"rounds={config.num_stage2_rounds if config.num_stage2_rounds > 0 else 'auto'} | "
        f"epochs_per_round={config.epochs_per_stage2_round if config.epochs_per_stage2_round > 0 else 'auto'} | "
        "refresh_unit=epoch | "
        f"num_prototypes={getattr(config, 'num_prototypes', 0)} | "
        f"state_dim={getattr(config, 'state_dim', 0)} | "
        f"tau_conf={getattr(config, 'tau_conf', 0.7)} | "
        f"joint_core={getattr(config, 'joint_core_mode', 'minimal')} | "
        f"active_pool_trim={getattr(config, 'active_pool_trim_enabled', False)} "
        f"(stage0={getattr(config, 'active_pool_trim_stage0_ratio', 0.0)}, "
        f"stage1={getattr(config, 'active_pool_trim_stage1_ratio', 0.0)})"
    )
    print(
        "Loss/Score: "
        f"lambda=(stage2_rec={config.stage2_lambda_rec}, "
        f"state={getattr(config, 'lambda_state_consistency', 1.0)}, "
        f"pull={getattr(config, 'lambda_proto_pull', 1.0)}, "
        f"repulse={getattr(config, 'lambda_proto_repulsion', 1.0)}) | "
        f"js={getattr(config, 'lambda_js_score', 1.0)} | "
        f"proto_recon={getattr(config, 'prototype_recon_weight', 0.5)} | "
        f"threshold_q={config.decision_quantile}"
    )
    print(
        "Visualization: "
        f"enabled={config.enable_stage_visualization} | "
        f"method={config.visualization_method} | "
        f"output_dir={os.path.abspath(os.path.join(run_dir, 'visualizations'))}"
    )


def print_eval_summary(
    config: Config,
    run_dir: str,
    run_name: str,
    checkpoint_path: str,
    experiment_name: str,
):
    train_step, test_step = _resolved_window_steps(config)
    print(f"Results directory: {os.path.abspath(run_dir)}")
    print(f"Run name: {run_name}")
    print(f"Dataset: {config.dataset}")
    print(f"Experiment: {experiment_name}")
    print(f"Checkpoint: {os.path.abspath(checkpoint_path)}")
    print(f"Data path: {os.path.abspath(config.data_path)}")
    print(f"Device: {config.device}")
    print(
        "Data/Model: "
        f"seq_len={config.seq_len} | "
        f"step={config.step} | "
        f"train_step={train_step} | "
        f"test_step={test_step} | "
        f"in_channels={config.in_channels} | "
        f"active_view={getattr(config, 'active_view', 'v1')} | "
        f"dual_feature={getattr(config, 'dual_view_feature_mode', 'avg')} | "
        f"dual_view_evidence=(center={getattr(config, 'dual_view_center_weight', 1.0)}, recon={getattr(config, 'dual_view_recon_weight', 0.5)}) | "
        f"{_encoder_config_summary(config)} | "
        f"{_encoder_backbone_summary(config)}"
    )
    print(
        "Evaluation: "
        f"method={getattr(config, 'stage2_method', 'separate_proto')} | "
        f"num_prototypes={getattr(config, 'num_prototypes', 0)} | "
        f"threshold_q={config.decision_quantile} | "
        f"js={getattr(config, 'lambda_js_score', 1.0)} | "
        f"proto_recon={getattr(config, 'prototype_recon_weight', 0.5)}"
    )


def load_checkpoint_config(checkpoint_path: str) -> Config:
    checkpoint_path = os.path.abspath(checkpoint_path)
    sidecar_config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    if os.path.exists(sidecar_config_path):
        with open(sidecar_config_path, "r", encoding="utf-8") as file:
            return Config.from_dict(json.load(file))

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return Config.from_dict(payload.get("config", {}))


def run_training(
    config,
    run_name: str,
    results_root: str,
    experiment_name: str,
    show_visualization_artifacts: bool = False,
):
    config = _as_config(config)
    run_dir = build_run_dir(run_name, results_root=results_root)
    config.save_dir = run_dir
    resolved_run_name = os.path.basename(run_dir)

    log_path = os.path.join(run_dir, "run.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = TeeStream(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            print_train_summary(config, run_dir, resolved_run_name, experiment_name)
            save_config_artifact(run_dir, config)

            solver = Solver(config)
            results = solver.train_pipeline()
            save_result_artifacts(run_dir, results)

            if show_visualization_artifacts:
                print_visualization_artifacts(run_dir, experiment_name)

            print(f"\n========== {experiment_name} Final Results ==========")
            for key, value in results["metrics"].items():
                print(f"{key}: {value:.6f}")
            print_score_family_results(results)
            print(f"\nRun artifacts saved under: {os.path.abspath(run_dir)}")

    return {"run_dir": run_dir, "results": results}


def run_evaluation(
    checkpoint_path: str,
    config=None,
    run_name: str = "eval",
    results_root: Optional[str] = None,
    experiment_name: str = "Evaluation",
    show_visualization_artifacts: bool = False,
):
    checkpoint_path = os.path.abspath(checkpoint_path)
    checkpoint_config = load_checkpoint_config(checkpoint_path)
    if config is None:
        effective_config = checkpoint_config
    else:
        effective_config = merge_configs(checkpoint_config, _as_override_dict(config))

    if results_root is None:
        results_root = os.path.dirname(checkpoint_path)

    run_dir = build_run_dir(run_name, results_root=results_root)
    effective_config.save_dir = run_dir
    resolved_run_name = os.path.basename(run_dir)

    log_path = os.path.join(run_dir, "eval.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = TeeStream(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            print_eval_summary(effective_config, run_dir, resolved_run_name, checkpoint_path, experiment_name)
            save_config_artifact(run_dir, effective_config)

            solver = Solver(effective_config)
            solver.load_checkpoint(checkpoint_path)
            results = solver.test()
            save_result_artifacts(run_dir, results)

            if show_visualization_artifacts:
                print_visualization_artifacts(run_dir, experiment_name)

            print(f"\n========== {experiment_name} Eval Results ==========")
            for key, value in results["metrics"].items():
                print(f"{key}: {value:.6f}")
            print_score_family_results(results)
            print(f"\nEval artifacts saved under: {os.path.abspath(run_dir)}")

    return {"run_dir": run_dir, "results": results}
