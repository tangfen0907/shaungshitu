"""Shape smoke tests for every runnable preset script.

This script checks the axis-view dual encoder contract without starting a full
training run:

* script dataset/data_path/in_channels/seq_len configuration
* one real DataLoader batch converted to [B, N, T]
* random model forward with stage="stage2"
* dual outputs: z1/z2/u1/u2 [B, D], q1/q2 [B, K],
  proto_dist_matrix1/2 [B, K], proto_dist1/2 [B], x_hat1/2 [B, N, 1]

Note: proto_dist1/proto_dist2 are the existing min-distance score fields with
shape [B]. The full per-prototype distances live in proto_dist_matrix1/2.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_factory.data_loader import get_loader_segment
from model.axis_view_encoder import PatchRelationDualEncoder
from model.main_net import AnomalyDetector
from utils.config import Config, build_dataset_config


@dataclass
class PresetCase:
    script: str
    dataset_key: str
    overrides: dict[str, Any]
    note: str = ""


DATASET_TRAIN_FILES = {
    "SKAB": "SKAB_train.npy",
    "SMAP": "SMAP_train.npy",
    "SMD": "SMD_train.npy",
    "PSM": "PSM_train.npy",
    "GECCO": "GECCO_train.npy",
    "Genesis": "Genesis_train.npy",
    "PUMP": "PUMP_train.npy",
    "SWAT": "SWaT_train.npy",
    "WaDi": "WaDi_train.npy",
    "HAI": "HAI_train.npy",
    "MSL": "MSL_train.npy",
}

STATIC_SCRIPT_DATASETS = {
    "main_SKAB.py": "SKAB",
    "main_SMAP.py": "SMAP",
    "main_SMD.py": "SMD",
    "main_PSM.py": "PSM",
    "main_GECCO.py": "GECCO",
    "main_Genesis.py": "Genesis",
    "main_Genesis_no_stage2_recon.py": "Genesis",
    "main_Genesis_testtrain.py": "Genesis",
    "main_PUMP.py": "PUMP",
}

EXPECTED_BUT_ABSENT = [
    "main_MSL.py",
    "main_SWaT.py",
    "main_WADI.py",
    "main_HAI.py",
]


def _module_name(script: str) -> str:
    return script[:-3] if script.endswith(".py") else script


def _load_script_case(script: str, dataset_key: str) -> PresetCase:
    module = importlib.import_module(_module_name(script))
    overrides = dict(getattr(module, "LOCAL_CONFIG_OVERRIDES", {}))
    return PresetCase(script=script, dataset_key=dataset_key, overrides=overrides)


def _load_genesis_v1_ablation_case() -> PresetCase:
    from main_Genesis import LOCAL_CONFIG_OVERRIDES

    overrides = dict(LOCAL_CONFIG_OVERRIDES)
    overrides.update(
        {
            "active_view": "v1",
            "stage2_method": "separate_proto",
            "lambda_cv_stage0": 0.0,
            "lambda_cv_stage1": 0.0,
            "lambda_cv_stage2": 0.0,
            "num_prototypes": 5,
            "enable_stage_visualization": False,
            "enable_stage1_plotly_visualization": False,
            "enable_stage2_train_plotly_visualization": False,
            "enable_stage1_recon_scoring": False,
            "visualization_max_points": 0,
            "device": "cpu",
            "num_workers": 0,
        }
    )
    return PresetCase(
        script="run_genesis_v1_ablation.py",
        dataset_key="Genesis",
        overrides=overrides,
        note="single-view v1 ablation",
    )


def _load_genesis_eval_case() -> Optional[PresetCase]:
    module = importlib.import_module("main_Genesis_eval")
    try:
        checkpoint_path = module.resolve_checkpoint_path()
        config = module.build_eval_config(checkpoint_path)
    except Exception as exc:
        print(f"[SKIP] main_Genesis_eval.py | checkpoint config unavailable: {exc}")
        return None
    overrides = {key: value for key, value in vars(config).items() if not key.startswith("_")}
    overrides["device"] = "cpu"
    overrides["num_workers"] = 0
    return PresetCase(
        script="main_Genesis_eval.py",
        dataset_key=str(getattr(config, "dataset", "Genesis")),
        overrides=overrides,
        note="checkpoint-config forward only; old checkpoint weights are not loaded",
    )


def collect_cases() -> list[PresetCase]:
    cases = [
        _load_script_case(script, dataset_key)
        for script, dataset_key in STATIC_SCRIPT_DATASETS.items()
        if (ROOT / script).is_file()
    ]
    if (ROOT / "run_genesis_v1_ablation.py").is_file():
        cases.append(_load_genesis_v1_ablation_case())
    if (ROOT / "main_Genesis_eval.py").is_file():
        eval_case = _load_genesis_eval_case()
        if eval_case is not None:
            cases.append(eval_case)
    return cases


def _as_abs(path: Union[str, os.PathLike[str]]) -> Path:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    return path_obj


def true_variable_count(config: Config) -> int:
    dataset = str(config.dataset)
    data_path = _as_abs(config.data_path)
    train_file = DATASET_TRAIN_FILES.get(dataset)
    if train_file is None:
        raise AssertionError(f"No train-file mapping for dataset={dataset!r}")
    train_path = data_path / train_file
    if not train_path.is_file() and dataset == "SWAT":
        alt = ROOT / "dataset" / "SWaT" / train_file
        if alt.is_file():
            train_path = alt
    if not train_path.is_file():
        raise AssertionError(f"Missing train data file: {train_path}")
    arr = np.load(train_path, mmap_mode="r")
    if arr.ndim != 2:
        raise AssertionError(f"Expected 2D train array, got {arr.shape} from {train_path}")
    return int(arr.shape[1])


def _extract_window(batch: Any) -> Any:
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, dict):
        for key in ("x", "window", "data"):
            if key in batch:
                return batch[key]
    return batch


def _to_channel_first_batch(batch: Any) -> torch.Tensor:
    x = _extract_window(batch)
    if isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        tensor = x.float()
    else:
        tensor = torch.tensor(x, dtype=torch.float32)
    if tensor.dim() != 3:
        raise AssertionError(f"DataLoader batch should be 3D before model input, got {tuple(tensor.shape)}")
    if tensor.shape[1] > tensor.shape[2]:
        tensor = tensor.transpose(1, 2).contiguous()
    return tensor


def dataloader_shape(config: Config) -> tuple[int, int, int]:
    loader = get_loader_segment(
        index=0,
        data_path=str(_as_abs(config.data_path)),
        batch_size=2,
        win_size=int(config.seq_len),
        step=int(getattr(config, "step", 1)),
        mode=str(getattr(config, "train_split_mode", "train")),
        dataset=str(config.dataset),
        entity_id=str(getattr(config, "entity_id", "")),
        spacecraft=str(getattr(config, "spacecraft", "")),
        metadata_path=str(getattr(config, "metadata_path", "")),
        scaler_fit_mode=str(getattr(config, "scaler_fit_mode", "train")),
        cache_windows=False,
        pin_memory=False,
    )
    batch = next(iter(loader))
    tensor = _to_channel_first_batch(batch)
    return tuple(int(dim) for dim in tensor.shape)


def _make_model(config: Config) -> AnomalyDetector:
    return AnomalyDetector(
        in_channels=int(config.in_channels),
        seq_len=int(config.seq_len),
        latent_dim=int(config.latent_dim),
        tcn_layers=tuple(config.tcn_layers),
        dropout=float(config.tcn_dropout),
        tcn_kernel_size=int(config.tcn_kernel_size),
        v2_first_kernel_size=int(getattr(config, "v2_first_kernel_size", 0)),
        tcn_activation=str(config.tcn_activation),
        use_attentive_pooling=bool(config.use_attentive_pooling),
        active_view=str(getattr(config, "active_view", "v1")),
        dual_view_feature_mode=str(getattr(config, "dual_view_feature_mode", "avg")),
        state_dim=int(getattr(config, "state_dim", 0)),
        num_prototypes=int(getattr(config, "num_prototypes", 0) or 1),
        proto_temperature=float(getattr(config, "proto_temperature", 0.2)),
        stage2_method=str(getattr(config, "stage2_method", "separate_proto")),
        readout_mode=str(getattr(config, "readout_mode", "attn_topk_max")),
        topk_ratio=float(getattr(config, "topk_ratio", 0.1)),
        topk_k=int(getattr(config, "topk_k", 0)),
        patch_len=int(getattr(config, "patch_len", 16)),
        patch_stride=int(getattr(config, "patch_stride", 8)),
    )


def _assert_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    actual = tuple(int(dim) for dim in tensor.shape)
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def _smoke_dual(model: AnomalyDetector, config: Config) -> None:
    batch_size = 2
    n = int(config.in_channels)
    t = int(config.seq_len)
    d = int(config.latent_dim)
    k = int(model.num_prototypes)

    if not isinstance(model.dual_encoder, PatchRelationDualEncoder):
        raise AssertionError("dual encoder is not PatchRelationDualEncoder")
    if int(model.dual_encoder.view1_patch_encoder.patch_embedding[1].in_features) != int(
        getattr(config, "patch_len", 16)
    ):
        raise AssertionError("patch embedding input dim should be patch_len, not N")
    if any(int(v) != 1 for v in model.encoder_v1_dilations + model.encoder_v2_dilations):
        raise AssertionError("dual axis-view encoder should use dilation=1 only")
    if int(model.reconstructor_v1.out_channels) != n or int(model.reconstructor_v2.out_channels) != n:
        raise AssertionError("dual reconstructors should output N channels")

    x = torch.randn(batch_size, n, t)
    with torch.no_grad():
        out = model(x, stage="stage2")

    for key in ("z1", "z2", "u1", "u2"):
        _assert_shape(key, out[key], (batch_size, d))
    patch_count = model.dual_encoder.num_patches(t)
    _assert_shape("h1_patch", out["h1_patch"], (batch_size, n, patch_count, model.dual_encoder.d1))
    _assert_shape("h2_var", out["h2_var"], (batch_size, n, model.dual_encoder.d2))
    _assert_shape("h2_patch", out["h2_patch"], (batch_size, n, patch_count, model.dual_encoder.d2))
    for key in ("x_hat", "x_hat1", "x_hat2", "x_hat2_raw"):
        _assert_shape(key, out[key], (batch_size, n, 1))
    for key in ("q1", "q2", "proto_dist_matrix1", "proto_dist_matrix2"):
        _assert_shape(key, out[key], (batch_size, k))
    for key in ("proto_dist1", "proto_dist2"):
        _assert_shape(key, out[key], (batch_size,))


def _smoke_single(model: AnomalyDetector, config: Config) -> None:
    batch_size = 2
    n = int(config.in_channels)
    t = int(config.seq_len)
    d = int(config.latent_dim)
    k = int(model.num_prototypes)

    x = torch.randn(batch_size, n, t)
    with torch.no_grad():
        out = model(x, stage="stage2")

    for key in ("z", "u"):
        _assert_shape(key, out[key], (batch_size, d))
    _assert_shape("x_hat", out["x_hat"], (batch_size, n, 1))
    _assert_shape("q", out["q"], (batch_size, k))
    _assert_shape("proto_dist_matrix", out["proto_dist_matrix"], (batch_size, k))
    _assert_shape("proto_dist", out["proto_dist"], (batch_size,))


def smoke_model(config: Config) -> None:
    torch.set_num_threads(1)
    model = _make_model(config)
    model.eval()
    if model.is_dual_view:
        _smoke_dual(model, config)
    else:
        _smoke_single(model, config)


def build_config(case: PresetCase) -> Config:
    config = build_dataset_config(case.dataset_key, case.overrides)
    config.device = "cpu"
    config.num_workers = 0
    return config


def main() -> int:
    rows = []
    failures = []

    for case in collect_cases():
        try:
            config = build_config(case)
            actual_n = true_variable_count(config)
            if int(config.in_channels) != actual_n:
                raise AssertionError(
                    f"in_channels={config.in_channels} but true variable count is {actual_n}"
                )
            batch_shape = dataloader_shape(config)
            expected_batch = (2, actual_n, int(config.seq_len))
            if batch_shape != expected_batch:
                raise AssertionError(f"DataLoader x shape expected {expected_batch}, got {batch_shape}")
            smoke_model(config)
            note = case.note
            if str(getattr(config, "active_view", "v1")).strip().lower() == "dual" and int(
                getattr(config, "v2_first_kernel_size", 0)
            ) != 0:
                note = (note + "; " if note else "") + "v2_first_kernel_size is legacy flatten-only"
            rows.append(
                {
                    "script": case.script,
                    "dataset": str(config.dataset),
                    "in_channels": int(config.in_channels),
                    "seq_len": int(config.seq_len),
                    "status": "PASS",
                    "issue": note or "none",
                }
            )
            print(
                f"[PASS] {case.script} | dataset={config.dataset} | "
                f"N={config.in_channels} | T={config.seq_len} | batch={batch_shape}"
            )
        except Exception as exc:
            failures.append((case.script, exc))
            rows.append(
                {
                    "script": case.script,
                    "dataset": case.dataset_key,
                    "in_channels": "-",
                    "seq_len": "-",
                    "status": "FAIL",
                    "issue": str(exc),
                }
            )
            print(f"[FAIL] {case.script} | {exc}")

    for script in EXPECTED_BUT_ABSENT:
        if not (ROOT / script).is_file():
            rows.append(
                {
                    "script": script,
                    "dataset": "-",
                    "in_channels": "-",
                    "seq_len": "-",
                    "status": "MISSING",
                    "issue": "no runnable script in this repository",
                }
            )
            print(f"[MISSING] {script} | no runnable script in this repository")

    print("\n| Script | Dataset | in_channels | seq_len | Smoke | Issue |")
    print("| -- | -- | --: | --: | -- | -- |")
    for row in rows:
        print(
            f"| {row['script']} | {row['dataset']} | {row['in_channels']} | "
            f"{row['seq_len']} | {row['status']} | {row['issue']} |"
        )

    if failures:
        print("\nFailures:")
        for script, exc in failures:
            print(f"- {script}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
