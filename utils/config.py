import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple


@dataclass
class Config:
    """Experiment configuration shared by dataset entrypoints and Solver."""

    # Data and model
    dataset: str = "SKAB"
    data_path: str = "dataset/SKAB"
    train_split_mode: str = "train"
    test_split_mode: str = "test"
    scaler_fit_mode: str = "train"
    entity_id: str = ""
    spacecraft: str = ""
    metadata_path: str = ""
    step: int = 1
    train_step: int = 1
    test_step: int = 1
    # New dual-view route: seq_len is the local history length L itself.
    seq_len: int = 20
    in_channels: int = 8
    tcn_layers: Tuple[int, ...] = (64, 128, 128)
    latent_dim: int = 64
    tcn_kernel_size: int = 3
    # Legacy single-view flatten path only.
    v2_first_kernel_size: int = 0
    tcn_dropout: float = 0.1
    tcn_activation: str = "relu"
    use_attentive_pooling: bool = False
    active_view: str = "v1"
    dual_view_feature_mode: str = "avg"
    stage2_method: str = "separate_proto"
    state_dim: int = 0
    num_prototypes: int = 0
    proto_temperature: float = 0.2
    proto_separation_margin: float = 1.0
    stage2_injected_margin: float = 1.0
    active_pool_trim_enabled: bool = False
    active_pool_trim_stage0_ratio: float = 0.0
    active_pool_trim_stage1_ratio: float = 0.0

    # Training schedule
    epoch_stage0: int = 10
    epoch_stage1: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # Stage 2 prototype training
    num_stage2_rounds: int = 3
    stage2_a_epochs: int = 1
    stage2_b_epochs: int = 1

    # Reconstruction and anomaly score composition
    lambda_rec: float = 1.0
    decision_quantile: float = 0.995
    robust_eps: float = 1e-6

    # Runtime and artifacts
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"
    save_dir: str = "results"
    checkpoint_name: str = "model_last.pt"
    cache_windows: bool = False
    left_pad_windows: bool = True
    pin_memory: bool = True
    enable_tf32: bool = True
    cudnn_benchmark: bool = True
    stage1_positive_offset: int = 1
    stage1_inject_context_len: int = 0
    lambda_stage1_triplet: float = 1.0
    stage1_ap_margin: float = 0.1
    stage1_triplet_margin: float = 0.3
    stage2_inject_context_len: int = 0
    core_ratio_A: float = 0.3
    min_core_per_proto: int = 1
    lambda_pull_A: float = 1.0
    lambda_sep_A: float = 0.1
    core_ratio_B: float = 0.5
    lambda_rec_B: float = 1.0
    lambda_ap_B: float = 0.2
    lambda_core_B: float = 0.5
    lambda_neg_B: float = 0.05
    stage2_ap_margin: float = 0.1
    boundary_quantile: float = 0.95
    negative_boundary_margin: float = 0.1
    use_negative_boundary_radius: bool = True
    enable_stage1_recon_scoring: bool = False
    enable_stage_visualization: bool = False
    visualization_max_points: int = 3000
    visualization_method: str = "tsne"
    visualization_tsne_perplexity: float = 30.0
    visualization_tsne_init: str = "pca"
    visualization_umap_n_neighbors: int = 30
    visualization_umap_min_dist: float = 0.1
    show_batch_progress: bool = False

    def to_dict(self) -> Dict:
        """Return a plain dict for artifact saving."""
        return asdict(self)

    @classmethod
    def from_dict(cls, kwargs: Dict) -> "Config":
        """Build Config from a dict and ignore unknown keys."""
        field_names = {item.name for item in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in kwargs.items() if k in field_names}
        return cls(**filtered)


DEFAULT_CONFIG: Dict = Config().to_dict()


def _default_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


_COMMON_EXPLICIT: Dict[str, object] = {
    "seq_len": 20,
    "step": 1,
    "train_step": 1,
    "test_step": 1,
    "latent_dim": 64,
    "tcn_kernel_size": 3,
    "v2_first_kernel_size": 0,
    "tcn_dropout": 0.1,
    "tcn_activation": "relu",
    "epoch_stage0": 10,
    "epoch_stage1": 10,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "num_stage2_rounds": 3,
    "stage2_a_epochs": 1,
    "stage2_b_epochs": 1,
    "stage2_method": "separate_proto",
    "state_dim": 0,
    "num_prototypes": 0,
    "proto_temperature": 0.2,
    "proto_separation_margin": 1.0,
    "stage2_injected_margin": 1.0,
    "active_pool_trim_enabled": False,
    "active_pool_trim_stage0_ratio": 0.0,
    "active_pool_trim_stage1_ratio": 0.0,
    "lambda_rec": 1.0,
    "decision_quantile": 0.995,
    "seed": 42,
    "device": _default_device(),
    "save_dir": "results",
    "checkpoint_name": "model_last.pt",
    "cache_windows": False,
    "left_pad_windows": True,
    "pin_memory": True,
    "enable_tf32": True,
    "cudnn_benchmark": True,
    "stage1_positive_offset": 1,
    "stage1_inject_context_len": 0,
    "lambda_stage1_triplet": 1.0,
    "stage1_ap_margin": 0.1,
    "stage1_triplet_margin": 0.3,
    "stage2_inject_context_len": 0,
    "core_ratio_A": 0.3,
    "min_core_per_proto": 1,
    "lambda_pull_A": 1.0,
    "lambda_sep_A": 0.1,
    "core_ratio_B": 0.5,
    "lambda_rec_B": 1.0,
    "lambda_ap_B": 0.2,
    "lambda_core_B": 0.5,
    "lambda_neg_B": 0.05,
    "stage2_ap_margin": 0.1,
    "boundary_quantile": 0.95,
    "negative_boundary_margin": 0.1,
    "use_negative_boundary_radius": True,
    "enable_stage_visualization": True,
    "visualization_max_points": 3000,
    "visualization_method": "tsne",
    "visualization_tsne_perplexity": 35.0,
}


_DATASET_PRESETS: Dict[str, Dict[str, object]] = {
    "SKAB": {
        "dataset": "SKAB",
        "data_path": os.path.join("dataset", "SKAB"),
        "seq_len": 20,
        "step": 1,
        "in_channels": 8,
        "tcn_layers": (64, 128, 128),
        "latent_dim": 64,
        "use_attentive_pooling": True,
        "epoch_stage0": 10,
        "epoch_stage1": 10,
        "batch_size": 128,
        "seed": 42,
        "num_workers": 8,
        "num_prototypes": 10,
    },
    "SMAP": {
        "dataset": "SMAP",
        "data_path": os.path.join("dataset", "SMAP"),
        "seq_len": 20,
        "step": 1,
        "in_channels": 25,
        "tcn_layers": (64, 128, 256, 256, 256),
        "latent_dim": 96,
        "use_attentive_pooling": True,
        "epoch_stage0": 10,
        "epoch_stage1": 10,
        "batch_size": 256,
        "seed": 42,
        "num_workers": 4,
        "num_prototypes": 50,
    },
    "GECCO": {
        "dataset": "GECCO",
        "data_path": os.path.join("dataset", "GECCO"),
        "seq_len": 20,
        "step": 1,
        "in_channels": 9,
        "tcn_layers": (64, 128, 128),
        "latent_dim": 64,
        "use_attentive_pooling": True,
        "epoch_stage0": 10,
        "epoch_stage1": 10,
        "batch_size": 256,
        "seed": 42,
        "num_workers": 4,
        "num_prototypes": 30,
    },
    "Genesis": {
        "dataset": "Genesis",
        "data_path": os.path.join("dataset", "Genesis"),
        "seq_len": 20,
        "step": 1,
        "in_channels": 18,
        "tcn_layers": (64, 128, 128),
        "latent_dim": 96,
        "use_attentive_pooling": True,
        "epoch_stage0": 10,
        "epoch_stage1": 10,
        "batch_size": 128,
        "seed": 42,
        "num_workers": 8,
        "num_prototypes": 10,
    },
    "PUMP": {
        "dataset": "PUMP",
        "data_path": os.path.join("dataset", "PUMP"),
        "seq_len": 20,
        "step": 1,
        "in_channels": 51,
        "tcn_layers": (64, 128, 128),
        "latent_dim": 192,
        "use_attentive_pooling": True,
        "epoch_stage0": 10,
        "epoch_stage1": 10,
        "batch_size": 256,
        "seed": 42,
        "num_workers": 4,
        "num_prototypes": 10,
    },
    "PSM": {
        "dataset": "PSM",
        "data_path": os.path.join("dataset", "PSM"),
        "seq_len": 20,
        "step": 1,
        "in_channels": 25,
        "tcn_layers": (64, 128, 128),
        "latent_dim": 96,
        "use_attentive_pooling": True,
        "epoch_stage0": 10,
        "epoch_stage1": 10,
        "batch_size": 256,
        "seed": 42,
        "num_workers": 4,
        "num_prototypes": 10,
    },
    "SMD": {
        "dataset": "SMD",
        "data_path": os.path.join("dataset", "SMD"),
        "seq_len": 20,
        "step": 1,
        "in_channels": 38,
        "tcn_layers": (64, 128, 256),
        "latent_dim": 160,
        "use_attentive_pooling": True,
        "epoch_stage0": 10,
        "epoch_stage1": 10,
        "batch_size": 256,
        "seed": 42,
        "num_workers": 4,
        "num_prototypes": 300,
    },
}


_DATASET_META: Dict[str, Dict[str, object]] = {
    "SKAB": {
        "run_name": "skab_experiment",
        "results_root": os.path.join("results", "skab"),
        "print_visualization_artifacts": False,
    },
    "SMAP": {
        "run_name": "smap_experiment",
        "results_root": os.path.join("results", "smap"),
        "print_visualization_artifacts": True,
    },
    "GECCO": {
        "run_name": "gecco_experiment",
        "results_root": os.path.join("results", "gecco"),
        "print_visualization_artifacts": False,
    },
    "Genesis": {
        "run_name": "genesis_experiment",
        "results_root": os.path.join("results", "genesis"),
        "print_visualization_artifacts": True,
    },
    "PUMP": {
        "run_name": "pump_experiment",
        "results_root": os.path.join("results", "pump"),
        "print_visualization_artifacts": False,
    },
    "PSM": {
        "run_name": "psm_experiment",
        "results_root": os.path.join("results", "psm"),
        "print_visualization_artifacts": False,
    },
    "SMD": {
        "run_name": "smd_experiment",
        "results_root": os.path.join("results", "smd"),
        "print_visualization_artifacts": False,
    },
}


_STAGE2_METHOD_DEFAULTS: Dict[str, Dict[str, object]] = {
    "separate_proto": {
        "num_stage2_rounds": 3,
        "stage2_a_epochs": 1,
        "stage2_b_epochs": 1,
        "core_ratio_A": 0.3,
        "min_core_per_proto": 1,
        "lambda_pull_A": 1.0,
        "lambda_sep_A": 0.1,
        "core_ratio_B": 0.5,
        "lambda_rec_B": 1.0,
        "lambda_ap_B": 0.2,
        "lambda_core_B": 0.5,
        "lambda_neg_B": 0.05,
        "stage2_ap_margin": 0.1,
        "boundary_quantile": 0.95,
        "negative_boundary_margin": 0.1,
        "use_negative_boundary_radius": True,
    },
}


def _normalize_stage2_method(method: object) -> str:
    normalized = str(method or "separate_proto").strip().lower()
    if normalized != "separate_proto":
        raise ValueError("Only stage2_method='separate_proto' is supported.")
    return normalized


def apply_stage2_method_defaults(
    values: Dict[str, object],
    explicit_overrides: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Apply defaults owned by the selected Stage2 prototype method."""
    merged = dict(values)
    method = _normalize_stage2_method(merged.get("stage2_method", "separate_proto"))
    merged["stage2_method"] = method
    defaults = _STAGE2_METHOD_DEFAULTS.get(method, {})
    explicit_keys = set((explicit_overrides or {}).keys())
    for key, value in defaults.items():
        if key not in explicit_keys:
            merged[key] = value
    return merged


def apply_model_defaults(values: Dict[str, object]) -> Dict[str, object]:
    """Resolve model fields that intentionally follow other model fields."""
    merged = dict(values)
    if int(merged.get("state_dim", 0) or 0) <= 0:
        merged["state_dim"] = int(merged.get("latent_dim", 0) or 0)
    return merged


def available_datasets() -> Tuple[str, ...]:
    return tuple(_DATASET_PRESETS.keys())


def _canonical_dataset_name(dataset: str) -> str:
    normalized = str(dataset).strip().lower()
    for key, preset in _DATASET_PRESETS.items():
        if normalized in {key.lower(), str(preset["dataset"]).strip().lower()}:
            return key
    raise KeyError(f"Unsupported dataset preset: {dataset}")


def build_dataset_config(dataset: str, overrides: Optional[Dict[str, object]] = None) -> Config:
    key = _canonical_dataset_name(dataset)
    merged = dict(_COMMON_EXPLICIT)
    merged.update(_DATASET_PRESETS[key])
    if overrides:
        merged.update(dict(overrides))
    merged = apply_stage2_method_defaults(merged, explicit_overrides=overrides)
    merged = apply_model_defaults(merged)
    return Config.from_dict(merged)


def dataset_entry_meta(dataset: str) -> Dict[str, object]:
    key = _canonical_dataset_name(dataset)
    return dict(_DATASET_META[key])
