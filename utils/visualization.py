from typing import Dict, List, Optional, Tuple

import os

import numpy as np

from utils.scoring import compute_separate_proto_anomaly_score
__all__ = ['_subsample_indices', '_build_train_window_anomaly_counts_for_visualization', '_build_train_window_anomaly_flags_for_visualization', '_has_valid_window_labels', '_project_visual_features_nd', '_visual_sample_indices', '_visual_state_array', '_reference_arrays_from_state', '_mask_from_state', '_labels_from_state', '_cluster_display_id', '_save_train_3d_visualization', '_save_dual_truth_visualizations', '_save_train_test_score_3d_visualization', '_save_stage2_component_score_visualizations']


@staticmethod
def _subsample_indices(num_items: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    num_items = int(num_items)
    max_points = max(1, int(max_points))
    if num_items <= max_points:
        return np.arange(num_items, dtype=np.int64)
    return np.sort(rng.choice(num_items, size=max_points, replace=False)).astype(np.int64)


def _dataset_chain_attr(dataset, name: str, default=None):
    seen = set()
    current = dataset
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, name):
            value = getattr(current, name)
            if value is not None:
                return value
        current = getattr(current, "base_dataset", None)
    return default


def _dataset_chain_has_train_labels(dataset) -> bool:
    return _dataset_chain_attr(dataset, "train_labels", None) is not None


def _build_train_window_anomaly_counts_for_visualization(self, dataset) -> np.ndarray:
    dataset_mode = str(_dataset_chain_attr(dataset, "mode", "train")).strip().lower()
    if dataset_mode != "train":
        return self._build_window_anomaly_counts(dataset)

    labels = getattr(dataset, "train_labels", None)
    if labels is None:
        if _dataset_chain_has_train_labels(dataset):
            return self._build_window_anomaly_counts(dataset)
        return np.zeros(len(dataset), dtype=np.int64)

    label_array = np.asarray(labels, dtype=np.float32).reshape(-1)
    counts = np.zeros(len(dataset), dtype=np.int64)
    step = int(getattr(dataset, "step", 1))
    win_size = int(getattr(dataset, "win_size", self.config.seq_len))
    for idx in range(len(dataset)):
        start = idx * step
        end = min(start + win_size, label_array.size)
        if start >= label_array.size or end <= start:
            continue
        counts[idx] = int(np.sum(label_array[start:end] > 0))
    return counts


def _build_train_window_anomaly_flags_for_visualization(self, dataset) -> np.ndarray:
    dataset_mode = str(_dataset_chain_attr(dataset, "mode", "train")).strip().lower()
    if dataset_mode != "train":
        return self._build_window_anomaly_flags(dataset)

    labels = getattr(dataset, "train_labels", None)
    if labels is None:
        if _dataset_chain_has_train_labels(dataset):
            return self._build_window_anomaly_flags(dataset)
        return np.zeros(len(dataset), dtype=np.int64)

    label_array = np.asarray(labels, dtype=np.float32).reshape(-1)
    flags = np.zeros(len(dataset), dtype=np.int64)
    step = int(getattr(dataset, "step", 1))
    win_size = int(getattr(dataset, "win_size", self.config.seq_len))
    for idx in range(len(dataset)):
        start = idx * step
        end = min(start + win_size, label_array.size)
        if start >= label_array.size or end <= start:
            continue
        flags[idx] = int(np.any(label_array[start:end] > 0))
    return flags


def _has_valid_window_labels(self, dataset) -> bool:
    for idx in range(len(dataset)):
        labels = self._extract_label_window(dataset[idx])
        if labels is None:
            continue
        try:
            label_array = np.asarray(labels, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        if label_array.size > 0:
            return True
    return False


def _project_visual_features_nd(
    self,
    features: np.ndarray,
    stage_name: str,
    n_components: int = 2,
) -> Tuple[np.ndarray, str]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={tuple(features.shape)}")

    target_dims = max(1, int(n_components))
    if features.shape[0] == 0:
        return np.empty((0, target_dims), dtype=np.float32), "pca"

    method = str(getattr(self.config, "visualization_method", "tsne")).strip().lower()
    random_state = int(self.config.seed) + (1 if stage_name.lower() == "stage2" else 0)
    max_dims = max(1, min(target_dims, int(features.shape[0]), int(features.shape[1])))

    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_components=max_dims,
                n_neighbors=int(getattr(self.config, "visualization_umap_n_neighbors", 30)),
                min_dist=float(getattr(self.config, "visualization_umap_min_dist", 0.1)),
                random_state=random_state,
            )
            coords = reducer.fit_transform(features).astype(np.float32)
            if coords.shape[1] < target_dims:
                coords = np.pad(coords, ((0, 0), (0, target_dims - coords.shape[1])), mode="constant")
            return coords, "umap"
        except Exception as exc:
            print(f"[Visualize] UMAP unavailable, fallback to t-SNE: {type(exc).__name__}: {exc}")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE

        if features.shape[0] > 3:
            perplexity = float(getattr(self.config, "visualization_tsne_perplexity", 30.0))
            max_valid_perplexity = max(1.0, float(features.shape[0] - 1) / 3.0)
            perplexity = max(1.0, min(perplexity, max_valid_perplexity))
            reducer = TSNE(
                n_components=max_dims,
                perplexity=perplexity,
                init=str(getattr(self.config, "visualization_tsne_init", "pca")),
                learning_rate="auto",
                random_state=random_state,
            )
            coords = reducer.fit_transform(features).astype(np.float32)
            if coords.shape[1] < target_dims:
                coords = np.pad(coords, ((0, 0), (0, target_dims - coords.shape[1])), mode="constant")
            return coords, "tsne"
        method = "pca"

    from sklearn.decomposition import PCA

    reducer = PCA(n_components=max_dims)
    coords = reducer.fit_transform(features).astype(np.float32)
    if coords.shape[1] < target_dims:
        coords = np.pad(coords, ((0, 0), (0, target_dims - coords.shape[1])), mode="constant")
    return coords, "pca"


def _visual_sample_indices(
    self,
    labels: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    num_items = int(labels.shape[0])
    max_points = max(1, int(max_points))
    if num_items <= max_points:
        return np.arange(num_items, dtype=np.int64)

    normal_idx = np.where(labels == 0)[0].astype(np.int64)
    anomaly_idx = np.where(labels != 0)[0].astype(np.int64)
    if normal_idx.size == 0 or anomaly_idx.size == 0:
        return self._subsample_indices(num_items, max_points, rng)

    anomaly_keep = min(int(anomaly_idx.size), max(1, int(round(max_points * 0.35))))
    normal_keep = max_points - anomaly_keep
    if normal_keep <= 0:
        normal_keep = 1
        anomaly_keep = max_points - normal_keep
    normal_keep = min(int(normal_idx.size), int(normal_keep))
    anomaly_keep = min(int(anomaly_idx.size), int(max_points - normal_keep))

    selected = np.concatenate(
        [
            rng.choice(normal_idx, size=normal_keep, replace=False).astype(np.int64),
            rng.choice(anomaly_idx, size=anomaly_keep, replace=False).astype(np.int64),
        ],
        axis=0,
    )
    if selected.size < max_points:
        remaining = np.setdiff1d(np.arange(num_items, dtype=np.int64), selected, assume_unique=False)
        extra_count = min(int(max_points - selected.size), int(remaining.size))
        if extra_count > 0:
            selected = np.concatenate(
                [selected, rng.choice(remaining, size=extra_count, replace=False).astype(np.int64)],
                axis=0,
            )
    return np.sort(selected.astype(np.int64))


@staticmethod
def _visual_state_array(
    state: Optional[Dict[str, object]],
    key: str,
    shape: Tuple[int, ...],
    dtype=np.float32,
) -> np.ndarray:
    if not isinstance(state, dict) or state.get(key) is None:
        return np.empty(shape, dtype=dtype)
    return np.asarray(state[key], dtype=dtype)


def _reference_arrays_from_state(
    self,
    state: Optional[Dict[str, object]],
    feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = self._visual_state_array(state, "cluster_centers", (0, feature_dim), dtype=np.float32)
    centers = centers.reshape(-1, feature_dim) if centers.size > 0 else np.empty((0, feature_dim), dtype=np.float32)
    inner_refs = self._visual_state_array(state, "inner_ref_points", (0, feature_dim), dtype=np.float32)
    inner_refs = inner_refs.reshape(-1, feature_dim) if inner_refs.size > 0 else np.empty((0, feature_dim), dtype=np.float32)
    outer_refs = self._visual_state_array(state, "outer_ref_points", (0, feature_dim), dtype=np.float32)
    outer_refs = outer_refs.reshape(-1, feature_dim) if outer_refs.size > 0 else np.empty((0, feature_dim), dtype=np.float32)
    global_center = self._visual_state_array(state, "global_center", (0, feature_dim), dtype=np.float32)
    global_center = (
        global_center.reshape(1, feature_dim)
        if global_center.size > 0
        else np.empty((0, feature_dim), dtype=np.float32)
    )
    return centers, inner_refs, outer_refs, global_center


@staticmethod
def _mask_from_state(
    state: Optional[Dict[str, object]],
    key: str,
    length: int,
) -> np.ndarray:
    if not isinstance(state, dict) or state.get(key) is None:
        return np.zeros(length, dtype=bool)
    mask = np.asarray(state[key], dtype=bool).reshape(-1)
    if mask.shape[0] != length:
        return np.zeros(length, dtype=bool)
    return mask


@staticmethod
def _labels_from_state(
    state: Optional[Dict[str, object]],
    length: int,
) -> np.ndarray:
    if not isinstance(state, dict) or state.get("cluster_labels") is None:
        return np.full(length, -1, dtype=np.int64)
    labels = np.asarray(state["cluster_labels"], dtype=np.int64).reshape(-1)
    if labels.shape[0] != length:
        return np.full(length, -1, dtype=np.int64)
    return labels


@staticmethod
def _cluster_display_id(cluster_id: int) -> int:
    """Use 1-based cluster labels in visualization text."""
    return int(cluster_id) + 1


def _prototype_view_key(feature_view: str = None) -> str:
    view = str(feature_view or "").strip().lower()
    if view in {"v2", "view2", "z2", "v2_flatten", "flatten"}:
        return "v2"
    return "v1"


def _current_prototype_centers_for_view(
    self,
    feature_view: str = None,
    feature_dim: int = 0,
) -> Tuple[np.ndarray, str]:
    method = str(getattr(self.config, "stage2_method", "")).strip().lower()
    if method not in {"separate_proto", "consensus_proto", "consensus_proto_v2", "paired_proto"}:
        return np.empty((0, int(feature_dim)), dtype=np.float32), ""

    model = getattr(self, "model", None)
    if model is None:
        return np.empty((0, int(feature_dim)), dtype=np.float32), ""

    view_key = _prototype_view_key(feature_view)
    is_paired = (
        method in {"paired_proto", "separate_proto", "consensus_proto", "consensus_proto_v2"}
        and bool(getattr(model, "is_dual_view", False))
        and str(getattr(model, "prototype_mode", "")).strip().lower() == "separate"
    )
    if is_paired:
        head = getattr(model, f"prototype_head_{view_key}", None)
        trace_name = "View2 Prototypes" if view_key == "v2" else "View1 Prototypes"
    else:
        head = getattr(model, "prototype_head", None)
        trace_name = "Prototypes"

    prototypes = getattr(head, "prototypes", None)
    if prototypes is None:
        return np.empty((0, int(feature_dim)), dtype=np.float32), ""

    centers = prototypes.detach().cpu().numpy().astype(np.float32)
    if centers.ndim != 2 or (int(feature_dim) > 0 and centers.shape[1] != int(feature_dim)):
        print(
            "[Visualize] Skip prototype overlay: "
            f"shape={tuple(centers.shape)} feature_dim={int(feature_dim)}"
        )
        return np.empty((0, int(feature_dim)), dtype=np.float32), ""
    return centers, trace_name


def _save_train_3d_visualization(
    self,
    stage_key: str,
    *,
    state: Optional[Dict[str, object]] = None,
    include_structure: bool = False,
    include_prediction: bool = False,
    file_label: str = "",
    feature_view: str = None,
):
    if not bool(getattr(self.config, "enable_stage_visualization", False)):
        return
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        print(f"[Visualize] Skip train 3D {stage_key}: {type(exc).__name__}: {exc}")
        return

    vis_dir = os.path.join(self.config.save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    rng = np.random.default_rng(int(self.config.seed) + {"stage0": 11, "stage1": 12, "stage2": 13}.get(str(stage_key), 19))
    max_points = int(getattr(self.config, "visualization_max_points", 3000))

    train_features = self._collect_feature_matrix(self.train_eval_loader, view=feature_view)
    test_features = self._collect_feature_matrix(self.test_loader, view=feature_view)
    train_flags_all = self._build_train_window_anomaly_flags_for_visualization(self.train_eval_loader.dataset)
    test_flags_all = self._build_window_anomaly_flags(self.test_loader.dataset)
    train_counts_all = self._build_train_window_anomaly_counts_for_visualization(self.train_eval_loader.dataset)
    test_counts_all = self._build_window_anomaly_counts(self.test_loader.dataset)
    train_idx = self._visual_sample_indices(train_flags_all, max_points, rng)
    test_idx = self._visual_sample_indices(test_flags_all, max_points, rng)
    train_sel = train_features[train_idx]
    test_sel = test_features[test_idx]

    if include_structure and state is None:
        state = self.stage2_structure if isinstance(self.stage2_structure, dict) else None

    centers, inner_refs, outer_refs, global_center = self._reference_arrays_from_state(
        state if include_structure else None,
        train_features.shape[1],
    )
    prototype_centers, prototype_trace_name = _current_prototype_centers_for_view(
        self,
        feature_view=feature_view,
        feature_dim=train_features.shape[1],
    )
    parts = [train_sel, test_sel]
    for ref in [centers, inner_refs, outer_refs, global_center]:
        if ref.size > 0:
            parts.append(ref)
    if prototype_centers.size > 0:
        parts.append(prototype_centers)
    stacked = np.concatenate(parts, axis=0)
    coords_all, method_name = self._project_visual_features_nd(stacked, stage_name=stage_key, n_components=3)
    train_end = int(train_sel.shape[0])
    train_coords = coords_all[:train_end]
    test_end = train_end + int(test_sel.shape[0])
    test_coords = coords_all[train_end:test_end]
    cursor = test_end

    def _take_ref(ref: np.ndarray) -> np.ndarray:
        nonlocal cursor
        if ref.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        coords = coords_all[cursor:cursor + ref.shape[0]]
        cursor += ref.shape[0]
        return coords

    center_coords = _take_ref(centers)
    inner_ref_coords = _take_ref(inner_refs)
    outer_ref_coords = _take_ref(outer_refs)
    global_center_coords = _take_ref(global_center)
    prototype_coords = _take_ref(prototype_centers)

    core_mask_all = self._mask_from_state(state, "core_mask", train_features.shape[0]) if include_structure else np.zeros(train_features.shape[0], dtype=bool)
    cluster_labels_all = self._labels_from_state(state, train_features.shape[0]) if include_structure else np.full(train_features.shape[0], -1, dtype=np.int64)
    sampled_true = train_flags_all[train_idx]
    sampled_counts = train_counts_all[train_idx]
    sampled_test_true = test_flags_all[test_idx]
    sampled_test_counts = test_counts_all[test_idx]
    sampled_core = core_mask_all[train_idx]
    sampled_clusters = cluster_labels_all[train_idx]

    train_pred_flags = np.zeros(train_idx.shape[0], dtype=np.int64)
    test_pred_flags = np.zeros(test_idx.shape[0], dtype=np.int64)
    center_threshold = float("nan")
    train_center_scores = np.zeros(train_features.shape[0], dtype=np.float32)
    test_center_scores = np.zeros(test_features.shape[0], dtype=np.float32)
    if include_prediction and centers.size > 0:
        train_center_scores, _ = self._nearest_center_scores(train_features, centers)
        test_center_scores, _ = self._nearest_center_scores(test_features, centers)
        train_pred_all, center_threshold = self._threshold_scores(train_center_scores, train_center_scores)
        test_pred_all, _ = self._threshold_scores(train_center_scores, test_center_scores)
        train_pred_flags = train_pred_all[train_idx]
        test_pred_flags = test_pred_all[test_idx]

    def _hover_rows(
        group: str,
        split_name: str,
        indices: np.ndarray,
        sample_indices: np.ndarray,
        true_flags: np.ndarray,
        counts: np.ndarray,
        pred_flags: np.ndarray,
        score_array: np.ndarray,
        clusters: Optional[np.ndarray] = None,
    ) -> List[str]:
        rows: List[str] = []
        for local_idx in indices.tolist():
            dataset_idx = int(sample_indices[local_idx])
            cluster_id = int(clusters[local_idx]) if clusters is not None else -1
            cluster_name = "unassigned" if cluster_id < 0 else f"C{self._cluster_display_id(cluster_id)}"
            rows.append(
                "<br>".join(
                    [
                        f"group={group}",
                        f"split={split_name}",
                        f"index={dataset_idx}",
                        f"truth={'anomaly' if int(true_flags[local_idx]) else 'normal'}",
                        f"pred_center={'anomaly' if int(pred_flags[local_idx]) else 'normal'}",
                        f"anomaly_count={int(counts[local_idx])}",
                        f"cluster={cluster_name}",
                        f"center_score={float(score_array[dataset_idx]) if score_array.size else 0.0:.6f}",
                    ]
                )
            )
        return rows

    fig = go.Figure()

    def _add_trace(
        coords_source: np.ndarray,
        mask: np.ndarray,
        name: str,
        color: str,
        symbol: str,
        size: int,
        opacity: float,
        hover_text: List[str],
        line_width: float = 0.0,
    ):
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if not np.any(mask):
            return
        indices = np.where(mask)[0].astype(np.int64)
        coords = coords_source[indices]
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                name=f"{name} ({int(indices.size)})",
                text=hover_text,
                hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                marker=dict(size=size, color=color, symbol=symbol, opacity=opacity, line=dict(width=line_width, color=color)),
            )
        )

    def _trace_hover(
        name: str,
        split_name: str,
        mask: np.ndarray,
        sample_indices: np.ndarray,
        true_flags: np.ndarray,
        counts: np.ndarray,
        pred_flags: np.ndarray,
        score_array: np.ndarray,
        clusters: Optional[np.ndarray] = None,
    ) -> List[str]:
        return _hover_rows(
            name,
            split_name,
            np.where(np.asarray(mask, dtype=bool).reshape(-1))[0].astype(np.int64),
            sample_indices,
            true_flags,
            counts,
            pred_flags,
            score_array,
            clusters=clusters,
        )

    _add_trace(
        train_coords,
        sampled_true == 0,
        "Train True Normal",
        "#16a34a",
        "circle",
        3,
        0.26,
        _trace_hover("Train True Normal", "train", sampled_true == 0, train_idx, sampled_true, sampled_counts, train_pred_flags, train_center_scores, sampled_clusters),
    )
    _add_trace(
        train_coords,
        sampled_true != 0,
        "Train True Anomaly",
        "#dc2626",
        "circle",
        5,
        0.62,
        _trace_hover("Train True Anomaly", "train", sampled_true != 0, train_idx, sampled_true, sampled_counts, train_pred_flags, train_center_scores, sampled_clusters),
    )
    _add_trace(
        test_coords,
        sampled_test_true == 0,
        "Test True Normal",
        "#65a30d",
        "square",
        3,
        0.34,
        _trace_hover("Test True Normal", "test", sampled_test_true == 0, test_idx, sampled_test_true, sampled_test_counts, test_pred_flags, test_center_scores),
    )
    _add_trace(
        test_coords,
        sampled_test_true != 0,
        "Test True Anomaly",
        "#ef4444",
        "square",
        5,
        0.74,
        _trace_hover("Test True Anomaly", "test", sampled_test_true != 0, test_idx, sampled_test_true, sampled_test_counts, test_pred_flags, test_center_scores),
    )
    if include_structure:
        _add_trace(
            train_coords,
            sampled_core,
            "Train Core",
            "#2563eb",
            "circle-open",
            6,
            0.95,
            _trace_hover("Train Core", "train", sampled_core, train_idx, sampled_true, sampled_counts, train_pred_flags, train_center_scores, sampled_clusters),
            line_width=1.2,
        )
    if include_prediction:
        _add_trace(
            train_coords,
            train_pred_flags == 0,
            "Train Pred Normal (center)",
            "#0891b2",
            "cross",
            5,
            0.56,
            _trace_hover("Train Pred Normal (center)", "train", train_pred_flags == 0, train_idx, sampled_true, sampled_counts, train_pred_flags, train_center_scores, sampled_clusters),
            line_width=1.1,
        )
        _add_trace(
            train_coords,
            train_pred_flags != 0,
            "Train Pred Anomaly (center)",
            "#a21caf",
            "x",
            6,
            0.88,
            _trace_hover("Train Pred Anomaly (center)", "train", train_pred_flags != 0, train_idx, sampled_true, sampled_counts, train_pred_flags, train_center_scores, sampled_clusters),
            line_width=1.2,
        )
        _add_trace(
            test_coords,
            test_pred_flags == 0,
            "Test Pred Normal (center)",
            "#0e7490",
            "cross",
            5,
            0.56,
            _trace_hover("Test Pred Normal (center)", "test", test_pred_flags == 0, test_idx, sampled_test_true, sampled_test_counts, test_pred_flags, test_center_scores),
            line_width=1.1,
        )
        _add_trace(
            test_coords,
            test_pred_flags != 0,
            "Test Pred Anomaly (center)",
            "#be185d",
            "x",
            6,
            0.88,
            _trace_hover("Test Pred Anomaly (center)", "test", test_pred_flags != 0, test_idx, sampled_test_true, sampled_test_counts, test_pred_flags, test_center_scores),
            line_width=1.2,
        )

    ref_specs = [
        ("Centers", center_coords, "#111111", "x", 6, "Center"),
        ("Inner", inner_ref_coords, "#0f766e", "diamond", 5, "Inner"),
        ("Outer", outer_ref_coords, "#f97316", "diamond", 5, "Outer"),
        ("Global C", global_center_coords, "#7c3aed", "square", 8, "Global C"),
    ]
    if prototype_coords.size > 0:
        ref_specs.append(
            (prototype_trace_name or "Prototypes", prototype_coords, "#111827", "diamond-open", 8, "Prototype")
        )
    for name, coords, color, symbol, size, label_prefix in ref_specs:
        if coords.size == 0:
            continue
        labels = [f"{label_prefix} {idx + 1}" for idx in range(coords.shape[0])]
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers+text" if name in {"Centers", "Global C", prototype_trace_name} else "markers",
                name=f"{name} ({coords.shape[0]})",
                text=labels,
                textposition="top center",
                hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                marker=dict(size=size, color=color, symbol=symbol, opacity=0.96),
            )
        )

    stage_title = {
        "stage0": "Preprocess End",
        "stage1": "Stage1 End",
        "stage2": "Stage2 Final",
    }.get(str(stage_key).lower(), str(stage_key))
    summary = (
        f"{stage_title} train+test 3D | method={method_name.upper()} | "
        f"train={int(train_idx.size)} | test={int(test_idx.size)} | "
        f"train_true_anomaly={int(np.sum(sampled_true != 0))} | "
        f"test_true_anomaly={int(np.sum(sampled_test_true != 0))}"
    )
    if include_structure:
        summary += f" | core={int(np.sum(sampled_core))} | centers={int(centers.shape[0])}"
    if prototype_centers.size > 0:
        summary += f" | prototypes={int(prototype_centers.shape[0])}"
    if include_prediction:
        summary += (
            f" | train_pred_anomaly={int(np.sum(train_pred_flags != 0))} | "
            f"test_pred_anomaly={int(np.sum(test_pred_flags != 0))} | "
            f"center_thr={center_threshold:.6f}"
        )
    fig.update_layout(
        template="plotly_white",
        title=dict(text=summary, font=dict(size=14)),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.02),
        margin=dict(l=0, r=220, b=0, t=48),
        scene=dict(
            xaxis_title=f"{method_name.upper()}-1",
            yaxis_title=f"{method_name.upper()}-2",
            zaxis_title=f"{method_name.upper()}-3",
            bgcolor="#ffffff",
        ),
    )
    safe_label = file_label or str(stage_key).lower()
    save_path = os.path.join(vis_dir, f"train_test3d_{safe_label}_{method_name}.html")
    fig.write_html(save_path, include_plotlyjs=True, full_html=True)
    print(f"[Visualize] Saved {save_path}")


def _save_dual_truth_visualizations(self, stage_key: str):
    """Save clean train/test truth-only latent views for the current model."""
    stage_key = str(stage_key).strip().lower()
    if self._is_dual_view_model():
        self._save_train_3d_visualization(
            stage_key,
            include_structure=False,
            include_prediction=False,
            file_label=f"{stage_key}_truth_view1",
            feature_view="v1",
        )
        self._save_train_3d_visualization(
            stage_key,
            include_structure=False,
            include_prediction=False,
            file_label=f"{stage_key}_truth_view2",
            feature_view="v2",
        )
    else:
        self._save_train_3d_visualization(
            stage_key,
            include_structure=False,
            include_prediction=False,
            file_label=f"{stage_key}_truth",
        )


def _save_train_test_score_3d_visualization(
    self,
    stage_key: str,
    *,
    score_name: str,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    file_label: str,
    feature_view: str = None,
):
    """Save a clean latent view colored by one anomaly-score component."""
    if not bool(getattr(self.config, "enable_stage_visualization", False)):
        return
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        print(f"[Visualize] Skip score 3D {stage_key}/{score_name}: {type(exc).__name__}: {exc}")
        return

    vis_dir = os.path.join(self.config.save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    rng = np.random.default_rng(int(self.config.seed) + 97)
    max_points = int(getattr(self.config, "visualization_max_points", 3000))

    train_features = self._collect_feature_matrix(self.train_eval_loader, view=feature_view)
    test_features = self._collect_feature_matrix(self.test_loader, view=feature_view)
    train_flags_all = self._build_train_window_anomaly_flags_for_visualization(self.train_eval_loader.dataset)
    test_flags_all = self._build_window_anomaly_flags(self.test_loader.dataset)
    train_counts_all = self._build_train_window_anomaly_counts_for_visualization(self.train_eval_loader.dataset)
    test_counts_all = self._build_window_anomaly_counts(self.test_loader.dataset)
    train_idx = self._visual_sample_indices(train_flags_all, max_points, rng)
    test_idx = self._visual_sample_indices(test_flags_all, max_points, rng)

    train_scores = np.asarray(train_scores, dtype=np.float32).reshape(-1)
    test_scores = np.asarray(test_scores, dtype=np.float32).reshape(-1)
    if train_scores.shape[0] != train_features.shape[0]:
        raise ValueError(f"train_scores length mismatch for {score_name}: {train_scores.shape[0]} vs {train_features.shape[0]}")
    if test_scores.shape[0] != test_features.shape[0]:
        raise ValueError(f"test_scores length mismatch for {score_name}: {test_scores.shape[0]} vs {test_features.shape[0]}")

    train_sel = train_features[train_idx]
    test_sel = test_features[test_idx]
    prototype_centers, prototype_trace_name = _current_prototype_centers_for_view(
        self,
        feature_view=feature_view,
        feature_dim=train_features.shape[1],
    )
    parts = [train_sel, test_sel]
    if prototype_centers.size > 0:
        parts.append(prototype_centers)
    stacked = np.concatenate(parts, axis=0)
    coords_all, method_name = self._project_visual_features_nd(
        stacked,
        stage_name=f"{stage_key}_{score_name}",
        n_components=3,
    )
    train_end = int(train_sel.shape[0])
    train_coords = coords_all[:train_end]
    test_end = train_end + int(test_sel.shape[0])
    test_coords = coords_all[train_end:test_end]
    prototype_coords = coords_all[test_end:] if prototype_centers.size > 0 else np.empty((0, 3), dtype=np.float32)
    train_score_sel = train_scores[train_idx]
    test_score_sel = test_scores[test_idx]
    sampled_scores = np.concatenate([train_score_sel, test_score_sel], axis=0)
    color_min = float(np.nanquantile(sampled_scores, 0.01))
    color_max = float(np.nanquantile(sampled_scores, 0.99))
    if not np.isfinite(color_min) or not np.isfinite(color_max) or color_min >= color_max:
        color_min = float(np.nanmin(sampled_scores)) if sampled_scores.size else 0.0
        color_max = float(np.nanmax(sampled_scores)) if sampled_scores.size else 1.0
        if color_min >= color_max:
            color_max = color_min + 1.0

    sampled_train_true = train_flags_all[train_idx]
    sampled_test_true = test_flags_all[test_idx]
    sampled_train_counts = train_counts_all[train_idx]
    sampled_test_counts = test_counts_all[test_idx]

    def _hover(split_name: str, sample_indices: np.ndarray, true_flags: np.ndarray, counts: np.ndarray, scores: np.ndarray) -> List[str]:
        rows: List[str] = []
        for local_idx, dataset_idx in enumerate(sample_indices.tolist()):
            rows.append(
                "<br>".join(
                    [
                        f"split={split_name}",
                        f"index={int(dataset_idx)}",
                        f"truth={'anomaly' if int(true_flags[local_idx]) else 'normal'}",
                        f"anomaly_count={int(counts[local_idx])}",
                        f"{score_name}={float(scores[local_idx]):.6f}",
                    ]
                )
            )
        return rows

    train_hover = _hover("train", train_idx, sampled_train_true, sampled_train_counts, train_score_sel)
    test_hover = _hover("test", test_idx, sampled_test_true, sampled_test_counts, test_score_sel)

    fig = go.Figure()

    def _add_trace(coords: np.ndarray, mask: np.ndarray, name: str, symbol: str, scores: np.ndarray, hover_text: List[str]):
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if not np.any(mask):
            return
        indices = np.where(mask)[0].astype(np.int64)
        pts = coords[indices]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=f"{name} ({int(indices.size)})",
                text=[hover_text[int(i)] for i in indices],
                hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                marker=dict(
                    size=4,
                    color=scores[indices],
                    coloraxis="coloraxis",
                    symbol=symbol,
                    opacity=0.74,
                ),
            )
        )

    _add_trace(train_coords, sampled_train_true == 0, "Train True Normal", "circle", train_score_sel, train_hover)
    _add_trace(train_coords, sampled_train_true != 0, "Train True Anomaly", "circle", train_score_sel, train_hover)
    _add_trace(test_coords, sampled_test_true == 0, "Test True Normal", "square", test_score_sel, test_hover)
    _add_trace(test_coords, sampled_test_true != 0, "Test True Anomaly", "square", test_score_sel, test_hover)
    if prototype_coords.size > 0:
        labels = [f"Prototype {idx + 1}" for idx in range(prototype_coords.shape[0])]
        fig.add_trace(
            go.Scatter3d(
                x=prototype_coords[:, 0],
                y=prototype_coords[:, 1],
                z=prototype_coords[:, 2],
                mode="markers+text",
                name=f"{prototype_trace_name or 'Prototypes'} ({prototype_coords.shape[0]})",
                text=labels,
                textposition="top center",
                hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                marker=dict(size=8, color="#111827", symbol="diamond-open", opacity=0.98),
            )
        )

    stage_title = {
        "stage0": "Stage0 End",
        "stage1": "Stage1 End",
        "stage2": "Stage2 Final",
    }.get(str(stage_key).lower(), str(stage_key))
    view_name = "view2" if feature_view == "v2" else "view1"
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=(
                f"{stage_title} {view_name} score={score_name} | method={method_name.upper()} | "
                f"train_mean={float(np.mean(train_scores)):.4f} | test_mean={float(np.mean(test_scores)):.4f} | "
                f"prototypes={int(prototype_centers.shape[0])}"
            ),
            font=dict(size=14),
        ),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.02),
        margin=dict(l=0, r=220, b=0, t=48),
        scene=dict(
            xaxis_title=f"{method_name.upper()}-1",
            yaxis_title=f"{method_name.upper()}-2",
            zaxis_title=f"{method_name.upper()}-3",
            bgcolor="#ffffff",
        ),
        coloraxis=dict(
            colorscale="Turbo",
            cmin=color_min,
            cmax=color_max,
            colorbar=dict(title=score_name),
        ),
    )
    safe_label = file_label or f"{stage_key}_score_{score_name}_{view_name}"
    save_path = os.path.join(vis_dir, f"train_test3d_{safe_label}_{method_name}.html")
    fig.write_html(save_path, include_plotlyjs=True, full_html=True)
    print(f"[Visualize] Saved {save_path}")


def _save_stage2_component_score_visualizations(self, stage_key: str = "stage2"):
    method = str(getattr(self.config, "stage2_method", "")).strip().lower()
    if method not in {"separate_proto", "consensus_proto", "consensus_proto_v2", "paired_proto"}:
        print(f"[Visualize] Skip Stage2 component scores: unsupported stage2_method={method}")
        return

    train_outputs = self._collect_consensus_proto_eval_outputs(self.train_eval_loader)
    test_outputs = self._collect_consensus_proto_eval_outputs(self.test_loader)
    recon_weight = float(getattr(self.config, "prototype_recon_weight", getattr(self.config, "dual_view_recon_weight", 0.5)))
    lambda_js = float(getattr(self.config, "lambda_js_score", getattr(self.config, "dual_score_weight_cv", 1.0)))
    _, train_components = compute_separate_proto_anomaly_score(
        train_outputs["proto_dist1"],
        train_outputs["proto_dist2"],
        train_outputs["recon1"],
        train_outputs["recon2"],
        train_outputs["q1"],
        train_outputs["q2"],
        recon_weight=recon_weight,
        lambda_js=lambda_js,
        eps=float(getattr(self.config, "robust_eps", 1e-6)),
    )
    _, test_components = compute_separate_proto_anomaly_score(
        test_outputs["proto_dist1"],
        test_outputs["proto_dist2"],
        test_outputs["recon1"],
        test_outputs["recon2"],
        test_outputs["q1"],
        test_outputs["q2"],
        recon_weight=recon_weight,
        lambda_js=lambda_js,
        eps=float(getattr(self.config, "robust_eps", 1e-6)),
    )

    specs = [
        ("view1_proto_dist", "v1", "stage2_score_view1_proto_dist"),
        ("view1_recon", "v1", "stage2_score_view1_recon"),
        ("view2_proto_dist", "v2", "stage2_score_view2_proto_dist"),
        ("view2_recon", "v2", "stage2_score_view2_recon"),
        ("proto_dist_gap", "v1", "stage2_score_proto_dist_gap_view1"),
        ("proto_dist_gap", "v2", "stage2_score_proto_dist_gap_view2"),
        ("js_conflict", "v1", "stage2_score_js_conflict_view1"),
        ("js_conflict", "v2", "stage2_score_js_conflict_view2"),
    ]
    for score_name, view, file_label in specs:
        self._save_train_test_score_3d_visualization(
            stage_key,
            score_name=score_name,
            train_scores=train_components[score_name],
            test_scores=test_components[score_name],
            file_label=file_label,
            feature_view=view,
        )
