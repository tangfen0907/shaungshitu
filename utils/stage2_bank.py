from typing import Dict, List, Optional

import numpy as np

from utils.clustering import (
    _cluster_boundary_radii,
    _cluster_features,
    _mask_to_indices_by_cluster,
    _nearest_other_clusters,
)
from utils.scoring import _rank01, _robust_positive_norm
from utils.stage2_triplet import _compute_stage2_triplet_stats

__all__ = [
    "_clip_ratio",
    "_clip_quantile",
    "_select_count",
    "_selection_config",
    "_normalize_bank_mode",
    "_ordered_pool",
    "_build_stage2_selection_metrics",
    "_effective_core_candidate_ratio",
    "_select_stage2_candidates",
    "_build_bank_summary",
    "_finalize_stage2_state",
    "_build_bank_preview",
    "initialize_stage2_structure",
    "refresh_stage2_bank",
]


def _clip_ratio(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _clip_quantile(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _select_count(
    cluster_size: int,
    ratio: float,
    min_count: int,
    max_available: int,
) -> int:
    if cluster_size <= 0 or max_available <= 0:
        return 0
    ratio = _clip_ratio(ratio)
    min_count = max(0, int(min_count))
    proposed = int(np.ceil(cluster_size * ratio))
    return int(min(max_available, max(min_count, proposed)))


def _selection_config(selection_config: Optional[Dict[str, object]]) -> Dict[str, object]:
    cfg = dict(selection_config or {})
    cfg.setdefault("use_recon", True)
    cfg.setdefault("sparse_k", 5)
    cfg.setdefault("core_expand_per_epoch", 0.0)
    cfg.setdefault("core_projection_quantile", 0.40)
    cfg.setdefault("core_sparse_weight", 0.5)
    cfg.setdefault("core_recon_weight", 0.3)
    cfg.setdefault("recon_cap", 2.5)
    cfg.setdefault("round_idx", 0)
    cfg.setdefault("num_stage2_rounds", -1)
    cfg.setdefault("progress_idx", -1)
    cfg.setdefault("total_progress_steps", -1)
    cfg.setdefault("core_enter_threshold", -1.0)
    cfg.setdefault("core_exit_threshold", -1.0)
    return cfg


def _refresh_progress_step(cfg: Dict[str, object]) -> tuple:
    progress_idx = int(cfg.get("progress_idx", -1))
    total_steps = int(cfg.get("total_progress_steps", -1))
    if progress_idx < 0 or total_steps <= 0:
        round_idx = max(1, int(cfg.get("round_idx", 1)))
        total_rounds = int(cfg.get("num_stage2_rounds", -1))
        if total_rounds <= 1:
            total_rounds = round_idx
        total_steps = max(1, total_rounds - 1)
        progress_idx = max(0, round_idx - 1)
    return max(0, progress_idx), max(1, total_steps)


def _effective_core_candidate_ratio(
    base_ratio: float,
    cfg: Dict[str, object],
    selection_mode: str,
) -> float:
    base_ratio = _clip_ratio(float(base_ratio))
    if selection_mode != "refresh":
        return base_ratio
    expand_per_epoch = max(0.0, float(cfg.get("core_expand_per_epoch", 0.0)))
    if expand_per_epoch <= 0.0:
        return base_ratio
    progress_idx, _ = _refresh_progress_step(cfg)
    return _clip_ratio(base_ratio + expand_per_epoch * float(progress_idx))


def _score_thresholds(tau_c: float, selection_config: Optional[Dict[str, object]]) -> tuple:
    cfg = _selection_config(selection_config)
    enter = float(cfg.get("core_enter_threshold", -1.0))
    if enter < 0.0:
        enter = float(tau_c)
    exit_ = float(cfg.get("core_exit_threshold", -1.0))
    if exit_ < 0.0:
        exit_ = enter
    exit_ = min(exit_, enter)
    return enter, exit_


def _normalize_bank_mode(bank_mode: Optional[str]) -> str:
    key = str(bank_mode or "score_threshold").strip().lower()
    if key in {"score_threshold", "score", "threshold", "scored"}:
        return "score_threshold"
    if key in {"direct", "direct_bank", "candidate"}:
        return "direct"
    raise ValueError(f"Unsupported Stage 2 bank update mode: {bank_mode}")


def _ordered_pool(
    candidate_indices: np.ndarray,
    distance_values: np.ndarray,
    prefer_mask: np.ndarray,
    pool_size: int,
) -> np.ndarray:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    distance_values = np.asarray(distance_values, dtype=np.float32).reshape(-1)
    prefer_mask = np.asarray(prefer_mask, dtype=bool).reshape(-1)
    pool_size = int(max(0, pool_size))
    if pool_size <= 0 or candidate_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    if prefer_mask.shape[0] != candidate_indices.shape[0]:
        prefer_mask = np.ones(candidate_indices.shape[0], dtype=bool)

    order = np.argsort(distance_values, kind="stable")
    ordered_indices = candidate_indices[order]
    ordered_prefer = ordered_indices[prefer_mask[order]]
    ordered_fallback = ordered_indices[~prefer_mask[order]]
    pool = np.concatenate([ordered_prefer, ordered_fallback], axis=0)
    return pool[:pool_size].astype(np.int64)


def _build_stage2_selection_metrics(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    inner_ref_points: np.ndarray,
    outer_ref_points: np.ndarray,
    inward_directions: np.ndarray,
    recon_scores: Optional[np.ndarray],
    selection_config: Optional[Dict[str, object]],
    eps: float,
) -> Dict[str, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64).reshape(-1)
    cluster_centers = np.asarray(cluster_centers, dtype=np.float32)
    inner_ref_points = np.asarray(inner_ref_points, dtype=np.float32)
    outer_ref_points = np.asarray(outer_ref_points, dtype=np.float32)
    assigned_centers = cluster_centers[cluster_labels]
    assigned_inner = inner_ref_points[cluster_labels]
    assigned_outer = outer_ref_points[cluster_labels]
    center_distance = np.linalg.norm(features - assigned_centers, axis=1).astype(np.float32)
    inner_distance = np.linalg.norm(features - assigned_inner, axis=1).astype(np.float32)
    outer_distance = np.linalg.norm(features - assigned_outer, axis=1).astype(np.float32)
    cfg = _selection_config(selection_config)
    local_sparse_score = np.zeros(features.shape[0], dtype=np.float32)
    local_sparse_norm = np.zeros(features.shape[0], dtype=np.float32)
    use_recon = bool(cfg.get("use_recon", True)) and recon_scores is not None
    if use_recon:
        recon_score = np.asarray(recon_scores, dtype=np.float32).reshape(-1)
        if recon_score.shape[0] != features.shape[0]:
            raise ValueError("recon_scores should align with features.")
        recon_norm = _robust_positive_norm(
            recon_score,
            cap=float(cfg.get("recon_cap", 2.5)),
        )
    else:
        recon_score = np.zeros(features.shape[0], dtype=np.float32)
        recon_norm = np.zeros(features.shape[0], dtype=np.float32)
    return {
        "center_distance": center_distance.astype(np.float32),
        "inner_distance": inner_distance.astype(np.float32),
        "outer_distance": outer_distance.astype(np.float32),
        "local_sparse_score": local_sparse_score.astype(np.float32),
        "local_sparse_norm": local_sparse_norm.astype(np.float32),
        "recon_score": recon_score.astype(np.float32),
        "recon_norm": recon_norm.astype(np.float32),
        "recon_norm_clipped": recon_norm.astype(np.float32),
    }


def _select_stage2_candidates(
    features: Optional[np.ndarray],
    cluster_labels: np.ndarray,
    selection_metrics: Dict[str, np.ndarray],
    core_ratio: float,
    min_core_per_cluster: int,
    selection_mode: str,
    selection_config: Optional[Dict[str, object]] = None,
) -> Dict[str, np.ndarray]:
    del features
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64).reshape(-1)
    num_samples = int(cluster_labels.shape[0])
    num_clusters = int(np.max(cluster_labels)) + 1 if cluster_labels.size > 0 else 0

    center_distance = np.asarray(selection_metrics["center_distance"], dtype=np.float32)
    inner_distance = np.asarray(selection_metrics["inner_distance"], dtype=np.float32)
    recon_norm = np.asarray(selection_metrics.get("recon_norm_clipped", np.zeros(num_samples)), dtype=np.float32)
    cfg = _selection_config(selection_config)
    use_recon = bool(cfg.get("use_recon", True)) and recon_norm.shape[0] == num_samples and bool(np.any(recon_norm > 0))
    core_recon_weight = float(cfg.get("core_recon_weight", 0.3)) if use_recon else 0.0
    effective_core_ratio = _effective_core_candidate_ratio(
        base_ratio=core_ratio,
        cfg=cfg,
        selection_mode=selection_mode,
    )

    core_candidate_mask = np.zeros(num_samples, dtype=bool)
    core_pool_mask = np.zeros(num_samples, dtype=bool)
    core_rank = np.zeros(num_samples, dtype=np.float32)
    selection_mode = str(selection_mode).strip().lower()

    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_size = int(cluster_indices.size)
        if cluster_size == 0:
            continue

        base_core_count = _select_count(
            cluster_size=cluster_size,
            ratio=effective_core_ratio,
            min_count=min_core_per_cluster,
            max_available=cluster_size,
        )
        core_count = base_core_count

        if selection_mode == "init":
            cluster_core_metric = center_distance[cluster_indices]
        else:
            cluster_core_metric = inner_distance[cluster_indices]
        if core_recon_weight > 0.0:
            cluster_core_metric = (
                _rank01(cluster_core_metric)
                + core_recon_weight * _rank01(recon_norm[cluster_indices])
            ).astype(np.float32)
        core_rank[cluster_indices] = cluster_core_metric.astype(np.float32)

        core_order = np.argsort(cluster_core_metric, kind="stable")
        selected_core = cluster_indices[core_order[:core_count]]
        core_pool_mask[selected_core] = True
        core_candidate_mask[selected_core] = True

    return {
        "core_candidate_mask": core_candidate_mask.astype(bool),
        "core_pool_mask": core_pool_mask.astype(bool),
        "core_rank": core_rank.astype(np.float32),
    }


def _build_bank_summary(state: Dict[str, object]) -> List[Dict[str, object]]:
    cluster_labels = np.asarray(state["cluster_labels"], dtype=np.int64)
    num_clusters = int(np.asarray(state["cluster_centers"], dtype=np.float32).shape[0])
    core_candidate_mask = np.asarray(state["core_candidate_mask"], dtype=bool)
    core_mask = np.asarray(state["core_mask"], dtype=bool)
    score_core = np.asarray(state["score_core"], dtype=np.float32)
    core_pool_mask = np.asarray(state.get("core_pool_mask", np.zeros_like(core_mask)), dtype=bool)

    summary: List[Dict[str, object]] = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        summary.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": int(cluster_indices.size),
                "core_pool_count": int(np.sum(core_pool_mask[cluster_indices])),
                "core_candidate_count": int(np.sum(core_candidate_mask[cluster_indices])),
                "core_bank_count": int(np.sum(core_mask[cluster_indices])),
                "mean_score_core": float(np.mean(score_core[cluster_indices])) if cluster_indices.size > 0 else 0.0,
            }
        )
    return summary


def _finalize_stage2_state(state: Dict[str, object]) -> Dict[str, object]:
    cluster_labels = np.asarray(state["cluster_labels"], dtype=np.int64)
    cluster_centers = np.asarray(state["cluster_centers"], dtype=np.float32)
    core_mask = np.asarray(state["core_mask"], dtype=bool)
    core_candidate_mask = np.asarray(state["core_candidate_mask"], dtype=bool)

    state["neutral_mask"] = ~core_mask
    state["margin_mask"] = np.asarray(state["neutral_mask"], dtype=bool).copy()
    state["core_indices_by_cluster"] = _mask_to_indices_by_cluster(core_mask, cluster_labels, cluster_centers.shape[0])
    state["core_candidate_indices_by_cluster"] = _mask_to_indices_by_cluster(
        core_candidate_mask,
        cluster_labels,
        cluster_centers.shape[0],
    )
    state["bank_summary"] = _build_bank_summary(state)
    return state


def _build_bank_preview(
    stage2_state: Dict[str, object],
    core_mask: np.ndarray,
    score_core: np.ndarray,
) -> Dict[str, object]:
    preview_state = dict(stage2_state)
    preview_state["core_mask"] = np.asarray(core_mask, dtype=bool)
    preview_state["score_core"] = np.asarray(score_core, dtype=np.float32)
    return {
        "bank_summary": _build_bank_summary(preview_state),
        "triplet_stats": _compute_stage2_triplet_stats(preview_state, core_mask=core_mask),
    }


def initialize_stage2_structure(
    features: np.ndarray,
    n_clusters: int,
    init_core_ratio: float,
    min_core_per_cluster: int,
    tau_c: float,
    outer_ref_scale: float = 1.0,
    random_state: int = 42,
    selection_config: Optional[Dict[str, object]] = None,
    recon_scores: Optional[np.ndarray] = None,
    bank_mode: str = "score_threshold",
    cluster_method: str = "kmeans",
    eps: float = 1e-6,
) -> Dict[str, object]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features should be [N, D] with N > 0.")

    num_samples = int(features.shape[0])
    cluster_labels, cluster_centers, cluster_meta = _cluster_features(
        features=features,
        cluster_method=cluster_method,
        n_clusters=int(max(1, min(int(n_clusters), num_samples))),
        random_state=int(random_state),
    )
    global_center = cluster_centers.mean(axis=0).astype(np.float32)

    cluster_radii = np.linalg.norm(cluster_centers - global_center[None, :], axis=1).astype(np.float32)
    inward_directions = (
        (global_center[None, :] - cluster_centers) / np.maximum(cluster_radii[:, None], float(eps))
    ).astype(np.float32)
    if cluster_radii.size == 0:
        inner_distance = 0.0
    else:
        inner_distance = 0.5 * float(np.min(cluster_radii))
    inner_ref_points = (global_center[None, :] - inner_distance * inward_directions).astype(np.float32)
    outer_ref_scale = max(0.0, float(outer_ref_scale))
    outer_ref_radii = _cluster_boundary_radii(
        features=features,
        cluster_labels=cluster_labels,
        cluster_centers=cluster_centers,
        eps=float(eps),
    )
    outer_ref_offsets = outer_ref_scale * outer_ref_radii
    outer_ref_points = (
        cluster_centers - outer_ref_offsets[:, None] * inward_directions
    ).astype(np.float32)
    nearest_other_cluster = _nearest_other_clusters(cluster_centers)

    selection_metrics = _build_stage2_selection_metrics(
        features=features,
        cluster_labels=cluster_labels,
        cluster_centers=cluster_centers,
        inner_ref_points=inner_ref_points,
        outer_ref_points=outer_ref_points,
        inward_directions=inward_directions,
        recon_scores=recon_scores,
        selection_config=selection_config,
        eps=float(eps),
    )
    selection_masks = _select_stage2_candidates(
        features=features,
        cluster_labels=cluster_labels,
        selection_metrics=selection_metrics,
        core_ratio=init_core_ratio,
        min_core_per_cluster=min_core_per_cluster,
        selection_mode="init",
        selection_config=selection_config,
    )

    core_mask = np.asarray(selection_masks["core_candidate_mask"], dtype=bool)
    score_core = np.zeros(num_samples, dtype=np.float32)
    score_core[core_mask] = float(tau_c)
    normalized_bank_mode = _normalize_bank_mode(bank_mode)

    state = {
        "cluster_centers": cluster_centers,
        "cluster_labels": cluster_labels,
        **cluster_meta,
        "global_center": global_center,
        "cluster_radii": cluster_radii,
        "inward_directions": inward_directions,
        "inner_ref_points": inner_ref_points,
        "outer_ref_points": outer_ref_points,
        "outer_ref_radius": outer_ref_radii,
        "outer_ref_offset": outer_ref_offsets,
        "nearest_other_cluster": nearest_other_cluster,
        "core_candidate_mask": np.asarray(selection_masks["core_candidate_mask"], dtype=bool),
        "core_pool_mask": np.asarray(selection_masks["core_pool_mask"], dtype=bool),
        "core_mask": core_mask,
        "score_core": score_core,
        "refresh_round": 0,
        "bank_mode": normalized_bank_mode,
        "candidate_source": "center_core_init",
        "inner_ref_distance": float(inner_distance),
        "outer_ref_scale": float(outer_ref_scale),
        "axis_projection": None,
        "outer_closeness": None,
        "local_sparse_score": None,
        "local_sparse_norm": None,
        "recon_score": selection_metrics.get("recon_score"),
        "recon_norm": selection_metrics.get("recon_norm"),
        "recon_norm_clipped": selection_metrics.get("recon_norm_clipped"),
        "core_rank": np.asarray(selection_masks["core_rank"], dtype=np.float32),
    }
    direct_preview = _build_bank_preview(
        stage2_state=state,
        core_mask=core_mask,
        score_core=score_core,
    )
    state["direct_bank_summary"] = direct_preview["bank_summary"]
    state["direct_triplet_stats"] = direct_preview["triplet_stats"]
    state.update(selection_metrics)
    return _finalize_stage2_state(state)


def refresh_stage2_bank(
    features: np.ndarray,
    stage2_state: Dict[str, object],
    core_candidate_ratio: float,
    min_core_per_cluster: int,
    a_c: float,
    b_c: float,
    floor_c: float,
    tau_c: float,
    round_idx: int,
    selection_config: Optional[Dict[str, object]] = None,
    recon_scores: Optional[np.ndarray] = None,
    bank_mode: str = "score_threshold",
    eps: float = 1e-6,
) -> Dict[str, object]:
    features = np.asarray(features, dtype=np.float32)
    cluster_labels = np.asarray(stage2_state["cluster_labels"], dtype=np.int64)
    cluster_centers = np.asarray(stage2_state["cluster_centers"], dtype=np.float32)
    inner_ref_points = np.asarray(stage2_state["inner_ref_points"], dtype=np.float32)
    outer_ref_points = np.asarray(stage2_state["outer_ref_points"], dtype=np.float32)
    inward_directions = np.asarray(stage2_state["inward_directions"], dtype=np.float32)
    score_core = np.asarray(stage2_state["score_core"], dtype=np.float32).copy()

    selection_metrics = _build_stage2_selection_metrics(
        features=features,
        cluster_labels=cluster_labels,
        cluster_centers=cluster_centers,
        inner_ref_points=inner_ref_points,
        outer_ref_points=outer_ref_points,
        inward_directions=inward_directions,
        recon_scores=recon_scores,
        selection_config=selection_config,
        eps=float(eps),
    )
    selection_masks = _select_stage2_candidates(
        features=features,
        cluster_labels=cluster_labels,
        selection_metrics=selection_metrics,
        core_ratio=core_candidate_ratio,
        min_core_per_cluster=min_core_per_cluster,
        selection_mode="refresh",
        selection_config=selection_config,
    )

    core_candidate_mask = np.asarray(selection_masks["core_candidate_mask"], dtype=bool)
    score_core += np.where(core_candidate_mask, float(a_c), -float(b_c)).astype(np.float32)
    score_core = np.maximum(score_core, float(floor_c)).astype(np.float32)
    normalized_bank_mode = _normalize_bank_mode(bank_mode)
    enter_threshold, exit_threshold = _score_thresholds(tau_c, selection_config)

    direct_core_mask = core_candidate_mask
    preview_state = dict(stage2_state)
    direct_preview = _build_bank_preview(
        stage2_state=preview_state,
        core_mask=direct_core_mask,
        score_core=score_core,
    )

    if normalized_bank_mode == "direct":
        core_mask = direct_core_mask
    else:
        previous_core_mask = np.asarray(
            stage2_state.get("core_mask", np.zeros_like(core_candidate_mask)),
            dtype=bool,
        ).reshape(-1)
        enter_mask = score_core >= enter_threshold
        keep_mask = previous_core_mask & (score_core >= exit_threshold)
        core_mask = enter_mask | keep_mask

    updated = dict(stage2_state)
    updated.update(selection_metrics)
    updated["core_candidate_mask"] = core_candidate_mask
    updated["core_pool_mask"] = np.asarray(selection_masks["core_pool_mask"], dtype=bool)
    updated["core_mask"] = core_mask
    updated["score_core"] = score_core
    updated["refresh_round"] = int(round_idx)
    updated["bank_mode"] = normalized_bank_mode
    updated["candidate_source"] = "reference_distance_refresh"
    updated["core_enter_threshold"] = float(enter_threshold)
    updated["core_exit_threshold"] = float(exit_threshold)
    updated["direct_bank_summary"] = direct_preview["bank_summary"]
    updated["direct_triplet_stats"] = direct_preview["triplet_stats"]
    updated["axis_projection"] = None
    updated["outer_closeness"] = None
    updated["core_rank"] = np.asarray(selection_masks["core_rank"], dtype=np.float32)
    return _finalize_stage2_state(updated)
