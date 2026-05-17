from typing import Dict, Tuple

import numpy as np

__all__ = [
    '_robust_scale',
    '_robust_centered',
    '_robust_positive',
    '_rank01',
    '_robust_positive_norm',
    'previous_indices_from_point_indices',
    'js_divergence_np',
    'entropy_np',
    'distance_vector_gap_np',
    'compute_separate_proto_component_scores',
    'compute_separate_proto_anomaly_score',
]


def _robust_scale(values: np.ndarray, eps: float) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0, 1.0
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= float(eps):
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= float(eps):
        scale = 1.0
    return median, scale
def _robust_centered(values: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    median, scale = _robust_scale(values, eps=float(eps))
    return ((values - float(median)) / max(float(scale), float(eps))).astype(np.float32)
def _robust_positive(values: np.ndarray, eps: float) -> np.ndarray:
    return np.maximum(_robust_centered(values, eps=float(eps)), 0.0).astype(np.float32)
def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= 1:
        return np.zeros(values.shape[0], dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float32)
    ranks[order] = np.arange(values.shape[0], dtype=np.float32)
    return (ranks / float(max(values.shape[0] - 1, 1))).astype(np.float32)
def _robust_positive_norm(values: np.ndarray, cap: float = 2.5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape[0], dtype=np.float32)
    safe_values = values.copy()
    fill_value = float(np.median(safe_values[finite]))
    safe_values[~finite] = fill_value
    q25, q75 = np.quantile(safe_values[finite], [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(safe_values[finite]))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    normalized = np.maximum((safe_values - float(np.median(safe_values[finite]))) / scale, 0.0)
    cap = max(0.0, float(cap))
    if cap > 0.0:
        normalized = np.minimum(normalized, cap)
    return normalized.astype(np.float32)


def previous_indices_from_point_indices(point_indices: np.ndarray) -> np.ndarray:
    point_indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    prev_indices = np.arange(point_indices.shape[0], dtype=np.int64)
    row_by_point = {int(point): int(row) for row, point in enumerate(point_indices)}
    for row, point in enumerate(point_indices):
        prev_indices[row] = int(row_by_point.get(int(point) - 1, row))
    return prev_indices


def js_divergence_np(q1: np.ndarray, q2: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    if q1.shape != q2.shape:
        raise ValueError("q1 and q2 should have the same shape.")
    q1 = np.maximum(q1, float(eps))
    q2 = np.maximum(q2, float(eps))
    q1 = q1 / np.maximum(np.sum(q1, axis=1, keepdims=True), float(eps))
    q2 = q2 / np.maximum(np.sum(q2, axis=1, keepdims=True), float(eps))
    m = 0.5 * (q1 + q2)
    js = 0.5 * np.sum(q1 * (np.log(q1) - np.log(np.maximum(m, float(eps)))), axis=1)
    js += 0.5 * np.sum(q2 * (np.log(q2) - np.log(np.maximum(m, float(eps)))), axis=1)
    return js.astype(np.float32)


def entropy_np(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    q = np.maximum(q, float(eps))
    q = q / np.maximum(np.sum(q, axis=1, keepdims=True), float(eps))
    return (-np.sum(q * np.log(q), axis=1)).astype(np.float32)


def distance_vector_gap_np(D1: np.ndarray, D2: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    D1 = np.asarray(D1, dtype=np.float32)
    D2 = np.asarray(D2, dtype=np.float32)
    if D1.shape != D2.shape:
        raise ValueError("D1 and D2 should have the same shape.")
    if D1.ndim != 2:
        raise ValueError("Distance vectors should be a 2D array [N, K].")
    D1_norm = D1 / np.maximum(np.mean(D1, axis=1, keepdims=True), float(eps))
    D2_norm = D2 / np.maximum(np.mean(D2, axis=1, keepdims=True), float(eps))
    return np.mean((D1_norm - D2_norm) ** 2, axis=1).astype(np.float32)


def compute_separate_proto_component_scores(
    proto_dist_matrix1: np.ndarray,
    proto_dist_matrix2: np.ndarray,
    recon1: np.ndarray,
    recon2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    q1_prev: np.ndarray = None,
    q2_prev: np.ndarray = None,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    D1 = np.asarray(proto_dist_matrix1, dtype=np.float32)
    D2 = np.asarray(proto_dist_matrix2, dtype=np.float32)
    if D1.shape != D2.shape:
        raise ValueError("Prototype distance matrices should have the same shape.")
    if D1.ndim != 2:
        raise ValueError("Prototype distance matrices should be [N, K].")

    recon1 = np.asarray(recon1, dtype=np.float32).reshape(-1)
    recon2 = np.asarray(recon2, dtype=np.float32).reshape(-1)
    if not (D1.shape[0] == recon1.shape[0] == recon2.shape[0]):
        raise ValueError("Distance matrices and reconstruction scores should align.")

    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    if q1_prev is None:
        q1_prev = q1
    if q2_prev is None:
        q2_prev = q2
    q1_prev = np.asarray(q1_prev, dtype=np.float32)
    q2_prev = np.asarray(q2_prev, dtype=np.float32)

    score_proto_v1 = np.min(D1, axis=1).astype(np.float32)
    score_proto_v2 = np.min(D2, axis=1).astype(np.float32)
    score_dist_gap = distance_vector_gap_np(D1, D2, eps=float(eps))
    score_cross_view_js = (
        js_divergence_np(q1, q2, eps=float(eps))
        + js_divergence_np(q1_prev, q2_prev, eps=float(eps))
    ).astype(np.float32)
    score_temporal_js = (
        js_divergence_np(q1, q1_prev, eps=float(eps))
        + js_divergence_np(q2, q2_prev, eps=float(eps))
    ).astype(np.float32)

    return {
        "score_recon_v1": recon1.astype(np.float32),
        "score_recon_v2": recon2.astype(np.float32),
        "score_proto_v1": score_proto_v1,
        "score_proto_v2": score_proto_v2,
        "score_dist_gap": score_dist_gap,
        "score_cross_view_js": score_cross_view_js,
        "score_temporal_js": score_temporal_js,
    }


def compute_separate_proto_anomaly_score(
    proto_dist1: np.ndarray,
    proto_dist2: np.ndarray,
    recon1: np.ndarray,
    recon2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    recon_weight: float = 0.5,
    lambda_js: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    proto_dist1 = np.asarray(proto_dist1, dtype=np.float32).reshape(-1)
    proto_dist2 = np.asarray(proto_dist2, dtype=np.float32).reshape(-1)
    recon1 = np.asarray(recon1, dtype=np.float32).reshape(-1)
    recon2 = np.asarray(recon2, dtype=np.float32).reshape(-1)
    if not (proto_dist1.shape == proto_dist2.shape == recon1.shape == recon2.shape):
        raise ValueError("Prototype distances and reconstruction scores should align.")
    conflict = js_divergence_np(q1, q2, eps=float(eps))
    entropy_v1 = entropy_np(q1, eps=float(eps))
    entropy_v2 = entropy_np(q2, eps=float(eps))
    e1 = proto_dist1 + float(recon_weight) * recon1
    e2 = proto_dist2 + float(recon_weight) * recon2
    recon_score = (0.5 * (recon1 + recon2)).astype(np.float32)
    proto_dist_gap = np.abs(proto_dist1 - proto_dist2).astype(np.float32)
    max_evidence = np.maximum(e1, e2).astype(np.float32)
    final_score = (max_evidence + float(lambda_js) * conflict).astype(np.float32)
    return final_score, {
        "final": final_score,
        "view1": e1.astype(np.float32),
        "view2": e2.astype(np.float32),
        "max_evidence": max_evidence,
        "js_conflict": conflict.astype(np.float32),
        "js": conflict.astype(np.float32),
        "proto_dist_gap": proto_dist_gap,
        "dist_v1": proto_dist1.astype(np.float32),
        "dist_v2": proto_dist2.astype(np.float32),
        "entropy_v1": entropy_v1,
        "entropy_v2": entropy_v2,
        "recon_score": recon_score,
        "view1_proto_dist": proto_dist1.astype(np.float32),
        "view2_proto_dist": proto_dist2.astype(np.float32),
        "view1_recon": recon1.astype(np.float32),
        "view2_recon": recon2.astype(np.float32),
    }
