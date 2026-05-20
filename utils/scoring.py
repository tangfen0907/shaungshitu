from typing import Dict, Iterable, Tuple

import numpy as np

__all__ = [
    'lag_indices_from_point_indices',
    'previous_indices_from_point_indices',
    'compute_separate_proto_component_scores',
    'compute_multilag_local_component_scores',
]


def lag_indices_from_point_indices(point_indices: np.ndarray, lag: int = 1) -> np.ndarray:
    point_indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    lag = max(1, int(lag))
    lag_indices = np.arange(point_indices.shape[0], dtype=np.int64)
    row_by_point = {int(point): int(row) for row, point in enumerate(point_indices)}
    for row, point in enumerate(point_indices):
        lag_indices[row] = int(row_by_point.get(int(point) - lag, row))
    return lag_indices


def previous_indices_from_point_indices(point_indices: np.ndarray) -> np.ndarray:
    return lag_indices_from_point_indices(point_indices, lag=1)


def _parse_lags(lags: Iterable[int]) -> Tuple[int, ...]:
    parsed = []
    for lag in lags or (1, 3, 5, 10):
        lag = int(lag)
        if lag > 0 and lag not in parsed:
            parsed.append(lag)
    return tuple(parsed or [1])


def _topk_mean(values: np.ndarray, k: int = 2) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Expected a [N, num_lags] distance matrix.")
    k = max(1, min(int(k), values.shape[1]))
    if k == values.shape[1]:
        return np.mean(values, axis=1).astype(np.float32)
    topk = np.partition(values, values.shape[1] - k, axis=1)[:, -k:]
    return np.mean(topk, axis=1).astype(np.float32)


def compute_multilag_local_component_scores(
    h1: np.ndarray,
    h2: np.ndarray,
    point_indices: np.ndarray,
    lags: Iterable[int] = (1, 3, 5, 10),
    topk: int = 2,
) -> Dict[str, np.ndarray]:
    """Compute local scores against multiple temporal lags.

    The original local score compares H(X_t) with H(X_{t-1}).  These scores
    compare H(X_t) with several older windows and aggregate the resulting
    distances.  Missing early-history rows fall back to the current row, giving
    zero distance for that missing lag instead of leaking across sequence
    boundaries.
    """
    h1 = np.asarray(h1, dtype=np.float32)
    h2 = np.asarray(h2, dtype=np.float32)
    if h1.shape != h2.shape:
        raise ValueError("View hidden states should have matching shapes.")
    if h1.ndim != 2:
        raise ValueError("Hidden states should be [N, D].")

    point_indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    if point_indices.shape[0] != h1.shape[0]:
        raise ValueError("point_indices and hidden states should align.")

    parsed_lags = _parse_lags(lags)
    dist1 = []
    dist2 = []
    for lag in parsed_lags:
        lag_idx = lag_indices_from_point_indices(point_indices, lag=lag)
        dist1.append(np.linalg.norm(h1 - h1[lag_idx], axis=1).astype(np.float32))
        dist2.append(np.linalg.norm(h2 - h2[lag_idx], axis=1).astype(np.float32))

    D1 = np.stack(dist1, axis=1).astype(np.float32)
    D2 = np.stack(dist2, axis=1).astype(np.float32)

    v1_mean = np.mean(D1, axis=1).astype(np.float32)
    v2_mean = np.mean(D2, axis=1).astype(np.float32)
    v1_max = np.max(D1, axis=1).astype(np.float32)
    v2_max = np.max(D2, axis=1).astype(np.float32)
    v1_topk = _topk_mean(D1, k=topk)
    v2_topk = _topk_mean(D2, k=topk)

    return {
        "score_local_multilag_mean_v1": v1_mean,
        "score_local_multilag_mean_v2": v2_mean,
        "score_local_multilag_mean_sum": (v1_mean + v2_mean).astype(np.float32),
        "score_local_multilag_max_v1": v1_max,
        "score_local_multilag_max_v2": v2_max,
        "score_local_multilag_max_sum": (v1_max + v2_max).astype(np.float32),
        "score_local_multilag_top2_v1": v1_topk,
        "score_local_multilag_top2_v2": v2_topk,
        "score_local_multilag_top2_sum": (v1_topk + v2_topk).astype(np.float32),
    }


def compute_separate_proto_component_scores(
    proto_dist_matrix1: np.ndarray,
    proto_dist_matrix2: np.ndarray,
    recon1: np.ndarray,
    recon2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    h1: np.ndarray = None,
    h2: np.ndarray = None,
    q1_prev: np.ndarray = None,
    q2_prev: np.ndarray = None,
    h1_prev: np.ndarray = None,
    h2_prev: np.ndarray = None,
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

    if h1 is None:
        h1 = np.zeros((D1.shape[0], 1), dtype=np.float32)
    if h2 is None:
        h2 = np.zeros((D2.shape[0], 1), dtype=np.float32)
    if h1_prev is None:
        h1_prev = h1
    if h2_prev is None:
        h2_prev = h2
    h1 = np.asarray(h1, dtype=np.float32)
    h2 = np.asarray(h2, dtype=np.float32)
    h1_prev = np.asarray(h1_prev, dtype=np.float32)
    h2_prev = np.asarray(h2_prev, dtype=np.float32)
    if not (h1.shape == h1_prev.shape and h2.shape == h2_prev.shape):
        raise ValueError("Current and previous hidden states should have matching shapes per view.")
    if not (h1.shape[0] == h2.shape[0] == D1.shape[0]):
        raise ValueError("Hidden states and prototype distance matrices should align.")

    score_proto_v1 = np.min(D1, axis=1).astype(np.float32)
    score_proto_v2 = np.min(D2, axis=1).astype(np.float32)
    d_proto_v1 = np.sqrt(np.maximum(score_proto_v1, 0.0)).astype(np.float32)
    d_proto_v2 = np.sqrt(np.maximum(score_proto_v2, 0.0)).astype(np.float32)
    score_local_v1 = np.linalg.norm(h1 - h1_prev, axis=1).astype(np.float32)
    score_local_v2 = np.linalg.norm(h2 - h2_prev, axis=1).astype(np.float32)
    score_local_sum = (score_local_v1 + score_local_v2).astype(np.float32)
    score_proto_ap_gap_v1 = np.maximum(d_proto_v1 - score_local_v1, 0.0).astype(np.float32)
    score_proto_ap_gap_v2 = np.maximum(d_proto_v2 - score_local_v2, 0.0).astype(np.float32)
    score_proto_ap_gap_sum = (score_proto_ap_gap_v1 + score_proto_ap_gap_v2).astype(np.float32)

    return {
        "score_recon_v1": recon1.astype(np.float32),
        "score_recon_v2": recon2.astype(np.float32),
        "score_proto_v1": score_proto_v1,
        "score_proto_v2": score_proto_v2,
        "score_local_v1": score_local_v1,
        "score_local_v2": score_local_v2,
        "score_local_sum": score_local_sum,
        "score_proto_ap_gap_v1": score_proto_ap_gap_v1,
        "score_proto_ap_gap_v2": score_proto_ap_gap_v2,
        "score_proto_ap_gap_sum": score_proto_ap_gap_sum,
    }
