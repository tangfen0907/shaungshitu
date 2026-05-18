from typing import Dict

import numpy as np

__all__ = [
    'previous_indices_from_point_indices',
    'compute_separate_proto_component_scores',
]


def previous_indices_from_point_indices(point_indices: np.ndarray) -> np.ndarray:
    point_indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    prev_indices = np.arange(point_indices.shape[0], dtype=np.int64)
    row_by_point = {int(point): int(row) for row, point in enumerate(point_indices)}
    for row, point in enumerate(point_indices):
        prev_indices[row] = int(row_by_point.get(int(point) - 1, row))
    return prev_indices


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
    d_ap_v1 = np.linalg.norm(h1 - h1_prev, axis=1).astype(np.float32)
    d_ap_v2 = np.linalg.norm(h2 - h2_prev, axis=1).astype(np.float32)
    score_proto_ap_gap_v1 = np.maximum(d_proto_v1 - d_ap_v1, 0.0).astype(np.float32)
    score_proto_ap_gap_v2 = np.maximum(d_proto_v2 - d_ap_v2, 0.0).astype(np.float32)
    score_proto_ap_gap_sum = (score_proto_ap_gap_v1 + score_proto_ap_gap_v2).astype(np.float32)

    return {
        "score_recon_v1": recon1.astype(np.float32),
        "score_recon_v2": recon2.astype(np.float32),
        "score_proto_v1": score_proto_v1,
        "score_proto_v2": score_proto_v2,
        "score_proto_ap_gap_v1": score_proto_ap_gap_v1,
        "score_proto_ap_gap_v2": score_proto_ap_gap_v2,
        "score_proto_ap_gap_sum": score_proto_ap_gap_sum,
    }
