from typing import Dict, List, Optional

import numpy as np

from utils.clustering import _mask_to_indices_by_cluster

__all__ = ["_compute_stage2_triplet_stats", "build_stage2_triplet_plan"]


def _compute_stage2_triplet_stats(
    stage2_state: Dict[str, object],
    core_mask: Optional[np.ndarray] = None,
) -> Dict[str, int]:
    cluster_labels = np.asarray(stage2_state["cluster_labels"], dtype=np.int64)
    cluster_centers = np.asarray(stage2_state["cluster_centers"], dtype=np.float32)
    nearest_other_cluster = np.asarray(stage2_state["nearest_other_cluster"], dtype=np.int64)

    if core_mask is None:
        core_indices_by_cluster = stage2_state["core_indices_by_cluster"]
    else:
        core_indices_by_cluster = _mask_to_indices_by_cluster(
            core_mask,
            cluster_labels,
            cluster_centers.shape[0],
        )

    skipped_core_anchor = 0
    valid_core_anchor = 0
    injected_negative2 = 0

    num_clusters = int(cluster_centers.shape[0])
    all_indices = np.arange(cluster_labels.shape[0], dtype=np.int64)
    for cluster_id in range(num_clusters):
        same_core = np.asarray(
            core_indices_by_cluster.get(cluster_id, np.empty(0, dtype=np.int64)),
            dtype=np.int64,
        )
        other_cluster = int(nearest_other_cluster[cluster_id]) if cluster_id < nearest_other_cluster.shape[0] else -1
        other_cluster_indices = (
            np.where(cluster_labels == other_cluster)[0].astype(np.int64, copy=False)
            if other_cluster >= 0
            else np.empty(0, dtype=np.int64)
        )
        other_core = (
            np.asarray(core_indices_by_cluster.get(other_cluster, np.empty(0, dtype=np.int64)), dtype=np.int64)
            if other_cluster >= 0
            else np.empty(0, dtype=np.int64)
        )
        if other_cluster_indices.size > 0 and other_core.size > 0:
            other_non_core = other_cluster_indices[~np.isin(other_cluster_indices, other_core)]
        else:
            other_non_core = other_cluster_indices
        if other_non_core.size <= 0:
            other_non_core = other_cluster_indices
        if other_non_core.size <= 0:
            other_non_core = all_indices[cluster_labels != cluster_id]

        if same_core.size > 0:
            if same_core.size > 1 and other_non_core.size > 0:
                valid_core_anchor += int(same_core.size)
                injected_negative2 += int(same_core.size)
            else:
                skipped_core_anchor += int(same_core.size)

    return {
        "valid_core_anchor": int(valid_core_anchor),
        "skipped_core_anchor": int(skipped_core_anchor),
        "num_rows": int(valid_core_anchor),
        "injected_negative2": int(injected_negative2),
    }


def build_stage2_triplet_plan(
    stage2_state: Dict[str, object],
    seed: int,
) -> Dict[str, object]:
    cluster_labels = np.asarray(stage2_state["cluster_labels"], dtype=np.int64)
    cluster_centers = np.asarray(stage2_state["cluster_centers"], dtype=np.float32)
    nearest_other_cluster = np.asarray(stage2_state["nearest_other_cluster"], dtype=np.int64)
    core_indices_by_cluster = stage2_state["core_indices_by_cluster"]
    rng = np.random.default_rng(int(seed))

    rows: List[List[int]] = []
    anchor_types: List[int] = []
    cluster_ids: List[int] = []
    negative1_cluster_ids: List[int] = []
    negative2_cluster_ids: List[int] = []
    negative2_is_injected: List[bool] = []
    skipped_core_anchor = 0
    valid_core_anchor = 0
    injected_negative2 = 0

    num_clusters = int(cluster_centers.shape[0])
    all_indices = np.arange(cluster_labels.shape[0], dtype=np.int64)
    for cluster_id in range(num_clusters):
        same_core = np.asarray(
            core_indices_by_cluster.get(cluster_id, np.empty(0, dtype=np.int64)),
            dtype=np.int64,
        )
        other_cluster = int(nearest_other_cluster[cluster_id]) if cluster_id < nearest_other_cluster.shape[0] else -1
        other_cluster_indices = (
            np.where(cluster_labels == other_cluster)[0].astype(np.int64, copy=False)
            if other_cluster >= 0
            else np.empty(0, dtype=np.int64)
        )
        other_core = (
            np.asarray(core_indices_by_cluster.get(other_cluster, np.empty(0, dtype=np.int64)), dtype=np.int64)
            if other_cluster >= 0
            else np.empty(0, dtype=np.int64)
        )
        if other_cluster_indices.size > 0 and other_core.size > 0:
            other_non_core = other_cluster_indices[~np.isin(other_cluster_indices, other_core)]
        else:
            other_non_core = other_cluster_indices
        if other_non_core.size <= 0:
            other_non_core = other_cluster_indices
        if other_non_core.size <= 0:
            other_non_core = all_indices[cluster_labels != cluster_id]

        for anchor_idx in same_core.tolist():
            positive_pool = same_core[same_core != anchor_idx]
            if positive_pool.size <= 0 or other_non_core.size <= 0:
                skipped_core_anchor += 1
                continue
            positive_idx = int(rng.choice(positive_pool))
            negative1_idx = int(rng.choice(other_non_core))
            negative2_idx = int(anchor_idx)
            injected_negative2 += 1
            rows.append([int(anchor_idx), positive_idx, negative1_idx, negative2_idx])
            anchor_types.append(0)
            cluster_ids.append(cluster_id)
            negative1_cluster_ids.append(int(cluster_labels[negative1_idx]))
            negative2_cluster_ids.append(cluster_id)
            negative2_is_injected.append(True)
            valid_core_anchor += 1

    if not rows:
        raise RuntimeError("Stage 2 triplet plan is empty. Current core bank cannot form valid triplets.")

    return {
        "indices": np.asarray(rows, dtype=np.int64),
        "anchor_types": np.asarray(anchor_types, dtype=np.int64),
        "cluster_ids": np.asarray(cluster_ids, dtype=np.int64),
        "negative1_cluster_ids": np.asarray(negative1_cluster_ids, dtype=np.int64),
        "negative2_cluster_ids": np.asarray(negative2_cluster_ids, dtype=np.int64),
        "negative2_is_injected": np.asarray(negative2_is_injected, dtype=bool),
        "stats": {
            "valid_core_anchor": int(valid_core_anchor),
            "skipped_core_anchor": int(skipped_core_anchor),
            "num_rows": int(len(rows)),
            "injected_negative2": int(injected_negative2),
        },
    }
