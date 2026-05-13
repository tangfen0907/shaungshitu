from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans

__all__ = ['_nearest_other_clusters', '_cluster_with_kmeans', '_cluster_features', '_mask_to_indices_by_cluster', '_cluster_boundary_radii']


def _nearest_other_clusters(cluster_centers: np.ndarray) -> np.ndarray:
    cluster_centers = np.asarray(cluster_centers, dtype=np.float32)
    num_clusters = int(cluster_centers.shape[0])
    if num_clusters <= 1:
        return np.full(num_clusters, -1, dtype=np.int64)
    distances = np.linalg.norm(
        cluster_centers[:, None, :] - cluster_centers[None, :, :],
        axis=-1,
    )
    np.fill_diagonal(distances, np.inf)
    return np.argmin(distances, axis=1).astype(np.int64)
def _cluster_with_kmeans(
    features: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    num_samples = int(features.shape[0])
    n_clusters = int(max(1, min(int(n_clusters), num_samples)))
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=int(random_state),
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(features).astype(np.int64)
    cluster_centers = kmeans.cluster_centers_.astype(np.float32)
    return cluster_labels, cluster_centers, {
        "cluster_method": "kmeans",
        "cluster_method_actual": "kmeans",
        "cluster_count": int(cluster_centers.shape[0]),
    }
def _cluster_features(
    features: np.ndarray,
    cluster_method: str,
    n_clusters: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    method = str(cluster_method or "kmeans").strip().lower()
    if method == "kmeans":
        return _cluster_with_kmeans(features, n_clusters=n_clusters, random_state=random_state)
    raise ValueError(f"Unsupported cluster_method: {cluster_method}")
def _mask_to_indices_by_cluster(mask: np.ndarray, cluster_labels: np.ndarray, num_clusters: int) -> Dict[int, np.ndarray]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64).reshape(-1)
    grouped: Dict[int, np.ndarray] = {}
    for cluster_id in range(int(num_clusters)):
        grouped[cluster_id] = np.where(mask & (cluster_labels == cluster_id))[0].astype(np.int64)
    return grouped
def _cluster_boundary_radii(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    eps: float,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64).reshape(-1)
    cluster_centers = np.asarray(cluster_centers, dtype=np.float32)
    num_clusters = int(cluster_centers.shape[0])
    radii = np.full(num_clusters, float(eps), dtype=np.float32)

    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if cluster_indices.size == 0:
            continue
        distances = np.linalg.norm(
            features[cluster_indices] - cluster_centers[cluster_id][None, :],
            axis=1,
        ).astype(np.float32)
        if distances.size == 0:
            continue
        radius = float(np.max(distances))
        if not np.isfinite(radius) or radius <= float(eps):
            radius = float(eps)
        radii[cluster_id] = max(radius, float(eps))
    return radii.astype(np.float32)
