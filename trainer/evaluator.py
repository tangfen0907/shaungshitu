from typing import Dict, Optional, Tuple

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics.Matthews_correlation_coefficient import MCC
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.f1_score_f1_pa import get_adjust_F1PA
from utils.scoring import (
    compute_separate_proto_component_scores,
    previous_indices_from_point_indices,
)

try:
    from metrics.vus.metrics import get_range_vus_roc
    _VUS_AVAILABLE = True
except Exception:
    get_range_vus_roc = None
    _VUS_AVAILABLE = False


__all__ = ['_extract_label_window', '_build_window_anomaly_flags', '_build_window_anomaly_counts', '_build_point_anomaly_labels', '_build_point_anomaly_counts', '_dataset_point_window_indices', '_build_current_point_indices', '_build_current_point_anomaly_labels', '_nearest_center_scores', '_threshold_scores', '_pairwise_distance_block', '_init_distance_stats', '_update_distance_stats', '_summarize_distance_stats', '_within_class_distance_stats', '_cross_class_distance_stats', '_sample_within_class_distances', '_sample_cross_class_distances', '_select_distance_analysis_subset', '_analyze_test_latent_distances', '_collect_reconstruction_scores', '_collect_dual_reconstruction_scores', '_cross_view_disagreement_from_evidence', '_save_stage1_reconstruction_scoring', '_format_index_count_preview', '_evaluate_score_family', '_normalize_score_pair', '_collect_cross_view_scores', '_collect_pointwise_separate_proto_eval_outputs', '_collect_separate_proto_eval_outputs', '_test_dual_separate_proto', 'run_evaluation']


def combine_all_evaluation_scores(y_test, pred_labels, anomaly_scores):
    y_test = np.asarray(y_test, dtype=np.int64).reshape(-1).copy()
    pred_labels = np.asarray(pred_labels, dtype=np.int64).reshape(-1).copy()
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float32).reshape(-1).copy()

    events_pred = convert_vector_to_events(pred_labels.copy())
    events_gt = convert_vector_to_events(y_test.copy())
    Trange = (0, len(y_test))
    affiliation = pr_from_events(events_pred, events_gt, Trange)

    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(
        pred_labels.copy(),
        y_test.copy(),
    )
    try:
        mcc_score = MCC(y_test.copy(), pred_labels.copy())
    except Exception:
        mcc_score = 0.0

    if _VUS_AVAILABLE:
        try:
            vus_results = get_range_vus_roc(anomaly_scores.copy(), y_test.copy(), 100)
        except Exception:
            vus_results = {}
    else:
        vus_results = {}

    return {
        "pa_accuracy": pa_accuracy,
        "pa_precision": pa_precision,
        "pa_recall": pa_recall,
        "pa_f_score": pa_f_score,
        "MCC_score": mcc_score,
        "Affiliation precision": affiliation["precision"],
        "Affiliation recall": affiliation["recall"],
        "R_AUC_ROC": vus_results.get("R_AUC_ROC", float("nan")),
        "R_AUC_PR": vus_results.get("R_AUC_PR", float("nan")),
        "VUS_ROC": vus_results.get("VUS_ROC", float("nan")),
        "VUS_PR": vus_results.get("VUS_PR", float("nan")),
    }


@staticmethod
def _extract_label_window(sample):
    """Return label_window from dataset samples shaped as (window, label_window)."""
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[1]
    return None


def _build_window_anomaly_flags(self, dataset) -> np.ndarray:
    flags = np.zeros(len(dataset), dtype=np.int64)
    for idx in range(len(dataset)):
        labels = self._extract_label_window(dataset[idx])
        if labels is None:
            continue
        label_array = np.asarray(labels, dtype=np.float32).reshape(-1)
        flags[idx] = int(np.any(label_array > 0))
    return flags


def _build_window_anomaly_counts(self, dataset) -> np.ndarray:
    counts = np.zeros(len(dataset), dtype=np.int64)
    for idx in range(len(dataset)):
        labels = self._extract_label_window(dataset[idx])
        if labels is None:
            continue
        label_array = np.asarray(labels, dtype=np.float32).reshape(-1)
        counts[idx] = int(np.sum(label_array > 0))
    return counts


def _build_point_anomaly_labels(self, dataset) -> np.ndarray:
    dataset_mode = str(getattr(dataset, "mode", "train")).strip().lower()
    if dataset_mode == "train":
        labels = getattr(dataset, "train_labels", None)
        data = getattr(dataset, "train", None)
    else:
        labels = getattr(dataset, "test_labels", None)
        data = getattr(dataset, "test", None)

    current = getattr(dataset, "base_dataset", None)
    while labels is None and current is not None:
        dataset_mode = str(getattr(current, "mode", dataset_mode)).strip().lower()
        if dataset_mode == "train":
            labels = getattr(current, "train_labels", None)
            data = getattr(current, "train", data)
        else:
            labels = getattr(current, "test_labels", None)
            data = getattr(current, "test", data)
        current = getattr(current, "base_dataset", getattr(current, "dataset", None))

    if labels is not None:
        return np.asarray(labels, dtype=np.float32).reshape(-1).astype(np.int64)
    if data is not None:
        return np.zeros(int(np.asarray(data).shape[0]), dtype=np.int64)

    win_size = int(getattr(dataset, "win_size", self.config.seq_len))
    step = int(getattr(dataset, "step", 1))
    total_points = max(0, (int(len(dataset)) - 1) * step + win_size)
    return np.zeros(total_points, dtype=np.int64)


def _build_point_anomaly_counts(self, dataset) -> np.ndarray:
    # Kept as a count-like array so the existing logging helper can be reused.
    return self._build_point_anomaly_labels(dataset).astype(np.int64)


@staticmethod
def _dataset_point_window_indices(dataset) -> np.ndarray:
    if hasattr(dataset, "original_indices"):
        return np.asarray(getattr(dataset, "original_indices"), dtype=np.int64).reshape(-1)
    if hasattr(dataset, "indices"):
        return np.asarray(getattr(dataset, "indices"), dtype=np.int64).reshape(-1)
    return np.arange(len(dataset), dtype=np.int64)


def _build_current_point_indices(self, dataset) -> np.ndarray:
    window_indices = self._dataset_point_window_indices(dataset)

    def _chain_attr(name: str, default=None):
        current = dataset
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if hasattr(current, name):
                value = getattr(current, name)
                if value is not None:
                    return value
            current = getattr(current, "base_dataset", getattr(current, "dataset", None))
        return default

    step = int(_chain_attr("step", 1))
    win_size = int(_chain_attr("win_size", getattr(self.config, "seq_len", 1)))
    current_point_offset = int(_chain_attr("current_point_offset", max(0, win_size - 1)))
    return window_indices.astype(np.int64) * step + current_point_offset


def _build_current_point_anomaly_labels(self, dataset) -> np.ndarray:
    point_indices = self._build_current_point_indices(dataset)
    point_label_source = self._build_point_anomaly_labels(dataset)
    labels = np.zeros(len(dataset), dtype=np.int64)
    for idx in range(len(dataset)):
        label_window = self._extract_label_window(dataset[idx])
        if label_window is not None:
            label_array = np.asarray(label_window, dtype=np.float32).reshape(-1)
            if label_array.size > 0:
                labels[idx] = int(label_array[-1] > 0)
                continue
        safe_idx = int(np.clip(point_indices[idx], 0, max(0, point_label_source.shape[0] - 1)))
        labels[idx] = int(point_label_source[safe_idx] > 0)
    return labels


@staticmethod
def _nearest_center_scores(features: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    if features.ndim != 2 or centers.ndim != 2 or centers.shape[0] == 0:
        return np.zeros(features.shape[0], dtype=np.float32), np.full(features.shape[0], -1, dtype=np.int64)
    distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2).astype(np.float32)
    assigned = np.argmin(distances, axis=1).astype(np.int64)
    scores = distances[np.arange(distances.shape[0]), assigned].astype(np.float32)
    return scores, assigned


def _threshold_scores(self, train_scores: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, float]:
    train_scores = np.asarray(train_scores, dtype=np.float32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    finite_train = train_scores[np.isfinite(train_scores)]
    if finite_train.size == 0:
        return np.zeros(scores.shape[0], dtype=np.int64), float("nan")
    threshold = float(np.quantile(finite_train, float(self.config.decision_quantile)))
    return (scores > threshold).astype(np.int64), threshold


@staticmethod
def _pairwise_distance_block(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left and right should be 2D feature arrays.")
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    dist_sq = left_norm + right_norm - (2.0 * left @ right.T)
    return np.sqrt(np.maximum(dist_sq, 0.0)).astype(np.float32)


@staticmethod
def _init_distance_stats() -> Dict[str, float]:
    return {
        "pair_count": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "min": float("inf"),
        "max": 0.0,
    }


@staticmethod
def _update_distance_stats(stats: Dict[str, float], values: np.ndarray):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return
    values64 = values.astype(np.float64, copy=False)
    stats["pair_count"] += int(values.size)
    stats["sum"] += float(np.sum(values64))
    stats["sumsq"] += float(np.sum(values64 * values64))
    stats["min"] = min(float(stats["min"]), float(np.min(values64)))
    stats["max"] = max(float(stats["max"]), float(np.max(values64)))


@staticmethod
def _summarize_distance_stats(stats: Dict[str, float], sample_values: Optional[np.ndarray] = None) -> Dict[str, float]:
    pair_count = int(stats.get("pair_count", 0))
    if pair_count <= 0:
        return {
            "pair_count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sample_size": 0,
        }

    mean = float(stats["sum"] / max(pair_count, 1))
    variance = max(float(stats["sumsq"] / max(pair_count, 1)) - (mean * mean), 0.0)
    summary = {
        "pair_count": pair_count,
        "mean": mean,
        "std": float(np.sqrt(variance)),
        "min": float(stats["min"]),
        "max": float(stats["max"]),
        "sample_size": 0,
    }

    if sample_values is not None:
        sample_values = np.asarray(sample_values, dtype=np.float32).reshape(-1)
        if sample_values.size > 0:
            summary.update(
                {
                    "sample_size": int(sample_values.size),
                    "sample_median": float(np.median(sample_values)),
                    "sample_q05": float(np.quantile(sample_values, 0.05)),
                    "sample_q25": float(np.quantile(sample_values, 0.25)),
                    "sample_q75": float(np.quantile(sample_values, 0.75)),
                    "sample_q95": float(np.quantile(sample_values, 0.95)),
                }
            )
    return summary


def _within_class_distance_stats(self, features: np.ndarray, block_size: int = 512) -> Dict[str, float]:
    features = np.asarray(features, dtype=np.float32)
    stats = self._init_distance_stats()
    num_items = int(features.shape[0])
    if num_items < 2:
        return stats

    block_size = max(1, int(block_size))
    for start in range(0, num_items, block_size):
        end = min(num_items, start + block_size)
        left_block = features[start:end]
        same_block = self._pairwise_distance_block(left_block, left_block)
        upper = np.triu_indices(end - start, k=1)
        self._update_distance_stats(stats, same_block[upper])

        for other_start in range(end, num_items, block_size):
            other_end = min(num_items, other_start + block_size)
            right_block = features[other_start:other_end]
            cross_block = self._pairwise_distance_block(left_block, right_block)
            self._update_distance_stats(stats, cross_block.reshape(-1))
    return stats


def _cross_class_distance_stats(
    self,
    left_features: np.ndarray,
    right_features: np.ndarray,
    block_size: int = 512,
) -> Dict[str, float]:
    left_features = np.asarray(left_features, dtype=np.float32)
    right_features = np.asarray(right_features, dtype=np.float32)
    stats = self._init_distance_stats()
    if left_features.shape[0] == 0 or right_features.shape[0] == 0:
        return stats

    block_size = max(1, int(block_size))
    for start in range(0, left_features.shape[0], block_size):
        end = min(left_features.shape[0], start + block_size)
        left_block = left_features[start:end]
        for other_start in range(0, right_features.shape[0], block_size):
            other_end = min(right_features.shape[0], other_start + block_size)
            right_block = right_features[other_start:other_end]
            cross_block = self._pairwise_distance_block(left_block, right_block)
            self._update_distance_stats(stats, cross_block.reshape(-1))
    return stats


@staticmethod
def _sample_within_class_distances(
    features: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    num_items = int(features.shape[0])
    if num_items < 2 or max_pairs <= 0:
        return np.empty(0, dtype=np.float32)

    total_pairs = (num_items * (num_items - 1)) // 2
    sample_size = int(min(max_pairs, total_pairs))
    row_idx = rng.integers(0, num_items, size=sample_size, dtype=np.int64)
    col_idx = rng.integers(0, num_items - 1, size=sample_size, dtype=np.int64)
    col_idx += (col_idx >= row_idx).astype(np.int64)
    return np.linalg.norm(features[row_idx] - features[col_idx], axis=1).astype(np.float32)


@staticmethod
def _sample_cross_class_distances(
    left_features: np.ndarray,
    right_features: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    left_features = np.asarray(left_features, dtype=np.float32)
    right_features = np.asarray(right_features, dtype=np.float32)
    if left_features.shape[0] == 0 or right_features.shape[0] == 0 or max_pairs <= 0:
        return np.empty(0, dtype=np.float32)

    total_pairs = int(left_features.shape[0] * right_features.shape[0])
    sample_size = int(min(max_pairs, total_pairs))
    left_idx = rng.integers(0, left_features.shape[0], size=sample_size, dtype=np.int64)
    right_idx = rng.integers(0, right_features.shape[0], size=sample_size, dtype=np.int64)
    return np.linalg.norm(left_features[left_idx] - right_features[right_idx], axis=1).astype(np.float32)


def _select_distance_analysis_subset(
    self,
    features: np.ndarray,
    labels: np.ndarray,
    max_points: int = 8000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    indices = np.arange(labels.shape[0], dtype=np.int64)
    summary = {
        "original_point_count": int(labels.shape[0]),
        "analyzed_point_count": int(labels.shape[0]),
        "used_subset": False,
        "max_points": int(max_points),
    }
    if labels.shape[0] <= int(max_points):
        return features, labels, indices, summary

    rng = np.random.default_rng(int(self.config.seed) + 2048)
    normal_idx = indices[labels == 0]
    anomaly_idx = indices[labels == 1]
    if normal_idx.size == 0 or anomaly_idx.size == 0:
        selected = np.sort(rng.choice(indices, size=int(max_points), replace=False).astype(np.int64))
    else:
        keep_anomaly = min(anomaly_idx.size, max(1, int(max_points // 3)))
        keep_normal = min(normal_idx.size, int(max_points) - keep_anomaly)
        if keep_normal < int(max_points) - keep_anomaly and anomaly_idx.size > keep_anomaly:
            keep_anomaly = min(anomaly_idx.size, int(max_points) - keep_normal)
        anomaly_selected = (
            anomaly_idx
            if keep_anomaly >= anomaly_idx.size
            else np.sort(rng.choice(anomaly_idx, size=keep_anomaly, replace=False).astype(np.int64))
        )
        normal_selected = (
            normal_idx
            if keep_normal >= normal_idx.size
            else np.sort(rng.choice(normal_idx, size=keep_normal, replace=False).astype(np.int64))
        )
        selected = np.sort(np.concatenate([normal_selected, anomaly_selected], axis=0)).astype(np.int64)

    summary["analyzed_point_count"] = int(selected.size)
    summary["used_subset"] = True
    return features[selected], labels[selected], selected, summary


def _analyze_test_latent_distances(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    subset_features, subset_labels, subset_indices, scope = self._select_distance_analysis_subset(features, labels)
    normal_features = subset_features[subset_labels == 0]
    anomaly_features = subset_features[subset_labels == 1]

    rng = np.random.default_rng(int(self.config.seed) + 4096)
    sample_pairs = 200000
    block_size = 512

    nn_stats = self._within_class_distance_stats(normal_features, block_size=block_size)
    na_stats = self._cross_class_distance_stats(normal_features, anomaly_features, block_size=block_size)
    aa_stats = self._within_class_distance_stats(anomaly_features, block_size=block_size)

    nn_sample = self._sample_within_class_distances(normal_features, sample_pairs, rng)
    na_sample = self._sample_cross_class_distances(normal_features, anomaly_features, sample_pairs, rng)
    aa_sample = self._sample_within_class_distances(anomaly_features, sample_pairs, rng)

    summary = {
        "scope": {
            **scope,
            "normal_count": int(normal_features.shape[0]),
            "anomaly_count": int(anomaly_features.shape[0]),
            "label_semantics": {"0": "normal", "1": "anomaly"},
        },
        "normal_normal": self._summarize_distance_stats(nn_stats, nn_sample),
        "normal_anomaly": self._summarize_distance_stats(na_stats, na_sample),
        "anomaly_anomaly": self._summarize_distance_stats(aa_stats, aa_sample),
    }

    scope_desc = "subset" if bool(scope["used_subset"]) else "full_test_set"
    print(
        "[Test][Latent Distance] "
        f"scope={scope_desc} | "
        f"points={int(scope['analyzed_point_count'])}/{int(scope['original_point_count'])} | "
        f"normal={int(normal_features.shape[0])} | "
        f"anomaly={int(anomaly_features.shape[0])}"
    )
    for group_name, group_summary in [
        ("normal-normal", summary["normal_normal"]),
        ("normal-anomaly", summary["normal_anomaly"]),
        ("anomaly-anomaly", summary["anomaly_anomaly"]),
    ]:
        print(
            f"[Test][Latent Distance] {group_name}: "
            f"pairs={int(group_summary['pair_count'])} | "
            f"mean={float(group_summary['mean']):.6f} | "
            f"std={float(group_summary['std']):.6f}"
        )

    return {
        "summary": summary,
        "analyzed_indices": subset_indices.astype(np.int64),
        "analyzed_labels": subset_labels.astype(np.int64),
        "analyzed_features": subset_features.astype(np.float32),
        "samples": {
            "normal_normal": nn_sample.astype(np.float32),
            "normal_anomaly": na_sample.astype(np.float32),
            "anomaly_anomaly": aa_sample.astype(np.float32),
        },
    }


def _collect_reconstruction_scores(self, loader: DataLoader) -> np.ndarray:
    was_training = self.model.training
    self.model.eval()

    scores = []
    with torch.inference_mode():
        for batch in loader:
            x = self._prepare_batch(batch)
            outputs = self.model(x, stage="stage0")
            batch_scores = self._mse_per_sample_from_outputs(outputs, x)
            scores.append(batch_scores.detach().cpu().numpy())

    if was_training:
        self.model.train()

    if not scores:
        raise RuntimeError("No windows were available for reconstruction scoring.")

    return np.concatenate(scores, axis=0).astype(np.float32)


def _collect_dual_reconstruction_scores(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    if not self._is_dual_view_model():
        scores = self._collect_reconstruction_scores(loader)
        return scores, scores

    was_training = self.model.training
    self.model.eval()

    scores_v1 = []
    scores_v2 = []
    with torch.inference_mode():
        for batch in loader:
            x = self._prepare_batch(batch)
            outputs = self.model(x, stage="stage0")
            scores_v1.append(self._mse_per_sample(x, outputs["x_hat1"], view="v1").detach().cpu().numpy())
            scores_v2.append(self._mse_per_sample(x, outputs["x_hat2"], view="v2").detach().cpu().numpy())

    if was_training:
        self.model.train()

    if not scores_v1 or not scores_v2:
        raise RuntimeError("No windows were available for dual reconstruction scoring.")

    return (
        np.concatenate(scores_v1, axis=0).astype(np.float32),
        np.concatenate(scores_v2, axis=0).astype(np.float32),
    )


@staticmethod
def _cross_view_disagreement_from_evidence(view1_score: np.ndarray, view2_score: np.ndarray) -> np.ndarray:
    view1_score = np.asarray(view1_score, dtype=np.float32).reshape(-1)
    view2_score = np.asarray(view2_score, dtype=np.float32).reshape(-1)
    if view1_score.shape[0] != view2_score.shape[0]:
        raise ValueError("View evidence scores should have the same length.")
    return np.abs(view1_score - view2_score).astype(np.float32)


def _save_stage1_reconstruction_scoring(self) -> Dict[str, object]:
    if not bool(getattr(self.config, "enable_stage1_recon_scoring", False)):
        return {}

    print("========== Stage1 Reconstruction-only Scoring ==========")
    train_score = self._collect_reconstruction_scores(self.train_eval_loader)
    test_score = self._collect_reconstruction_scores(self.test_loader)
    threshold = float(np.quantile(train_score, self.config.decision_quantile))
    pred_labels = (test_score > threshold).astype(np.int64)
    y_true = self._build_current_point_anomaly_labels(self.test_loader.dataset)
    y_count = y_true.astype(np.int64)
    point_indices = self._build_current_point_indices(self.test_loader.dataset)
    metrics = combine_all_evaluation_scores(y_true.copy(), pred_labels.copy(), test_score.copy())
    metrics = {str(key): float(value) for key, value in metrics.items()}

    pred_anomaly_idx = np.where(pred_labels == 1)[0].astype(np.int64)
    pred_anomaly_counts = y_count[pred_anomaly_idx] if pred_anomaly_idx.size > 0 else np.empty(0, dtype=np.int64)
    print(
        "[Stage1][ReconScore] "
        f"score=last_point_reconstruction_mse | "
        f"threshold_train_q={float(self.config.decision_quantile):.4f} | "
        f"threshold={threshold:.6f} | "
        f"train_score_mean={float(np.mean(train_score)):.6f} | "
        f"test_score_mean={float(np.mean(test_score)):.6f} | "
        f"pred_anomaly={int(pred_anomaly_idx.size)}"
    )
    print(
        "[Stage1][ReconScore] Pred anomaly point indices with labels: "
        f"{self._format_index_count_preview(point_indices[pred_anomaly_idx], pred_anomaly_counts)}"
    )
    for key, value in metrics.items():
        print(f"[Stage1][ReconScore] {key}: {value:.6f}")

    artifact_summary = {
        "score_name": "stage1_last_point_reconstruction_mse",
        "threshold": threshold,
        "threshold_quantile": float(self.config.decision_quantile),
        "metrics": metrics,
        "train_score_mean": float(np.mean(train_score)),
        "train_score_std": float(np.std(train_score)),
        "test_score_mean": float(np.mean(test_score)),
        "test_score_std": float(np.std(test_score)),
        "pred_anomaly_count": int(pred_anomaly_idx.size),
        "test_window_count": int(test_score.shape[0]),
        "test_point_count": int(test_score.shape[0]),
    }
    os.makedirs(self.config.save_dir, exist_ok=True)
    with open(os.path.join(self.config.save_dir, "stage1_recon_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(artifact_summary, file, ensure_ascii=False, indent=2)

    arrays = {
        "stage1_recon_train_scores.npy": train_score,
        "stage1_recon_test_scores.npy": test_score,
        "stage1_recon_pred_labels.npy": pred_labels,
        "stage1_recon_y_true.npy": y_true,
        "stage1_recon_point_indices.npy": point_indices,
    }
    for filename, value in arrays.items():
        with open(os.path.join(self.config.save_dir, filename), "wb") as file:
            np.save(file, value)

    return artifact_summary


@staticmethod
def _format_index_count_preview(indices: np.ndarray, counts: np.ndarray, max_items: int = 80) -> str:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if indices.size == 0 or counts.size == 0:
        return "[]"
    size = min(indices.size, counts.size)
    items = [f"{int(indices[i])}:{int(counts[i])}" for i in range(size)]
    if size <= int(max_items):
        return "[" + ", ".join(items) + "]"
    head_count = max(1, int(max_items) // 2)
    suffix_count = max(1, int(max_items) - head_count)
    head = ", ".join(items[:head_count])
    suffix = ", ".join(items[-suffix_count:])
    return f"[{head}, ... , {suffix}]"


def _evaluate_score_family(
    self,
    *,
    label: str,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    y_true: np.ndarray,
    y_count: np.ndarray,
    item_indices: np.ndarray = None,
    item_name: str = "window",
) -> Tuple[Dict[str, float], np.ndarray, float]:
    train_scores = np.asarray(train_scores, dtype=np.float32).reshape(-1)
    test_scores = np.asarray(test_scores, dtype=np.float32).reshape(-1)
    pred_labels, threshold = self._threshold_scores(train_scores, test_scores)
    metrics = combine_all_evaluation_scores(y_true.copy(), pred_labels.copy(), test_scores.copy())
    metrics = {str(key): float(value) for key, value in metrics.items()}
    pred_anomaly_idx = np.where(pred_labels == 1)[0].astype(np.int64)
    pred_anomaly_counts = y_count[pred_anomaly_idx] if pred_anomaly_idx.size > 0 else np.empty(0, dtype=np.int64)
    display_indices = (
        np.asarray(item_indices, dtype=np.int64).reshape(-1)[pred_anomaly_idx]
        if item_indices is not None
        else pred_anomaly_idx
    )

    print(
        f"[Test][{label}] "
        f"threshold_train_q={float(self.config.decision_quantile):.4f} | "
        f"threshold={threshold:.6f} | "
        f"train_mean={float(np.mean(train_scores)):.6f} | "
        f"test_mean={float(np.mean(test_scores)):.6f} | "
        f"pred_anomaly={int(pred_anomaly_idx.size)}"
    )
    print(
        f"[Test][{label}] Pred anomaly {item_name} indices with label/count: "
        f"{self._format_index_count_preview(display_indices, pred_anomaly_counts)}"
    )
    for key, value in metrics.items():
        print(f"[Test][{label}] {key}: {value:.6f}")
    return metrics, pred_labels, threshold


@staticmethod
def _normalize_score_pair(train_score: np.ndarray, test_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train_score = np.asarray(train_score, dtype=np.float32).reshape(-1)
    test_score = np.asarray(test_score, dtype=np.float32).reshape(-1)
    median = float(np.median(train_score)) if train_score.size > 0 else 0.0
    q25, q75 = np.quantile(train_score, [0.25, 0.75]) if train_score.size > 0 else (0.0, 1.0)
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(train_score))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return (
        np.maximum((train_score - median) / max(scale, 1e-6), 0.0).astype(np.float32),
        np.maximum((test_score - median) / max(scale, 1e-6), 0.0).astype(np.float32),
    )


def _collect_cross_view_scores(self, loader: DataLoader) -> np.ndarray:
    if not self._is_dual_view_model():
        return np.zeros(len(loader.dataset), dtype=np.float32)
    was_training = self.model.training
    self.model.eval()
    scores = []
    with torch.inference_mode():
        for batch in loader:
            x = self._prepare_batch(batch)
            z1, z2 = self.model.encode_views(x)
            batch_scores = torch.mean((z1 - z2) ** 2, dim=1)
            scores.append(batch_scores.detach().cpu().numpy())
    if was_training:
        self.model.train()
    return np.concatenate(scores, axis=0).astype(np.float32)


def _collect_pointwise_separate_proto_eval_outputs(self, loader: DataLoader) -> Dict[str, np.ndarray]:
    """
    Collect one current-point score vector per local L-window.

    New route semantics: each dataset sample is X_t=[x_{t-L+1},...,x_t]
    and the model outputs H_t only. Therefore the correct label is the last
    label in the returned label window, not window-any and not an average over
    duplicate appearances from a T=100 point-level encoder.
    """
    if not self._is_dual_view_model():
        raise RuntimeError("Current-point prototype scoring requires the dual-view model.")

    dataset = loader.dataset
    window_indices = self._dataset_point_window_indices(dataset)

    def _chain_attr(name: str, default=None):
        current = dataset
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if hasattr(current, name):
                value = getattr(current, name)
                if value is not None:
                    return value
            current = getattr(current, "base_dataset", getattr(current, "dataset", None))
        return default

    step = int(_chain_attr("step", 1))
    win_size = int(_chain_attr("win_size", getattr(self.config, "seq_len", 1)))
    current_point_offset = int(_chain_attr("current_point_offset", max(0, win_size - 1)))

    collected = {
        "u1": [],
        "u2": [],
        "u_joint": [],
        "q1": [],
        "q2": [],
        "proto_dist_matrix1": [],
        "proto_dist_matrix2": [],
        "proto_dist1": [],
        "proto_dist2": [],
        "pred1": [],
        "pred2": [],
        "conf1": [],
        "conf2": [],
        "recon1": [],
        "recon2": [],
        "point_labels": [],
        "point_indices": [],
    }

    cursor = 0
    was_training = self.model.training
    self.model.eval()

    with torch.inference_mode():
        for batch in loader:
            x = self._prepare_batch(batch)
            outputs = self.model(x, stage="test")
            h1 = outputs["H1"]
            h2 = outputs["H2"]
            u_joint = torch.nn.functional.normalize(torch.cat([h1, h2], dim=1), dim=1)
            recon1 = self._mse_per_sample(x, outputs["x_hat1"], view="v1")
            recon2 = self._mse_per_sample(x, outputs["x_hat2"], view="v2")

            batch_size = int(x.size(0))
            batch_window_indices = window_indices[cursor:cursor + batch_size]
            point_indices = batch_window_indices.astype(np.int64) * int(step) + current_point_offset

            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                labels_raw = batch[1]
                if isinstance(labels_raw, torch.Tensor):
                    labels_np = labels_raw.detach().cpu().numpy()
                else:
                    labels_np = np.asarray(labels_raw)
                labels_np = labels_np.reshape(batch_size, -1)
                point_labels = (labels_np[:, -1] > 0).astype(np.int64)
            else:
                point_label_source = self._build_point_anomaly_labels(dataset)
                safe_indices = np.clip(point_indices, 0, max(0, point_label_source.shape[0] - 1))
                point_labels = point_label_source[safe_indices].astype(np.int64)

            collected["u1"].append(h1.detach().cpu().numpy())
            collected["u2"].append(h2.detach().cpu().numpy())
            collected["u_joint"].append(u_joint.detach().cpu().numpy())
            collected["q1"].append(outputs["q1"].detach().cpu().numpy())
            collected["q2"].append(outputs["q2"].detach().cpu().numpy())
            collected["proto_dist_matrix1"].append(outputs["proto_dist_matrix1"].detach().cpu().numpy())
            collected["proto_dist_matrix2"].append(outputs["proto_dist_matrix2"].detach().cpu().numpy())
            collected["proto_dist1"].append(outputs["proto_dist1"].detach().cpu().numpy())
            collected["proto_dist2"].append(outputs["proto_dist2"].detach().cpu().numpy())
            collected["pred1"].append(outputs["proto_pred1"].detach().cpu().numpy())
            collected["pred2"].append(outputs["proto_pred2"].detach().cpu().numpy())
            collected["conf1"].append(outputs["proto_conf1"].detach().cpu().numpy())
            collected["conf2"].append(outputs["proto_conf2"].detach().cpu().numpy())
            collected["recon1"].append(recon1.detach().cpu().numpy())
            collected["recon2"].append(recon2.detach().cpu().numpy())
            collected["point_labels"].append(point_labels.astype(np.int64))
            collected["point_indices"].append(point_indices.astype(np.int64))
            cursor += batch_size

    if was_training:
        self.model.train()

    if not collected["u1"]:
        raise RuntimeError("Current-point scoring found no encoded windows.")

    outputs = {}
    for key, values in collected.items():
        dtype = np.int64 if key in {"point_labels", "point_indices"} else np.float32
        outputs[key] = np.concatenate(values, axis=0).astype(dtype)
    return outputs

def _collect_separate_proto_eval_outputs(self, loader: DataLoader) -> Dict[str, np.ndarray]:
    was_training = self.model.training
    self.model.eval()
    collected = {
        "u1": [],
        "u2": [],
        "u_joint": [],
        "q1": [],
        "q2": [],
        "proto_dist_matrix1": [],
        "proto_dist_matrix2": [],
        "proto_dist1": [],
        "proto_dist2": [],
        "pred1": [],
        "pred2": [],
        "conf1": [],
        "conf2": [],
        "recon1": [],
        "recon2": [],
    }
    with torch.inference_mode():
        for batch in loader:
            x = self._prepare_batch(batch)
            outputs = self.model(x, stage="test")
            if self._is_dual_view_model():
                u1 = outputs["u1"]
                u2 = outputs["u2"]
                q1 = outputs["q1"]
                q2 = outputs["q2"]
                proto_dist_matrix1 = outputs["proto_dist_matrix1"]
                proto_dist_matrix2 = outputs["proto_dist_matrix2"]
                proto_dist1 = outputs["proto_dist1"]
                proto_dist2 = outputs["proto_dist2"]
                pred1 = outputs["proto_pred1"]
                pred2 = outputs["proto_pred2"]
                conf1 = outputs["proto_conf1"]
                conf2 = outputs["proto_conf2"]
                recon1 = self._mse_per_sample(x, outputs["x_hat1"], view="v1")
                recon2 = self._mse_per_sample(x, outputs["x_hat2"], view="v2")
                if self._uses_separate_prototypes():
                    u_joint = torch.nn.functional.normalize(torch.cat([u1, u2], dim=1), dim=1)
                else:
                    u_joint = torch.nn.functional.normalize(0.5 * (u1 + u2), dim=1)
            else:
                u1 = outputs["u"]
                u2 = outputs["u"]
                q1 = outputs["q"]
                q2 = outputs["q"]
                proto_dist_matrix1 = outputs["proto_dist_matrix"]
                proto_dist_matrix2 = outputs["proto_dist_matrix"]
                proto_dist1 = outputs["proto_dist"]
                proto_dist2 = outputs["proto_dist"]
                pred1 = outputs["proto_pred"]
                pred2 = outputs["proto_pred"]
                conf1 = outputs["proto_conf"]
                conf2 = outputs["proto_conf"]
                recon = self._mse_per_sample(x, outputs["x_hat"], view=self._normalize_reconstruction_view())
                recon1 = recon
                recon2 = recon
                u_joint = torch.nn.functional.normalize(u1, dim=1)
            collected["u1"].append(u1.detach().cpu().numpy())
            collected["u2"].append(u2.detach().cpu().numpy())
            collected["u_joint"].append(u_joint.detach().cpu().numpy())
            collected["q1"].append(q1.detach().cpu().numpy())
            collected["q2"].append(q2.detach().cpu().numpy())
            collected["proto_dist_matrix1"].append(proto_dist_matrix1.detach().cpu().numpy())
            collected["proto_dist_matrix2"].append(proto_dist_matrix2.detach().cpu().numpy())
            collected["proto_dist1"].append(proto_dist1.detach().cpu().numpy())
            collected["proto_dist2"].append(proto_dist2.detach().cpu().numpy())
            collected["pred1"].append(pred1.detach().cpu().numpy())
            collected["pred2"].append(pred2.detach().cpu().numpy())
            collected["conf1"].append(conf1.detach().cpu().numpy())
            collected["conf2"].append(conf2.detach().cpu().numpy())
            collected["recon1"].append(recon1.detach().cpu().numpy())
            collected["recon2"].append(recon2.detach().cpu().numpy())
    if was_training:
        self.model.train()
    return {key: np.concatenate(value, axis=0).astype(np.float32) for key, value in collected.items()}


def _test_dual_separate_proto(self):
    self._uses_separate_prototypes()
    family_label = "Separate"
    print(f"========== Testing Dual-View {family_label} Prototypes ==========")
    train_outputs = self._collect_pointwise_separate_proto_eval_outputs(self.train_eval_loader)
    test_outputs = self._collect_pointwise_separate_proto_eval_outputs(self.test_loader)

    train_prev = previous_indices_from_point_indices(train_outputs["point_indices"])
    test_prev = previous_indices_from_point_indices(test_outputs["point_indices"])
    eps = float(getattr(self.config, "robust_eps", 1e-6))

    train_components = compute_separate_proto_component_scores(
        train_outputs["proto_dist_matrix1"],
        train_outputs["proto_dist_matrix2"],
        train_outputs["recon1"],
        train_outputs["recon2"],
        train_outputs["q1"],
        train_outputs["q2"],
        q1_prev=train_outputs["q1"][train_prev],
        q2_prev=train_outputs["q2"][train_prev],
        eps=eps,
    )
    components = compute_separate_proto_component_scores(
        test_outputs["proto_dist_matrix1"],
        test_outputs["proto_dist_matrix2"],
        test_outputs["recon1"],
        test_outputs["recon2"],
        test_outputs["q1"],
        test_outputs["q2"],
        q1_prev=test_outputs["q1"][test_prev],
        q2_prev=test_outputs["q2"][test_prev],
        eps=eps,
    )

    y_true = test_outputs["point_labels"]
    y_count = y_true.astype(np.int64)
    print(
        f"[Test][Labels] point labels: normal={int(np.sum(y_true == 0))} | "
        f"anomaly={int(np.sum(y_true == 1))} | total={int(y_true.shape[0])}"
    )
    distance_analysis = self._analyze_test_latent_distances(test_outputs["u_joint"], y_true)
    proto_head = self.model.prototype_head_v1

    print(
        f"{family_label} component scoring: "
        "no fused score emitted | "
        "scores=score_recon_v1,score_recon_v2,score_proto_v1,score_proto_v2,"
        "score_dist_gap,score_cross_view_js,score_temporal_js | "
        f"K={int(getattr(proto_head, 'num_prototypes', 0))} | "
        f"state_dim={int(getattr(self.model, 'state_dim', 0))}"
    )

    components.update(
        {
            "view1_conf": test_outputs["conf1"],
            "view2_conf": test_outputs["conf2"],
            "view1_pred": test_outputs["pred1"],
            "view2_pred": test_outputs["pred2"],
            "q1": test_outputs["q1"],
            "q2": test_outputs["q2"],
        }
    )
    train_components.update(
        {
            "view1_conf": train_outputs["conf1"],
            "view2_conf": train_outputs["conf2"],
            "view1_pred": train_outputs["pred1"],
            "view2_pred": train_outputs["pred2"],
        }
    )

    score_specs = [
        ("score_recon_v1", f"{family_label}ScoreReconV1"),
        ("score_recon_v2", f"{family_label}ScoreReconV2"),
        ("score_proto_v1", f"{family_label}ScoreProtoV1"),
        ("score_proto_v2", f"{family_label}ScoreProtoV2"),
        ("score_dist_gap", f"{family_label}ScoreDistGap"),
        ("score_cross_view_js", f"{family_label}ScoreCrossViewJS"),
        ("score_temporal_js", f"{family_label}ScoreTemporalJS"),
    ]
    component_families = {}
    for score_key, score_label in score_specs:
        family_metrics, family_pred, family_threshold = self._evaluate_score_family(
            label=score_label,
            train_scores=train_components[score_key],
            test_scores=components[score_key],
            y_true=y_true,
            y_count=y_count,
            item_indices=test_outputs["point_indices"],
            item_name="point",
        )
        component_families[score_key] = {
            "metrics": family_metrics,
            "threshold": family_threshold,
            "train_scores": train_components[score_key],
            "test_scores": components[score_key],
            "pred_labels": family_pred,
        }

    return {
        "metrics": {},
        "threshold": None,
        "y_true": y_true,
        "pred_labels": None,
        "scores": None,
        "components": components,
        "train_components": train_components,
        "score_name": "separate_proto_component_scores_only",
        "test_features": test_outputs["u_joint"],
        "distance_analysis": distance_analysis,
        "component_families": component_families,
        "banks": self._export_bank_artifacts(),
    }


def run_evaluation(solver):
    self = solver
    return self._test_dual_separate_proto()
