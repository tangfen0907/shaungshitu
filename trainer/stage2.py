from typing import Dict, List, Optional, Tuple

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_factory.triplet_dataset import _extract_window, _to_channel_first_tensor
from utils.clustering import _cluster_features, _nearest_other_clusters
from utils.losses import (
    consensus_teacher_distribution,
    js_divergence,
    prototype_usage_balance_loss,
    prototype_repulsion_loss,
    prototype_separation_loss,
    state_consistency_teacher_loss,
)


__all__ = [
    "_stage2_total_epochs",
    "_stage2_method",
    "_uses_separate_prototypes",
    "_resolve_stage2_schedule",
    "_zero_stage2_loss",
    "_collect_consensus_state_features",
    "_collect_paired_state_features",
    "_build_paired_kmeans_features",
    "_collect_consensus_proto_outputs",
    "_initialize_consensus_prototypes",
    "_initialize_paired_prototypes",
    "_build_consensus_proto_state",
    "_cached_joint_core_label_arrays",
    "_log_joint_core_label_diagnostics",
    "_refresh_consensus_joint_core",
    "_build_stage2_consensus_loader",
    "_paired_prototype_relation_loss",
    "_consensus_proto_batch_loss",
    "_stage2_consensus_proto_epoch",
    "_run_stage2_consensus_proto_refinement",
    "run_stage2_consensus_proto_refinement",
]


class _Stage2ConsensusDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return int(len(self.base_dataset))

    def __getitem__(self, idx: int):
        window = _to_channel_first_tensor(_extract_window(self.base_dataset[int(idx)]))
        return window, torch.tensor(int(idx), dtype=torch.long)


def _stage2_total_epochs(self) -> int:
    rounds, epochs_per_round = self._resolve_stage2_schedule()
    return rounds * epochs_per_round


def _stage2_method(self) -> str:
    method = str(getattr(self.config, "stage2_method", "separate_proto")).strip().lower()
    aliases = {
        "prototype": "separate_proto",
        "consensus": "separate_proto",
        "consensus_proto": "separate_proto",
        "consensus_proto_v2": "separate_proto",
        "consensus_proto_balanced": "separate_proto",
        "balanced_consensus_proto": "separate_proto",
        "paired": "separate_proto",
        "paired_proto": "separate_proto",
        "paired_prototype": "separate_proto",
        "separate": "separate_proto",
        "separate_prototype": "separate_proto",
    }
    method = aliases.get(method, method)
    if method in {"shared_proto", "shared_prototype", "common_proto", "common_prototype"}:
        raise ValueError("Shared/common prototype mode has been removed. Use 'separate_proto'.")
    if method != "separate_proto":
        raise ValueError("stage2_method should be 'separate_proto'.")
    return method


def _uses_separate_prototypes(self) -> bool:
    return bool(self._is_dual_view_model())


def _resolve_stage2_schedule(self) -> Tuple[int, int]:
    rounds = int(getattr(self.config, "num_stage2_rounds", -1))
    epochs_per_round = int(getattr(self.config, "epochs_per_stage2_round", -1))
    legacy_total_epochs = max(1, int(getattr(self.config, "epoch_stage2", 1)))
    legacy_block_epochs = max(1, int(getattr(self.config, "stage2_block_epochs", 5)))

    if rounds > 0 and epochs_per_round > 0:
        return rounds, epochs_per_round
    if rounds > 0:
        return rounds, max(1, int(np.ceil(legacy_total_epochs / float(rounds))))
    if epochs_per_round > 0:
        return max(1, int(np.ceil(legacy_total_epochs / float(epochs_per_round)))), epochs_per_round
    return max(1, int(np.ceil(legacy_total_epochs / float(legacy_block_epochs)))), legacy_block_epochs


def _zero_stage2_loss(self) -> torch.Tensor:
    return torch.zeros((), device=self.device)


def _collect_consensus_state_features(self, loader: DataLoader) -> np.ndarray:
    was_training = self.model.training
    self.model.eval()
    features: List[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            x = self._prepare_batch(batch)
            if self._is_dual_view_model():
                z1, z2 = self.model.encode_views(x)
                u1, u2 = self.model.state_from_views(z1, z2)
                u = F.normalize(0.5 * (u1 + u2), dim=1)
            else:
                z = self.model.encode(x)
                u = F.normalize(self.model.project_state(z), dim=1)
            features.append(u.detach().cpu().numpy())
    if was_training:
        self.model.train()
    if not features:
        raise RuntimeError("No windows were available for consensus prototype initialization.")
    return np.concatenate(features, axis=0).astype(np.float32)


def _collect_paired_state_features(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    if not self._is_dual_view_model():
        raise RuntimeError("separate_proto requires a dual-view model.")
    was_training = self.model.training
    self.model.eval()
    features_v1: List[np.ndarray] = []
    features_v2: List[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            x = self._prepare_batch(batch)
            z1, z2 = self.model.encode_views(x)
            u1, u2 = self.model.state_from_views(z1, z2)
            features_v1.append(u1.detach().cpu().numpy())
            features_v2.append(u2.detach().cpu().numpy())
    if was_training:
        self.model.train()
    if not features_v1 or not features_v2:
        raise RuntimeError("No windows were available for separate prototype initialization.")
    return (
        np.concatenate(features_v1, axis=0).astype(np.float32),
        np.concatenate(features_v2, axis=0).astype(np.float32),
    )


def _build_paired_kmeans_features(
    self,
    features_v1: np.ndarray,
    features_v2: np.ndarray,
) -> np.ndarray:
    eps = float(getattr(self.config, "robust_eps", 1e-6))

    def _normalize_then_standardize(features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norm, eps)
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        arr = (arr - mean) / np.maximum(std, eps)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    view1 = _normalize_then_standardize(features_v1)
    view2 = _normalize_then_standardize(features_v2)
    scale = np.float32(0.70710678118)
    return np.concatenate([scale * view1, scale * view2], axis=1).astype(np.float32)


def _collect_consensus_proto_outputs(self, loader: DataLoader) -> Dict[str, np.ndarray]:
    was_training = self.model.training
    self.model.eval()
    collected: Dict[str, List[np.ndarray]] = {
        "u1": [],
        "u2": [],
        "u_cons": [],
        "q1": [],
        "q2": [],
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
            outputs = self.model(x, stage="stage2")
            if self._is_dual_view_model():
                u1 = outputs["u1"]
                u2 = outputs["u2"]
                q1 = outputs["q1"]
                q2 = outputs["q2"]
                proto_dist1 = outputs["proto_dist1"]
                proto_dist2 = outputs["proto_dist2"]
                pred1 = outputs["proto_pred1"]
                pred2 = outputs["proto_pred2"]
                conf1 = outputs["proto_conf1"]
                conf2 = outputs["proto_conf2"]
                recon1 = self._mse_per_sample(x, outputs["x_hat1"], view="v1")
                recon2 = self._mse_per_sample(x, outputs["x_hat2"], view="v2")
                if self._uses_separate_prototypes():
                    u_cons = F.normalize(torch.cat([u1, u2], dim=1), dim=1)
                else:
                    u_cons = F.normalize(0.5 * (u1 + u2), dim=1)
            else:
                u1 = outputs["u"]
                u2 = outputs["u"]
                q1 = outputs["q"]
                q2 = outputs["q"]
                proto_dist1 = outputs["proto_dist"]
                proto_dist2 = outputs["proto_dist"]
                pred1 = outputs["proto_pred"]
                pred2 = outputs["proto_pred"]
                conf1 = outputs["proto_conf"]
                conf2 = outputs["proto_conf"]
                recon = self._mse_per_sample(x, outputs["x_hat"], view=self._normalize_reconstruction_view())
                recon1 = recon
                recon2 = recon
                u_cons = F.normalize(u1, dim=1)
            collected["u1"].append(u1.detach().cpu().numpy())
            collected["u2"].append(u2.detach().cpu().numpy())
            collected["u_cons"].append(u_cons.detach().cpu().numpy())
            collected["q1"].append(q1.detach().cpu().numpy())
            collected["q2"].append(q2.detach().cpu().numpy())
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
    result: Dict[str, np.ndarray] = {}
    for key, values in collected.items():
        if not values:
            raise RuntimeError(f"Separate prototype collection produced no values for {key}.")
        result[key] = np.concatenate(values, axis=0)
    return result


def _initialize_consensus_prototypes(self):
    features = self._collect_consensus_state_features(self.train_eval_loader)
    num_prototypes = int(getattr(self.model.prototype_head, "num_prototypes", 1))
    labels, centers, cluster_meta = _cluster_features(
        features=features,
        cluster_method="kmeans",
        n_clusters=max(1, min(num_prototypes, features.shape[0])),
        random_state=int(getattr(self.config, "seed", 42)),
    )
    if centers.shape[0] != num_prototypes:
        raise RuntimeError(
            "Single-view prototype count does not match the model head. "
            f"head={num_prototypes}, kmeans={centers.shape[0]}"
        )
    self.model.init_prototypes_from_centers(centers)
    print(
        "[Stage2-Single] initialized learnable prototypes | "
        f"cluster_actual={cluster_meta.get('cluster_method_actual', 'kmeans')} | "
        f"K={int(centers.shape[0])} | "
        f"state_dim={int(centers.shape[1])} | "
        f"temperature={float(getattr(self.config, 'proto_temperature', 0.2)):.4f}"
    )
    self._refresh_consensus_joint_core(round_idx=0)


def _initialize_paired_prototypes(self):
    features_v1, features_v2 = self._collect_paired_state_features(self.train_eval_loader)
    num_prototypes = int(
        getattr(
            getattr(self.model, "prototype_head_v1", self.model.prototype_head),
            "num_prototypes",
            1,
        )
    )
    joint_features = self._build_paired_kmeans_features(features_v1, features_v2)
    labels, _, cluster_meta = _cluster_features(
        features=joint_features,
        cluster_method="kmeans",
        n_clusters=max(1, min(num_prototypes, joint_features.shape[0])),
        random_state=int(getattr(self.config, "seed", 42)),
    )
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    label_count = int(labels.max()) + 1 if labels.size else 0
    if label_count != num_prototypes:
        raise RuntimeError(
            "Separate prototype count does not match the model heads. "
            f"head={num_prototypes}, kmeans={label_count}"
        )

    centers_v1 = np.zeros((num_prototypes, features_v1.shape[1]), dtype=np.float32)
    centers_v2 = np.zeros((num_prototypes, features_v2.shape[1]), dtype=np.float32)
    for proto_id in range(num_prototypes):
        mask = labels == proto_id
        if not np.any(mask):
            fallback_idx = int(proto_id % max(1, features_v1.shape[0]))
            centers_v1[proto_id] = features_v1[fallback_idx]
            centers_v2[proto_id] = features_v2[fallback_idx]
            continue
        centers_v1[proto_id] = features_v1[mask].mean(axis=0)
        centers_v2[proto_id] = features_v2[mask].mean(axis=0)

    self.model.init_separate_prototypes_from_centers(centers_v1, centers_v2)
    print(
        "[Stage2-Separate] initialized separate prototypes | "
        f"cluster_actual={cluster_meta.get('cluster_method_actual', 'kmeans')} | "
        "init=joint_kmeans_l2_zscore | "
        f"K={int(num_prototypes)} | "
        f"state_dim={int(centers_v1.shape[1])} | "
        f"temperature={float(getattr(self.config, 'proto_temperature', 0.2)):.4f}"
    )
    self._refresh_consensus_joint_core(round_idx=0)


def _build_consensus_proto_state(
    self,
    outputs: Dict[str, np.ndarray],
    joint_core_mask: np.ndarray,
    round_idx: int,
) -> Dict[str, object]:
    use_separate = self._uses_separate_prototypes()
    centers_v1 = None
    centers_v2 = None
    if use_separate:
        centers_v1 = self.model.prototype_head_v1.prototypes.detach().cpu().numpy().astype(np.float32)
        centers_v2 = self.model.prototype_head_v2.prototypes.detach().cpu().numpy().astype(np.float32)
        centers = (0.5 * (centers_v1 + centers_v2)).astype(np.float32)
    else:
        centers = self.model.prototype_head.prototypes.detach().cpu().numpy().astype(np.float32)
    q_avg = 0.5 * (
        np.asarray(outputs["q1"], dtype=np.float32)
        + np.asarray(outputs["q2"], dtype=np.float32)
    )
    labels = np.argmax(q_avg, axis=1).astype(np.int64)
    core_mask = np.asarray(joint_core_mask, dtype=bool).reshape(-1)
    global_center = centers.mean(axis=0).astype(np.float32)
    cluster_radii = np.linalg.norm(centers - global_center[None, :], axis=1).astype(np.float32)
    nearest_other = _nearest_other_clusters(centers)
    score_core = np.zeros(labels.shape[0], dtype=np.float32)
    score_core[core_mask] = 1.0

    bank_summary: List[Dict[str, object]] = []
    for cluster_id in range(int(centers.shape[0])):
        cluster_indices = np.where(labels == cluster_id)[0]
        bank_summary.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": int(cluster_indices.size),
                "core_count": int(np.sum(core_mask[cluster_indices])) if cluster_indices.size else 0,
                "mean_score_core": float(np.mean(score_core[cluster_indices])) if cluster_indices.size else 0.0,
            }
        )

    state = {
        "stage2_method": self._stage2_method(),
        "cluster_centers": centers,
        "cluster_labels": labels,
        "global_center": global_center,
        "cluster_radii": cluster_radii,
        "nearest_other_cluster": nearest_other,
        "core_mask": core_mask.copy(),
        "score_core": score_core,
        "refresh_round": int(round_idx),
        "bank_mode": "separate_proto",
        "bank_summary": bank_summary,
        "proto_conf1": np.asarray(outputs["conf1"], dtype=np.float32),
        "proto_conf2": np.asarray(outputs["conf2"], dtype=np.float32),
        "proto_pred1": np.asarray(outputs["pred1"], dtype=np.int64),
        "proto_pred2": np.asarray(outputs["pred2"], dtype=np.int64),
        "proto_dist1": np.asarray(outputs["proto_dist1"], dtype=np.float32),
        "proto_dist2": np.asarray(outputs["proto_dist2"], dtype=np.float32),
        "recon1": np.asarray(outputs["recon1"], dtype=np.float32),
        "recon2": np.asarray(outputs["recon2"], dtype=np.float32),
    }
    if centers_v1 is not None and centers_v2 is not None:
        state["cluster_centers_v1"] = centers_v1
        state["cluster_centers_v2"] = centers_v2

    self.cluster_centers = centers
    self.cluster_labels = labels
    self.core_mask = core_mask
    self.cluster_radii = cluster_radii
    self.nearest_other_clusters = nearest_other
    self.global_center = global_center
    self.score_core = score_core
    self.stage2_refresh_round = int(round_idx)
    self.bank_summary = bank_summary
    self.cluster_bank = state
    self.stage2_structure = state
    if centers.shape[1] == int(getattr(self.config, "latent_dim", centers.shape[1])):
        try:
            self.model.svdd.update_centers(centers)
        except Exception:
            pass
    return state


def _cached_joint_core_label_arrays(self, dataset):
    cache = getattr(self, "_joint_core_label_cache", None)
    cache_key = (id(dataset), int(len(dataset)))
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        return cache.get("flags"), cache.get("counts")

    try:
        if hasattr(self, "_build_train_window_anomaly_flags_for_visualization"):
            flags = self._build_train_window_anomaly_flags_for_visualization(dataset)
            counts = self._build_train_window_anomaly_counts_for_visualization(dataset)
        else:
            flags = self._build_window_anomaly_flags(dataset)
            counts = self._build_window_anomaly_counts(dataset)
    except Exception as exc:
        print(f"[Stage2-CoreDiag] skipped: cannot build train label windows ({type(exc).__name__}: {exc})")
        return None, None

    flags = np.asarray(flags, dtype=np.int64).reshape(-1)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if flags.shape[0] != int(len(dataset)) or counts.shape[0] != int(len(dataset)):
        print(
            "[Stage2-CoreDiag] skipped: label length mismatch | "
            f"labels={flags.shape[0]} counts={counts.shape[0]} dataset={int(len(dataset))}"
        )
        return None, None

    self._joint_core_label_cache = {
        "key": cache_key,
        "flags": flags,
        "counts": counts,
    }
    return flags, counts


def _log_joint_core_label_diagnostics(self, state: Dict[str, object], round_idx: int):
    if not bool(getattr(self.config, "enable_joint_core_label_diagnostics", False)):
        return None

    dataset = self.train_eval_loader.dataset
    flags, counts = self._cached_joint_core_label_arrays(dataset)
    if flags is None or counts is None:
        return None

    core_mask = np.asarray(state.get("core_mask", []), dtype=bool).reshape(-1)
    cluster_labels = np.asarray(state.get("cluster_labels", []), dtype=np.int64).reshape(-1)
    if core_mask.shape[0] != flags.shape[0] or cluster_labels.shape[0] != flags.shape[0]:
        print(
            "[Stage2-CoreDiag] skipped: state length mismatch | "
            f"core={core_mask.shape[0]} labels={cluster_labels.shape[0]} train={flags.shape[0]}"
        )
        return None

    def _ratio(numerator: int, denominator: int) -> float:
        return float(numerator) / float(denominator) if int(denominator) > 0 else 0.0

    total = int(flags.shape[0])
    active_anomaly = int(np.sum(flags > 0))
    core_total = int(np.sum(core_mask))
    core_anomaly = int(np.sum(flags[core_mask] > 0)) if core_total else 0
    noncore_mask = ~core_mask
    noncore_total = int(np.sum(noncore_mask))
    noncore_anomaly = int(np.sum(flags[noncore_mask] > 0)) if noncore_total else 0
    active_ratio = _ratio(active_anomaly, total)
    core_ratio = _ratio(core_anomaly, core_total)
    noncore_ratio = _ratio(noncore_anomaly, noncore_total)

    cluster_centers = np.asarray(state.get("cluster_centers", np.zeros((0, 0))))
    num_proto = int(cluster_centers.shape[0])
    per_proto = []
    for proto_id in range(max(0, num_proto)):
        assigned_mask = cluster_labels == proto_id
        assigned_total = int(np.sum(assigned_mask))
        assigned_anomaly = int(np.sum(flags[assigned_mask] > 0)) if assigned_total else 0
        proto_core_mask = assigned_mask & core_mask
        proto_core_total = int(np.sum(proto_core_mask))
        proto_core_anomaly = int(np.sum(flags[proto_core_mask] > 0)) if proto_core_total else 0
        per_proto.append(
            {
                "proto_id": int(proto_id),
                "assigned_total": assigned_total,
                "assigned_anomaly": assigned_anomaly,
                "assigned_anomaly_ratio": _ratio(assigned_anomaly, assigned_total),
                "core_total": proto_core_total,
                "core_anomaly": proto_core_anomaly,
                "core_anomaly_ratio": _ratio(proto_core_anomaly, proto_core_total),
            }
        )

    diagnostic = {
        "round": int(round_idx),
        "active_total": total,
        "active_anomaly": active_anomaly,
        "active_anomaly_ratio": active_ratio,
        "core_total": core_total,
        "core_anomaly": core_anomaly,
        "core_anomaly_ratio": core_ratio,
        "noncore_total": noncore_total,
        "noncore_anomaly": noncore_anomaly,
        "noncore_anomaly_ratio": noncore_ratio,
        "core_vs_active_ratio": core_ratio / active_ratio if active_ratio > 0.0 else 0.0,
        "per_proto": per_proto,
    }

    history = list(getattr(self, "joint_core_label_diagnostics", []))
    history.append(diagnostic)
    self.joint_core_label_diagnostics = history
    state["joint_core_label_diagnostics"] = history
    state["latest_joint_core_label_diagnostic"] = diagnostic

    core_ratios = [round(float(item["core_anomaly_ratio"]), 4) for item in per_proto]
    core_anom_counts = [int(item["core_anomaly"]) for item in per_proto]
    print(
        "[Stage2-CoreDiag] "
        f"round={int(round_idx)} | "
        f"active_anom={active_anomaly}/{total} ({active_ratio:.4%}) | "
        f"core_anom={core_anomaly}/{core_total} ({core_ratio:.4%}) | "
        f"noncore_anom={noncore_anomaly}/{noncore_total} ({noncore_ratio:.4%}) | "
        f"core_vs_active={diagnostic['core_vs_active_ratio']:.3f}"
    )
    print(
        "[Stage2-CoreDiag] "
        f"round={int(round_idx)} | "
        f"core_anom_by_proto={core_anom_counts} | "
        f"core_anom_ratio_by_proto={core_ratios}"
    )

    try:
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, "joint_core_label_diagnostics.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[Stage2-CoreDiag] could not save diagnostics: {type(exc).__name__}: {exc}")

    return diagnostic


def _refresh_consensus_joint_core(self, round_idx: int):
    outputs = self._collect_consensus_proto_outputs(self.train_eval_loader)
    tau_conf = float(getattr(self.config, "tau_conf", 0.7))
    pred1 = np.asarray(outputs["pred1"], dtype=np.int64).reshape(-1)
    pred2 = np.asarray(outputs["pred2"], dtype=np.int64).reshape(-1)
    conf1 = np.asarray(outputs["conf1"], dtype=np.float32).reshape(-1)
    conf2 = np.asarray(outputs["conf2"], dtype=np.float32).reshape(-1)
    joint_core = (conf1 > tau_conf) & (conf2 > tau_conf) & (pred1 == pred2)
    mode = str(getattr(self.config, "joint_core_mode", "minimal")).strip().lower()
    if mode == "robust":
        ref_mask = joint_core if np.any(joint_core) else np.ones_like(joint_core, dtype=bool)
        dist_q = min(max(float(getattr(self.config, "joint_core_dist_quantile", 0.8)), 0.0), 1.0)
        recon_q = min(max(float(getattr(self.config, "joint_core_recon_quantile", 0.8)), 0.0), 1.0)
        dist1_thr = float(np.quantile(np.asarray(outputs["proto_dist1"])[ref_mask], dist_q))
        dist2_thr = float(np.quantile(np.asarray(outputs["proto_dist2"])[ref_mask], dist_q))
        recon1_thr = float(np.quantile(np.asarray(outputs["recon1"])[ref_mask], recon_q))
        recon2_thr = float(np.quantile(np.asarray(outputs["recon2"])[ref_mask], recon_q))
        joint_core = (
            joint_core
            & (np.asarray(outputs["proto_dist1"]) <= dist1_thr)
            & (np.asarray(outputs["proto_dist2"]) <= dist2_thr)
            & (np.asarray(outputs["recon1"]) <= recon1_thr)
            & (np.asarray(outputs["recon2"]) <= recon2_thr)
        )
    if self._stage2_method() == "separate_proto" and bool(
        getattr(self.config, "stage2_balanced_core", False)
    ):
        labels_for_balance = np.argmax(
            0.5 * (
                np.asarray(outputs["q1"], dtype=np.float32)
                + np.asarray(outputs["q2"], dtype=np.float32)
            ),
            axis=1,
        ).astype(np.int64)
        core_count = int(np.sum(joint_core))
        if core_count > 0:
            max_fraction = min(max(float(getattr(self.config, "stage2_balanced_core_max_fraction", 1.0)), 0.0), 1.0)
            min_per_proto = max(0, int(getattr(self.config, "stage2_balanced_core_min_per_proto", 0)))
            cap = max(min_per_proto, int(np.ceil(core_count * max_fraction))) if max_fraction < 1.0 else core_count
            if cap < core_count:
                core_score = (
                    np.asarray(outputs["proto_dist1"], dtype=np.float32)
                    + np.asarray(outputs["proto_dist2"], dtype=np.float32)
                    + np.asarray(outputs["recon1"], dtype=np.float32)
                    + np.asarray(outputs["recon2"], dtype=np.float32)
                )
                balanced_core = np.zeros_like(joint_core, dtype=bool)
                num_proto = int(outputs["q1"].shape[1]) if np.ndim(outputs["q1"]) == 2 else int(np.max(labels_for_balance) + 1)
                for proto_id in range(max(1, num_proto)):
                    proto_indices = np.where(joint_core & (labels_for_balance == proto_id))[0]
                    if proto_indices.size == 0:
                        continue
                    keep_count = min(int(proto_indices.size), int(cap))
                    order = np.argsort(core_score[proto_indices], kind="mergesort")
                    balanced_core[proto_indices[order[:keep_count]]] = True
                joint_core = balanced_core
    self.consensus_joint_core_mask = joint_core.astype(bool)
    self.consensus_proto_train_outputs = outputs
    state = self._build_consensus_proto_state(outputs, self.consensus_joint_core_mask, round_idx=round_idx)
    self._log_joint_core_label_diagnostics(state, round_idx=round_idx)
    labels = np.asarray(state["cluster_labels"], dtype=np.int64)
    core_counts = np.bincount(labels[self.consensus_joint_core_mask], minlength=state["cluster_centers"].shape[0])
    prefix = "[Stage2-Separate]" if self._uses_separate_prototypes() else "[Stage2-Single]"
    print(
        f"{prefix} refreshed joint core | "
        f"round={int(round_idx)} | "
        f"mode={mode} | "
        f"tau_conf={tau_conf:.4f} | "
        f"joint_core={int(np.sum(self.consensus_joint_core_mask))}/{int(joint_core.shape[0])} | "
        f"core_by_proto={core_counts.astype(int).tolist()}"
    )


def _build_stage2_consensus_loader(self) -> DataLoader:
    dataset = _Stage2ConsensusDataset(self.train_loader.dataset)
    return DataLoader(
        dataset=dataset,
        batch_size=self.config.batch_size,
        shuffle=True,
        num_workers=self._effective_num_workers(),
        drop_last=False,
        pin_memory=self._pin_memory(),
    )


def _paired_prototype_relation_loss(
    self,
    prototypes_v1: torch.Tensor,
    prototypes_v2: torch.Tensor,
) -> torch.Tensor:
    if prototypes_v1.size(0) <= 1 or prototypes_v2.size(0) <= 1:
        return self._zero_stage2_loss()
    eps = float(getattr(self.config, "robust_eps", 1e-6))
    p1 = F.normalize(prototypes_v1, dim=1, eps=eps)
    p2 = F.normalize(prototypes_v2, dim=1, eps=eps)
    sim1 = torch.matmul(p1, p1.t())
    sim2 = torch.matmul(p2, p2.t())
    mask = ~torch.eye(sim1.size(0), dtype=torch.bool, device=sim1.device)
    return F.mse_loss(sim1[mask], sim2[mask])


def _consensus_proto_batch_loss(self, raw_batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if not hasattr(self, "consensus_joint_core_mask"):
        raise RuntimeError("Separate prototype joint core has not been initialized.")
    x, indices = raw_batch
    x = x.float().to(self.device, non_blocking=self._pin_memory())
    indices = indices.long().to(self.device, non_blocking=self._pin_memory())
    outputs = self.model(x, stage="stage2")
    use_separate = self._uses_separate_prototypes()
    if not self._is_dual_view_model():
        outputs = dict(outputs)
        outputs["u1"] = outputs["u"]
        outputs["u2"] = outputs["u"]
        outputs["q1"] = outputs["q"]
        outputs["q2"] = outputs["q"]
        outputs["proto_dist1"] = outputs["proto_dist"]
        outputs["proto_dist2"] = outputs["proto_dist"]
        outputs["proto_pred1"] = outputs["proto_pred"]
        outputs["proto_pred2"] = outputs["proto_pred"]
        outputs["proto_conf1"] = outputs["proto_conf"]
        outputs["proto_conf2"] = outputs["proto_conf"]

    rec_v1, rec_v2 = self._dual_reconstruction_losses_from_outputs(outputs, x)
    loss_rec = 0.5 * (rec_v1 + rec_v2)

    core_np = np.asarray(self.consensus_joint_core_mask, dtype=bool)
    batch_core = torch.as_tensor(
        core_np[indices.detach().cpu().numpy()],
        dtype=torch.bool,
        device=self.device,
    )
    if use_separate:
        loss_state = js_divergence(
            outputs["q1"],
            outputs["q2"],
            eps=float(getattr(self.config, "robust_eps", 1e-8)),
        ).mean()
    elif torch.any(batch_core):
        q1_core = outputs["q1"][batch_core]
        q2_core = outputs["q2"][batch_core]
        q_teacher = consensus_teacher_distribution(
            q1_core,
            q2_core,
            sharpen_temperature=float(getattr(self.config, "q_cons_sharpen_temperature", 0.5)),
        )
        loss_state = state_consistency_teacher_loss(q1_core, q2_core, q_teacher)
    else:
        loss_state = self._zero_stage2_loss()

    if torch.any(batch_core):
        q1_core = outputs["q1"][batch_core]
        q2_core = outputs["q2"][batch_core]
        q_teacher = consensus_teacher_distribution(
            q1_core,
            q2_core,
            sharpen_temperature=float(getattr(self.config, "q_cons_sharpen_temperature", 0.5)),
        )
        target = torch.argmax(q_teacher, dim=1)
        if use_separate:
            target1 = target.long().clamp(0, self.model.prototype_head_v1.prototypes.size(0) - 1)
            target2 = target.long().clamp(0, self.model.prototype_head_v2.prototypes.size(0) - 1)
            proto1 = self.model.prototype_head_v1.prototypes[target1]
            proto2 = self.model.prototype_head_v2.prototypes[target2]
            pull1 = torch.sum((outputs["u1"][batch_core] - proto1) ** 2, dim=1)
            pull2 = torch.sum((outputs["u2"][batch_core] - proto2) ** 2, dim=1)
            loss_pull = 0.5 * (pull1 + pull2).mean()
        else:
            target_single = target.long().clamp(0, self.model.prototype_head.prototypes.size(0) - 1)
            proto = self.model.prototype_head.prototypes[target_single]
            pull1 = torch.sum((outputs["u1"][batch_core] - proto) ** 2, dim=1)
            pull2 = torch.sum((outputs["u2"][batch_core] - proto) ** 2, dim=1)
            loss_pull = 0.5 * (pull1 + pull2).mean()
    else:
        loss_pull = self._zero_stage2_loss()

    lambda_repulsion = float(getattr(self.config, "lambda_proto_repulsion", 1.0))
    if lambda_repulsion != 0.0:
        x_negative = self._inject_stage1_negative_batch(x, stage="stage2")
        neg_outputs = self.model(x_negative, stage="stage2")
        if not self._is_dual_view_model():
            neg_outputs = dict(neg_outputs)
            neg_outputs["u1"] = neg_outputs["u"]
            neg_outputs["u2"] = neg_outputs["u"]
        margin = float(getattr(self.config, "proto_repulsion_margin", 1.0))
        if use_separate:
            rep1 = prototype_repulsion_loss(neg_outputs["u1"], self.model.prototype_head_v1.prototypes, margin)
            rep2 = prototype_repulsion_loss(neg_outputs["u2"], self.model.prototype_head_v2.prototypes, margin)
        else:
            rep1 = prototype_repulsion_loss(neg_outputs["u1"], self.model.prototype_head.prototypes, margin)
            rep2 = prototype_repulsion_loss(neg_outputs["u2"], self.model.prototype_head.prototypes, margin)
        loss_repulsion = 0.5 * (rep1 + rep2)
    else:
        loss_repulsion = self._zero_stage2_loss()

    lambda_separation = float(getattr(self.config, "lambda_proto_separation", 0.3))
    if lambda_separation != 0.0:
        if use_separate:
            sep1 = prototype_separation_loss(
                self.model.prototype_head_v1.prototypes,
                float(getattr(self.config, "proto_separation_margin", 1.0)),
                force_weight=float(getattr(self.config, "proto_separation_force_weight", 0.1)),
                eps=float(getattr(self.config, "robust_eps", 1e-6)),
            )
            sep2 = prototype_separation_loss(
                self.model.prototype_head_v2.prototypes,
                float(getattr(self.config, "proto_separation_margin", 1.0)),
                force_weight=float(getattr(self.config, "proto_separation_force_weight", 0.1)),
                eps=float(getattr(self.config, "robust_eps", 1e-6)),
            )
            loss_proto_separation = 0.5 * (sep1 + sep2)
        else:
            loss_proto_separation = prototype_separation_loss(
                self.model.prototype_head.prototypes,
                float(getattr(self.config, "proto_separation_margin", 1.0)),
                force_weight=float(getattr(self.config, "proto_separation_force_weight", 0.1)),
                eps=float(getattr(self.config, "robust_eps", 1e-6)),
            )
    else:
        loss_proto_separation = self._zero_stage2_loss()

    lambda_usage_balance = float(getattr(self.config, "lambda_proto_usage_balance", 0.0))
    if lambda_usage_balance != 0.0:
        loss_usage_balance = prototype_usage_balance_loss(
            outputs["q1"],
            outputs["q2"],
            eps=float(getattr(self.config, "robust_eps", 1e-8)),
        )
    else:
        loss_usage_balance = self._zero_stage2_loss()

    lambda_relation = float(getattr(self.config, "lambda_proto_relation_consistency", 0.0))
    if use_separate and lambda_relation != 0.0:
        loss_proto_relation = self._paired_prototype_relation_loss(
            self.model.prototype_head_v1.prototypes,
            self.model.prototype_head_v2.prototypes,
        )
    else:
        loss_proto_relation = self._zero_stage2_loss()

    with torch.no_grad():
        def _min_proto_dist(prototypes: torch.Tensor) -> torch.Tensor:
            if prototypes.size(0) <= 1:
                return self._zero_stage2_loss()
            proto_dist = torch.cdist(prototypes, prototypes, p=2.0)
            proto_dist = proto_dist.masked_fill(
                torch.eye(prototypes.size(0), dtype=torch.bool, device=prototypes.device),
                float("inf"),
            )
            return proto_dist.min()

        if use_separate:
            proto_min_dist = torch.minimum(
                _min_proto_dist(self.model.prototype_head_v1.prototypes),
                _min_proto_dist(self.model.prototype_head_v2.prototypes),
            )
        else:
            proto_min_dist = _min_proto_dist(self.model.prototype_head.prototypes)

    loss = (
        float(getattr(self.config, "lambda_state_consistency", 1.0)) * loss_state
        + float(getattr(self.config, "lambda_proto_pull", 1.0)) * loss_pull
        + lambda_repulsion * loss_repulsion
        + lambda_separation * loss_proto_separation
        + lambda_usage_balance * loss_usage_balance
        + lambda_relation * loss_proto_relation
        + float(getattr(self.config, "stage2_lambda_rec", 0.0)) * loss_rec
    )
    return loss, {
        "loss": loss,
        "loss_state": loss_state,
        "loss_pull": loss_pull,
        "loss_repulsion": loss_repulsion,
        "loss_proto_separation": loss_proto_separation,
        "loss_usage_balance": loss_usage_balance,
        "loss_proto_relation": loss_proto_relation,
        "loss_rec": loss_rec,
        "rec_v1": rec_v1,
        "rec_v2": rec_v2,
        "proto_min_dist": proto_min_dist,
        "batch_joint_core": batch_core.float().sum(),
    }


def _stage2_consensus_proto_epoch(
    self,
    loader: DataLoader,
    round_idx: int,
    epoch_in_round: int,
    global_epoch: int,
    total_epochs: int,
):
    self.model.train()
    totals = {
        "loss": 0.0,
        "loss_state": 0.0,
        "loss_pull": 0.0,
        "loss_repulsion": 0.0,
        "loss_proto_separation": 0.0,
        "loss_usage_balance": 0.0,
        "loss_proto_relation": 0.0,
        "loss_rec": 0.0,
        "proto_min_dist": 0.0,
        "batch_joint_core": 0.0,
    }
    log_label = "Stage2-Separate" if self._uses_separate_prototypes() else "Stage2-Single"
    progress = tqdm(
        loader,
        desc=f"{log_label} R{round_idx} E{epoch_in_round}",
        leave=False,
        disable=not self._show_batch_progress(),
    )
    for batch in progress:
        loss, batch_losses = self._consensus_proto_batch_loss(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        for key in totals:
            totals[key] += float(batch_losses[key].item())
        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "state": f"{batch_losses['loss_state'].item():.4f}",
                "pull": f"{batch_losses['loss_pull'].item():.4f}",
                "rep": f"{batch_losses['loss_repulsion'].item():.4f}",
                "sep": f"{batch_losses['loss_proto_separation'].item():.4f}",
                "bal": f"{batch_losses['loss_usage_balance'].item():.4f}",
                "rel": f"{batch_losses['loss_proto_relation'].item():.4f}",
                "pmin": f"{batch_losses['proto_min_dist'].item():.4f}",
                "rec": f"{batch_losses['loss_rec'].item():.4f}",
            }
        )
    denom = max(1, len(loader))
    self._log_epoch(
        log_label,
        global_epoch,
        total_epochs,
        {
            "joint_core": int(np.sum(getattr(self, "consensus_joint_core_mask", np.zeros(0, dtype=bool)))),
            "loss": totals["loss"] / denom,
            "loss_state": totals["loss_state"] / denom,
            "loss_pull": totals["loss_pull"] / denom,
            "loss_repulsion": totals["loss_repulsion"] / denom,
            "loss_proto_separation": totals["loss_proto_separation"] / denom,
            "loss_usage_balance": totals["loss_usage_balance"] / denom,
            "loss_proto_relation": totals["loss_proto_relation"] / denom,
            "proto_min_dist": totals["proto_min_dist"] / denom,
            "loss_rec": totals["loss_rec"] / denom,
        },
    )


def _run_stage2_consensus_proto_refinement(
    self,
    num_stage2_rounds: int,
    epochs_per_round: int,
    total_stage2_epochs: int,
):
    prefix = "[Stage2-Separate]" if self._uses_separate_prototypes() else "[Stage2-Single]"
    if self._uses_separate_prototypes():
        self._initialize_paired_prototypes()
    else:
        self._initialize_consensus_prototypes()

    global_epoch = 1
    for round_idx in range(1, num_stage2_rounds + 1):
        print(f"{prefix} Starting round {round_idx}/{num_stage2_rounds}")
        for epoch_in_round in range(1, epochs_per_round + 1):
            if global_epoch == 1:
                print(
                    f"{prefix} Epoch refresh {global_epoch}/{total_stage2_epochs}: "
                    "use initialized prototypes and joint core"
                )
            else:
                print(
                    f"{prefix} Epoch refresh {global_epoch}/{total_stage2_epochs}: "
                    "refresh joint core from current q1/q2"
                )
                self._refresh_consensus_joint_core(round_idx=global_epoch)
            loader = self._build_stage2_consensus_loader()
            self._stage2_consensus_proto_epoch(
                loader=loader,
                round_idx=round_idx,
                epoch_in_round=epoch_in_round,
                global_epoch=global_epoch,
                total_epochs=total_stage2_epochs,
            )
            global_epoch += 1
    self._refresh_consensus_joint_core(round_idx=total_stage2_epochs)


def run_stage2_consensus_proto_refinement(
    solver,
    num_stage2_rounds=None,
    epochs_per_round=None,
    total_stage2_epochs=None,
):
    if num_stage2_rounds is None or epochs_per_round is None:
        num_stage2_rounds, epochs_per_round = solver._resolve_stage2_schedule()
    if total_stage2_epochs is None:
        total_stage2_epochs = solver._stage2_total_epochs()
    return solver._run_stage2_consensus_proto_refinement(
        num_stage2_rounds=int(num_stage2_rounds),
        epochs_per_round=int(epochs_per_round),
        total_stage2_epochs=int(total_stage2_epochs),
    )
