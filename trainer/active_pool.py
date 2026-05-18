import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


__all__ = [
    "_active_pool_trim_enabled",
    "_ensure_train_active_mask",
    "_current_active_train_mask",
    "_current_active_train_indices",
    "_active_pool_summary_text",
    "_build_active_pool_dataset",
    "_refresh_active_train_loaders",
    "_collect_active_pool_feature_matrix",
    "_active_pool_stage_ratio",
    "_write_active_pool_artifacts",
    "_trim_active_training_pool",
]


class _ActivePoolDataset(Dataset):
    """Dataset view over the original training windows selected by active pool."""

    def __init__(self, base_dataset: Dataset, original_indices: np.ndarray):
        self.base_dataset = base_dataset
        self.original_indices = np.asarray(original_indices, dtype=np.int64).reshape(-1)

    def __len__(self) -> int:
        return int(self.original_indices.shape[0])

    def __getitem__(self, idx: int):
        return self.base_dataset[int(self.original_indices[int(idx)])]


def _active_pool_trim_enabled(self) -> bool:
    return bool(getattr(self.config, "active_pool_trim_enabled", False))


def _ensure_train_active_mask(self) -> np.ndarray:
    base_dataset = getattr(self, "full_train_dataset", None)
    if base_dataset is None:
        base_dataset = self.train_loader.dataset
        self.full_train_dataset = base_dataset

    total = int(len(base_dataset))
    mask = getattr(self, "train_active_mask", None)
    if mask is None or np.asarray(mask).reshape(-1).shape[0] != total:
        mask = np.ones(total, dtype=bool)
        self.train_active_mask = mask
    else:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        self.train_active_mask = mask
    if not hasattr(self, "active_pool_history"):
        self.active_pool_history = []
    return self.train_active_mask


def _current_active_train_mask(self) -> np.ndarray:
    return self._ensure_train_active_mask().copy()


def _current_active_train_indices(self) -> np.ndarray:
    return np.flatnonzero(self._ensure_train_active_mask()).astype(np.int64)


def _active_pool_summary_text(self) -> str:
    mask = self._ensure_train_active_mask()
    return f"{int(np.sum(mask))}/{int(mask.shape[0])}"


def _build_active_pool_dataset(self) -> Dataset:
    return _ActivePoolDataset(self.full_train_dataset, self._current_active_train_indices())


def _refresh_active_train_loaders(self):
    dataset = self._build_active_pool_dataset()
    self.train_loader = DataLoader(
        dataset=dataset,
        batch_size=self.config.batch_size,
        shuffle=True,
        num_workers=self._effective_num_workers(),
        drop_last=False,
        pin_memory=self._pin_memory(),
        generator=self._make_loader_generator(401),
    )
    self.train_eval_loader = DataLoader(
        dataset=dataset,
        batch_size=self.config.batch_size,
        shuffle=False,
        num_workers=self._effective_num_workers(),
        drop_last=False,
        pin_memory=self._pin_memory(),
        generator=self._make_loader_generator(402),
    )


def _collect_active_pool_feature_matrix(self) -> np.ndarray:
    was_training = self.model.training
    self.model.eval()
    features: List[np.ndarray] = []
    with torch.inference_mode():
        for batch in self.train_eval_loader:
            x = self._prepare_batch(batch)
            if self._is_dual_view_model():
                z1, z2 = self.model.encode_views(x)
                feat = F.normalize(
                    torch.cat(
                        [
                            F.normalize(z1, dim=1),
                            F.normalize(z2, dim=1),
                        ],
                        dim=1,
                    ),
                    dim=1,
                )
            else:
                feat = F.normalize(self.model.encode(x), dim=1)
            features.append(feat.detach().cpu().numpy())
    if was_training:
        self.model.train()
    if not features:
        raise RuntimeError("Active-pool feature collection found no active training windows.")
    return np.concatenate(features, axis=0).astype(np.float32)


def _active_pool_stage_ratio(self, stage_key: str) -> float:
    stage_key = str(stage_key).strip().lower()
    if stage_key in {"stage0", "a0", "0"}:
        return float(getattr(self.config, "active_pool_trim_stage0_ratio", 0.0))
    if stage_key in {"stage1", "a1", "1"}:
        return float(getattr(self.config, "active_pool_trim_stage1_ratio", 0.0))
    return 0.0


def _write_active_pool_artifacts(self, stage_key: str, scores_full: np.ndarray, removed_indices: np.ndarray):
    os.makedirs(self.config.save_dir, exist_ok=True)
    np.save(os.path.join(self.config.save_dir, "train_active_mask.npy"), self._ensure_train_active_mask())
    np.save(os.path.join(self.config.save_dir, f"active_pool_{stage_key}_scores.npy"), scores_full)
    np.save(os.path.join(self.config.save_dir, f"active_pool_{stage_key}_removed_indices.npy"), removed_indices)
    history_path = os.path.join(self.config.save_dir, "active_pool_history.json")
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(getattr(self, "active_pool_history", []), file, ensure_ascii=False, indent=2)


def _trim_active_training_pool(self, stage_key: str):
    if not self._active_pool_trim_enabled():
        return

    stage_key = str(stage_key).strip().lower()
    ratio = min(max(self._active_pool_stage_ratio(stage_key), 0.0), 1.0)
    if ratio <= 0.0:
        return

    mask = self._ensure_train_active_mask()
    active_indices = np.flatnonzero(mask).astype(np.int64)
    active_count = int(active_indices.shape[0])
    if active_count <= 1:
        print(f"[ActivePool] {stage_key} trim skipped | active_pool={active_count}/{int(mask.shape[0])}")
        return

    features = self._collect_active_pool_feature_matrix()
    if features.shape[0] != active_count:
        raise RuntimeError(
            "Active-pool feature count does not match active mask: "
            f"{features.shape[0]} vs {active_count}"
        )

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    center = np.median(features, axis=0, keepdims=True).astype(np.float32)
    scores = np.linalg.norm(features - center, axis=1).astype(np.float32)
    remove_count = int(np.ceil(active_count * ratio))
    remove_count = min(max(remove_count, 1), active_count - 1)
    order = np.argsort(scores, kind="mergesort")
    remove_local = order[-remove_count:]
    removed_indices = np.sort(active_indices[remove_local]).astype(np.int64)
    score_threshold = float(np.min(scores[remove_local])) if remove_local.size else float("nan")

    new_mask = mask.copy()
    new_mask[removed_indices] = False
    self.train_active_mask = new_mask
    self._refresh_active_train_loaders()

    scores_full = np.full(mask.shape[0], np.nan, dtype=np.float32)
    scores_full[active_indices] = scores
    history_item: Dict[str, object] = {
        "stage": stage_key,
        "ratio": float(ratio),
        "active_before": int(active_count),
        "removed": int(remove_count),
        "active_after": int(np.sum(new_mask)),
        "total": int(new_mask.shape[0]),
        "score_threshold": score_threshold,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
    }
    self.active_pool_history.append(history_item)
    self._write_active_pool_artifacts(stage_key, scores_full, removed_indices)

    print(
        f"[ActivePool] {stage_key} trim | "
        f"ratio={ratio:.4f} | "
        f"removed={int(remove_count)}/{int(active_count)} | "
        f"active_pool={int(np.sum(new_mask))}/{int(new_mask.shape[0])} | "
        f"score_thr={score_threshold:.6f} | "
        f"score_mean={float(np.mean(scores)):.6f}"
    )
