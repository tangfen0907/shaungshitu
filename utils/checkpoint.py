from typing import Dict

import os

import numpy as np
import torch


__all__ = [
    '_export_bank_artifacts',
    '_save_checkpoint',
    '_save_stage_checkpoint',
    'load_checkpoint',
]


def _export_bank_artifacts(self) -> Dict[str, object]:
    if self.core_mask is None:
        return {}
    stage2_state = self.stage2_structure if isinstance(self.stage2_structure, dict) else {}
    artifacts = {
        "prototype_centers": self.cluster_centers,
        "global_center": self.global_center,
        "core_mask": self.core_mask,
        "cluster_labels": self.cluster_labels,
        "cluster_core_radius": self.cluster_radii,
        "cluster_scoring_radius": self.cluster_radii,
        "nearest_other_cluster": self.nearest_other_clusters,
        "score_core": self.score_core,
        "summary": self.bank_summary,
        "proto_conf1": stage2_state.get("proto_conf1"),
        "proto_conf2": stage2_state.get("proto_conf2"),
        "proto_pred1": stage2_state.get("proto_pred1"),
        "proto_pred2": stage2_state.get("proto_pred2"),
        "proto_dist1": stage2_state.get("proto_dist1"),
        "proto_dist2": stage2_state.get("proto_dist2"),
        "recon1": stage2_state.get("recon1"),
        "recon2": stage2_state.get("recon2"),
        "joint_core_label_diagnostics": getattr(self, "joint_core_label_diagnostics", None),
    }
    return artifacts


def _checkpoint_payload(self, stage_label: str = None) -> Dict[str, object]:
    payload = {
        "model_state_dict": self.model.state_dict(),
        "config": self.config.to_dict(),
        "cluster_centers": self.cluster_centers,
        "cluster_labels": self.cluster_labels,
        "cluster_radii": self.cluster_radii,
        "global_center": self.global_center,
        "core_mask": self.core_mask,
        "nearest_other_clusters": self.nearest_other_clusters,
        "score_core": self.score_core,
        "stage2_refresh_round": self.stage2_refresh_round,
        "bank_summary": self.bank_summary,
        "cluster_bank": self.cluster_bank,
        "train_active_mask": getattr(self, "train_active_mask", None),
        "active_pool_history": getattr(self, "active_pool_history", []),
        "joint_core_label_diagnostics": getattr(self, "joint_core_label_diagnostics", []),
    }
    if stage_label is not None:
        payload["stage_label"] = str(stage_label)
    return payload


def _save_checkpoint_file(self, checkpoint_name: str, message: str, stage_label: str = None):
    checkpoint_path = os.path.join(self.config.save_dir, checkpoint_name)
    payload = _checkpoint_payload(self, stage_label=stage_label)
    torch.save(payload, checkpoint_path)
    print(f"{message}: {checkpoint_path}")


def _save_stage_checkpoint(self, stage_name: str):
    stage_name = str(stage_name).strip().lower()
    if not stage_name:
        raise ValueError("stage_name must be a non-empty string.")
    checkpoint_name = f"model_{stage_name}.pt"
    _save_checkpoint_file(
        self,
        checkpoint_name=checkpoint_name,
        message=f"[{stage_name.upper()}] model checkpoint saved to",
        stage_label=stage_name,
    )


def _save_checkpoint(self):
    _save_checkpoint_file(
        self,
        checkpoint_name=self.config.checkpoint_name,
        message="Final model checkpoint saved to",
        stage_label="final",
    )


def load_checkpoint(self, checkpoint_path: str):
    payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
    model_state_dict = dict(payload["model_state_dict"])
    model_state_dict.pop("svdd.cluster_centers", None)
    model_state_dict.pop("svdd.cluster_priors", None)
    self.model.load_state_dict(model_state_dict, strict=False)

    self.cluster_labels = payload.get("cluster_labels", None)
    self.cluster_centers = payload.get("cluster_centers", None)
    self.cluster_radii = payload.get("cluster_radii", None)
    self.global_center = payload.get("global_center", None)
    self.core_mask = payload.get("core_mask", None)
    self.nearest_other_clusters = payload.get("nearest_other_clusters", None)
    self.score_core = payload.get("score_core", None)
    self.stage2_refresh_round = int(payload.get("stage2_refresh_round", -1))
    self.bank_summary = list(payload.get("bank_summary", []))
    self.cluster_bank = payload.get("cluster_bank", None)
    self.stage2_structure = self.cluster_bank
    train_active_mask = payload.get("train_active_mask", None)
    if train_active_mask is not None:
        self.train_active_mask = np.asarray(train_active_mask, dtype=bool).reshape(-1)
        if hasattr(self, "_refresh_active_train_loaders"):
            self._refresh_active_train_loaders()
    self.active_pool_history = list(payload.get("active_pool_history", []))
    self.joint_core_label_diagnostics = list(payload.get("joint_core_label_diagnostics", []))
    if self.cluster_bank is not None:
        self._apply_cluster_bank(self.cluster_bank)
    elif self.cluster_centers is not None:
        self.cluster_centers = np.asarray(self.cluster_centers, dtype=np.float32)
        self.model.svdd.update_centers(self.cluster_centers)
