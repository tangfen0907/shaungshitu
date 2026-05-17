from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.clustering import _cluster_boundary_radii, _cluster_features, _nearest_other_clusters


__all__ = [
    "_stage2_total_epochs",
    "_stage2_method",
    "_uses_separate_prototypes",
    "_resolve_stage2_schedule",
    "_zero_stage2_loss",
    "_collect_last_point_features",
    "_hungarian_match_from_overlap",
    "_initialize_aligned_separate_prototypes",
    "_build_stage2_loader",
    "_set_stage2_train_phase",
    "_select_a_core",
    "_stage2_a_batch_loss",
    "_stage2_a_collect_calibration_data",
    "_stage2_a_ema_update_one_view",
    "_run_stage2_a_epoch",
    "_select_b_core",
    "_stage2_b_batch_loss",
    "_run_stage2_b_epoch",
    "_refresh_aligned_stage2_state",
    "_run_stage2_ab_refinement",
    "run_stage2_ab_refinement",
]


def _stage2_total_epochs(self) -> int:
    rounds, a_epochs, b_epochs = self._resolve_stage2_schedule()
    return int(rounds) * (int(a_epochs) + int(b_epochs))


def _stage2_method(self) -> str:
    method = str(getattr(self.config, "stage2_method", "separate_proto")).strip().lower()
    if method != "separate_proto":
        raise ValueError("Only stage2_method='separate_proto' is supported.")
    return method


def _uses_separate_prototypes(self) -> bool:
    if not self._is_dual_view_model():
        raise RuntimeError("separate_proto requires a dual-view model.")
    return True


def _resolve_stage2_schedule(self) -> Tuple[int, int, int]:
    rounds = max(1, int(getattr(self.config, "num_stage2_rounds", 3)))
    a_epochs = max(1, int(getattr(self.config, "stage2_a_epochs", 1)))
    b_epochs = max(1, int(getattr(self.config, "stage2_b_epochs", 1)))
    return rounds, a_epochs, b_epochs


def _zero_stage2_loss(self) -> torch.Tensor:
    return torch.zeros((), device=self.device)


def _collect_last_point_features(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Collect one current-point embedding per local L-window."""
    self._uses_separate_prototypes()
    was_training = self.model.training
    self.model.eval()
    features_v1: List[np.ndarray] = []
    features_v2: List[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            x = self._prepare_batch(batch)
            z1, z2 = self.model.encode_views(x)
            features_v1.append(z1.detach().cpu().numpy())
            features_v2.append(z2.detach().cpu().numpy())
    if was_training:
        self.model.train()
    if not features_v1 or not features_v2:
        raise RuntimeError("No local-window features were available for Stage2 initialization.")
    z1 = np.concatenate(features_v1, axis=0).astype(np.float32)
    z2 = np.concatenate(features_v2, axis=0).astype(np.float32)
    if z1.shape != z2.shape:
        raise RuntimeError(f"Stage2 init feature shapes should match, got {z1.shape} vs {z2.shape}.")
    return z1, z2


def _hungarian_match_from_overlap(
    self,
    labels_v1: np.ndarray,
    labels_v2: np.ndarray,
    num_prototypes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_v1 = np.asarray(labels_v1, dtype=np.int64).reshape(-1)
    labels_v2 = np.asarray(labels_v2, dtype=np.int64).reshape(-1)
    if labels_v1.shape != labels_v2.shape:
        raise ValueError("labels_v1 and labels_v2 should have the same shape.")

    overlap = np.zeros((int(num_prototypes), int(num_prototypes)), dtype=np.int64)
    for left, right in zip(labels_v1, labels_v2):
        overlap[int(left), int(right)] += 1

    row_ind, col_ind = linear_sum_assignment(-overlap)
    match = np.full(int(num_prototypes), -1, dtype=np.int64)
    match[row_ind.astype(np.int64)] = col_ind.astype(np.int64)
    if np.any(match < 0):
        raise RuntimeError("Hungarian prototype matching did not assign every View1 prototype.")

    inverse_match = np.empty_like(match)
    inverse_match[match] = np.arange(int(num_prototypes), dtype=np.int64)
    return overlap, match, inverse_match


def _initialize_aligned_separate_prototypes(self):
    z1, z2 = self._collect_last_point_features(self.train_eval_loader)
    num_prototypes = int(getattr(self.model.prototype_head_v1, "num_prototypes", 1))
    labels1, centers1, meta1 = _cluster_features(
        features=z1,
        cluster_method="kmeans",
        n_clusters=num_prototypes,
        random_state=int(getattr(self.config, "seed", 42)),
    )
    labels2, centers2, meta2 = _cluster_features(
        features=z2,
        cluster_method="kmeans",
        n_clusters=num_prototypes,
        random_state=int(getattr(self.config, "seed", 42)) + 1,
    )
    if centers1.shape[0] != num_prototypes or centers2.shape[0] != num_prototypes:
        raise RuntimeError(
            "Independent KMeans should produce one center per prototype head entry: "
            f"expected={num_prototypes}, got v1={centers1.shape[0]}, v2={centers2.shape[0]}."
        )

    overlap, match, inverse_match = self._hungarian_match_from_overlap(labels1, labels2, num_prototypes)
    aligned_centers2 = centers2[match]
    labels2_aligned = inverse_match[np.asarray(labels2, dtype=np.int64)]

    self.model.init_separate_prototypes_from_centers(centers1, aligned_centers2)
    self.stage2_init_labels_v1 = np.asarray(labels1, dtype=np.int64)
    self.stage2_init_labels_v2 = np.asarray(labels2, dtype=np.int64)
    self.stage2_init_labels_v2_aligned = np.asarray(labels2_aligned, dtype=np.int64)
    self.stage2_proto_match = np.asarray(match, dtype=np.int64)
    self.stage2_proto_inverse_match = np.asarray(inverse_match, dtype=np.int64)
    self.stage2_proto_overlap = np.asarray(overlap, dtype=np.int64)

    matched_count = int(overlap[np.arange(num_prototypes), match].sum())
    total_count = int(labels1.shape[0])
    print(
        "[Stage2-Init] aligned separate prototypes | "
        f"features={tuple(z1.shape)} | "
        f"K={num_prototypes} | "
        f"kmeans_v1={meta1.get('cluster_method_actual', 'kmeans')} | "
        f"kmeans_v2={meta2.get('cluster_method_actual', 'kmeans')} | "
        f"matched_overlap={matched_count}/{total_count} | "
        f"match={match.astype(int).tolist()}"
    )


def _build_stage2_loader(self) -> DataLoader:
    return self.train_loader


def _set_stage2_train_phase(self, phase: str):
    phase_key = str(phase).strip().upper()
    if phase_key not in {"A", "B"}:
        raise ValueError("Stage2 phase should be 'A' or 'B'.")
    self._uses_separate_prototypes()

    p1 = self.model.prototype_head_v1.prototypes
    p2 = self.model.prototype_head_v2.prototypes
    encoder_modules = [
        self.model.dual_encoder,
        self.model.reconstructor_v1,
        self.model.reconstructor_v2,
    ]

    if phase_key == "A":
        # Stage2-A is prototype refresh/calibration. It freezes encoders and
        # updates P1/P2 explicitly from high-quality core means, not by a
        # gradient loss or optimizer.step().
        self.model.eval()
        for module in encoder_modules:
            module.requires_grad_(False)
            module.eval()
        p1.requires_grad_(False)
        p2.requires_grad_(False)
        self.optimizer = None
        return

    for module in encoder_modules:
        module.requires_grad_(True)
        module.train()
    p1.requires_grad_(False)
    p2.requires_grad_(False)
    parameters = []
    for module in encoder_modules:
        parameters.extend(list(module.parameters()))
    self.optimizer = Adam(
        parameters,
        lr=float(getattr(self.config, "lr", 1e-3)),
        weight_decay=float(getattr(self.config, "weight_decay", 1e-5)),
    )


def _select_a_core(
    self,
    z_flat: torch.Tensor,
    prototypes: torch.Tensor,
    ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compatibility helper: distance-only per-prototype core selection."""
    if z_flat.dim() != 2 or prototypes.dim() != 2:
        raise ValueError("A-core selection expects z_flat [N, D] and prototypes [K, D].")
    ratio = min(max(float(ratio), 0.0), 1.0)
    if ratio <= 0.0:
        return torch.zeros(z_flat.size(0), dtype=torch.bool, device=z_flat.device), torch.zeros(
            z_flat.size(0), dtype=torch.long, device=z_flat.device
        )

    with torch.no_grad():
        dist_sq = torch.cdist(z_flat.detach(), prototypes.detach(), p=2.0).pow(2)
        nearest_dist_sq, labels = torch.min(dist_sq, dim=1)
        core_mask = torch.zeros(z_flat.size(0), dtype=torch.bool, device=z_flat.device)
        for proto_id in range(int(prototypes.size(0))):
            proto_indices = torch.nonzero(labels == proto_id, as_tuple=False).flatten()
            if proto_indices.numel() == 0:
                continue
            keep_count = max(1, int(np.ceil(float(proto_indices.numel()) * ratio)))
            keep_count = min(keep_count, int(proto_indices.numel()))
            order = torch.argsort(nearest_dist_sq[proto_indices], stable=True)
            core_mask[proto_indices[order[:keep_count]]] = True
    return core_mask, labels


def _stage2_a_batch_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    raise RuntimeError(
        "Stage2-A is now prototype refresh/calibration. It has no per-batch "
        "gradient loss and does not call optimizer.step()."
    )


def _prototype_min_pairwise_distance(prototypes: torch.Tensor) -> float:
    if prototypes.size(0) <= 1:
        return 0.0
    dist = torch.cdist(prototypes, prototypes, p=2.0)
    dist.fill_diagonal_(float("inf"))
    value = torch.min(dist).item()
    return float(value) if np.isfinite(value) else 0.0


def _format_float_list(values: np.ndarray, precision: int = 4) -> List[float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return [round(float(v), int(precision)) for v in values.tolist()]


def _stage2_a_collect_calibration_data(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
    collected: Dict[str, List[torch.Tensor]] = {
        "h1": [],
        "h2": [],
        "pred1": [],
        "pred2": [],
        "score": [],
        "dist_score": [],
        "align_score": [],
        "recon_score": [],
    }
    alpha = float(getattr(self.config, "alpha_A", 1.0))
    beta = float(getattr(self.config, "beta_A", 1.0))
    gamma = float(getattr(self.config, "gamma_A", 0.5))

    progress = tqdm(
        loader,
        desc="Stage2-A collect",
        leave=False,
        disable=not self._show_batch_progress(),
    )
    with torch.inference_mode():
        for batch in progress:
            x = self._prepare_batch(batch)
            outputs = self.model(x, stage="stage2", detach_prototypes=True)
            h1 = outputs["H1"]
            h2 = outputs["H2"]
            q1 = outputs["q1"]
            q2 = outputs["q2"]
            dist1_sq = outputs["proto_dist_matrix1"].min(dim=1).values
            dist2_sq = outputs["proto_dist_matrix2"].min(dim=1).values
            dist_score = 0.5 * (dist1_sq + dist2_sq)
            align_score = torch.mean((q1 - q2) ** 2, dim=1)
            if gamma != 0.0:
                recon1 = self._mse_per_sample(x, outputs["x_hat1"], view="v1")
                recon2 = self._mse_per_sample(x, outputs["x_hat2"], view="v2")
                recon_score = 0.5 * (recon1 + recon2)
            else:
                recon_score = torch.zeros_like(dist_score)
            score = alpha * dist_score + beta * align_score + gamma * recon_score

            collected["h1"].append(h1.detach().cpu())
            collected["h2"].append(h2.detach().cpu())
            collected["pred1"].append(outputs["proto_pred1"].detach().cpu().long())
            collected["pred2"].append(outputs["proto_pred2"].detach().cpu().long())
            collected["score"].append(score.detach().cpu())
            collected["dist_score"].append(dist_score.detach().cpu())
            collected["align_score"].append(align_score.detach().cpu())
            collected["recon_score"].append(recon_score.detach().cpu())

    if not collected["h1"]:
        raise RuntimeError("Stage2-A calibration found no training samples.")
    return {key: torch.cat(value, dim=0) for key, value in collected.items()}


def _stage2_a_ema_update_one_view(
    self,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    score: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    ratio: float,
    min_core_per_proto: int,
    momentum: float,
    view_name: str,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    num_prototypes = int(prototypes.size(0))
    updated = prototypes.clone()
    core_counts = np.zeros(num_prototypes, dtype=np.int64)
    mean_scores = np.full(num_prototypes, np.nan, dtype=np.float32)
    update_norms = np.zeros(num_prototypes, dtype=np.float32)
    skipped: List[int] = []
    ratio = min(max(float(ratio), 0.0), 1.0)
    min_core_per_proto = max(1, int(min_core_per_proto))

    for proto_id in range(num_prototypes):
        proto_indices = torch.nonzero(labels == proto_id, as_tuple=False).flatten()
        count = int(proto_indices.numel())
        if count < min_core_per_proto or ratio <= 0.0:
            skipped.append(proto_id)
            continue
        keep_count = int(np.ceil(float(count) * ratio))
        keep_count = max(min_core_per_proto, keep_count)
        keep_count = min(keep_count, count)
        order = torch.argsort(score[proto_indices], stable=True)
        core_indices = proto_indices[order[:keep_count]]
        core_mean = embeddings[core_indices].mean(dim=0)
        new_value = float(momentum) * prototypes[proto_id] + (1.0 - float(momentum)) * core_mean
        updated[proto_id] = new_value
        core_counts[proto_id] = int(keep_count)
        mean_scores[proto_id] = float(score[core_indices].mean().item())
        update_norms[proto_id] = float(torch.norm(new_value - prototypes[proto_id], p=2).item())

    if skipped:
        print(
            f"[Stage2-A][Warning] {view_name} skipped prototype(s) without enough core: "
            f"{skipped} | min_core_per_proto={min_core_per_proto}"
        )
    return updated, core_counts, mean_scores, update_norms, skipped


def _run_stage2_a_epoch(
    self,
    loader: DataLoader,
    *,
    round_idx: int,
    epoch_in_phase: int,
    global_epoch: int,
    total_epochs: int,
):
    self._set_stage2_train_phase("A")
    data = self._stage2_a_collect_calibration_data(loader)

    p1_old = self.model.prototype_head_v1.prototypes.detach().cpu()
    p2_old = self.model.prototype_head_v2.prototypes.detach().cpu()
    ratio = float(getattr(self.config, "core_ratio_A", 0.3))
    min_core = int(getattr(self.config, "min_core_per_proto", 1))
    momentum = float(getattr(self.config, "proto_momentum", 0.8))
    momentum = min(max(momentum, 0.0), 0.999)
    pair_strength = float(getattr(self.config, "pair_align_strength", 0.2))
    pair_strength = min(max(pair_strength, 0.0), 1.0)

    p1_new, core_count_v1, mean_score_v1, update_norm_v1, skipped_v1 = self._stage2_a_ema_update_one_view(
        data["h1"],
        data["pred1"],
        data["score"],
        p1_old,
        ratio=ratio,
        min_core_per_proto=min_core,
        momentum=momentum,
        view_name="View1",
    )
    p2_new, core_count_v2, mean_score_v2, update_norm_v2, skipped_v2 = self._stage2_a_ema_update_one_view(
        data["h2"],
        data["pred2"],
        data["score"],
        p2_old,
        ratio=ratio,
        min_core_per_proto=min_core,
        momentum=momentum,
        view_name="View2",
    )

    # Semantic goal: P1[k] and P2[k] should represent the same normal operating
    # mode. Implementation: use progressive pair alignment to avoid
    # destabilizing early training when the two view spaces are not yet fully
    # calibrated.
    p_avg = 0.5 * (p1_new + p2_new)
    p1_final = (1.0 - pair_strength) * p1_new + pair_strength * p_avg
    p2_final = (1.0 - pair_strength) * p2_new + pair_strength * p_avg

    with torch.no_grad():
        self.model.prototype_head_v1.prototypes.copy_(
            p1_final.to(device=self.model.prototype_head_v1.prototypes.device, dtype=self.model.prototype_head_v1.prototypes.dtype)
        )
        self.model.prototype_head_v2.prototypes.copy_(
            p2_final.to(device=self.model.prototype_head_v2.prototypes.device, dtype=self.model.prototype_head_v2.prototypes.dtype)
        )

    update_norm_final_v1 = torch.norm(p1_final - p1_old, dim=1).numpy().astype(np.float32)
    update_norm_final_v2 = torch.norm(p2_final - p2_old, dim=1).numpy().astype(np.float32)
    mean_score_by_proto = np.nanmean(np.stack([mean_score_v1, mean_score_v2], axis=0), axis=0)
    finite_mean_score = mean_score_by_proto[np.isfinite(mean_score_by_proto)]

    diagnostics = {
        "samples": int(data["h1"].size(0)),
        "core_ratio_v1": float(core_count_v1.sum() / max(1, int(data["h1"].size(0)))),
        "core_ratio_v2": float(core_count_v2.sum() / max(1, int(data["h2"].size(0)))),
        "prototype_min_dist_v1": _prototype_min_pairwise_distance(p1_final),
        "prototype_min_dist_v2": _prototype_min_pairwise_distance(p2_final),
        "pair_mse": float(F.mse_loss(p1_final, p2_final).item()),
        "mean_core_score": float(np.mean(finite_mean_score)) if finite_mean_score.size else 0.0,
        "prototype_update_norm": float(0.5 * (np.mean(update_norm_final_v1) + np.mean(update_norm_final_v2))),
        "dist_score_mean": float(data["dist_score"].mean().item()),
        "align_score_mean": float(data["align_score"].mean().item()),
        "recon_score_mean": float(data["recon_score"].mean().item()),
        "skipped_proto_v1": int(len(skipped_v1)),
        "skipped_proto_v2": int(len(skipped_v2)),
    }
    self.stage2_a_last_diagnostics = {
        **diagnostics,
        "core_count_by_proto_v1": core_count_v1,
        "core_count_by_proto_v2": core_count_v2,
        "mean_core_score_by_proto": mean_score_by_proto,
        "prototype_update_norm_v1": update_norm_final_v1,
        "prototype_update_norm_v2": update_norm_final_v2,
        "ema_update_norm_v1": update_norm_v1,
        "ema_update_norm_v2": update_norm_v2,
    }

    print(
        "[Stage2-A] prototype refresh/calibration | "
        f"round={round_idx} | epoch_in_phase={epoch_in_phase} | "
        f"proto_momentum={momentum:.3f} | pair_align_strength={pair_strength:.3f} | "
        f"core_ratio_A={ratio:.3f} | min_core_per_proto={min_core}"
    )
    print(
        "[Stage2-A] diagnostics | "
        f"prototype_min_dist_v1={diagnostics['prototype_min_dist_v1']:.6f} | "
        f"prototype_min_dist_v2={diagnostics['prototype_min_dist_v2']:.6f} | "
        f"pair_mse={diagnostics['pair_mse']:.6f} | "
        f"prototype_update_norm={diagnostics['prototype_update_norm']:.6f}"
    )
    print(
        "[Stage2-A] core_count_by_proto_v1="
        f"{core_count_v1.astype(int).tolist()} | "
        "core_count_by_proto_v2="
        f"{core_count_v2.astype(int).tolist()}"
    )
    print(
        "[Stage2-A] mean_core_score_by_proto="
        f"{_format_float_list(np.nan_to_num(mean_score_by_proto, nan=0.0))} | "
        "prototype_update_norm_v1="
        f"{_format_float_list(update_norm_final_v1)} | "
        "prototype_update_norm_v2="
        f"{_format_float_list(update_norm_final_v2)}"
    )

    self._log_epoch("Stage2-A", global_epoch, total_epochs, diagnostics)


def _select_b_core(self, outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    q1 = outputs["q1"]
    q2 = outputs["q2"]
    dist_score = 0.5 * (
        outputs["proto_dist_matrix1"].min(dim=1).values
        + outputs["proto_dist_matrix2"].min(dim=1).values
    )
    align = torch.mean((q1 - q2) ** 2, dim=-1)
    score = (
        float(getattr(self.config, "alpha_B", 1.0)) * dist_score
        + float(getattr(self.config, "beta_B", 1.0)) * align
    )

    ratio = min(max(float(getattr(self.config, "core_ratio_B", 0.5)), 0.0), 1.0)
    core_count = 0 if ratio <= 0.0 else max(1, int(np.ceil(float(score.numel()) * ratio)))
    core_count = min(core_count, int(score.numel()))
    core_mask = torch.zeros_like(score, dtype=torch.bool)
    if core_count > 0:
        indices = torch.topk(score.detach(), k=core_count, largest=False, sorted=False).indices
        core_mask[indices] = True
    return core_mask, {
        "q1": q1,
        "q2": q2,
        "dist_score": dist_score,
        "align": align,
        "score": score,
    }


def _stage2_b_batch_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    x = self._prepare_batch(batch)
    outputs = self.model(x, stage="stage2")
    rec_v1, rec_v2 = self._dual_reconstruction_losses_from_outputs(outputs, x)
    loss_rec = 0.5 * (rec_v1 + rec_v2)

    core_mask, cached = self._select_b_core(outputs)
    q1 = cached["q1"]
    q2 = cached["q2"]
    align = cached["align"]
    q_avg = 0.5 * (q1 + q2)
    core_labels = torch.argmax(q_avg, dim=-1)
    h1 = outputs["H1"]
    h2 = outputs["H2"]
    p1 = self.model.prototype_head_v1.prototypes
    p2 = self.model.prototype_head_v2.prototypes

    if bool(core_mask.any()):
        selected_labels = core_labels[core_mask]
        pull_v1 = torch.sum((h1[core_mask] - p1.detach()[selected_labels]) ** 2, dim=1).mean()
        pull_v2 = torch.sum((h2[core_mask] - p2.detach()[selected_labels]) ** 2, dim=1).mean()
        loss_pull = 0.5 * (pull_v1 + pull_v2)
    else:
        pull_v1 = self._zero_stage2_loss()
        pull_v2 = self._zero_stage2_loss()
        loss_pull = self._zero_stage2_loss()

    loss_align = align.mean()
    loss_delta = self._zero_stage2_loss()

    x_negative = self._inject_last_context(x, stage="stage2")
    outputs_negative = self.model(x_negative, stage="stage2", detach_prototypes=True)
    z1_negative = outputs_negative["H1"]
    z2_negative = outputs_negative["H2"]
    min_dist1_negative = torch.cdist(z1_negative, p1.detach(), p=2.0).pow(2).min(dim=1).values
    min_dist2_negative = torch.cdist(z2_negative, p2.detach(), p=2.0).pow(2).min(dim=1).values
    margin = float(getattr(self.config, "margin_anom", 1.0))
    anom_v1 = F.relu(margin - min_dist1_negative).mean()
    anom_v2 = F.relu(margin - min_dist2_negative).mean()
    loss_anom = 0.5 * (anom_v1 + anom_v2)

    loss = (
        float(getattr(self.config, "lambda_rec_B", 1.0)) * loss_rec
        + float(getattr(self.config, "lambda_pull_B", 0.5)) * loss_pull
        + float(getattr(self.config, "lambda_align_B", 0.05)) * loss_align
        + float(getattr(self.config, "lambda_anom_B", 0.05)) * loss_anom
    )
    return loss, {
        "loss": loss,
        "loss_rec": loss_rec,
        "rec_v1": rec_v1,
        "rec_v2": rec_v2,
        "loss_pull": loss_pull,
        "pull_v1": pull_v1,
        "pull_v2": pull_v2,
        "loss_align": loss_align,
        "loss_delta": loss_delta,
        "loss_anom": loss_anom,
        "anom_v1": anom_v1,
        "anom_v2": anom_v2,
        "core_ratio": core_mask.float().mean(),
        "score_mean": cached["score"].mean(),
        "align_mean": align.mean(),
        "delta_mean": loss_delta,
    }


def _run_stage2_b_epoch(
    self,
    loader: DataLoader,
    *,
    round_idx: int,
    epoch_in_phase: int,
    global_epoch: int,
    total_epochs: int,
):
    self._set_stage2_train_phase("B")
    self.model.train()
    totals = {
        "loss": 0.0,
        "loss_rec": 0.0,
        "rec_v1": 0.0,
        "rec_v2": 0.0,
        "loss_pull": 0.0,
        "pull_v1": 0.0,
        "pull_v2": 0.0,
        "loss_align": 0.0,
        "loss_delta": 0.0,
        "loss_anom": 0.0,
        "anom_v1": 0.0,
        "anom_v2": 0.0,
        "core_ratio": 0.0,
        "score_mean": 0.0,
        "align_mean": 0.0,
        "delta_mean": 0.0,
    }
    progress = tqdm(
        loader,
        desc=f"Stage2-B R{round_idx} E{epoch_in_phase}",
        leave=False,
        disable=not self._show_batch_progress(),
    )
    for batch in progress:
        loss, metrics = self._stage2_b_batch_loss(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        for key in totals:
            totals[key] += float(metrics[key].item())
        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "rec": f"{metrics['loss_rec'].item():.4f}",
                "pull": f"{metrics['loss_pull'].item():.4f}",
                "align": f"{metrics['loss_align'].item():.4f}",
                "anom": f"{metrics['loss_anom'].item():.4f}",
            }
        )
    denom = max(1, len(loader))
    self._log_epoch(
        "Stage2-B",
        global_epoch,
        total_epochs,
        {key: value / denom for key, value in totals.items()},
    )


def _refresh_aligned_stage2_state(self, round_idx: int):
    self._uses_separate_prototypes()
    was_training = self.model.training
    self.model.eval()
    collected: Dict[str, List[np.ndarray]] = {
        "h1": [],
        "h2": [],
        "q1": [],
        "q2": [],
        "pred1": [],
        "pred2": [],
        "conf1": [],
        "conf2": [],
        "dist1": [],
        "dist2": [],
        "recon1": [],
        "recon2": [],
    }
    with torch.inference_mode():
        for batch in self.train_eval_loader:
            x = self._prepare_batch(batch)
            outputs = self.model(x, stage="stage2")
            collected["h1"].append(outputs["H1"].detach().cpu().numpy())
            collected["h2"].append(outputs["H2"].detach().cpu().numpy())
            collected["q1"].append(outputs["q1"].detach().cpu().numpy())
            collected["q2"].append(outputs["q2"].detach().cpu().numpy())
            collected["pred1"].append(outputs["proto_pred1"].detach().cpu().numpy())
            collected["pred2"].append(outputs["proto_pred2"].detach().cpu().numpy())
            collected["conf1"].append(outputs["proto_conf1"].detach().cpu().numpy())
            collected["conf2"].append(outputs["proto_conf2"].detach().cpu().numpy())
            collected["dist1"].append(outputs["proto_dist1"].detach().cpu().numpy())
            collected["dist2"].append(outputs["proto_dist2"].detach().cpu().numpy())
            collected["recon1"].append(self._mse_per_sample(x, outputs["x_hat1"], view="v1").detach().cpu().numpy())
            collected["recon2"].append(self._mse_per_sample(x, outputs["x_hat2"], view="v2").detach().cpu().numpy())
    if was_training:
        self.model.train()

    arrays = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    centers_v1 = self.model.prototype_head_v1.prototypes.detach().cpu().numpy().astype(np.float32)
    centers_v2 = self.model.prototype_head_v2.prototypes.detach().cpu().numpy().astype(np.float32)
    centers = (0.5 * (centers_v1 + centers_v2)).astype(np.float32)
    q_avg = 0.5 * (arrays["q1"] + arrays["q2"])
    labels = np.argmax(q_avg, axis=1).astype(np.int64)
    avg_features = 0.5 * (arrays["h1"].astype(np.float32) + arrays["h2"].astype(np.float32))
    core_mask = np.ones(labels.shape[0], dtype=bool)
    score_core = (0.5 * (arrays["dist1"] + arrays["dist2"])).astype(np.float32)
    radii = _cluster_boundary_radii(
        avg_features,
        labels,
        centers,
        eps=float(getattr(self.config, "robust_eps", 1e-6)),
    )
    global_center = centers.mean(axis=0).astype(np.float32)
    nearest_other = _nearest_other_clusters(centers)
    summary = [
        {
            "round": int(round_idx),
            "core_count": int(core_mask.sum()),
            "sample_count": int(labels.shape[0]),
            "mean_score_core": float(np.mean(score_core)) if score_core.size else 0.0,
        }
    ]
    state = {
        "cluster_centers": centers,
        "cluster_centers_v1": centers_v1,
        "cluster_centers_v2": centers_v2,
        "cluster_labels": labels,
        "core_mask": core_mask,
        "cluster_radii": radii,
        "global_center": global_center,
        "nearest_other_cluster": nearest_other,
        "score_core": score_core,
        "refresh_round": int(round_idx),
        "bank_mode": "stage2_ab_aligned_local_window",
        "bank_summary": summary,
        "proto_conf1": arrays["conf1"].astype(np.float32),
        "proto_conf2": arrays["conf2"].astype(np.float32),
        "proto_pred1": arrays["pred1"].astype(np.int64),
        "proto_pred2": arrays["pred2"].astype(np.int64),
        "proto_dist1": arrays["dist1"].astype(np.float32),
        "proto_dist2": arrays["dist2"].astype(np.float32),
        "recon1": arrays["recon1"].astype(np.float32),
        "recon2": arrays["recon2"].astype(np.float32),
    }
    self._apply_cluster_bank(state)
    print(
        "[Stage2] refreshed aligned prototype state | "
        f"round={int(round_idx)} | "
        f"samples={int(labels.shape[0])} | "
        f"mean_proto_score={float(np.mean(score_core)):.6f}"
    )


def _run_stage2_ab_refinement(
    self,
    num_stage2_rounds: int,
    stage2_a_epochs: int,
    stage2_b_epochs: int,
    total_stage2_epochs: int,
):
    self._initialize_aligned_separate_prototypes()
    loader = self._build_stage2_loader()
    global_epoch = 1
    for round_idx in range(1, int(num_stage2_rounds) + 1):
        print(f"[Stage2] Starting round {round_idx}/{int(num_stage2_rounds)}")
        for epoch_in_phase in range(1, int(stage2_a_epochs) + 1):
            self._run_stage2_a_epoch(
                loader=loader,
                round_idx=round_idx,
                epoch_in_phase=epoch_in_phase,
                global_epoch=global_epoch,
                total_epochs=total_stage2_epochs,
            )
            global_epoch += 1
        for epoch_in_phase in range(1, int(stage2_b_epochs) + 1):
            self._run_stage2_b_epoch(
                loader=loader,
                round_idx=round_idx,
                epoch_in_phase=epoch_in_phase,
                global_epoch=global_epoch,
                total_epochs=total_stage2_epochs,
            )
            global_epoch += 1
    self._refresh_aligned_stage2_state(round_idx=int(num_stage2_rounds))


def run_stage2_ab_refinement(
    solver,
    num_stage2_rounds=None,
    stage2_a_epochs=None,
    stage2_b_epochs=None,
    total_stage2_epochs=None,
):
    if num_stage2_rounds is None or stage2_a_epochs is None or stage2_b_epochs is None:
        num_stage2_rounds, stage2_a_epochs, stage2_b_epochs = solver._resolve_stage2_schedule()
    if total_stage2_epochs is None:
        total_stage2_epochs = solver._stage2_total_epochs()
    return solver._run_stage2_ab_refinement(
        num_stage2_rounds=int(num_stage2_rounds),
        stage2_a_epochs=int(stage2_a_epochs),
        stage2_b_epochs=int(stage2_b_epochs),
        total_stage2_epochs=int(total_stage2_epochs),
    )
