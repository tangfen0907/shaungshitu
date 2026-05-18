from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_factory.triplet_dataset import Stage1AdjacentPairDataset
from utils.clustering import _cluster_features, _nearest_other_clusters

# This route removes strong cross-view alignment from the dual-view prototype
# pipeline. A one-shot Hungarian reorder is allowed only at initialization so
# prototype IDs start in a readable order; after that, the two views keep
# independent representation and prototype spaces.

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
    "_build_stage2_b_loader",
    "_set_stage2_train_phase",
    "_select_a_core",
    "_select_a_core_mask_by_proto",
    "_prototype_core_pull_loss",
    "_prototype_assignment_pull_loss",
    "_prototype_separation_loss",
    "_stage2_a_batch_loss",
    "_stage2_a_collect_calibration_data",
    "_stage2_a_ema_update_one_view",
    "_run_stage2_a_epoch",
    "_select_b_core",
    "_select_per_proto_core_mask",
    "_prepare_stage2_b_batch",
    "_per_proto_radius_quantiles",
    "_refresh_stage2_boundary_radii",
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
    counts_v1 = np.bincount(labels1.astype(np.int64), minlength=num_prototypes).astype(int).tolist()
    counts_v2 = np.bincount(labels2.astype(np.int64), minlength=num_prototypes).astype(int).tolist()
    overlap_nonzero = int(np.count_nonzero(overlap))
    overlap_max = int(np.max(overlap)) if overlap.size else 0
    print(
        "[Stage2-Init] independent KMeans prototypes | "
        f"H1_train={tuple(z1.shape)} | "
        f"H2_train={tuple(z2.shape)} | "
        f"K={num_prototypes} | "
        f"kmeans_v1={meta1.get('cluster_method_actual', 'kmeans')} | "
        f"kmeans_v2={meta2.get('cluster_method_actual', 'kmeans')}"
    )
    print(
        "[Stage2-Init] assignment counts | "
        f"view1={counts_v1} | "
        f"view2={counts_v2}"
    )
    print(
        "[Stage2-Init] one-shot ID alignment only | "
        f"overlap_nonzero={overlap_nonzero}/{int(overlap.size)} | "
        f"overlap_max={overlap_max} | "
        f"matched_overlap={matched_count}/{total_count} | "
        f"match={match.astype(int).tolist()} | "
        "no coordinate alignment loss will be used"
    )


def _build_stage2_loader(self) -> DataLoader:
    return self.train_loader


def _build_stage2_b_loader(self) -> DataLoader:
    """
    Stage2-B needs paired local windows: X_t and the previous X_{t-1}.

    Stage2-A intentionally keeps using _build_stage2_loader() unchanged.  This
    B-only loader reuses the Stage1 adjacent-pair dataset so the positive/past
    window construction stays consistent without changing the A-phase sample
    stream.
    """
    base_dataset = getattr(self, "full_train_dataset", self.train_loader.dataset)
    active_mask = None
    if hasattr(self, "_current_active_train_mask"):
        mask = self._current_active_train_mask()
        if np.asarray(mask).reshape(-1).shape[0] == len(base_dataset):
            active_mask = mask

    step = int(getattr(base_dataset, "step", 1))
    if step != 1:
        print(
            "[Stage2-B][Warning] x_prev is intended to be X_{t-1}. "
            f"The base training dataset step is {step}, so adjacent dataset "
            "indices may be separated by more than one real timestamp."
        )

    try:
        pair_dataset = Stage1AdjacentPairDataset(
            base_dataset=base_dataset,
            positive_offset=1,
            positive_direction="past",
            active_mask=active_mask,
            in_channels=int(getattr(self.config, "in_channels", 0)),
            seq_len=int(getattr(self.config, "seq_len", 0)),
        )
    except RuntimeError as exc:
        print(
            "[Stage2-B][Warning] Could not build adjacent X_t/X_{t-1} loader; "
            f"falling back to the ordinary Stage2 loader with x_prev=x_t. reason={exc}"
        )
        return self._build_stage2_loader()

    return DataLoader(
        dataset=pair_dataset,
        batch_size=self.config.batch_size,
        shuffle=True,
        num_workers=self._effective_num_workers(),
        drop_last=False,
        pin_memory=self._pin_memory(),
        generator=self._make_loader_generator(301),
    )


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
        # Stage2-A learns only P1/P2 while keeping the feature geometry fixed.
        self.model.eval()
        for module in encoder_modules:
            module.requires_grad_(False)
            module.eval()
        p1.requires_grad_(True)
        p2.requires_grad_(True)
        self.optimizer = Adam(
            [p1, p2],
            lr=float(getattr(self.config, "lr", 1e-3)),
            weight_decay=float(getattr(self.config, "weight_decay", 1e-5)),
        )
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


def _select_a_core_mask_by_proto(
    self,
    score: torch.Tensor,
    labels: torch.Tensor,
    num_prototypes: int,
) -> torch.Tensor:
    if score.dim() != 1 or labels.dim() != 1:
        raise ValueError("Stage2-A core selection expects score [B] and labels [B].")
    if score.shape[0] != labels.shape[0]:
        raise ValueError(f"Stage2-A score/label size mismatch: {score.shape[0]} vs {labels.shape[0]}.")

    ratio = min(max(float(getattr(self.config, "core_ratio_A", 0.3)), 0.0), 1.0)
    min_core = max(1, int(getattr(self.config, "min_core_per_proto", 1)))
    core_mask = torch.zeros_like(score, dtype=torch.bool)
    with torch.no_grad():
        for proto_id in range(int(num_prototypes)):
            proto_indices = torch.nonzero(labels == proto_id, as_tuple=False).flatten()
            count = int(proto_indices.numel())
            if count <= 0:
                continue
            keep_count = max(min_core, int(np.ceil(float(count) * ratio)))
            keep_count = min(keep_count, count)
            order = torch.argsort(score.detach()[proto_indices], stable=True)
            core_mask[proto_indices[order[:keep_count]]] = True
    return core_mask


def _prototype_core_pull_loss(
    self,
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    core_mask: torch.Tensor,
) -> torch.Tensor:
    if not bool(core_mask.any()):
        return prototypes.sum() * 0.0
    selected_labels = labels[core_mask].long()
    selected_embeddings = embeddings.detach()[core_mask]
    selected_prototypes = prototypes[selected_labels]
    return torch.sum((selected_embeddings - selected_prototypes) ** 2, dim=1).mean()


def _prototype_assignment_pull_loss(
    self,
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    selected_prototypes = prototypes[labels.long()]
    return torch.sum((embeddings.detach() - selected_prototypes) ** 2, dim=1).mean()


def _prototype_separation_loss(self, prototypes: torch.Tensor) -> torch.Tensor:
    if int(prototypes.size(0)) <= 1:
        return prototypes.sum() * 0.0
    pair_indices = torch.triu_indices(
        int(prototypes.size(0)),
        int(prototypes.size(0)),
        offset=1,
        device=prototypes.device,
    )
    pairwise_dist = torch.cdist(prototypes, prototypes, p=2.0)[pair_indices[0], pair_indices[1]]
    margin = float(getattr(self.config, "proto_separation_margin", 1.0))
    return F.relu(margin - pairwise_dist).pow(2).mean()


def _stage2_a_batch_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    x = self._prepare_batch(batch)
    with torch.no_grad():
        outputs = self.model(x, stage="stage2", detach_prototypes=True)
        h1 = outputs["H1"].detach()
        h2 = outputs["H2"].detach()

    p1 = self.model.prototype_head_v1.prototypes
    p2 = self.model.prototype_head_v2.prototypes

    dist_mat1 = torch.cdist(h1, p1, p=2.0).pow(2)
    dist_mat2 = torch.cdist(h2, p2, p=2.0).pow(2)
    min_dist1, pred1 = torch.min(dist_mat1, dim=1)
    min_dist2, pred2 = torch.min(dist_mat2, dim=1)

    pull_v1 = self._prototype_assignment_pull_loss(h1, p1, pred1)
    pull_v2 = self._prototype_assignment_pull_loss(h2, p2, pred2)
    loss_pull_a = 0.5 * (pull_v1 + pull_v2)
    sep_v1 = self._prototype_separation_loss(p1)
    sep_v2 = self._prototype_separation_loss(p2)
    loss_sep_a = 0.5 * (sep_v1 + sep_v2)
    loss = (
        float(getattr(self.config, "lambda_pull_A", 1.0)) * loss_pull_a
        + float(getattr(self.config, "lambda_sep_A", 0.1)) * loss_sep_a
    )
    assign_count_v1 = torch.bincount(pred1.detach(), minlength=int(p1.size(0))).to(dtype=loss.dtype)
    assign_count_v2 = torch.bincount(pred2.detach(), minlength=int(p2.size(0))).to(dtype=loss.dtype)
    return loss, {
        "loss": loss,
        "loss_A": loss,
        "loss_pull_A": loss_pull_a,
        "loss_pull_v1": pull_v1,
        "loss_pull_v2": pull_v2,
        "loss_sep_A": loss_sep_a,
        "loss_sep_v1": sep_v1,
        "loss_sep_v2": sep_v2,
        "assign_count_v1": assign_count_v1,
        "assign_count_v2": assign_count_v2,
        "empty_proto_count_v1": (assign_count_v1 == 0).float().sum(),
        "empty_proto_count_v2": (assign_count_v2 == 0).float().sum(),
        "prototype_min_dist_v1": torch.as_tensor(
            _prototype_min_pairwise_distance(p1.detach()),
            device=p1.device,
            dtype=p1.dtype,
        ),
        "prototype_min_dist_v2": torch.as_tensor(
            _prototype_min_pairwise_distance(p2.detach()),
            device=p2.device,
            dtype=p2.dtype,
        ),
        "min_proto_dist_v1_mean": torch.sqrt(min_dist1.clamp_min(0.0) + 1e-12).mean(),
        "min_proto_dist_v2_mean": torch.sqrt(min_dist2.clamp_min(0.0) + 1e-12).mean(),
        "pair_mse_diag": F.mse_loss(p1, p2).detach(),
        "batch_size": torch.as_tensor(float(x.size(0)), device=x.device, dtype=loss.dtype),
    }


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
    totals = {
        "loss": 0.0,
        "loss_A": 0.0,
        "loss_pull_A": 0.0,
        "loss_pull_v1": 0.0,
        "loss_pull_v2": 0.0,
        "loss_sep_A": 0.0,
        "loss_sep_v1": 0.0,
        "loss_sep_v2": 0.0,
        "empty_proto_count_v1": 0.0,
        "empty_proto_count_v2": 0.0,
    }
    sample_weighted_totals = {
        "min_proto_dist_v1_mean": 0.0,
        "min_proto_dist_v2_mean": 0.0,
    }
    total_samples = 0.0
    assign_count_v1_total = None
    assign_count_v2_total = None

    progress = tqdm(
        loader,
        desc=f"Stage2-A R{round_idx} E{epoch_in_phase}",
        leave=False,
        disable=not self._show_batch_progress(),
    )
    for batch in progress:
        loss, metrics = self._stage2_a_batch_loss(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        batch_size = float(metrics["batch_size"].item())
        total_samples += batch_size
        for key in totals:
            totals[key] += float(metrics[key].item())
        for key in sample_weighted_totals:
            sample_weighted_totals[key] += float(metrics[key].item()) * batch_size
        batch_assign_v1 = metrics["assign_count_v1"].detach().cpu().numpy().astype(np.int64)
        batch_assign_v2 = metrics["assign_count_v2"].detach().cpu().numpy().astype(np.int64)
        if assign_count_v1_total is None:
            assign_count_v1_total = np.zeros_like(batch_assign_v1, dtype=np.int64)
            assign_count_v2_total = np.zeros_like(batch_assign_v2, dtype=np.int64)
        assign_count_v1_total += batch_assign_v1
        assign_count_v2_total += batch_assign_v2
        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "pull": f"{metrics['loss_pull_A'].item():.4f}",
                "sep": f"{metrics['loss_sep_A'].item():.4f}",
            }
        )

    denom = max(1, len(loader))
    sample_denom = max(1.0, total_samples)
    p1 = self.model.prototype_head_v1.prototypes.detach()
    p2 = self.model.prototype_head_v2.prototypes.detach()
    diagnostics = {
        "loss": totals["loss"] / denom,
        "loss_A": totals["loss_A"] / denom,
        "loss_pull_A": totals["loss_pull_A"] / denom,
        "loss_pull_v1": totals["loss_pull_v1"] / denom,
        "loss_pull_v2": totals["loss_pull_v2"] / denom,
        "loss_sep_A": totals["loss_sep_A"] / denom,
        "loss_sep_v1": totals["loss_sep_v1"] / denom,
        "loss_sep_v2": totals["loss_sep_v2"] / denom,
        "empty_proto_count_v1": int(np.sum(assign_count_v1_total == 0)) if assign_count_v1_total is not None else 0,
        "empty_proto_count_v2": int(np.sum(assign_count_v2_total == 0)) if assign_count_v2_total is not None else 0,
        "pair_mse_diag": float(F.mse_loss(p1, p2).item()),
        "prototype_min_dist_v1": _prototype_min_pairwise_distance(p1),
        "prototype_min_dist_v2": _prototype_min_pairwise_distance(p2),
        "min_proto_dist_v1_mean": sample_weighted_totals["min_proto_dist_v1_mean"] / sample_denom,
        "min_proto_dist_v2_mean": sample_weighted_totals["min_proto_dist_v2_mean"] / sample_denom,
    }
    self.stage2_a_last_diagnostics = dict(diagnostics)

    print(
        "[Stage2-A] learnable prototype training | "
        f"round={round_idx} | epoch_in_phase={epoch_in_phase} | "
        f"lambda_pull_A={float(getattr(self.config, 'lambda_pull_A', 1.0)):.3f} | "
        f"lambda_sep_A={float(getattr(self.config, 'lambda_sep_A', 0.1)):.3f}"
    )
    print(
        "[Stage2-A] diagnostics | "
        f"loss={diagnostics['loss']:.6f} | "
        f"loss_pull_A={diagnostics['loss_pull_A']:.6f} | "
        f"loss_pull_v1={diagnostics['loss_pull_v1']:.6f} | "
        f"loss_pull_v2={diagnostics['loss_pull_v2']:.6f} | "
        f"loss_sep_A={diagnostics['loss_sep_A']:.6f} | "
        f"loss_sep_v1={diagnostics['loss_sep_v1']:.6f} | "
        f"loss_sep_v2={diagnostics['loss_sep_v2']:.6f} | "
        f"empty_proto_count_v1={diagnostics['empty_proto_count_v1']:.6f} | "
        f"empty_proto_count_v2={diagnostics['empty_proto_count_v2']:.6f} | "
        f"pair_mse_diag={diagnostics['pair_mse_diag']:.6f} | "
        f"prototype_min_dist_v1={diagnostics['prototype_min_dist_v1']:.6f} | "
        f"prototype_min_dist_v2={diagnostics['prototype_min_dist_v2']:.6f} | "
        f"min_proto_dist_v1_mean={diagnostics['min_proto_dist_v1_mean']:.6f} | "
        f"min_proto_dist_v2_mean={diagnostics['min_proto_dist_v2_mean']:.6f}"
    )
    print(
        "[Stage2-A] assignment counts | "
        f"view1={(assign_count_v1_total.astype(int).tolist() if assign_count_v1_total is not None else [])} | "
        f"view2={(assign_count_v2_total.astype(int).tolist() if assign_count_v2_total is not None else [])}"
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


def _select_per_proto_core_mask(
    self,
    score: torch.Tensor,
    labels: torch.Tensor,
    num_prototypes: int,
) -> torch.Tensor:
    if score.dim() != 1 or labels.dim() != 1:
        raise ValueError("Per-prototype core selection expects score [B] and labels [B].")
    if score.shape[0] != labels.shape[0]:
        raise ValueError(f"Per-prototype score/label mismatch: {score.shape[0]} vs {labels.shape[0]}.")

    ratio = min(max(float(getattr(self.config, "core_ratio_B", 0.5)), 0.0), 1.0)
    core_mask = torch.zeros_like(score, dtype=torch.bool)
    if ratio <= 0.0:
        return core_mask

    with torch.no_grad():
        for proto_id in range(int(num_prototypes)):
            proto_indices = torch.nonzero(labels == proto_id, as_tuple=False).flatten()
            count = int(proto_indices.numel())
            if count <= 0:
                continue
            keep_count = max(1, int(np.ceil(float(count) * ratio)))
            keep_count = min(keep_count, count)
            order = torch.argsort(score.detach()[proto_indices], stable=True)
            core_mask[proto_indices[order[:keep_count]]] = True
    return core_mask


def _prepare_stage2_b_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Return (x_t, x_prev, has_prev) for Stage2-B.

    The B loader normally yields paired tensors from Stage1AdjacentPairDataset.
    If an ordinary window batch slips through, keep the code runnable by using
    x_prev=x_t and marking has_prev=False so AP/core terms can be skipped.
    """
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        left, right = batch[0], batch[1]
        left_dim = int(left.dim()) if isinstance(left, torch.Tensor) else int(np.ndim(left))
        right_dim = int(right.dim()) if isinstance(right, torch.Tensor) else int(np.ndim(right))
        if left_dim == 3 and right_dim == 3:
            return self._prepare_batch(left), self._prepare_batch(right), True

    x_t = self._prepare_batch(batch)
    return x_t, x_t, False


def _per_proto_radius_quantiles(
    self,
    labels: np.ndarray,
    min_distances: np.ndarray,
    num_prototypes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    min_distances = np.asarray(min_distances, dtype=np.float32).reshape(-1)
    if labels.shape[0] != min_distances.shape[0]:
        raise ValueError("Prototype labels and distances should have the same number of samples.")

    num_prototypes = int(num_prototypes)
    quantile = min(max(float(getattr(self.config, "boundary_quantile", 0.95)), 0.0), 1.0)
    eps = float(getattr(self.config, "robust_eps", 1e-6))
    radii = np.full(num_prototypes, np.nan, dtype=np.float32)
    counts = np.bincount(labels, minlength=num_prototypes).astype(np.int64)

    for proto_id in range(num_prototypes):
        proto_dist = min_distances[labels == proto_id]
        if proto_dist.size == 0:
            continue
        radius = float(np.quantile(proto_dist, quantile))
        if np.isfinite(radius):
            radii[proto_id] = max(radius, eps)

    finite = np.isfinite(radii)
    fallback = (
        float(np.median(radii[finite]))
        if bool(np.any(finite))
        else float(getattr(self.config, "stage2_injected_margin", 1.0))
    )
    fallback = max(float(fallback), eps)
    radii[~finite] = fallback
    return radii.astype(np.float32), counts


def _refresh_stage2_boundary_radii(self):
    self._uses_separate_prototypes()
    was_training = self.model.training
    self.model.eval()
    collected: Dict[str, List[np.ndarray]] = {
        "pred1": [],
        "pred2": [],
        "dist1": [],
        "dist2": [],
    }
    with torch.inference_mode():
        for batch in self.train_eval_loader:
            x = self._prepare_batch(batch)
            outputs = self.model(x, stage="stage2", detach_prototypes=True)
            collected["pred1"].append(outputs["proto_pred1"].detach().cpu().numpy())
            collected["pred2"].append(outputs["proto_pred2"].detach().cpu().numpy())
            collected["dist1"].append(outputs["proto_dist1"].detach().cpu().numpy())
            collected["dist2"].append(outputs["proto_dist2"].detach().cpu().numpy())
    if was_training:
        self.model.train()

    arrays = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    num_prototypes = int(self.model.prototype_head_v1.prototypes.size(0))
    radii_v1, counts_v1 = self._per_proto_radius_quantiles(arrays["pred1"], arrays["dist1"], num_prototypes)
    radii_v2, counts_v2 = self._per_proto_radius_quantiles(arrays["pred2"], arrays["dist2"], num_prototypes)
    self.stage2_boundary_radii_v1 = radii_v1
    self.stage2_boundary_radii_v2 = radii_v2
    self.stage2_boundary_assignment_counts_v1 = counts_v1
    self.stage2_boundary_assignment_counts_v2 = counts_v2
    print(
        "[Stage2-B] refreshed independent boundary radii | "
        f"quantile={float(getattr(self.config, 'boundary_quantile', 0.95)):.3f} | "
        f"radius_v1_mean={float(np.mean(radii_v1)):.6f} | "
        f"radius_v2_mean={float(np.mean(radii_v2)):.6f} | "
        f"assign_v1={counts_v1.astype(int).tolist()} | "
        f"assign_v2={counts_v2.astype(int).tolist()}"
    )


def _stage2_b_batch_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    x_t, x_prev, has_prev = self._prepare_stage2_b_batch(batch)

    outputs_t = self.model(x_t, stage="stage2", detach_prototypes=True)
    outputs_p = self.model(x_prev, stage="stage2", detach_prototypes=True)
    x_negative = self._inject_last_context(x_t, stage="stage2")
    outputs_n = self.model(x_negative, stage="stage2", detach_prototypes=True)

    rec_v1, rec_v2 = self._dual_reconstruction_losses_from_outputs(outputs_t, x_t)
    loss_rec = 0.5 * (rec_v1 + rec_v2)

    H1_t, H2_t = outputs_t["H1"], outputs_t["H2"]
    H1_p, H2_p = outputs_p["H1"], outputs_p["H2"]
    H1_n, H2_n = outputs_n["H1"], outputs_n["H2"]
    P1 = self.model.prototype_head_v1.prototypes.detach()
    P2 = self.model.prototype_head_v2.prototypes.detach()

    D1_t = torch.cdist(H1_t, P1, p=2.0).pow(2)
    D2_t = torch.cdist(H2_t, P2, p=2.0).pow(2)
    D1_n = torch.cdist(H1_n, P1, p=2.0).pow(2)
    D2_n = torch.cdist(H2_n, P2, p=2.0).pow(2)

    stage2_ap_margin = float(getattr(self.config, "stage2_ap_margin", 0.1))
    if bool(has_prev):
        ap_dist_v1 = torch.norm(H1_t - H1_p, dim=1)
        ap_dist_v2 = torch.norm(H2_t - H2_p, dim=1)
        loss_ap_v1 = F.relu(ap_dist_v1 - stage2_ap_margin).pow(2).mean()
        loss_ap_v2 = F.relu(ap_dist_v2 - stage2_ap_margin).pow(2).mean()
        loss_ap = 0.5 * (loss_ap_v1 + loss_ap_v2)
        ap_active_ratio_v1 = (ap_dist_v1 > stage2_ap_margin).float().mean()
        ap_active_ratio_v2 = (ap_dist_v2 > stage2_ap_margin).float().mean()
    else:
        loss_ap = H1_t.sum() * 0.0
        ap_dist_v1 = torch.zeros_like(H1_t[:, 0])
        ap_dist_v2 = torch.zeros_like(H2_t[:, 0])
        loss_ap_v1 = loss_ap
        loss_ap_v2 = loss_ap
        ap_active_ratio_v1 = loss_ap
        ap_active_ratio_v2 = loss_ap

    min_d1_t_sq, k1_t = torch.min(D1_t, dim=1)
    min_d2_t_sq, k2_t = torch.min(D2_t, dim=1)
    min_d1_n_sq, k1_n = torch.min(D1_n, dim=1)
    min_d2_n_sq, k2_n = torch.min(D2_n, dim=1)
    min_d1_t = torch.sqrt(min_d1_t_sq.clamp_min(0.0) + 1e-12)
    min_d2_t = torch.sqrt(min_d2_t_sq.clamp_min(0.0) + 1e-12)
    min_d1_n = torch.sqrt(min_d1_n_sq.clamp_min(0.0) + 1e-12)
    min_d2_n = torch.sqrt(min_d2_n_sq.clamp_min(0.0) + 1e-12)

    num_prototypes = int(P1.size(0))
    core_v1 = self._select_per_proto_core_mask(min_d1_t_sq, k1_t, num_prototypes)
    core_v2 = self._select_per_proto_core_mask(min_d2_t_sq, k2_t, num_prototypes)
    joint_core = core_v1 & core_v2

    if bool(joint_core.any()):
        selected_k1 = k1_t[joint_core].long()
        selected_k2 = k2_t[joint_core].long()
        loss_core_v1 = torch.sum((H1_t[joint_core] - P1[selected_k1]) ** 2, dim=1).mean()
        loss_core_v2 = torch.sum((H2_t[joint_core] - P2[selected_k2]) ** 2, dim=1).mean()
        loss_core = 0.5 * (loss_core_v1 + loss_core_v2)
    else:
        selected_k1 = k1_t.new_empty((0,), dtype=torch.long)
        selected_k2 = k2_t.new_empty((0,), dtype=torch.long)
        loss_core_v1 = H1_t.sum() * 0.0
        loss_core_v2 = H2_t.sum() * 0.0
        loss_core = H1_t.sum() * 0.0

    use_radius = bool(getattr(self.config, "use_negative_boundary_radius", True))
    negative_boundary_margin = float(getattr(self.config, "negative_boundary_margin", 0.1))
    if use_radius and hasattr(self, "stage2_boundary_radii_v1") and hasattr(self, "stage2_boundary_radii_v2"):
        radii_v1 = torch.as_tensor(self.stage2_boundary_radii_v1, device=x_t.device, dtype=H1_t.dtype)
        radii_v2 = torch.as_tensor(self.stage2_boundary_radii_v2, device=x_t.device, dtype=H2_t.dtype)
        target_v1 = radii_v1[k1_n] + negative_boundary_margin
        target_v2 = radii_v2[k2_n] + negative_boundary_margin
    else:
        fixed_margin = float(getattr(self.config, "stage2_injected_margin", getattr(self.config, "margin_anom", 1.0)))
        target_v1 = torch.full_like(min_d1_n, fixed_margin)
        target_v2 = torch.full_like(min_d2_n, fixed_margin)
    loss_neg_v1 = F.relu(target_v1 - min_d1_n).pow(2).mean()
    loss_neg_v2 = F.relu(target_v2 - min_d2_n).pow(2).mean()
    loss_neg = 0.5 * (loss_neg_v1 + loss_neg_v2)
    neg_boundary_active_ratio_v1 = (min_d1_n < target_v1).float().mean()
    neg_boundary_active_ratio_v2 = (min_d2_n < target_v2).float().mean()

    loss = (
        float(getattr(self.config, "lambda_rec_B", 1.0)) * loss_rec
        + float(getattr(self.config, "lambda_ap_B", 0.2)) * loss_ap
        + float(getattr(self.config, "lambda_core_B", getattr(self.config, "lambda_pull_B", 0.5))) * loss_core
        + float(getattr(self.config, "lambda_neg_B", getattr(self.config, "lambda_anom_B", 0.05))) * loss_neg
    )
    same_proto = k1_t == k2_t
    total_count = torch.as_tensor(float(x_t.size(0)), device=x_t.device, dtype=loss.dtype)
    joint_core_count = joint_core.float().sum().to(dtype=loss.dtype)
    core_count_per_proto_v1 = torch.bincount(
        selected_k1.detach(),
        minlength=num_prototypes,
    ).to(device=x_t.device, dtype=loss.dtype)
    core_count_per_proto_v2 = torch.bincount(
        selected_k2.detach(),
        minlength=num_prototypes,
    ).to(device=x_t.device, dtype=loss.dtype)
    return loss, {
        "loss": loss,
        "loss_B": loss,
        "loss_rec": loss_rec,
        "loss_ap": loss_ap,
        "loss_ap_v1": loss_ap_v1,
        "loss_ap_v2": loss_ap_v2,
        "loss_core": loss_core,
        "loss_core_v1": loss_core_v1,
        "loss_core_v2": loss_core_v2,
        "loss_neg": loss_neg,
        "loss_neg_v1": loss_neg_v1,
        "loss_neg_v2": loss_neg_v2,
        "ap_dist_v1_mean": ap_dist_v1.mean(),
        "ap_dist_v2_mean": ap_dist_v2.mean(),
        "ap_active_ratio_v1": ap_active_ratio_v1,
        "ap_active_ratio_v2": ap_active_ratio_v2,
        "stage2_ap_margin": torch.as_tensor(stage2_ap_margin, device=x_t.device, dtype=loss.dtype),
        "same_proto_ratio": same_proto.float().mean(),
        "core_count": joint_core_count,
        "joint_core_count": joint_core_count,
        "total_count": total_count,
        "core_ratio": joint_core_count / total_count.clamp_min(1.0),
        "core_count_per_proto_v1": core_count_per_proto_v1,
        "core_count_per_proto_v2": core_count_per_proto_v2,
        "normal_min_proto_dist_v1_mean": min_d1_t.mean(),
        "normal_min_proto_dist_v2_mean": min_d2_t.mean(),
        "neg_min_proto_dist_v1_mean": min_d1_n.mean(),
        "neg_min_proto_dist_v2_mean": min_d2_n.mean(),
        "neg_boundary_active_ratio_v1": neg_boundary_active_ratio_v1,
        "neg_boundary_active_ratio_v2": neg_boundary_active_ratio_v2,
        "prototype_min_dist_v1": torch.as_tensor(
            _prototype_min_pairwise_distance(P1),
            device=x_t.device,
            dtype=loss.dtype,
        ),
        "prototype_min_dist_v2": torch.as_tensor(
            _prototype_min_pairwise_distance(P2),
            device=x_t.device,
            dtype=loss.dtype,
        ),
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
    self._refresh_stage2_boundary_radii()
    self.model.train()
    totals = {
        "loss": 0.0,
        "loss_B": 0.0,
        "loss_rec": 0.0,
        "loss_ap": 0.0,
        "loss_ap_v1": 0.0,
        "loss_ap_v2": 0.0,
        "loss_core": 0.0,
        "loss_core_v1": 0.0,
        "loss_core_v2": 0.0,
        "loss_neg": 0.0,
        "loss_neg_v1": 0.0,
        "loss_neg_v2": 0.0,
        "ap_dist_v1_mean": 0.0,
        "ap_dist_v2_mean": 0.0,
        "ap_active_ratio_v1": 0.0,
        "ap_active_ratio_v2": 0.0,
        "stage2_ap_margin": 0.0,
        "same_proto_ratio": 0.0,
        "core_ratio": 0.0,
        "normal_min_proto_dist_v1_mean": 0.0,
        "normal_min_proto_dist_v2_mean": 0.0,
        "neg_min_proto_dist_v1_mean": 0.0,
        "neg_min_proto_dist_v2_mean": 0.0,
        "neg_boundary_active_ratio_v1": 0.0,
        "neg_boundary_active_ratio_v2": 0.0,
        "prototype_min_dist_v1": 0.0,
        "prototype_min_dist_v2": 0.0,
    }
    joint_core_count_total = 0.0
    total_count = 0.0
    core_count_per_proto_v1_total = None
    core_count_per_proto_v2_total = None
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
            value = float(metrics[key].item())
            if key in {
                "same_proto_ratio",
                "core_ratio",
                "normal_min_proto_dist_v1_mean",
                "normal_min_proto_dist_v2_mean",
                "neg_min_proto_dist_v1_mean",
                "neg_min_proto_dist_v2_mean",
                "neg_boundary_active_ratio_v1",
                "neg_boundary_active_ratio_v2",
            }:
                value *= float(metrics["total_count"].item())
            totals[key] += value
        batch_core_count = float(metrics["joint_core_count"].item())
        batch_total_count = float(metrics["total_count"].item())
        joint_core_count_total += batch_core_count
        total_count += batch_total_count
        batch_core_count_per_proto_v1 = metrics["core_count_per_proto_v1"].detach().cpu().numpy().astype(np.int64)
        batch_core_count_per_proto_v2 = metrics["core_count_per_proto_v2"].detach().cpu().numpy().astype(np.int64)
        if core_count_per_proto_v1_total is None:
            core_count_per_proto_v1_total = np.zeros_like(batch_core_count_per_proto_v1, dtype=np.int64)
            core_count_per_proto_v2_total = np.zeros_like(batch_core_count_per_proto_v2, dtype=np.int64)
        core_count_per_proto_v1_total += batch_core_count_per_proto_v1
        core_count_per_proto_v2_total += batch_core_count_per_proto_v2
        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "rec": f"{metrics['loss_rec'].item():.4f}",
                "ap": f"{metrics['loss_ap'].item():.4f}",
                "neg": f"{metrics['loss_neg'].item():.4f}",
                "core": f"{metrics['loss_core'].item():.4f}",
            }
        )
    denom = max(1, len(loader))
    sample_denom = max(1.0, total_count)
    logs = {key: value / denom for key, value in totals.items()}
    for key in {
        "same_proto_ratio",
        "core_ratio",
        "normal_min_proto_dist_v1_mean",
        "normal_min_proto_dist_v2_mean",
        "neg_min_proto_dist_v1_mean",
        "neg_min_proto_dist_v2_mean",
        "neg_boundary_active_ratio_v1",
        "neg_boundary_active_ratio_v2",
    }:
        logs[key] = totals[key] / sample_denom
    logs["core_count"] = int(joint_core_count_total)
    logs["joint_core_count"] = int(joint_core_count_total)
    logs["total_count"] = int(total_count)
    logs["core_ratio"] = joint_core_count_total / sample_denom
    radii_v1 = np.asarray(getattr(self, "stage2_boundary_radii_v1", np.empty(0)), dtype=np.float32)
    radii_v2 = np.asarray(getattr(self, "stage2_boundary_radii_v2", np.empty(0)), dtype=np.float32)
    if radii_v1.size:
        logs.update(
            {
                "radius_v1_mean": float(np.mean(radii_v1)),
                "radius_v1_min": float(np.min(radii_v1)),
                "radius_v1_max": float(np.max(radii_v1)),
            }
        )
    if radii_v2.size:
        logs.update(
            {
                "radius_v2_mean": float(np.mean(radii_v2)),
                "radius_v2_min": float(np.min(radii_v2)),
                "radius_v2_max": float(np.max(radii_v2)),
            }
        )
    self._log_epoch(
        "Stage2-B",
        global_epoch,
        total_epochs,
        logs,
    )
    print(
        f"[Stage2-B] Epoch {global_epoch}/{total_epochs} | "
        f"core_count_per_proto_v1: "
        f"{(core_count_per_proto_v1_total.astype(int).tolist() if core_count_per_proto_v1_total is not None else [])} | "
        f"core_count_per_proto_v2: "
        f"{(core_count_per_proto_v2_total.astype(int).tolist() if core_count_per_proto_v2_total is not None else [])}"
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
    centers = centers_v1
    labels = arrays["pred1"].astype(np.int64)
    core_mask = np.ones(labels.shape[0], dtype=bool)
    score_core = arrays["dist1"].astype(np.float32)
    final_a_score_v1 = np.square(arrays["dist1"].astype(np.float32))
    final_a_score_v2 = np.square(arrays["dist2"].astype(np.float32))
    num_prototypes = int(centers_v1.shape[0])
    final_a_core_mask_v1 = self._select_a_core_mask_by_proto(
        torch.from_numpy(final_a_score_v1),
        torch.from_numpy(arrays["pred1"].astype(np.int64)),
        num_prototypes,
    ).cpu().numpy().astype(bool)
    final_a_core_mask_v2 = self._select_a_core_mask_by_proto(
        torch.from_numpy(final_a_score_v2),
        torch.from_numpy(arrays["pred2"].astype(np.int64)),
        num_prototypes,
    ).cpu().numpy().astype(bool)

    full_train_dataset = getattr(self, "full_train_dataset", self.train_eval_loader.dataset)
    full_train_count = int(len(full_train_dataset))
    active_indices = (
        self._current_active_train_indices()
        if hasattr(self, "_current_active_train_indices")
        else np.arange(labels.shape[0], dtype=np.int64)
    )
    active_indices = np.asarray(active_indices, dtype=np.int64).reshape(-1)
    if active_indices.shape[0] != labels.shape[0]:
        active_indices = np.arange(labels.shape[0], dtype=np.int64)
        full_train_count = max(full_train_count, int(labels.shape[0]))
    final_a_core_mask_v1_full = np.zeros(full_train_count, dtype=bool)
    final_a_core_mask_v2_full = np.zeros(full_train_count, dtype=bool)
    valid_active = (active_indices >= 0) & (active_indices < full_train_count)
    final_a_core_mask_v1_full[active_indices[valid_active]] = final_a_core_mask_v1[valid_active]
    final_a_core_mask_v2_full[active_indices[valid_active]] = final_a_core_mask_v2[valid_active]
    radii_v1, counts_v1 = self._per_proto_radius_quantiles(arrays["pred1"], arrays["dist1"], num_prototypes)
    radii_v2, counts_v2 = self._per_proto_radius_quantiles(arrays["pred2"], arrays["dist2"], num_prototypes)
    radii = radii_v1
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
        "cluster_radii_v1": radii_v1,
        "cluster_radii_v2": radii_v2,
        "global_center": global_center,
        "nearest_other_cluster": nearest_other,
        "score_core": score_core,
        "refresh_round": int(round_idx),
        "bank_mode": "stage2_ab_proto_no_view_align_local_window",
        "bank_summary": summary,
        "proto_conf1": arrays["conf1"].astype(np.float32),
        "proto_conf2": arrays["conf2"].astype(np.float32),
        "proto_pred1": arrays["pred1"].astype(np.int64),
        "proto_pred2": arrays["pred2"].astype(np.int64),
        "proto_dist1": arrays["dist1"].astype(np.float32),
        "proto_dist2": arrays["dist2"].astype(np.float32),
        "recon1": arrays["recon1"].astype(np.float32),
        "recon2": arrays["recon2"].astype(np.float32),
        "final_a_score_v1": final_a_score_v1,
        "final_a_score_v2": final_a_score_v2,
        "final_a_core_mask_v1": final_a_core_mask_v1,
        "final_a_core_mask_v2": final_a_core_mask_v2,
        "final_a_core_mask_v1_full": final_a_core_mask_v1_full,
        "final_a_core_mask_v2_full": final_a_core_mask_v2_full,
    }
    self._apply_cluster_bank(state)
    print(
        "[Stage2] refreshed independent prototype state | "
        f"round={int(round_idx)} | "
        f"samples={int(labels.shape[0])} | "
        f"mean_proto_score_v1={float(np.mean(arrays['dist1'])):.6f} | "
        f"mean_proto_score_v2={float(np.mean(arrays['dist2'])):.6f} | "
        f"assign_v1={counts_v1.astype(int).tolist()} | "
        f"assign_v2={counts_v2.astype(int).tolist()} | "
        f"final_a_core_v1={int(final_a_core_mask_v1.sum())} | "
        f"final_a_core_v2={int(final_a_core_mask_v2.sum())}"
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
    loader_b = self._build_stage2_b_loader()
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
                loader=loader_b,
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
