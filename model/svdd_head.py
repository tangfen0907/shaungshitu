from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class MultiCenterSVDD(nn.Module):
    """Multi-center SVDD state used by the legacy score path."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.register_buffer("global_c", torch.zeros(self.latent_dim))
        self.register_buffer("cluster_centers", torch.zeros(1, self.latent_dim))
        self.register_buffer("cluster_priors", torch.ones(1))

    def update_centers(
        self,
        new_centers: Union[np.ndarray, torch.Tensor],
        features: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        if isinstance(new_centers, np.ndarray):
            centers = torch.from_numpy(new_centers).float()
        elif isinstance(new_centers, torch.Tensor):
            centers = new_centers.detach().float().cpu()
        else:
            raise TypeError("new_centers must be a numpy.ndarray or torch.Tensor.")

        if centers.dim() != 2 or centers.size(-1) != self.latent_dim:
            raise ValueError(
                f"new_centers should be [m, {self.latent_dim}], got {tuple(centers.shape)}."
            )
        if centers.size(0) == 0:
            raise ValueError("At least one center is required for SVDD.")

        priors = torch.ones(centers.size(0), dtype=torch.float32) / centers.size(0)
        if features is None:
            global_center = centers.mean(dim=0)
        elif isinstance(features, np.ndarray):
            global_center = torch.from_numpy(features).float().mean(dim=0)
        elif isinstance(features, torch.Tensor):
            global_center = features.detach().float().cpu().mean(dim=0)
        else:
            raise TypeError("features must be None, numpy.ndarray, or torch.Tensor.")

        self.cluster_centers = centers.to(self.global_c.device)
        self.cluster_priors = priors.to(self.global_c.device)
        self.global_c = global_center.to(self.global_c.device)

    @torch.no_grad()
    def ema_update_cluster_centers(
        self,
        cluster_ids: torch.Tensor,
        batch_cluster_means: torch.Tensor,
        momentum: float,
    ):
        if batch_cluster_means.numel() == 0:
            return
        momentum = min(max(float(momentum), 0.0), 1.0)
        cluster_ids = cluster_ids.detach().long().to(self.cluster_centers.device)
        batch_cluster_means = batch_cluster_means.detach().to(self.cluster_centers.device)
        updated = self.cluster_centers.clone()
        updated[cluster_ids] = momentum * updated[cluster_ids] + (1.0 - momentum) * batch_cluster_means
        self.cluster_centers = updated

    @torch.no_grad()
    def ema_update_global_center(self, batch_features: torch.Tensor, momentum: float):
        if batch_features.numel() == 0:
            return
        momentum = min(max(float(momentum), 0.0), 1.0)
        batch_mean = batch_features.detach().to(self.global_c.device).mean(dim=0)
        self.global_c = momentum * self.global_c + (1.0 - momentum) * batch_mean

    def compute_gravity(self, z: torch.Tensor, tau: float) -> torch.Tensor:
        tau = max(float(tau), 1e-6)
        distances = torch.cdist(z, self.cluster_centers, p=2.0) ** 2
        log_prior = torch.log(self.cluster_priors.clamp_min(1e-12)).unsqueeze(0)
        weighted_logits = -distances / tau + log_prior
        return -torch.logsumexp(weighted_logits, dim=1)
