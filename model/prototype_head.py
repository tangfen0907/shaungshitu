from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeHead(nn.Module):
    """Per-view learnable prototypes.

    One instance owns one prototype table with shape [K, D]. In dual-view
    mode the model creates two independent instances, one for View1 and one
    for View2. The parameters are never shared between views.
    """

    def __init__(self, num_prototypes: int, state_dim: int, temperature: float = 0.2):
        super().__init__()
        self.num_prototypes = int(max(1, num_prototypes))
        self.state_dim = int(state_dim)
        self.temperature = float(max(temperature, 1e-6))
        self.prototypes = nn.Parameter(torch.empty(self.num_prototypes, self.state_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    @torch.no_grad()
    def init_from_centers(self, centers: Union[np.ndarray, torch.Tensor]):
        if isinstance(centers, np.ndarray):
            centers = torch.from_numpy(centers).float()
        elif isinstance(centers, torch.Tensor):
            centers = centers.detach().float().cpu()
        else:
            raise TypeError("centers must be a numpy array or torch tensor.")
        if centers.dim() != 2 or centers.size(0) != self.num_prototypes or centers.size(1) != self.state_dim:
            raise ValueError(
                f"Prototype centers should be [{self.num_prototypes}, {self.state_dim}], "
                f"got {tuple(centers.shape)}."
            )
        self.prototypes.copy_(centers.to(device=self.prototypes.device, dtype=self.prototypes.dtype))

    def distances(self, u: torch.Tensor, detach_prototypes: bool = False) -> torch.Tensor:
        if u.size(-1) != self.state_dim:
            raise ValueError(
                f"PrototypeHead expected last dim D={self.state_dim}, got {int(u.size(-1))}."
            )
        leading_shape = u.shape[:-1]
        u_flat = u.reshape(-1, self.state_dim)
        prototypes = self.prototypes.detach() if bool(detach_prototypes) else self.prototypes
        dist_sq_flat = torch.cdist(u_flat, prototypes, p=2.0) ** 2
        return dist_sq_flat.reshape(*leading_shape, self.num_prototypes)

    def forward(self, u: torch.Tensor, detach_prototypes: bool = False) -> dict:
        if u.size(-1) != self.state_dim:
            raise ValueError(
                f"PrototypeHead expected last dim D={self.state_dim}, got {int(u.size(-1))}."
            )
        leading_shape = u.shape[:-1]
        u_flat = u.reshape(-1, self.state_dim)
        prototypes = self.prototypes.detach() if bool(detach_prototypes) else self.prototypes
        dist_sq_flat = torch.cdist(u_flat, prototypes, p=2.0) ** 2
        logits_flat = -dist_sq_flat / max(float(self.temperature), 1e-6)
        q_flat = F.softmax(logits_flat, dim=-1)
        conf_flat, pred_flat = torch.max(q_flat, dim=-1)
        min_dist_sq_flat = torch.gather(dist_sq_flat, 1, pred_flat.view(-1, 1)).squeeze(1)
        min_dist_flat = torch.sqrt(min_dist_sq_flat.clamp_min(0.0) + 1e-12)

        dist_sq = dist_sq_flat.reshape(*leading_shape, self.num_prototypes)
        q = q_flat.reshape(*leading_shape, self.num_prototypes)
        pred = pred_flat.reshape(*leading_shape)
        conf = conf_flat.reshape(*leading_shape)
        min_dist = min_dist_flat.reshape(*leading_shape)
        return {
            "dist_sq": dist_sq,
            "q": q,
            "pred": pred,
            "conf": conf,
            "min_dist": min_dist,
        }


class DualViewPrototypeHeads(nn.Module):
    """Separate prototype heads for View1 and View2."""

    def __init__(self, num_prototypes: int, state_dim: int, temperature: float = 0.2):
        super().__init__()
        self.prototype_head_v1 = PrototypeHead(num_prototypes, state_dim, temperature)
        self.prototype_head_v2 = PrototypeHead(num_prototypes, state_dim, temperature)

    @torch.no_grad()
    def init_from_centers(
        self,
        centers_v1: Union[np.ndarray, torch.Tensor],
        centers_v2: Union[np.ndarray, torch.Tensor],
    ):
        self.prototype_head_v1.init_from_centers(centers_v1)
        self.prototype_head_v2.init_from_centers(centers_v2)

    def forward(self, u1: torch.Tensor, u2: torch.Tensor, detach_prototypes: bool = False) -> dict:
        return {
            "view1": self.prototype_head_v1(u1, detach_prototypes=detach_prototypes),
            "view2": self.prototype_head_v2(u2, detach_prototypes=detach_prototypes),
        }
