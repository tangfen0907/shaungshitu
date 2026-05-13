import torch
import torch.nn.functional as F


def _squared_l2_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance for each sample in a batch."""
    return torch.sum((x - y) ** 2, dim=-1)


def sharpen_distribution(q: torch.Tensor, temperature: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    temperature = max(float(temperature), float(eps))
    q = q.clamp_min(float(eps))
    sharpened = q ** (1.0 / temperature)
    return sharpened / sharpened.sum(dim=1, keepdim=True).clamp_min(float(eps))


def js_divergence(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q1 = q1.clamp_min(float(eps))
    q2 = q2.clamp_min(float(eps))
    q1 = q1 / q1.sum(dim=1, keepdim=True).clamp_min(float(eps))
    q2 = q2 / q2.sum(dim=1, keepdim=True).clamp_min(float(eps))
    m = 0.5 * (q1 + q2)
    kl1 = torch.sum(q1 * (torch.log(q1) - torch.log(m.clamp_min(float(eps)))), dim=1)
    kl2 = torch.sum(q2 * (torch.log(q2) - torch.log(m.clamp_min(float(eps)))), dim=1)
    return 0.5 * (kl1 + kl2)


def consensus_teacher_distribution(
    q1: torch.Tensor,
    q2: torch.Tensor,
    sharpen_temperature: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    q_avg = 0.5 * (q1 + q2)
    return sharpen_distribution(q_avg, temperature=sharpen_temperature, eps=eps).detach()


def state_consistency_teacher_loss(
    q1: torch.Tensor,
    q2: torch.Tensor,
    q_teacher: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if q1.numel() == 0:
        return torch.zeros((), device=q_teacher.device, dtype=q_teacher.dtype)
    q_teacher = q_teacher.detach().clamp_min(float(eps))
    q_teacher = q_teacher / q_teacher.sum(dim=1, keepdim=True).clamp_min(float(eps))
    log_q1 = torch.log(q1.clamp_min(float(eps)))
    log_q2 = torch.log(q2.clamp_min(float(eps)))
    loss1 = F.kl_div(log_q1, q_teacher, reduction="batchmean")
    loss2 = F.kl_div(log_q2, q_teacher, reduction="batchmean")
    return 0.5 * (loss1 + loss2)


def consensus_prototype_pull_loss(
    u1: torch.Tensor,
    u2: torch.Tensor,
    prototypes: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if u1.numel() == 0:
        return torch.zeros((), device=prototypes.device, dtype=prototypes.dtype)
    target = target.long().clamp(0, prototypes.size(0) - 1)
    target_proto = prototypes[target]
    pull1 = _squared_l2_distance(u1, target_proto)
    pull2 = _squared_l2_distance(u2, target_proto)
    return 0.5 * (pull1 + pull2).mean()


def prototype_usage_balance_loss(
    q1: torch.Tensor,
    q2: torch.Tensor = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL(mean prototype usage || uniform), used only by anti-collapse variants."""
    if q1.numel() == 0:
        return torch.zeros((), device=q1.device, dtype=q1.dtype)
    if q2 is None:
        q_mean = q1.mean(dim=0)
    else:
        q_mean = 0.5 * (q1.mean(dim=0) + q2.mean(dim=0))
    q_mean = q_mean.clamp_min(float(eps))
    q_mean = q_mean / q_mean.sum().clamp_min(float(eps))
    num_proto = max(1, int(q_mean.numel()))
    return torch.sum(q_mean * torch.log(q_mean * float(num_proto)))


def prototype_repulsion_loss(
    u: torch.Tensor,
    prototypes: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if u.numel() == 0:
        return torch.zeros((), device=prototypes.device, dtype=prototypes.dtype)
    dist = torch.cdist(u, prototypes, p=2.0)
    min_dist = torch.min(dist, dim=1).values
    return F.relu(float(margin) - min_dist).pow(2).mean()


def prototype_separation_loss(
    prototypes: torch.Tensor,
    margin: float,
    force_weight: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    if prototypes.size(0) <= 1 or float(margin) <= 0.0:
        return torch.zeros((), device=prototypes.device, dtype=prototypes.dtype)
    diff = prototypes.unsqueeze(1) - prototypes.unsqueeze(0)
    dist_sq = torch.sum(diff.pow(2), dim=-1)
    mask = torch.triu(
        torch.ones(prototypes.size(0), prototypes.size(0), dtype=torch.bool, device=prototypes.device),
        diagonal=1,
    )
    pairwise_dist_sq = dist_sq[mask]
    margin_sq = float(margin) ** 2
    hinge = F.relu(margin_sq - pairwise_dist_sq).pow(2) / max(margin_sq, float(eps))

    force_weight = max(float(force_weight), 0.0)
    if force_weight <= 0.0:
        return hinge.mean()
    repulsive_force = F.relu(margin_sq / pairwise_dist_sq.clamp_min(float(eps)) - 1.0).clamp_max(10.0)
    return (hinge + force_weight * repulsive_force).mean()
