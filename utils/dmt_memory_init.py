from typing import Iterable

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


def _batch_to_x(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


@torch.no_grad()
def collect_h1_tokens(model, loader: Iterable, device: torch.device, max_tokens: int = 200000) -> torch.Tensor:
    """
    Collect DMT-M1 encoder tokens for K-means memory initialization.

    model(x, mode="pretrain") returns H1: [B, C, P, D].
    Returned tokens have shape [num_tokens, D] on CPU.
    """
    model.eval()
    max_tokens = int(max_tokens)
    chunks = []
    total = 0
    for batch in loader:
        x = _batch_to_x(batch).float().to(device)
        out = model(x, mode="pretrain")
        tokens = out["H1"].reshape(-1, out["H1"].size(-1)).detach().cpu()
        chunks.append(tokens)
        total += tokens.size(0)
        if total >= max_tokens:
            break

    if not chunks:
        raise RuntimeError("no H1 tokens were collected; check the data loader")

    tokens = torch.cat(chunks, dim=0)
    if tokens.size(0) > max_tokens:
        index = torch.randperm(tokens.size(0))[:max_tokens]
        tokens = tokens[index]
    return tokens.contiguous()


def kmeans_init(tokens: torch.Tensor, n_memory: int, seed: int = 42) -> torch.Tensor:
    """
    Run MiniBatchKMeans on encoder tokens.

    Args:
        tokens: Tensor [num_tokens, D].
        n_memory: Number of memory items K.
        seed: Random seed for K-means.

    Returns:
        centers: Tensor [K, D], dtype float32.
    """
    if tokens.dim() != 2:
        raise ValueError(f"tokens must be [num_tokens, D], got shape {tuple(tokens.shape)}")
    n_memory = int(n_memory)
    if tokens.size(0) < n_memory:
        raise ValueError(f"not enough tokens for K-means: tokens={tokens.size(0)}, n_memory={n_memory}")

    data = tokens.detach().cpu().numpy().astype(np.float32, copy=False)
    kmeans = MiniBatchKMeans(
        n_clusters=n_memory,
        random_state=int(seed),
        batch_size=min(4096, max(n_memory * 16, 256)),
        n_init=10,
    )
    kmeans.fit(data)
    return torch.from_numpy(kmeans.cluster_centers_.astype(np.float32, copy=False))
