import torch
import torch.nn as nn


class Reconstructor(nn.Module):
    """
    Decode either:
        - local-window/current-point latent vectors [B, D] into [B, C, output_len]
        - point-level latent sequences [B, T, D] into [B, T, C]

    In the new dual-view route output_len=1, so H_t reconstructs only the
    current point x_t, not the whole input context window.
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int = None,
        seq_len: int = None,
        hidden_dim: int = 128,
        output_len: int = 1,
        in_channels: int = None,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if out_channels is None:
            raise ValueError("out_channels is required.")
        self.out_channels = int(out_channels)
        self.seq_len = seq_len
        self.output_len = int(output_len)

        self.window_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, self.out_channels * self.output_len),
        )
        self.point_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.out_channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            x_hat = self.window_decoder(z)
            return x_hat.view(z.size(0), self.out_channels, self.output_len)
        if z.dim() == 3:
            return self.point_decoder(z)

        raise ValueError(f"Reconstructor expects [B, D] or [B, T, D], got {tuple(z.shape)}")
