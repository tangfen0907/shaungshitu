import torch
import torch.nn as nn


class Reconstructor(nn.Module):
    """
    Decode a window-level latent vector into the raw-window target space.
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

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, self.out_channels * self.output_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"Reconstructor expects [B, D], got {tuple(z.shape)}")

        x_hat = self.decoder(z)
        return x_hat.view(z.size(0), self.out_channels, self.output_len)
