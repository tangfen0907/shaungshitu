from typing import Dict

import torch
import torch.nn as nn


class PointwiseDualEncoder(nn.Module):
    """
    Local-window dual-view encoder.

    New route:
        X_t: [B, M, L]
        -> F1_t: [B, D1], F2_t: [B, D2]
        -> H1_t/H2_t: [B, d_model]

    A single length-L local window is one training/evaluation sample and only
    produces the representation of its current point t (the last column). There
    is intentionally no [B, T, d] point-level output here.

    View1 keeps the M x L matrix structure:
        - horizontal per-variable long kernel L: M values
        - horizontal per-variable short kernel floor(L/2): M * (L - floor(L/2) + 1) values
        - vertical full-variable summary per time column: L values

      D1 = M * (L - floor(L/2) + 2) + L
      For even L this is M * (L/2 + 2) + L.

    View2 flattens in time-major order:
        [x_{t-L+1,1}, ..., x_{t-L+1,M}, ..., x_{t,M}]
        - long kernel L*M: 1 value
        - short kernel floor(L*M/2): L*M - floor(L*M/2) + 1 values

      D2 = L*M - floor(L*M/2) + 2
      For even L*M this is L*M/2 + 2.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        history_len: int = 20,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.in_channels = max(1, int(in_channels))  # M
        self.d_model = int(d_model)
        self.history_len = max(1, int(history_len))  # L, now the full input window length.
        self.short_len = max(1, self.history_len // 2)
        self.flat_len = self.in_channels * self.history_len
        self.flat_short_len = max(1, self.flat_len // 2)

        self.view1_horizontal_long = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.history_len,
            groups=self.in_channels,
            bias=False,
        )
        self.view1_horizontal_short = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.short_len,
            groups=self.in_channels,
            bias=False,
        )
        self.view1_vertical = nn.Linear(self.in_channels, 1, bias=False)

        self.view2_long = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.flat_len,
            bias=False,
        )
        self.view2_short = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.flat_short_len,
            bias=False,
        )

        self.view1_dim = self.in_channels * (1 + (self.history_len - self.short_len + 1)) + self.history_len
        self.view2_dim = 1 + (self.flat_len - self.flat_short_len + 1)

        # The teacher's design treats F1/F2 as direct horizontal concatenations
        # of weighted/conv responses. Do not reshape them vertically and do not
        # turn this into a token sequence; Linear consumes [B, D] directly.
        del activation
        self.dropout = nn.Dropout(float(dropout))
        self.view1_projection = nn.Linear(self.view1_dim, self.d_model)
        self.view2_projection = nn.Linear(self.view2_dim, self.d_model)
        self._init_weights()

    def _init_weights(self):
        for module in (
            self.view1_horizontal_long,
            self.view1_horizontal_short,
            self.view2_long,
            self.view2_short,
        ):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.view1_vertical.weight)
        if self.view1_vertical.bias is not None:
            nn.init.zeros_(self.view1_vertical.bias)

    def _validate_input(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"PointwiseDualEncoder expects [B, M, L], got {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"PointwiseDualEncoder expected M={self.in_channels}, got M={int(x.size(1))}"
            )
        if x.size(2) != self.history_len:
            raise ValueError(
                "The new local-window encoder requires the dataloader window length to equal L. "
                f"Expected L={self.history_len}, got L={int(x.size(2))}. "
                "Set seq_len to the intended local history length."
            )

    def _view1_features(self, x: torch.Tensor) -> torch.Tensor:
        # Horizontal, per-variable along time.
        # Long: [B, M, L] -> [B, M, 1]
        h_long = self.view1_horizontal_long(x)
        # Short valid windows. Flip so the feature order follows right-to-left
        # sliding: most recent short segment first.
        h_short = torch.flip(self.view1_horizontal_short(x), dims=(-1,))
        horizontal = torch.cat([h_long, h_short], dim=-1).reshape(x.size(0), -1)

        # Vertical, per-time-column along variables: [B, L, M] -> [B, L]
        vertical = self.view1_vertical(x.transpose(1, 2).contiguous()).squeeze(-1)
        return torch.cat([horizontal, vertical], dim=-1)

    def _view2_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = int(x.size(0))
        # Time-major flatten: [B, M, L] -> [B, L, M] -> [B, 1, L*M]
        x_flat = x.transpose(1, 2).contiguous().reshape(batch_size, 1, self.flat_len)
        v2_long = self.view2_long(x_flat).reshape(batch_size, 1)
        v2_short = torch.flip(self.view2_short(x_flat), dims=(-1,)).reshape(batch_size, -1)
        f2 = torch.cat([v2_long, v2_short], dim=-1)
        return {"x_flat": x_flat.transpose(1, 2).contiguous(), "F2": f2}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._validate_input(x)
        f1 = self._view1_features(x)
        view2 = self._view2_features(x)
        f2 = view2["F2"]

        # The concatenated 1 x D feature vectors are already standard Linear
        # inputs: nn.Linear(D, d_model) acts on the last dimension of [B, D].
        f1_project = self.dropout(f1)
        f2_project = self.dropout(f2)
        h1 = self.view1_projection(f1_project)
        h2 = self.view2_projection(f2_project)

        return {
            "F1": f1,
            "H1": h1,
            "x_flat": view2["x_flat"],
            "F2": f2,
            "H2": h2,
        }
