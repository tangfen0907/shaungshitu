from typing import Union

import numpy as np
import torch
import torch.nn as nn

from model.axis_view_encoder import PointwiseDualEncoder
from model.prototype_head import DualViewPrototypeHeads, PrototypeHead
from model.reconstructor import Reconstructor
from model.svdd_head import MultiCenterSVDD
from model.tcn_encoder import TCNEncoder


class DualTCNEncoder(nn.Module):
    """Dual-view wrapper that keeps the separate-prototype route but uses TCNs.

    View1 is the ordinary channel-first local window [B, M, L].
    View2 follows the legacy flattened view [B, 1, L*M].
    """

    def __init__(
        self,
        in_channels: int,
        history_len: int,
        d_model: int,
        tcn_layers,
        tcn_kernel_size: int = 3,
        v2_first_kernel_size: int = 0,
        dropout: float = 0.1,
        activation: str = "relu",
        use_attentive_pooling: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.history_len = int(history_len)
        self.flat_len = int(in_channels) * int(history_len)
        layers = tuple(tcn_layers)
        self.encoder_v1 = TCNEncoder(
            in_channels=self.in_channels,
            latent_dim=d_model,
            tcn_layers=layers,
            kernel_size=tuple(int(tcn_kernel_size) for _ in layers),
            dropout=dropout,
            activation=activation,
            use_attentive_pooling=use_attentive_pooling,
            dilations=tuple(2 ** idx for idx, _ in enumerate(layers)),
        )
        first_kernel_size = int(v2_first_kernel_size) if int(v2_first_kernel_size) > 0 else int(tcn_kernel_size)
        self.encoder_v2 = TCNEncoder(
            in_channels=1,
            latent_dim=d_model,
            tcn_layers=layers,
            kernel_size=tuple(
                first_kernel_size if idx == 0 else int(tcn_kernel_size)
                for idx, _ in enumerate(layers)
            ),
            dropout=dropout,
            activation=activation,
            use_attentive_pooling=use_attentive_pooling,
            dilations=tuple(1 if idx == 0 else self.in_channels * (2 ** (idx - 1)) for idx, _ in enumerate(layers)),
        )

    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() != 3:
            raise ValueError(f"DualTCNEncoder expects [B, M, L], got {tuple(x.shape)}")
        if x.size(1) != self.in_channels or x.size(2) != self.history_len:
            raise ValueError(
                f"DualTCNEncoder expected [B,{self.in_channels},{self.history_len}], got {tuple(x.shape)}"
            )
        batch_size = int(x.size(0))
        x_flat = x.transpose(1, 2).contiguous().reshape(batch_size, 1, self.flat_len)
        h1 = self.encoder_v1(x)
        h2 = self.encoder_v2(x_flat)
        return {
            "F1": h1,
            "H1": h1,
            "x_flat": x_flat.transpose(1, 2).contiguous(),
            "F2": h2,
            "H2": h2,
        }


class AnomalyDetector(nn.Module):
    """Top-level anomaly detector.

    Dual-view mode now uses local-window View1/View2 encodings:

        X_t: [B, M, L] -> H1_t/H2_t: [B, d_model]

    A length-L window is one sample for the current point t. There is no
    point-level [B, T, d] sequence output in the dual-view route.
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        latent_dim: int,
        tcn_layers,
        gravity_tau: float = 1.0,
        dropout: float = 0.1,
        tcn_kernel_size: int = 3,
        v2_first_kernel_size: int = 0,
        tcn_activation: str = "relu",
        use_attentive_pooling: bool = False,
        active_view: str = "v1",
        dual_view_feature_mode: str = "avg",
        dual_encoder_type: str = "axis",
        state_dim: int = 0,
        num_prototypes: int = 1,
        proto_temperature: float = 0.2,
        stage2_method: str = "separate_proto",
    ):
        super().__init__()
        self.gravity_tau = gravity_tau
        self.raw_in_channels = int(in_channels)
        self.raw_seq_len = int(seq_len)
        self.active_view = self._normalize_active_view(active_view)
        self.dual_view_feature_mode = self._normalize_dual_feature_mode(dual_view_feature_mode)
        self.dual_encoder_type = self._normalize_dual_encoder_type(dual_encoder_type)
        self.is_dual_view = self.active_view == "dual"
        self.tcn_kernel_size = int(tcn_kernel_size)
        self.v2_first_kernel_size = int(v2_first_kernel_size)
        self.state_dim = int(state_dim) if int(state_dim) > 0 else int(latent_dim)
        if self.state_dim != int(latent_dim):
            raise ValueError(
                "Prototype state uses identity projection, so state_dim must equal latent_dim. "
                f"got state_dim={self.state_dim}, latent_dim={int(latent_dim)}."
            )
        self.num_prototypes = int(max(1, num_prototypes))
        self.stage2_method = str(stage2_method or "separate_proto").strip().lower()
        self.prototype_mode = "separate" if self.is_dual_view else "single"

        if self.is_dual_view:
            self.encoder_in_channels = self.raw_in_channels
            self.encoder_seq_len = self.raw_seq_len
            if self.dual_encoder_type == "tcn":
                self.dual_encoder = DualTCNEncoder(
                    in_channels=self.raw_in_channels,
                    history_len=self.raw_seq_len,
                    d_model=latent_dim,
                    tcn_layers=tcn_layers,
                    tcn_kernel_size=tcn_kernel_size,
                    v2_first_kernel_size=v2_first_kernel_size,
                    dropout=dropout,
                    activation=tcn_activation,
                    use_attentive_pooling=use_attentive_pooling,
                )
            else:
                self.dual_encoder = PointwiseDualEncoder(
                    in_channels=self.raw_in_channels,
                    d_model=latent_dim,
                    history_len=self.raw_seq_len,
                    dropout=dropout,
                    activation=tcn_activation,
                )
            self.encoder_v2_in_channels = self.raw_in_channels
            self.encoder_v2_seq_len = self.raw_seq_len
            self.reconstructor_v1 = Reconstructor(
                latent_dim=latent_dim,
                out_channels=self.raw_in_channels,
                output_len=1,
                hidden_dim=max(128, latent_dim * 2),
            )
            self.reconstructor_v2 = Reconstructor(
                latent_dim=latent_dim,
                out_channels=self.raw_in_channels,
                output_len=1,
                hidden_dim=max(128, latent_dim * 2),
            )
            self._sample_proto_shape_logged = False
        elif self.active_view == "v1":
            encoder_in_channels = self.raw_in_channels
            self.encoder_seq_len = self.raw_seq_len
            self._build_single_view_modules(
                encoder_in_channels=encoder_in_channels,
                latent_dim=latent_dim,
                tcn_layers=tcn_layers,
                tcn_kernel_size=tcn_kernel_size,
                v2_first_kernel_size=v2_first_kernel_size,
                dropout=dropout,
                tcn_activation=tcn_activation,
                use_attentive_pooling=use_attentive_pooling,
                view="v1",
            )
        elif self.active_view == "v2_flatten":
            encoder_in_channels, self.encoder_seq_len = self._view_encoder_shape(self.active_view)
            self._build_single_view_modules(
                encoder_in_channels=encoder_in_channels,
                latent_dim=latent_dim,
                tcn_layers=tcn_layers,
                tcn_kernel_size=tcn_kernel_size,
                v2_first_kernel_size=v2_first_kernel_size,
                dropout=dropout,
                tcn_activation=tcn_activation,
                use_attentive_pooling=use_attentive_pooling,
                view="v2_flatten",
            )
        else:
            raise ValueError(f"Unsupported active_view: {active_view}")
        self.svdd = MultiCenterSVDD(latent_dim=latent_dim)
        if self.is_dual_view:
            self.prototype_heads = DualViewPrototypeHeads(
                num_prototypes=self.num_prototypes,
                state_dim=self.state_dim,
                temperature=proto_temperature,
            )
            self.prototype_head_v1 = self.prototype_heads.prototype_head_v1
            self.prototype_head_v2 = self.prototype_heads.prototype_head_v2
            self.prototype_head_v1.prototypes.requires_grad_(False)
            self.prototype_head_v2.prototypes.requires_grad_(False)
        else:
            self.prototype_head = PrototypeHead(
                num_prototypes=self.num_prototypes,
                state_dim=self.state_dim,
                temperature=proto_temperature,
            )

    def _build_single_view_modules(
        self,
        encoder_in_channels: int,
        latent_dim: int,
        tcn_layers,
        tcn_kernel_size: int,
        v2_first_kernel_size: int,
        dropout: float,
        tcn_activation: str,
        use_attentive_pooling: bool,
        view: str,
    ):
        view = self._normalize_active_view(view)
        self.encoder_in_channels = int(encoder_in_channels)
        self.encoder_dilations = (
            self._v2_flatten_dilations(tcn_layers)
            if view == "v2_flatten"
            else self._default_tcn_dilations(tcn_layers)
        )
        self.encoder_kernel_sizes = (
            self._v2_flatten_kernel_sizes(
                tcn_layers=tcn_layers,
                base_kernel_size=tcn_kernel_size,
                first_kernel_size=v2_first_kernel_size,
            )
            if view == "v2_flatten"
            else tuple(int(tcn_kernel_size) for _ in tuple(tcn_layers))
        )
        self.encoder = TCNEncoder(
            in_channels=self.encoder_in_channels,
            latent_dim=latent_dim,
            tcn_layers=tcn_layers,
            kernel_size=self.encoder_kernel_sizes,
            dropout=dropout,
            activation=tcn_activation,
            use_attentive_pooling=use_attentive_pooling,
            dilations=self.encoder_dilations,
        )
        recon_out_channels = 1 if view == "v2_flatten" else self.raw_in_channels
        recon_output_len = self.raw_in_channels if view == "v2_flatten" else 1
        self.reconstructor = Reconstructor(
            latent_dim=latent_dim,
            out_channels=recon_out_channels,
            output_len=recon_output_len,
            hidden_dim=max(128, latent_dim * 2),
        )

    @staticmethod
    def _default_tcn_dilations(tcn_layers) -> tuple:
        return tuple(2 ** layer_idx for layer_idx, _ in enumerate(tuple(tcn_layers)))

    def _v2_flatten_dilations(self, tcn_layers) -> tuple:
        channel_period = max(1, int(self.raw_in_channels))
        # V2 is time-major flattened: every C scalar positions form one
        # original time step, so deeper dilations follow that channel period.
        return tuple(
            1 if layer_idx == 0 else channel_period * (2 ** (layer_idx - 1))
            for layer_idx, _ in enumerate(tuple(tcn_layers))
        )

    def _v2_flatten_kernel_sizes(
        self,
        tcn_layers,
        base_kernel_size: int,
        first_kernel_size: int,
    ) -> tuple:
        layers = tuple(tcn_layers)
        if not layers:
            return tuple()
        base_kernel_size = max(1, int(base_kernel_size))
        first_kernel_size = int(first_kernel_size)
        if first_kernel_size <= 0:
            first_kernel_size = base_kernel_size
        return tuple(
            first_kernel_size if layer_idx == 0 else base_kernel_size
            for layer_idx, _ in enumerate(layers)
        )

    @staticmethod
    def _normalize_active_view(active_view: str) -> str:
        key = str(active_view or "v1").strip().lower()
        aliases = {
            "view1": "v1",
            "raw": "v1",
            "time": "v1",
            "flatten": "v2_flatten",
            "v2-a": "v2_flatten",
            "v2a": "v2_flatten",
            "dual_flatten": "dual",
            "dual_v2_flatten": "dual",
        }
        return aliases.get(key, key)

    @staticmethod
    def _normalize_dual_feature_mode(mode: str) -> str:
        key = str(mode or "avg").strip().lower()
        aliases = {
            "mean": "avg",
            "average": "avg",
            "view1": "v1",
            "view2": "v2",
        }
        key = aliases.get(key, key)
        if key not in {"avg", "v1", "v2"}:
            raise ValueError("dual_view_feature_mode should be one of: avg, v1, v2.")
        return key

    @staticmethod
    def _normalize_dual_encoder_type(encoder_type: str) -> str:
        key = str(encoder_type or "axis").strip().lower()
        aliases = {
            "pointwise": "axis",
            "local_window": "axis",
            "local-window": "axis",
            "axis_view": "axis",
            "axis-view": "axis",
            "dual_tcn": "tcn",
            "tcns": "tcn",
        }
        key = aliases.get(key, key)
        if key not in {"axis", "tcn"}:
            raise ValueError("dual_encoder_type should be 'axis' or 'tcn'.")
        return key

    def _view_encoder_shape(self, view: str) -> tuple:
        view = self._normalize_active_view(view)
        if view == "v1":
            return self.raw_in_channels, self.raw_seq_len
        if view == "v2_flatten":
            return 1, self.raw_seq_len * self.raw_in_channels
        raise ValueError(f"Unsupported view: {view}")

    def _build_encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        return self._build_view_input(x, self.active_view)

    def _build_view_input(self, x: torch.Tensor, view: str) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"AnomalyDetector expects raw input [B, C, L], got {tuple(x.shape)}")
        if x.size(1) != self.raw_in_channels:
            raise ValueError(
                f"AnomalyDetector expected raw C={self.raw_in_channels}, got C={int(x.size(1))}"
            )

        view = self._normalize_active_view(view)
        if view == "v1":
            return x
        if view == "v2_flatten":
            batch_size = x.size(0)
            return x.transpose(1, 2).reshape(batch_size, 1, x.size(1) * x.size(2)).contiguous()
        raise ValueError(f"Unsupported view: {view}")

    def _combine_dual_features(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if self.dual_view_feature_mode == "v1":
            return z1
        if self.dual_view_feature_mode == "v2":
            return z2
        return 0.5 * (z1 + z2)

    def encode_views(self, x: torch.Tensor) -> tuple:
        if not self.is_dual_view:
            z = self.encode(x)
            return z, z
        features = self.dual_encoder(x)
        return features["H1"], features["H2"]

    def project_state(self, z: torch.Tensor, view: str = None) -> torch.Tensor:
        if not self.is_dual_view:
            return z
        view_key = str(view or "v1").strip().lower()
        if view_key in {"view1", "z1"}:
            view_key = "v1"
        if view_key in {"view2", "z2", "v2_flatten", "flatten"}:
            view_key = "v2"
        if view_key in {"v1", "v2"}:
            return z
        raise ValueError("view should be 'v1' or 'v2'.")

    def state_from_views(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple:
        if not self.is_dual_view:
            u = self.project_state(z1)
            return u, u
        return self.project_state(z1, view="v1"), self.project_state(z2, view="v2")

    @torch.no_grad()
    def init_prototypes_from_centers(self, centers: Union[np.ndarray, torch.Tensor]):
        if self.is_dual_view:
            raise RuntimeError("Use init_separate_prototypes_from_centers in dual-view mode.")
        self.prototype_head.init_from_centers(centers)

    @torch.no_grad()
    def init_separate_prototypes_from_centers(
        self,
        centers_v1: Union[np.ndarray, torch.Tensor],
        centers_v2: Union[np.ndarray, torch.Tensor],
    ):
        if not self.is_dual_view:
            raise RuntimeError("Separate dual-view prototypes require dual-view mode.")
        self.prototype_head_v1.init_from_centers(centers_v1)
        self.prototype_head_v2.init_from_centers(centers_v2)


    def encode(self, x: torch.Tensor, view: str = None) -> torch.Tensor:
        if self.is_dual_view:
            z1, z2 = self.encode_views(x)
            view_key = str(view or self.dual_view_feature_mode).strip().lower()
            if view_key in {"view1", "z1"}:
                view_key = "v1"
            if view_key in {"view2", "z2"}:
                view_key = "v2"
            if view_key == "v1":
                return z1
            if view_key == "v2":
                return z2
            return self._combine_dual_features(z1, z2)
        return self.encoder(self._build_encoder_input(x))

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        if self.is_dual_view:
            raise RuntimeError("Use reconstruct_view(z, view='v1' or 'v2') in dual-view mode.")
        return self.reconstructor(z)

    def reconstruct_view(self, z: torch.Tensor, view: str = None) -> torch.Tensor:
        if not self.is_dual_view:
            return self.reconstruct(z)
        view_key = str(view or "v1").strip().lower()
        if view_key in {"view1", "z1"}:
            view_key = "v1"
        if view_key in {"view2", "z2"}:
            view_key = "v2"
        if view_key == "v1":
            return self.reconstructor_v1(z)
        if view_key == "v2":
            return self.reconstructor_v2(z)
        raise ValueError("view should be 'v1' or 'v2'.")

    def forward(self, x: torch.Tensor, stage: str = "stage1", detach_prototypes: bool = False):
        """Return reconstruction and optional prototype outputs for the requested stage."""
        if self.is_dual_view:
            dual_features = self.dual_encoder(x)
            H1 = dual_features["H1"]
            H2 = dual_features["H2"]
            z1 = H1
            z2 = H2
            z = self._combine_dual_features(z1, z2)
            x_hat1 = self.reconstructor_v1(z1)
            x_hat2 = self.reconstructor_v2(z2)
            x_hat = 0.5 * (x_hat1 + x_hat2)
        else:
            z = self.encode(x)
            x_hat = self.reconstruct(z)

        outputs = {
            "z": z,
            "x_hat": x_hat,
        }
        if self.is_dual_view:
            outputs.update(
                {
                    "z1": z1,
                    "z2": z2,
                    "H1": H1,
                    "H2": H2,
                    "F1": dual_features["F1"],
                    "F2": dual_features["F2"],
                    "x_flat": dual_features["x_flat"],
                    "x_hat1": x_hat1,
                    "x_hat2": x_hat2,
                }
            )

        if stage in {"stage2", "test", "separate_proto"}:
            if self.is_dual_view:
                u1 = H1
                u2 = H2
                proto1 = self.prototype_head_v1(u1, detach_prototypes=detach_prototypes)
                proto2 = self.prototype_head_v2(u2, detach_prototypes=detach_prototypes)
                q1 = proto1["q"]
                q2 = proto2["q"]
                proto_conf1, proto_pred1 = torch.max(q1, dim=-1)
                proto_conf2, proto_pred2 = torch.max(q2, dim=-1)
                proto_dist1 = proto1["min_dist"]
                proto_dist2 = proto2["min_dist"]
                if not self._sample_proto_shape_logged:
                    print(
                        "[Stage2-SampleProto] forward shapes | "
                        f"H1={tuple(H1.shape)} | "
                        f"u1={tuple(u1.shape)} | "
                        f"q1={tuple(q1.shape)} | "
                        f"proto_dist1={tuple(proto_dist1.shape)}"
                    )
                    self._sample_proto_shape_logged = True
                outputs.update(
                    {
                        "u1": u1,
                        "u2": u2,
                        "q1": q1,
                        "q2": q2,
                        "proto_dist_matrix1": proto1["dist_sq"],
                        "proto_dist_matrix2": proto2["dist_sq"],
                        "proto_dist1": proto_dist1,
                        "proto_dist2": proto_dist2,
                        "proto_pred1": proto_pred1,
                        "proto_pred2": proto_pred2,
                        "proto_conf1": proto_conf1,
                        "proto_conf2": proto_conf2,
                    }
                )
            else:
                u = self.project_state(z)
                proto = self.prototype_head(u)
                outputs.update(
                    {
                        "u": u,
                        "q": proto["q"],
                        "proto_dist_matrix": proto["dist_sq"],
                        "proto_dist": proto["min_dist"],
                        "proto_pred": proto["pred"],
                        "proto_conf": proto["conf"],
                    }
                )

        if stage in {"stage2", "test"}:
            global_dist = torch.sum((z - self.svdd.global_c.unsqueeze(0)) ** 2, dim=1)
            gravity = self.svdd.compute_gravity(z, tau=self.gravity_tau)
            outputs["global_dist"] = global_dist
            outputs["gravity"] = gravity

        return outputs
