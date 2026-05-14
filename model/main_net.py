from typing import Union

import numpy as np
import torch
import torch.nn as nn

from model.axis_view_encoder import PatchRelationDualEncoder
from model.prototype_head import DualViewPrototypeHeads, PrototypeHead
from model.reconstructor import Reconstructor
from model.svdd_head import MultiCenterSVDD
from model.tcn_encoder import TCNEncoder


class AnomalyDetector(nn.Module):
    """Top-level anomaly detector.

    Dual-view mode uses independent View1/View2 prototype heads. The
    compatibility global z remains for the existing training/evaluation flow.
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
        state_dim: int = 0,
        num_prototypes: int = 1,
        proto_temperature: float = 0.2,
        stage2_method: str = "separate_proto",
        readout_mode: str = "attn_topk_max",
        topk_ratio: float = 0.1,
        topk_k: int = 0,
        patch_len: int = 16,
        patch_stride: int = 8,
    ):
        super().__init__()
        self.gravity_tau = gravity_tau
        self.raw_in_channels = int(in_channels)
        self.raw_seq_len = int(seq_len)
        self.active_view = self._normalize_active_view(active_view)
        self.dual_view_feature_mode = self._normalize_dual_feature_mode(dual_view_feature_mode)
        self.is_dual_view = self.active_view == "dual"
        self.tcn_kernel_size = int(tcn_kernel_size)
        self.v2_first_kernel_size = int(v2_first_kernel_size)
        self.readout_mode = str(readout_mode or "attn_topk_max").strip().lower()
        self.topk_ratio = float(topk_ratio)
        self.topk_k = int(topk_k)
        self.patch_len = max(1, int(patch_len))
        self.patch_stride = max(1, int(patch_stride))
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
            axis_kernels = (3, 5, 7)
            axis_num_blocks = 3
            self.encoder_v1_dilations = tuple(1 for _ in range(axis_num_blocks))
            self.encoder_v2_dilations = tuple(1 for _ in range(axis_num_blocks))
            self.encoder_v1_kernel_sizes = axis_kernels
            self.encoder_v2_kernel_sizes = axis_kernels
            self.dual_encoder = PatchRelationDualEncoder(
                latent_dim=latent_dim,
                tcn_layers=tcn_layers,
                patch_len=self.patch_len,
                patch_stride=self.patch_stride,
                patch_blocks=axis_num_blocks,
                relation_layers=1,
                relation_heads=4,
                kernels=axis_kernels,
                dropout=dropout,
                activation=tcn_activation,
                readout=self.readout_mode,
                topk_ratio=self.topk_ratio,
                topk_k=self.topk_k,
            )
            self.encoder_v2_in_channels = self.raw_in_channels
            self.encoder_v2_seq_len = self.raw_seq_len
            self.reconstructor_v1 = Reconstructor(
                latent_dim=latent_dim,
                out_channels=self.raw_in_channels,
                hidden_dim=max(128, latent_dim * 2),
            )
            self.reconstructor_v2 = Reconstructor(
                latent_dim=latent_dim,
                out_channels=self.raw_in_channels,
                output_len=1,
                hidden_dim=max(128, latent_dim * 2),
            )
            token_dim = int(tuple(tcn_layers)[0])
            self.token_projector_v1 = (
                nn.Identity()
                if token_dim == self.state_dim
                else nn.Linear(token_dim, self.state_dim)
            )
            self.token_projector_v2 = (
                nn.Identity()
                if token_dim == self.state_dim
                else nn.Linear(token_dim, self.state_dim)
            )
            self._token_proto_shape_logged = False
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

    def _v2_last_to_raw_layout(self, x_hat_v2: torch.Tensor) -> torch.Tensor:
        if (
            x_hat_v2.dim() == 3
            and x_hat_v2.size(1) == self.raw_in_channels
            and x_hat_v2.size(2) == 1
        ):
            return x_hat_v2
        # TODO: remove this legacy [B, 1, C] branch after old v2_flatten
        # configs/checkpoints are no longer needed.
        if x_hat_v2.dim() == 3 and x_hat_v2.size(1) == 1 and x_hat_v2.size(2) == self.raw_in_channels:
            return x_hat_v2.transpose(1, 2).contiguous()
        raise ValueError(
            "V2 reconstruction should be raw [B, C, 1] or legacy [B, 1, C], "
            f"got {tuple(x_hat_v2.shape)}."
        )

    def _legacy_v2_to_raw_layout(self, x_hat_v2: torch.Tensor) -> torch.Tensor:
        if x_hat_v2.dim() != 3 or x_hat_v2.size(1) != 1 or x_hat_v2.size(2) != self.raw_in_channels:
            raise ValueError(
                "V2 reconstruction should be [B, 1, C] before converting to raw layout, "
                f"got {tuple(x_hat_v2.shape)}."
            )
        return x_hat_v2.transpose(1, 2).contiguous()

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
        return features["z1_global"], features["z2_global"]

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

    def _token_topk_mean(self, token_dist: torch.Tensor) -> torch.Tensor:
        if token_dist.dim() < 2:
            return token_dist.reshape(token_dist.size(0), -1).mean(dim=1)
        flat = token_dist.reshape(token_dist.size(0), -1)
        token_count = int(flat.size(1))
        if token_count <= 0:
            return flat.mean(dim=1)
        if self.topk_k > 0:
            k = max(1, min(int(self.topk_k), token_count))
        else:
            ratio = self.topk_ratio if self.topk_ratio > 0 else 1.0
            k = max(1, min(token_count, int(np.ceil(float(token_count) * float(ratio)))))
        if k >= token_count:
            return flat.mean(dim=1)
        return torch.topk(flat, k=k, dim=1, largest=True, sorted=False).values.mean(dim=1)

    def forward(self, x: torch.Tensor, stage: str = "stage1", detach_prototypes: bool = False):
        """Return reconstruction and optional prototype outputs for the requested stage."""
        if self.is_dual_view:
            dual_features = self.dual_encoder(x)
            z1 = dual_features["z1_global"]
            z2 = dual_features["z2_global"]
            z = self._combine_dual_features(z1, z2)
            x_hat1 = self.reconstructor_v1(z1)
            x_hat2 = self.reconstructor_v2(z2)
            # New axis-view V2 reconstructs the final multivariate state in
            # the same raw layout as V1: [B, C, 1].
            x_hat2_raw = self._v2_last_to_raw_layout(x_hat2)
            x_hat = 0.5 * (x_hat1 + x_hat2_raw)
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
                    "x_hat1": x_hat1,
                    "x_hat2": x_hat2,
                    "x_hat2_raw": x_hat2_raw,
                    "h1_patch": dual_features["h1_patch"],
                    "h2_var": dual_features["h2_var"],
                    "h2_patch": dual_features["h2_patch"],
                }
            )

        if stage in {"stage2", "test", "separate_proto"}:
            if self.is_dual_view:
                h1_patch = dual_features["h1_patch"]
                h2_patch = dual_features["h2_patch"]
                u1_token = self.token_projector_v1(h1_patch)
                u2_token = self.token_projector_v2(h2_patch)
                proto1 = self.prototype_head_v1(u1_token, detach_prototypes=detach_prototypes)
                proto2 = self.prototype_head_v2(u2_token, detach_prototypes=detach_prototypes)
                q1 = proto1["q"].mean(dim=(1, 2))
                q2 = proto2["q"].mean(dim=(1, 2))
                proto_conf1, proto_pred1 = torch.max(q1, dim=-1)
                proto_conf2, proto_pred2 = torch.max(q2, dim=-1)
                proto_dist1 = self._token_topk_mean(proto1["min_dist"])
                proto_dist2 = self._token_topk_mean(proto2["min_dist"])
                u1 = u1_token.mean(dim=(1, 2))
                u2 = u2_token.mean(dim=(1, 2))
                if not self._token_proto_shape_logged:
                    print(
                        "[Stage2-TokenProto] forward shapes | "
                        f"h1_patch={tuple(h1_patch.shape)} | "
                        f"u1_token={tuple(u1_token.shape)} | "
                        f"q1_token={tuple(proto1['q'].shape)} | "
                        f"proto_dist1_token={tuple(proto1['min_dist'].shape)} | "
                        f"q1={tuple(q1.shape)} | "
                        f"proto_dist1={tuple(proto_dist1.shape)}"
                    )
                    self._token_proto_shape_logged = True
                outputs.update(
                    {
                        "u1": u1,
                        "u2": u2,
                        "u1_token": u1_token,
                        "u2_token": u2_token,
                        "q1": q1,
                        "q2": q2,
                        "q1_token": proto1["q"],
                        "q2_token": proto2["q"],
                        "proto_dist_matrix1": proto1["dist_sq"].mean(dim=(1, 2)),
                        "proto_dist_matrix2": proto2["dist_sq"].mean(dim=(1, 2)),
                        "proto_dist_matrix1_token": proto1["dist_sq"],
                        "proto_dist_matrix2_token": proto2["dist_sq"],
                        "proto_dist1": proto_dist1,
                        "proto_dist2": proto_dist2,
                        "proto_dist1_token": proto1["min_dist"],
                        "proto_dist2_token": proto2["min_dist"],
                        "proto_pred1": proto_pred1,
                        "proto_pred2": proto_pred2,
                        "proto_pred1_token": proto1["pred"],
                        "proto_pred2_token": proto2["pred"],
                        "proto_conf1": proto_conf1,
                        "proto_conf2": proto_conf2,
                        "proto_conf1_token": proto1["conf"],
                        "proto_conf2_token": proto2["conf"],
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
