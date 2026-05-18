import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_factory.data_loader import get_loader_segment
from data_factory.triplet_dataset import _extract_window
from model.main_net import AnomalyDetector
from trainer import active_pool as active_pool_methods
from trainer import evaluator as evaluator_methods
from trainer import stage0 as stage0_methods
from trainer import stage1 as stage1_methods
from trainer import stage2 as stage2_methods
from trainer import train_loop as train_loop_methods
from trainer.evaluator import run_evaluation
from trainer.train_loop import run_train_loop
from utils import checkpoint as checkpoint_methods
from utils import visualization as visualization_methods
from utils.anomaly_injection import TimeSeriesAnomalyInjector
from utils.config import Config



def _batch_to_channel_first_tensor(
    batch,
    device,
    non_blocking: bool = False,
    in_channels: int = None,
    seq_len: int = None,
) -> torch.Tensor:
    batch = _extract_window(batch)
    if isinstance(batch, np.ndarray):
        tensor = torch.from_numpy(batch).float()
    elif isinstance(batch, torch.Tensor):
        tensor = batch.float()
    else:
        tensor = torch.tensor(batch, dtype=torch.float32)

    if tensor.dim() != 3:
        raise ValueError(f"Batch tensor should be 3D, got {tuple(tensor.shape)}")

    if in_channels is not None and seq_len is not None:
        in_channels = int(in_channels)
        seq_len = int(seq_len)
        if tensor.shape[1] == seq_len and tensor.shape[2] == in_channels:
            tensor = tensor.transpose(1, 2).contiguous()
        elif tensor.shape[1] == in_channels and tensor.shape[2] == seq_len:
            tensor = tensor.contiguous()
        elif tensor.shape[1] > tensor.shape[2]:
            tensor = tensor.transpose(1, 2).contiguous()
    elif tensor.shape[1] > tensor.shape[2]:
        tensor = tensor.transpose(1, 2).contiguous()

    return tensor.to(device, non_blocking=bool(non_blocking))


def extract_all_features(model, dataloader, device, view: str = None) -> np.ndarray:
    was_training = model.training
    model.eval()

    features = []
    non_blocking = getattr(device, "type", "") == "cuda"
    with torch.inference_mode():
        for batch in dataloader:
            x = _batch_to_channel_first_tensor(
                batch,
                device,
                non_blocking=non_blocking,
                in_channels=getattr(model, "raw_in_channels", None),
                seq_len=getattr(model, "raw_seq_len", None),
            )
            z = model.encode(x, view=view)
            features.append(z.detach().cpu().numpy())

    if was_training:
        model.train()

    if not features:
        raise RuntimeError("Feature extraction failed because the dataloader yielded no windows.")

    return np.concatenate(features, axis=0)




class Solver:
    """Three-stage anomaly detection pipeline with fixed-skeleton Stage 2."""

    def __init__(self, config):
        self.config = config if isinstance(config, Config) else Config.from_dict(config)
        self.device = self._build_device()
        self._configure_torch_backends()
        self._set_random_seed(self.config.seed)
        self._warned_windows_num_workers = False

        self.train_loader, self.test_loader = self._build_base_dataloaders()
        self.full_train_dataset = self.train_loader.dataset
        self.train_active_mask = np.ones(len(self.full_train_dataset), dtype=bool)
        self.active_pool_history: List[Dict[str, object]] = []
        self.train_eval_loader = DataLoader(
            dataset=self.train_loader.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self._effective_num_workers(),
            drop_last=False,
            pin_memory=self._pin_memory(),
            generator=self._make_loader_generator(103),
        )

        self._align_config_with_data()
        self.model = AnomalyDetector(
            in_channels=self.config.in_channels,
            seq_len=self.config.seq_len,
            latent_dim=self.config.latent_dim,
            tcn_layers=self.config.tcn_layers,
            dropout=self.config.tcn_dropout,
            tcn_kernel_size=self.config.tcn_kernel_size,
            v2_first_kernel_size=getattr(self.config, "v2_first_kernel_size", 0),
            tcn_activation=self.config.tcn_activation,
            use_attentive_pooling=self.config.use_attentive_pooling,
            active_view=getattr(self.config, "active_view", "v1"),
            dual_view_feature_mode=getattr(self.config, "dual_view_feature_mode", "avg"),
            state_dim=int(getattr(self.config, "state_dim", 0)),
            num_prototypes=int(
                getattr(self.config, "num_prototypes", 0)
                if int(getattr(self.config, "num_prototypes", 0)) > 0
                else 1
            ),
            proto_temperature=float(getattr(self.config, "proto_temperature", 0.2)),
            stage2_method=str(getattr(self.config, "stage2_method", "separate_proto")),
        ).to(self.device)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.recon_loss_fn = nn.MSELoss(reduction="mean")

        self.injector = TimeSeriesAnomalyInjector()
        self.cluster_labels = None
        self.core_mask = None
        self.cluster_centers = None
        self.cluster_radii = None
        self.nearest_other_clusters = None
        self.global_center = None
        self.score_core = None
        self.stage2_structure = None
        self.stage2_refresh_round = -1
        self.bank_summary: List[Dict[str, object]] = []
        self.cluster_bank = None
        self.current_stage2_view = "v1"
        os.makedirs(self.config.save_dir, exist_ok=True)

    def _build_device(self) -> torch.device:
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _configure_torch_backends(self):
        if self.device.type == "cuda":
            if bool(getattr(self.config, "enable_tf32", True)):
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass
            try:
                torch.backends.cudnn.benchmark = bool(getattr(self.config, "cudnn_benchmark", True))
            except Exception:
                pass

    def _pin_memory(self) -> bool:
        return self.device.type == "cuda" and bool(getattr(self.config, "pin_memory", True))

    def _make_loader_generator(self, seed_offset: int = 0) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(int(self.config.seed) + int(seed_offset))
        return generator

    @staticmethod
    def _set_random_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _effective_num_workers(self) -> int:
        requested = max(0, int(getattr(self.config, "num_workers", 0)))
        if os.name == "nt" and requested > 0:
            if not self._warned_windows_num_workers:
                print(
                    "[DataLoader] Windows detected; using num_workers=0 for Solver "
                    "loaders to avoid shared-memory file mapping failures."
                )
                self._warned_windows_num_workers = True
            return 0
        return requested

    def _normalize_cluster_bank(self, bank: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if bank is None:
            return None

        normalized = dict(bank)
        required = [
            "cluster_centers",
            "cluster_labels",
            "core_mask",
            "cluster_radii",
            "global_center",
            "nearest_other_cluster",
            "score_core",
        ]
        missing = [key for key in required if key not in normalized]
        if missing:
            raise ValueError(f"Stage 2 structure is missing fields: {missing}")

        centers = np.asarray(normalized["cluster_centers"], dtype=np.float32)
        labels = np.asarray(normalized["cluster_labels"], dtype=np.int64).reshape(-1)
        if centers.ndim != 2 or centers.shape[0] == 0:
            raise ValueError("cluster_centers should be [K, D] with K > 0.")

        normalized["cluster_centers"] = centers
        normalized["cluster_labels"] = labels
        normalized["core_mask"] = np.asarray(normalized["core_mask"], dtype=bool).reshape(-1)
        normalized["cluster_radii"] = np.asarray(normalized["cluster_radii"], dtype=np.float32).reshape(-1)
        normalized["global_center"] = np.asarray(normalized["global_center"], dtype=np.float32).reshape(-1)
        normalized["nearest_other_cluster"] = np.asarray(normalized["nearest_other_cluster"], dtype=np.int64).reshape(-1)
        normalized["score_core"] = np.asarray(normalized["score_core"], dtype=np.float32).reshape(-1)
        normalized.setdefault("bank_mode", "separate_proto")
        normalized.setdefault("refresh_round", int(max(self.stage2_refresh_round, 0)))
        normalized.setdefault("bank_summary", [])
        return normalized

    def _apply_cluster_bank(self, bank: Dict[str, object]):
        bank = self._normalize_cluster_bank(bank)
        if bank is None:
            raise RuntimeError("Stage 2 structure is not available.")

        self.cluster_centers = np.asarray(bank["cluster_centers"], dtype=np.float32)
        self.cluster_labels = np.asarray(bank["cluster_labels"], dtype=np.int64)
        self.core_mask = np.asarray(bank["core_mask"], dtype=bool)
        self.cluster_radii = np.asarray(bank["cluster_radii"], dtype=np.float32)
        self.nearest_other_clusters = np.asarray(bank["nearest_other_cluster"], dtype=np.int64)
        self.global_center = np.asarray(bank["global_center"], dtype=np.float32)
        self.score_core = np.asarray(bank["score_core"], dtype=np.float32)
        self.stage2_refresh_round = int(bank.get("refresh_round", 0))
        self.bank_summary = list(bank.get("bank_summary", []))
        self.cluster_bank = bank
        self.stage2_structure = bank
        self.model.svdd.update_centers(self.cluster_centers)

    def _resolve_data_path(self) -> str:
        if os.path.exists(self.config.data_path):
            return self.config.data_path

        fallback = os.path.join("dataset", self.config.dataset)
        if os.path.exists(fallback):
            return fallback

        raise FileNotFoundError(
            f"Could not find data path: {self.config.data_path}; fallback also missing: {fallback}"
        )

    def _build_base_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        data_path = self._resolve_data_path()
        train_split_mode = str(getattr(self.config, "train_split_mode", "train")).strip() or "train"
        test_split_mode = str(getattr(self.config, "test_split_mode", "test")).strip() or "test"
        loader_kwargs = {
            "entity_id": getattr(self.config, "entity_id", ""),
            "spacecraft": getattr(self.config, "spacecraft", ""),
            "metadata_path": getattr(self.config, "metadata_path", ""),
            "scaler_fit_mode": getattr(self.config, "scaler_fit_mode", "train"),
            "left_pad_windows": bool(
                getattr(
                    self.config,
                    "left_pad_windows",
                    str(getattr(self.config, "active_view", "")).strip().lower() == "dual",
                )
            ),
        }
        base_step = max(1, int(getattr(self.config, "step", 1)))
        train_step = int(getattr(self.config, "train_step", -1))
        test_step = int(getattr(self.config, "test_step", -1))
        train_step = train_step if train_step > 0 else base_step
        test_step = test_step if test_step > 0 else base_step

        train_loader = get_loader_segment(
            index=0,
            data_path=data_path,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=train_step,
            mode=train_split_mode,
            dataset=self.config.dataset,
            cache_windows=bool(getattr(self.config, "cache_windows", False)),
            pin_memory=self._pin_memory(),
            generator=self._make_loader_generator(101),
            **loader_kwargs,
        )
        test_loader = get_loader_segment(
            index=0,
            data_path=data_path,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=test_step,
            mode=test_split_mode,
            dataset=self.config.dataset,
            cache_windows=bool(getattr(self.config, "cache_windows", False)),
            pin_memory=self._pin_memory(),
            generator=self._make_loader_generator(102),
            **loader_kwargs,
        )
        if train_split_mode.lower() != "train":
            print(
                "[Data] Training split override | "
                f"train_split_mode={train_split_mode} | "
                f"scaler_fit_mode={getattr(self.config, 'scaler_fit_mode', 'train')} | "
                "labels, if present, are ignored by the training losses"
            )
        return train_loader, test_loader

    def _align_config_with_data(self):
        sample = _extract_window(self.train_loader.dataset[0])
        if isinstance(sample, np.ndarray):
            window = torch.from_numpy(sample).float()
        elif isinstance(sample, torch.Tensor):
            window = sample.detach().clone().float()
        else:
            window = torch.tensor(sample, dtype=torch.float32)
        if window.dim() != 2:
            raise ValueError(f"Window tensor should be 2D, got {tuple(window.shape)}")
        # Repo loaders return windows as [L, M]. This explicit interpretation is
        # important now that L=20 can be smaller than M (e.g. PSM/PUMP/SMD), so
        # the old shape[0] > shape[1] heuristic would flip the axes wrongly.
        self.config.seq_len = int(window.shape[0])
        self.config.in_channels = int(window.shape[1])
        if int(getattr(self.config, "state_dim", 0)) <= 0:
            self.config.state_dim = int(getattr(self.config, "latent_dim", 0))

    def _prepare_batch(self, batch) -> torch.Tensor:
        x = _extract_window(batch)
        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            tensor = x.float()
        else:
            tensor = torch.tensor(x, dtype=torch.float32)

        if tensor.dim() != 3:
            raise ValueError(f"Batch tensor should be 3D, got {tuple(tensor.shape)}")

        in_channels = int(getattr(self.config, "in_channels", 0))
        seq_len = int(getattr(self.config, "seq_len", 0))
        if tensor.shape[1] == seq_len and tensor.shape[2] == in_channels:
            tensor = tensor.transpose(1, 2).contiguous()
        elif tensor.shape[1] == in_channels and tensor.shape[2] == seq_len:
            tensor = tensor.contiguous()
        elif tensor.shape[1] > tensor.shape[2]:
            tensor = tensor.transpose(1, 2).contiguous()

        return tensor.to(self.device, non_blocking=self._pin_memory())

    @staticmethod
    def _last_timestep(x: torch.Tensor) -> torch.Tensor:
        return x[:, :, -1:]

    @staticmethod
    def _v2_last_timestep(x: torch.Tensor) -> torch.Tensor:
        return x[:, :, -1].unsqueeze(1)

    def _uses_legacy_v2_reconstruction(self, view: Optional[str] = None) -> bool:
        if self._normalize_reconstruction_view(view) != "v2":
            return False
        active_view = str(getattr(self.model, "active_view", getattr(self.config, "active_view", "v1"))).lower()
        # TODO: delete this compatibility path after legacy v2_flatten runs are retired.
        return (not self._is_dual_view_model()) and active_view == "v2_flatten"

    def _normalize_reconstruction_view(self, view: Optional[str] = None) -> str:
        key = str(view or "").strip().lower()
        if key in {"avg", "mean", "dual"}:
            return "v1"
        if key in {"v2", "view2", "z2", "flatten", "v2_flatten"}:
            return "v2"
        if key in {"v1", "view1", "z1", "raw", "time"}:
            return "v1"

        active_view = str(getattr(self.model, "active_view", getattr(self.config, "active_view", "v1"))).lower()
        return "v2" if active_view == "v2_flatten" else "v1"

    def _target_last_for_view(self, target: torch.Tensor, view: Optional[str] = None) -> torch.Tensor:
        if self._uses_legacy_v2_reconstruction(view):
            return self._v2_last_timestep(target)
        return self._last_timestep(target)

    def _prediction_last_for_view(self, x_hat: torch.Tensor, view: Optional[str] = None) -> torch.Tensor:
        if self._uses_legacy_v2_reconstruction(view):
            if x_hat.dim() == 3 and x_hat.size(1) == 1:
                return x_hat
            return self._v2_last_timestep(x_hat)
        return self._last_timestep(x_hat)

    def _uses_full_window_reconstruction(
        self,
        x_hat: torch.Tensor,
        target: torch.Tensor,
        view: Optional[str] = None,
    ) -> bool:
        if self._uses_legacy_v2_reconstruction(view):
            return False
        return tuple(x_hat.shape) == tuple(target.shape)

    def _target_for_reconstruction(
        self,
        x_hat: torch.Tensor,
        target: torch.Tensor,
        view: Optional[str] = None,
    ) -> torch.Tensor:
        if self._uses_full_window_reconstruction(x_hat, target, view):
            return target
        return self._target_last_for_view(target, view)

    def _prediction_for_reconstruction(
        self,
        x_hat: torch.Tensor,
        target: torch.Tensor,
        view: Optional[str] = None,
    ) -> torch.Tensor:
        if self._uses_full_window_reconstruction(x_hat, target, view):
            return x_hat
        return self._prediction_last_for_view(x_hat, view)

    def _reconstruction_loss(
        self,
        x_hat: torch.Tensor,
        target: torch.Tensor,
        view: Optional[str] = None,
    ) -> torch.Tensor:
        return self.recon_loss_fn(
            self._prediction_for_reconstruction(x_hat, target, view),
            self._target_for_reconstruction(x_hat, target, view),
        )

    def _mse_per_sample(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        view: Optional[str] = None,
    ) -> torch.Tensor:
        error = (
            self._target_for_reconstruction(x_hat, x, view)
            - self._prediction_for_reconstruction(x_hat, x, view)
        ) ** 2
        return torch.mean(error, dim=(1, 2))

    def _mse_per_sample_from_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if "x_hat1" in outputs and "x_hat2" in outputs:
            return torch.stack(
                [
                    self._mse_per_sample(target, outputs["x_hat1"], view="v1"),
                    self._mse_per_sample(target, outputs["x_hat2"], view="v2"),
                ],
                dim=0,
            ).mean(dim=0)
        return self._mse_per_sample(
            target,
            outputs["x_hat"],
            view=self._normalize_reconstruction_view(),
        )

    def _stage1_reconstruction_loss(
        self,
        x_hat: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        view: Optional[str] = None,
    ) -> torch.Tensor:
        return self._reconstruction_loss(x_hat, target, view=view)

    def _reconstruction_loss_from_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if "x_hat1" in outputs and "x_hat2" in outputs:
            return torch.stack(
                [
                    self._stage1_reconstruction_loss(outputs["x_hat1"], target, mask, view="v1"),
                    self._stage1_reconstruction_loss(outputs["x_hat2"], target, mask, view="v2"),
                ]
            ).mean()
        return self._stage1_reconstruction_loss(
            outputs["x_hat"],
            target,
            mask,
            view=self._normalize_reconstruction_view(),
        )

    def _dual_reconstruction_losses_from_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "x_hat1" not in outputs or "x_hat2" not in outputs:
            loss = self._stage1_reconstruction_loss(
                outputs["x_hat"],
                target,
                mask,
                view=self._normalize_reconstruction_view(),
            )
            return loss, loss
        return (
            self._stage1_reconstruction_loss(outputs["x_hat1"], target, mask, view="v1"),
            self._stage1_reconstruction_loss(outputs["x_hat2"], target, mask, view="v2"),
        )

    def _cross_view_consistency_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if "z1" not in outputs or "z2" not in outputs:
            reference = outputs.get("z")
            if reference is None:
                return torch.zeros((), device=self.device)
            return torch.zeros((), device=reference.device, dtype=reference.dtype)
        return torch.mean((outputs["z1"] - outputs["z2"]) ** 2)

    def _stage1_triplet_embedding_loss(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        z_negative: torch.Tensor,
    ) -> torch.Tensor:
        d_ap = torch.sqrt(torch.sum((z_anchor - z_positive) ** 2, dim=1) + 1e-12)
        d_an = torch.sqrt(torch.sum((z_anchor - z_negative) ** 2, dim=1) + 1e-12)
        ap_margin = float(getattr(self.config, "stage1_ap_margin", 0.1))
        margin = float(getattr(self.config, "stage1_triplet_margin", 0.3))
        return (torch.relu(d_ap - ap_margin) ** 2 + torch.relu(margin - d_an) ** 2).mean()

    def _is_dual_view_model(self) -> bool:
        return bool(getattr(self.model, "is_dual_view", False))

    def _collect_feature_matrix(self, loader: DataLoader, view: str = None) -> np.ndarray:
        return extract_all_features(self.model, loader, self.device, view=view)

    def train_pipeline(self):
        return run_train_loop(self)

    def test(self):
        return run_evaluation(self)


def _attach_solver_methods():
    modules = (
        active_pool_methods,
        train_loop_methods,
        stage0_methods,
        stage1_methods,
        stage2_methods,
        evaluator_methods,
        visualization_methods,
        checkpoint_methods,
    )
    skip = {
        "run_train_loop",
        "run_evaluation",
        "run_stage0_epoch",
        "run_stage1_epoch",
        "run_stage2_ab_refinement",
    }
    for module in modules:
        for name in getattr(module, "__all__", ()): 
            if name in skip:
                continue
            setattr(Solver, name, getattr(module, name))


_attach_solver_methods()
