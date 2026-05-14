import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from data_factory.data_loader import get_loader_segment
from model.dmt_model import DMTPatchMemoryModel
from utils.dmt_memory_init import collect_h1_tokens, kmeans_init


class DMTSolver:
    """
    Independent DMT-M1 solver.

    It does not reuse the old Solver or old memory module. The four modes are:
        pretrain      -> train patch encoder + weak decoder
        init_memory   -> collect H1 tokens and run K-means
        memory_train  -> load fixed memory and train memory-guided reconstruction
        test          -> window-level score/evaluation
    """

    def __init__(self, args):
        self.args = args
        self.device = self._resolve_device(getattr(args, "device", "cuda:0"))
        self.dataset = str(args.dataset)
        self.save_dir = Path(args.model_save_path) / self.dataset
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.pretrain_ckpt = self.save_dir / "checkpoint_pretrain.pth"
        self.memory_path = self.save_dir / "memory_v1.pt"
        self.memory_ckpt = self.save_dir / "checkpoint_memory.pth"
        self._set_seed(int(args.seed))

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _resolve_device(self, requested: str) -> torch.device:
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            print(f"[DMT] CUDA requested ({requested}) but unavailable; using CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    def _loader(self, mode: str):
        return get_loader_segment(
            index=None,
            data_path=self.args.data_path,
            batch_size=int(self.args.batch_size),
            win_size=int(self.args.win_size),
            step=1,
            mode=mode,
            dataset=self.args.dataset,
            scaler_fit_mode=getattr(self.args, "scaler_fit_mode", "train"),
            cache_windows=bool(getattr(self.args, "cache_windows", False)),
            pin_memory=bool(getattr(self.args, "pin_memory", False)),
        )

    def _new_model(self, memory_init=None) -> DMTPatchMemoryModel:
        model = DMTPatchMemoryModel(
            in_channels=int(self.args.input_c),
            seq_len=int(self.args.win_size),
            patch_len=int(self.args.patch_len),
            d_model=int(self.args.d_model),
            n_heads=int(self.args.n_heads),
            num_layers=int(self.args.num_layers),
            n_memory=int(self.args.n_memory),
            temperature=float(self.args.temperature),
            topk_ratio=float(self.args.topk_ratio),
            memory_init=memory_init,
            memory_trainable=bool(getattr(self.args, "memory_trainable", False)),
            dropout=float(getattr(self.args, "dropout", 0.1)),
        )
        return model.to(self.device)

    def _batch_to_x(self, batch) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return x.float().to(self.device)

    def _batch_to_label(self, batch, bsz: int):
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            return torch.zeros(bsz, dtype=torch.float32)
        labels = batch[1].float()
        return labels.reshape(labels.size(0), -1).max(dim=1).values

    def _save_model(self, model: DMTPatchMemoryModel, path: Path) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": vars(self.args),
            },
            path,
        )

    def _load_checkpoint(self, path: Path) -> Dict[str, object]:
        if not path.exists():
            raise FileNotFoundError(f"DMT checkpoint not found: {path}")
        return torch.load(path, map_location=self.device)

    def _load_pretrain_weights(self, model: DMTPatchMemoryModel) -> None:
        ckpt = self._load_checkpoint(self.pretrain_ckpt)
        state = ckpt["model_state_dict"]
        model.load_state_dict(state, strict=True)

    def _load_pretrain_weights_keep_memory(self, model: DMTPatchMemoryModel) -> None:
        ckpt = self._load_checkpoint(self.pretrain_ckpt)
        state = {
            key: value
            for key, value in ckpt["model_state_dict"].items()
            if not key.startswith("memory_v1.memory")
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [key for key in missing if not key.startswith("memory_v1.memory")]
        if missing or unexpected:
            raise RuntimeError(f"pretrain load mismatch: missing={missing}, unexpected={unexpected}")

    def _load_memory_centers(self) -> torch.Tensor:
        if not self.memory_path.exists():
            raise FileNotFoundError(f"DMT memory file not found: {self.memory_path}")
        memory = torch.load(self.memory_path, map_location="cpu")
        if isinstance(memory, dict) and "memory" in memory:
            memory = memory["memory"]
        return memory.float()

    def _optimizer(self, model: DMTPatchMemoryModel):
        return torch.optim.Adam(model.parameters(), lr=float(self.args.lr))

    def _max_batches(self, name: str) -> int:
        return int(getattr(self.args, name, 0) or 0)

    def _should_stop(self, batch_idx: int, max_batches: int) -> bool:
        return max_batches > 0 and batch_idx + 1 >= max_batches

    def pretrain(self) -> None:
        train_loader = self._loader("train")
        model = self._new_model(memory_init=None)
        optimizer = self._optimizer(model)
        max_batches = self._max_batches("max_train_batches")

        for epoch in range(1, int(self.args.num_epochs) + 1):
            model.train()
            losses: List[float] = []
            for batch_idx, batch in enumerate(train_loader):
                x = self._batch_to_x(batch)  # [B, L, C]
                out = model(x, mode="pretrain")
                loss = F.mse_loss(out["x_hat"], x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

                if epoch == 1 and batch_idx == 0:
                    print(
                        "[DMT:pretrain] "
                        f"x={tuple(x.shape)} H1={tuple(out['H1'].shape)} "
                        f"x_hat={tuple(out['x_hat'].shape)} rec_loss={loss.item():.6f}"
                    )
                if self._should_stop(batch_idx, max_batches):
                    break
            print(f"[DMT:pretrain] epoch={epoch} rec_loss={np.mean(losses):.6f}")

        self._save_model(model, self.pretrain_ckpt)
        print(f"[DMT:pretrain] saved: {self.pretrain_ckpt}")

    def init_memory(self) -> None:
        train_loader = self._loader("train")
        model = self._new_model(memory_init=None)
        self._load_pretrain_weights(model)
        tokens = collect_h1_tokens(
            model,
            train_loader,
            self.device,
            max_tokens=int(self.args.max_memory_tokens),
        )
        centers = kmeans_init(tokens, int(self.args.n_memory), seed=int(self.args.seed))
        torch.save({"memory": centers, "config": vars(self.args)}, self.memory_path)
        print(f"[DMT:init_memory] collected_tokens={tuple(tokens.shape)}")
        print(f"[DMT:init_memory] memory_centers={tuple(centers.shape)}")
        print(f"[DMT:init_memory] saved: {self.memory_path}")

    def memory_train(self) -> None:
        memory = self._load_memory_centers()
        train_loader = self._loader("train")
        model = self._new_model(memory_init=memory)
        self._load_pretrain_weights_keep_memory(model)
        optimizer = self._optimizer(model)
        max_batches = self._max_batches("max_train_batches")

        for epoch in range(1, int(self.args.num_epochs) + 1):
            model.train()
            losses: List[float] = []
            rec_losses: List[float] = []
            ent_losses: List[float] = []
            for batch_idx, batch in enumerate(train_loader):
                x = self._batch_to_x(batch)  # [B, L, C]
                out = model(x, mode="memory_train")
                rec_loss = F.mse_loss(out["x_hat"], x)
                entropy_loss = -(out["attn"] * torch.log(out["attn"] + 1e-8)).mean()
                loss = rec_loss + float(self.args.lambda_ent) * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
                rec_losses.append(float(rec_loss.item()))
                ent_losses.append(float(entropy_loss.item()))

                if epoch == 1 and batch_idx == 0:
                    print(
                        "[DMT:memory_train] "
                        f"H1={tuple(out['H1'].shape)} attn={tuple(out['attn'].shape)} "
                        f"x_hat={tuple(out['x_hat'].shape)} rec_loss={rec_loss.item():.6f} "
                        f"entropy_loss={entropy_loss.item():.6f}"
                    )
                if self._should_stop(batch_idx, max_batches):
                    break
            print(
                "[DMT:memory_train] "
                f"epoch={epoch} loss={np.mean(losses):.6f} "
                f"rec_loss={np.mean(rec_losses):.6f} entropy_loss={np.mean(ent_losses):.6f}"
            )

        self._save_model(model, self.memory_ckpt)
        print(f"[DMT:memory_train] saved: {self.memory_ckpt}")

    @torch.no_grad()
    def _score_loader(self, model: DMTPatchMemoryModel, loader: Iterable) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        scores = []
        labels = []
        max_batches = self._max_batches("max_eval_batches")
        for batch_idx, batch in enumerate(loader):
            x = self._batch_to_x(batch)
            out = model(x, mode="test")
            scores.append(out["window_score"].detach().cpu().numpy())
            labels.append(self._batch_to_label(batch, x.size(0)).numpy())
            if batch_idx == 0:
                print(
                    "[DMT:test] "
                    f"LSD={tuple(out['LSD'].shape)} ISD={tuple(out['ISD'].shape)} "
                    f"window_score={tuple(out['window_score'].shape)}"
                )
            if self._should_stop(batch_idx, max_batches):
                break
        return np.concatenate(scores, axis=0), np.concatenate(labels, axis=0)

    def test(self) -> Dict[str, float]:
        memory = self._load_memory_centers()
        model = self._new_model(memory_init=memory)
        ckpt = self._load_checkpoint(self.memory_ckpt)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        train_loader = self._loader("train")
        val_loader = self._loader("val")
        test_loader = self._loader("test")

        train_scores, _ = self._score_loader(model, train_loader)
        val_scores, _ = self._score_loader(model, val_loader)
        combined_scores = np.concatenate([train_scores, val_scores], axis=0)
        threshold = np.percentile(combined_scores, 100.0 - float(self.args.anormly_ratio))

        test_scores, gt = self._score_loader(model, test_loader)
        pred = (test_scores > threshold).astype(np.int64)
        gt = gt.astype(np.int64)

        precision, recall, f1, _ = precision_recall_fscore_support(
            gt,
            pred,
            average="binary",
            zero_division=0,
        )
        accuracy = accuracy_score(gt, pred)
        auroc = self._safe_auc(roc_auc_score, gt, test_scores)
        auprc = self._safe_auc(average_precision_score, gt, test_scores)
        result = {
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auroc": float(auroc),
            "auprc": float(auprc),
        }
        print(
            "[DMT:test] "
            f"threshold={threshold:.6f} accuracy={accuracy:.4f} precision={precision:.4f} "
            f"recall={recall:.4f} f1={f1:.4f} auroc={auroc:.4f} auprc={auprc:.4f}"
        )
        return result

    def _safe_auc(self, fn, gt: np.ndarray, scores: np.ndarray) -> float:
        try:
            if len(np.unique(gt)) < 2:
                return float("nan")
            return float(fn(gt, scores))
        except ValueError:
            return float("nan")

    def run(self):
        mode = str(self.args.mode)
        if mode == "pretrain":
            return self.pretrain()
        if mode == "init_memory":
            return self.init_memory()
        if mode == "memory_train":
            return self.memory_train()
        if mode == "test":
            return self.test()
        raise ValueError(f"unknown DMT mode: {mode}")
