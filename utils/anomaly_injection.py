import random
from typing import Union

import numpy as np
import torch


class TimeSeriesAnomalyInjector:
    """
    Stage 1 pseudo-anomaly injector for a single window shaped as [C, L].

    The original version only used simple spike/shift/scale perturbations.
    This version keeps those modes and adds richer subsequence-level corruptions
    inspired by DACAD-style anomaly synthesis so the encoder sees more diverse
    temporal structure violations during Stage 1.
    """

    def __init__(
        self,
        p_spike: float = 0.15,
        p_shift: float = 0.15,
        p_scale: float = 0.15,
        p_trend: float = 0.2,
        p_contextual: float = 0.2,
        p_shapelet: float = 0.15,
        p_relational: float = 0.0,
    ):
        total = p_spike + p_shift + p_scale + p_trend + p_contextual + p_shapelet + p_relational
        if total <= 0:
            raise ValueError("Sum of anomaly injection probabilities must be positive.")

        self.p_spike = p_spike / total
        self.p_shift = p_shift / total
        self.p_scale = p_scale / total
        self.p_trend = p_trend / total
        self.p_contextual = p_contextual / total
        self.p_shapelet = p_shapelet / total
        self.p_relational = p_relational / total

    def __call__(
        self,
        x_window: Union[np.ndarray, torch.Tensor],
        return_mask: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x_window, torch.Tensor):
            return self._inject_tensor(x_window, return_mask=return_mask)
        if isinstance(x_window, np.ndarray):
            return self._inject_numpy(x_window, return_mask=return_mask)
        raise TypeError("x_window must be a numpy.ndarray or torch.Tensor.")

    def _choose_mode(self) -> str:
        modes = ["spike", "shift", "scale", "trend", "contextual", "shapelet", "relational"]
        probs = [
            self.p_spike,
            self.p_shift,
            self.p_scale,
            self.p_trend,
            self.p_contextual,
            self.p_shapelet,
            self.p_relational,
        ]
        return random.choices(modes, weights=probs, k=1)[0]

    @staticmethod
    def _sample_segment(length: int, min_ratio: float, max_ratio: float):
        if length <= 1:
            return 0, length

        min_len = max(2, int(round(length * min_ratio)))
        max_len = max(min_len, int(round(length * max_ratio)))
        max_len = min(max_len, length)
        seg_len = random.randint(min_len, max_len)
        start = random.randint(0, max(0, length - seg_len))
        return start, start + seg_len

    @staticmethod
    def _sample_dims(num_channels: int, full_prob: float = 0.3):
        if num_channels <= 1 or random.random() < full_prob:
            return list(range(num_channels))

        max_dims = max(1, num_channels // 3)
        num_dims = random.randint(1, max_dims)
        return random.sample(range(num_channels), num_dims)

    def _inject_tensor(self, x_window: torch.Tensor, return_mask: bool = False) -> torch.Tensor:
        if x_window.dim() == 3:
            injected = []
            masks = []
            for sample in x_window:
                sample_injected, sample_mask = self._inject_tensor(sample, return_mask=True)
                injected.append(sample_injected)
                masks.append(sample_mask)
            x_out = torch.stack(injected, dim=0)
            mask_out = torch.stack(masks, dim=0)
            return (x_out, mask_out) if return_mask else x_out

        x = x_window.detach().clone()
        if x.dim() != 2:
            raise ValueError(f"Expected input shape [C, L] or [B, C, L], got {tuple(x.shape)}")

        mask = torch.zeros_like(x, dtype=torch.bool)
        mode = self._choose_mode()
        if mode == "spike":
            return self._add_spike_tensor(x, mask, return_mask=return_mask)
        if mode == "shift":
            return self._add_shift_tensor(x, mask, return_mask=return_mask)
        if mode == "scale":
            return self._add_scale_tensor(x, mask, return_mask=return_mask)
        if mode == "trend":
            return self._add_trend_tensor(x, mask, return_mask=return_mask)
        if mode == "contextual":
            return self._add_contextual_tensor(x, mask, return_mask=return_mask)
        if mode == "relational":
            return self._add_relational_tensor(x, mask, return_mask=return_mask)
        return self._add_shapelet_tensor(x, mask, return_mask=return_mask)

    def _inject_numpy(self, x_window: np.ndarray, return_mask: bool = False) -> np.ndarray:
        if x_window.ndim == 3:
            injected = []
            masks = []
            for sample in x_window:
                sample_injected, sample_mask = self._inject_numpy(sample, return_mask=True)
                injected.append(sample_injected)
                masks.append(sample_mask)
            x_out = np.stack(injected, axis=0)
            mask_out = np.stack(masks, axis=0)
            return (x_out, mask_out) if return_mask else x_out

        x = np.array(x_window, copy=True)
        if x.ndim != 2:
            raise ValueError(f"Expected input shape [C, L] or [B, C, L], got {x.shape}")

        mask = np.zeros_like(x, dtype=bool)
        mode = self._choose_mode()
        if mode == "spike":
            return self._add_spike_numpy(x, mask, return_mask=return_mask)
        if mode == "shift":
            return self._add_shift_numpy(x, mask, return_mask=return_mask)
        if mode == "scale":
            return self._add_scale_numpy(x, mask, return_mask=return_mask)
        if mode == "trend":
            return self._add_trend_numpy(x, mask, return_mask=return_mask)
        if mode == "contextual":
            return self._add_contextual_numpy(x, mask, return_mask=return_mask)
        if mode == "relational":
            return self._add_relational_numpy(x, mask, return_mask=return_mask)
        return self._add_shapelet_numpy(x, mask, return_mask=return_mask)

    @staticmethod
    def _finish_tensor(x: torch.Tensor, mask: torch.Tensor, return_mask: bool):
        return (x, mask) if return_mask else x

    @staticmethod
    def _finish_numpy(x: np.ndarray, mask: np.ndarray, return_mask: bool):
        return (x, mask) if return_mask else x

    def inject_relational_batch(
        self,
        x_batch: torch.Tensor,
        p: float = 1.0,
        max_shift_ratio: float = 0.2,
        max_channels: int = 0,
        mode_weights=None,
        return_mask: bool = False,
    ):
        if not isinstance(x_batch, torch.Tensor):
            raise TypeError("x_batch must be a torch.Tensor.")
        if x_batch.dim() != 3:
            raise ValueError(f"Expected batch shape [B, C, L], got {tuple(x_batch.shape)}")

        x = x_batch.detach().clone()
        bsz, num_channels, length = x.shape
        point_mask = torch.zeros_like(x, dtype=torch.bool)
        if bsz <= 0 or num_channels <= 0 or length <= 1:
            return (x, point_mask) if return_mask else x

        p = min(max(float(p), 0.0), 1.0)
        max_selected_channels = int(max_channels)
        if max_selected_channels <= 0:
            max_selected_channels = max(1, num_channels // 4)
        max_selected_channels = max(1, min(max_selected_channels, num_channels))
        max_shift = max(1, int(round(length * max(0.0, float(max_shift_ratio)))))
        modes = ["time_shift", "channel_replace", "channel_shuffle"]
        if mode_weights is None:
            weights = [1.0, 1.0, 1.0]
        else:
            weights = [max(0.0, float(w)) for w in mode_weights]
            if len(weights) != len(modes) or sum(weights) <= 0.0:
                weights = [1.0, 1.0, 1.0]

        for batch_idx in range(bsz):
            if random.random() > p:
                continue

            mode = random.choices(modes, weights=weights, k=1)[0]
            num_selected = random.randint(1, max_selected_channels)
            if mode == "channel_shuffle" and num_channels > 1:
                num_selected = max(2, min(num_selected, num_channels))
            channels = random.sample(range(num_channels), num_selected)

            if mode == "time_shift" or (mode == "channel_replace" and bsz <= 1):
                shift = random.randint(1, max_shift)
                if random.random() < 0.5:
                    shift = -shift
                original = x_batch[batch_idx, channels, :]
                shifted = original.clone()
                if shift > 0:
                    shifted[:, shift:] = original[:, :-shift]
                    shifted[:, :shift] = original[:, :1]
                else:
                    shift_abs = abs(shift)
                    shifted[:, :-shift_abs] = original[:, shift_abs:]
                    shifted[:, -shift_abs:] = original[:, -1:]
                x[batch_idx, channels, :] = shifted
            elif mode == "channel_replace":
                source_idx = random.randrange(bsz - 1)
                if source_idx >= batch_idx:
                    source_idx += 1
                x[batch_idx, channels, :] = x_batch[source_idx, channels, :]
            else:
                if len(channels) <= 1:
                    shift = random.randint(1, max_shift)
                    x[batch_idx, channels, :] = torch.roll(x[batch_idx, channels, :], shifts=shift, dims=-1)
                else:
                    permuted = channels[:]
                    while permuted == channels:
                        random.shuffle(permuted)
                    x[batch_idx, channels, :] = x_batch[batch_idx, permuted, :]

            point_mask[batch_idx, channels, :] = True

        return (x, point_mask) if return_mask else x

    def _add_spike_tensor(self, x: torch.Tensor, mask: torch.Tensor, return_mask: bool = False) -> torch.Tensor:
        c, l = x.shape
        num_points = random.randint(1, max(1, l // 10))
        std = x.std(unbiased=False).clamp_min(1e-6)
        amplitude = random.uniform(3.0, 6.0) * std

        for _ in range(num_points):
            ch = random.randrange(c)
            pos = random.randrange(l)
            sign = 1.0 if random.random() > 0.5 else -1.0
            x[ch, pos] = x[ch, pos] + sign * amplitude
            mask[ch, pos] = True
        return self._finish_tensor(x, mask, return_mask)

    def _add_shift_tensor(self, x: torch.Tensor, mask: torch.Tensor, return_mask: bool = False) -> torch.Tensor:
        dims, l = x.shape
        start, end = self._sample_segment(l, min_ratio=0.1, max_ratio=0.35)
        target_dims = self._sample_dims(dims)
        std = x.std(unbiased=False).clamp_min(1e-6)
        offset = random.uniform(-3.0, 3.0) * std
        x[target_dims, start:end] = x[target_dims, start:end] + offset
        mask[target_dims, start:end] = True
        return self._finish_tensor(x, mask, return_mask)

    def _add_scale_tensor(self, x: torch.Tensor, mask: torch.Tensor, return_mask: bool = False) -> torch.Tensor:
        dims, l = x.shape
        start, end = self._sample_segment(l, min_ratio=0.12, max_ratio=0.5)
        target_dims = self._sample_dims(dims)
        scale = random.choice([random.uniform(1.5, 2.5), random.uniform(0.2, 0.7)])
        x[target_dims, start:end] = x[target_dims, start:end] * scale
        mask[target_dims, start:end] = True
        return self._finish_tensor(x, mask, return_mask)

    def _add_trend_tensor(self, x: torch.Tensor, mask: torch.Tensor, return_mask: bool = False) -> torch.Tensor:
        dims, l = x.shape
        start, end = self._sample_segment(l, min_ratio=0.25, max_ratio=0.9)
        if random.random() < 0.5:
            end = l
        target_dims = self._sample_dims(dims)
        seg_len = max(1, end - start)
        std = x.std(unbiased=False).clamp_min(1e-6)
        base = random.uniform(0.5, 2.5) * std
        sign = 1.0 if random.random() > 0.5 else -1.0
        ramp = torch.linspace(0.0, 1.0, steps=seg_len, device=x.device, dtype=x.dtype)
        x[target_dims, start:end] = x[target_dims, start:end] + sign * base * ramp.unsqueeze(0)
        mask[target_dims, start:end] = True
        return self._finish_tensor(x, mask, return_mask)

    def _add_contextual_tensor(self, x: torch.Tensor, mask: torch.Tensor, return_mask: bool = False) -> torch.Tensor:
        dims, l = x.shape
        start, end = self._sample_segment(l, min_ratio=0.08, max_ratio=0.2)
        target_dims = self._sample_dims(dims, full_prob=0.2)
        std = x.std(unbiased=False).clamp_min(1e-6)
        offset = random.uniform(-2.5, 2.5) * std
        local_scale = random.uniform(1.5, 2.2)
        x[target_dims, start:end] = x[target_dims, start:end] * local_scale + offset
        mask[target_dims, start:end] = True
        return self._finish_tensor(x, mask, return_mask)

    def _add_shapelet_tensor(self, x: torch.Tensor, mask: torch.Tensor, return_mask: bool = False) -> torch.Tensor:
        dims, l = x.shape
        start, end = self._sample_segment(l, min_ratio=0.15, max_ratio=0.5)
        target_dims = self._sample_dims(dims, full_prob=0.2)
        anchor_pos = random.randint(0, max(0, l - 1))
        base = x[target_dims, anchor_pos].unsqueeze(1).expand(-1, end - start)
        std = x.std(unbiased=False).clamp_min(1e-6)
        noise = torch.randn_like(base) * (0.05 * std)
        x[target_dims, start:end] = base + noise
        mask[target_dims, start:end] = True
        return self._finish_tensor(x, mask, return_mask)

    def _add_relational_tensor(self, x: torch.Tensor, mask: torch.Tensor, return_mask: bool = False) -> torch.Tensor:
        dims, l = x.shape
        if dims <= 0 or l <= 1:
            return self._finish_tensor(x, mask, return_mask)
        mode = random.choice(["time_shift", "channel_shuffle"])
        target_dims = self._sample_dims(dims, full_prob=0.0)
        if mode == "time_shift" or dims == 1:
            max_shift = max(1, int(round(l * 0.2)))
            shift = random.randint(1, max_shift)
            if random.random() < 0.5:
                shift = -shift
            x[target_dims, :] = torch.roll(x[target_dims, :], shifts=shift, dims=-1)
            mask[target_dims, :] = True
            return self._finish_tensor(x, mask, return_mask)

        if len(target_dims) < 2:
            target_dims = random.sample(range(dims), min(2, dims))
        permuted = target_dims[:]
        while permuted == target_dims:
            random.shuffle(permuted)
        x[target_dims, :] = x[permuted, :].clone()
        mask[target_dims, :] = True
        return self._finish_tensor(x, mask, return_mask)

    def _add_spike_numpy(self, x: np.ndarray, mask: np.ndarray, return_mask: bool = False) -> np.ndarray:
        c, l = x.shape
        num_points = random.randint(1, max(1, l // 10))
        std = max(float(x.std()), 1e-6)
        amplitude = random.uniform(3.0, 6.0) * std

        for _ in range(num_points):
            ch = random.randrange(c)
            pos = random.randrange(l)
            sign = 1.0 if random.random() > 0.5 else -1.0
            x[ch, pos] = x[ch, pos] + sign * amplitude
            mask[ch, pos] = True
        return self._finish_numpy(x, mask, return_mask)

    def _add_shift_numpy(self, x: np.ndarray, mask: np.ndarray, return_mask: bool = False) -> np.ndarray:
        dims, l = x.shape
        start, end = self._sample_segment(l, min_ratio=0.1, max_ratio=0.35)
        target_dims = self._sample_dims(dims)
        std = max(float(x.std()), 1e-6)
        offset = random.uniform(-3.0, 3.0) * std
        x[target_dims, start:end] = x[target_dims, start:end] + offset
        mask[target_dims, start:end] = True
        return self._finish_numpy(x, mask, return_mask)

    def _add_scale_numpy(self, x: np.ndarray, mask: np.ndarray, return_mask: bool = False) -> np.ndarray:
        dims, l = x.shape
        start, end = self._sample_segment(l, min_ratio=0.12, max_ratio=0.5)
        target_dims = self._sample_dims(dims)
        scale = random.choice([random.uniform(1.5, 2.5), random.uniform(0.2, 0.7)])
        x[target_dims, start:end] = x[target_dims, start:end] * scale
        mask[target_dims, start:end] = True
        return self._finish_numpy(x, mask, return_mask)

    def _add_trend_numpy(self, x: np.ndarray, mask: np.ndarray, return_mask: bool = False) -> np.ndarray:
        dims, l = x.shape
        start, end = self._sample_segment(l, min_ratio=0.25, max_ratio=0.9)
        if random.random() < 0.5:
            end = l
        target_dims = self._sample_dims(dims)
        seg_len = max(1, end - start)
        std = max(float(x.std()), 1e-6)
        base = random.uniform(0.5, 2.5) * std
        sign = 1.0 if random.random() > 0.5 else -1.0
        ramp = np.linspace(0.0, 1.0, num=seg_len, dtype=x.dtype)
        x[target_dims, start:end] = x[target_dims, start:end] + sign * base * ramp[None, :]
        mask[target_dims, start:end] = True
        return self._finish_numpy(x, mask, return_mask)

    def _add_contextual_numpy(self, x: np.ndarray, mask: np.ndarray, return_mask: bool = False) -> np.ndarray:
        dims, l = x.shape
        start, end = self._sample_segment(l, min_ratio=0.08, max_ratio=0.2)
        target_dims = self._sample_dims(dims, full_prob=0.2)
        std = max(float(x.std()), 1e-6)
        offset = random.uniform(-2.5, 2.5) * std
        local_scale = random.uniform(1.5, 2.2)
        x[target_dims, start:end] = x[target_dims, start:end] * local_scale + offset
        mask[target_dims, start:end] = True
        return self._finish_numpy(x, mask, return_mask)

    def _add_shapelet_numpy(self, x: np.ndarray, mask: np.ndarray, return_mask: bool = False) -> np.ndarray:
        dims, l = x.shape
        start, end = self._sample_segment(l, min_ratio=0.15, max_ratio=0.5)
        target_dims = self._sample_dims(dims, full_prob=0.2)
        anchor_pos = random.randint(0, max(0, l - 1))
        base = np.repeat(x[target_dims, anchor_pos][:, None], end - start, axis=1)
        std = max(float(x.std()), 1e-6)
        noise = np.random.randn(*base.shape).astype(x.dtype, copy=False) * (0.05 * std)
        x[target_dims, start:end] = base + noise
        mask[target_dims, start:end] = True
        return self._finish_numpy(x, mask, return_mask)

    def _add_relational_numpy(self, x: np.ndarray, mask: np.ndarray, return_mask: bool = False) -> np.ndarray:
        dims, l = x.shape
        if dims <= 0 or l <= 1:
            return self._finish_numpy(x, mask, return_mask)
        mode = random.choice(["time_shift", "channel_shuffle"])
        target_dims = self._sample_dims(dims, full_prob=0.0)
        if mode == "time_shift" or dims == 1:
            max_shift = max(1, int(round(l * 0.2)))
            shift = random.randint(1, max_shift)
            if random.random() < 0.5:
                shift = -shift
            x[target_dims, :] = np.roll(x[target_dims, :], shift=shift, axis=-1)
            mask[target_dims, :] = True
            return self._finish_numpy(x, mask, return_mask)

        if len(target_dims) < 2:
            target_dims = random.sample(range(dims), min(2, dims))
        permuted = target_dims[:]
        while permuted == target_dims:
            random.shuffle(permuted)
        x[target_dims, :] = x[permuted, :].copy()
        mask[target_dims, :] = True
        return self._finish_numpy(x, mask, return_mask)
