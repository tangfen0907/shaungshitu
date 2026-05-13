"""
轻量级日志工具。

原文件依赖旧版 TensorFlow、scipy.misc 以及 Python2 的 StringIO，
会在当前 PyTorch 环境下触发编辑器导入报错。这里改为使用
PyTorch 官方兼容的 TensorBoard SummaryWriter，并尽量保持原有接口：

1. scalar_summary(tag, value, step)
2. image_summary(tag, images, step)
3. histo_summary(tag, values, step, bins=1000)
"""

from typing import Iterable

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """训练日志记录器。"""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def scalar_summary(self, tag: str, value: float, step: int):
        """记录标量。"""
        self.writer.add_scalar(tag, float(value), global_step=step)
        self.writer.flush()

    def image_summary(self, tag: str, images: Iterable, step: int):
        """
        记录一组图像。

        输入支持：
        1. 单张图像 [H, W] / [C, H, W]
        2. 图像列表
        3. numpy.ndarray 或 torch.Tensor
        """
        if isinstance(images, (np.ndarray, torch.Tensor)):
            image_list = [images]
        else:
            image_list = list(images)

        for idx, image in enumerate(image_list):
            if isinstance(image, np.ndarray):
                image_tensor = torch.from_numpy(image)
            elif isinstance(image, torch.Tensor):
                image_tensor = image.detach().cpu()
            else:
                image_tensor = torch.tensor(image)

            # 若是灰度图 [H, W]，补成 [1, H, W]。
            if image_tensor.dim() == 2:
                image_tensor = image_tensor.unsqueeze(0)

            # 若是 [H, W, C]，转为 [C, H, W]。
            if image_tensor.dim() == 3 and image_tensor.shape[-1] in (1, 3) and image_tensor.shape[0] not in (1, 3):
                image_tensor = image_tensor.permute(2, 0, 1).contiguous()

            self.writer.add_image(f"{tag}/{idx}", image_tensor, global_step=step)

        self.writer.flush()

    def histo_summary(self, tag: str, values, step: int, bins: int = 1000):
        """记录直方图。"""
        if isinstance(values, np.ndarray):
            hist_values = values
        elif isinstance(values, torch.Tensor):
            hist_values = values.detach().cpu().numpy()
        else:
            hist_values = np.asarray(values)

        self.writer.add_histogram(tag, hist_values, global_step=step, bins=bins)
        self.writer.flush()

    def close(self):
        """关闭底层 writer。"""
        self.writer.close()
