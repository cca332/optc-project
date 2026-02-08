from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch


def clip_by_l2_norm_(tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = tensor.norm(p=2)
    if norm > max_norm:
        tensor.mul_(max_norm / (norm + 1e-12))
    return tensor


def add_gaussian_noise_(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return tensor
    noise = torch.randn_like(tensor) * float(sigma)
    tensor.add_(noise)
    return tensor


@dataclass
class FeatureDPConfig:
    enabled: bool
    clip_C: float
    noise_sigma: float


@dataclass
class GradDPConfig:
    enabled: bool
    base_clip_C0: float
    importance_alpha: float
    noise_sigma0: float
    noise_schedule: object | None = None


def dp_features(x: torch.Tensor, cfg: FeatureDPConfig) -> torch.Tensor:
    if not cfg.enabled:
        return x
    x = x.clone()
    # per-sample clipping + noise (最小骨架实现)
    if x.ndim == 2:
        for i in range(x.shape[0]):
            clip_by_l2_norm_(x[i], cfg.clip_C)
            add_gaussian_noise_(x[i], cfg.noise_sigma)
    else:
        clip_by_l2_norm_(x, cfg.clip_C)
        add_gaussian_noise_(x, cfg.noise_sigma)
    return x


def dp_gradients(param_grads: List[torch.Tensor], cfg: GradDPConfig, importance: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
    """对参数梯度做裁剪+噪声（最小骨架版）。

    importance: 可选 [G] 因子（均值约 1），用于自适应裁剪/噪声。
    """
    if not cfg.enabled:
        return param_grads
    out = []
    for i, g in enumerate(param_grads):
        gg = g.clone()
        C = float(cfg.base_clip_C0)
        sigma = float(cfg.noise_sigma0)
        if importance is not None and i < importance.numel():
            factor = float(importance[i].item()) ** float(cfg.importance_alpha)
            C *= factor
            sigma *= (1.0 / max(factor, 1e-6))
        clip_by_l2_norm_(gg, C)
        add_gaussian_noise_(gg, sigma)
        out.append(gg)
    return out
