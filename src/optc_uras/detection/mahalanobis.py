from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import torch


@dataclass
class CovConfig:
    mode: str = "full"         # full | diag
    shrinkage: float = 1e-3


def fit_gaussian(x: torch.Tensor, cfg: CovConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """拟合 benign 分布：均值与协方差。"""
    mu = x.mean(dim=0)
    x0 = x - mu
    if cfg.mode == "diag":
        var = x0.var(dim=0) + float(cfg.shrinkage)
        cov = torch.diag(var)
    else:
        cov = (x0.t() @ x0) / max(x.shape[0] - 1, 1)
        cov = cov + torch.eye(cov.shape[0], device=x.device) * float(cfg.shrinkage)
    return mu, cov


def mahalanobis_score(x: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor, use_squared: bool = True) -> torch.Tensor:
    """Mahalanobis 距离（平方）作为异常分数。"""
    x0 = x - mu
    inv = torch.linalg.pinv(cov)
    d2 = torch.einsum("bi,ij,bj->b", x0, inv, x0)
    return d2 if use_squared else torch.sqrt(d2 + 1e-12)
