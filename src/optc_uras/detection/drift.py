from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class DriftConfig:
    enabled: bool
    margin: float
    ema_alpha: float
    update_covariance: str = "diag_ema"


def maybe_update(mu: torch.Tensor, cov: torch.Tensor, x: torch.Tensor, score: float, threshold: float, cfg: DriftConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    if not cfg.enabled:
        return mu, cov
    # 明显 benign 才更新：score < threshold - margin
    if score >= (threshold - float(cfg.margin)):
        return mu, cov
    a = float(cfg.ema_alpha)
    new_mu = (1 - a) * mu + a * x
    if cfg.update_covariance == "diag_ema":
        x0 = (x - new_mu)
        var = torch.diag(cov)
        new_var = (1 - a) * var + a * (x0 * x0)
        new_cov = torch.diag(new_var)
    else:
        new_cov = cov
    return new_mu, new_cov
