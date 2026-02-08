from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import torch


@dataclass
class ATCConfig:
    enabled: bool
    lambda_c: float
    lambda_r: float
    lambda_p: float


def atc_threshold(base_tau: float, confidence: float, route_uncertainty: float, privacy_risk: float, cfg: ATCConfig) -> float:
    """ATC：样本阈值自适应（越可信 -> 阈值更低；越不确定/风险高 -> 更高）。"""
    if not cfg.enabled:
        return float(base_tau)
    tau = float(base_tau) * (1.0 - float(cfg.lambda_c) * float(confidence))
    tau = tau + float(cfg.lambda_r) * float(route_uncertainty) + float(cfg.lambda_p) * float(privacy_risk)
    return max(tau, 0.0)
