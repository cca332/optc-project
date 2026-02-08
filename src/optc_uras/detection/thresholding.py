from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import torch


def quantile_threshold(scores: torch.Tensor, q: float) -> float:
    q = float(q)
    return float(torch.quantile(scores.detach().cpu(), q).item())
