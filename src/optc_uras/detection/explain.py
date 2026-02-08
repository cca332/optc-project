from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExplainConfig:
    enabled: bool
    top_k: int
    alpha_reliability: float


class ViewStyleProjectors(nn.Module):
    """每个视图一个线性投影，把 view-level 向量映射到 style 空间，用于证据排名（非归因/非定位）。"""

    def __init__(self, view_names: Sequence[str], in_dim: int, style_dim: int):
        super().__init__()
        self.view_names = list(view_names)
        self.proj = nn.ModuleDict({v: nn.Linear(in_dim, style_dim) for v in self.view_names})

    def forward(self, view_vecs: torch.Tensor) -> torch.Tensor:
        # view_vecs: [V,d] -> [V,style_dim]
        outs = []
        for i, v in enumerate(self.view_names):
            outs.append(self.proj[v](view_vecs[i]))
        return torch.stack(outs, dim=0)


def evidence_ranking(view_vecs: torch.Tensor, w: torch.Tensor, delta: torch.Tensor, proj: ViewStyleProjectors, cfg: ExplainConfig) -> List[Tuple[str, float]]:
    if not cfg.enabled:
        return []
    with torch.no_grad():
        # map each view to style space
        v_style = proj(view_vecs)  # [V,style]
        v_style = F.normalize(v_style, dim=-1)
        delta = F.normalize(delta, dim=-1)
        cos = (v_style * delta.unsqueeze(0)).sum(dim=-1)  # [V]
        weight = (w.clamp_min(1e-6) ** float(cfg.alpha_reliability))
        score = cos * weight
        pairs = [(proj.view_names[i], float(score[i].item())) for i in range(len(proj.view_names))]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[: int(cfg.top_k)]
