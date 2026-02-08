from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentHeads(nn.Module):
    """Step2 学生侧：从 Step1 融合表示 z 生成每个子空间向量。"""

    def __init__(self, in_dim: int, num_subspaces: int, subspace_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.num_subspaces = int(num_subspaces)
        self.subspace_dim = int(subspace_dim)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_subspaces * self.subspace_dim),
        )

    def forward(self, z: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        # z: [B, d]
        y = self.net(z).reshape(z.shape[0], self.num_subspaces, self.subspace_dim)
        return F.normalize(y, dim=-1) if normalize else y


def uras_from_subspaces(subspace_vecs: torch.Tensor, route_p: torch.Tensor) -> torch.Tensor:
    """URAS: 按路由分布加权后拼接。

    subspace_vecs: [B,M,d_s]
    route_p: [B,M]
    returns: [B, M*d_s]
    """
    weighted = subspace_vecs * route_p.unsqueeze(-1)
    return weighted.reshape(weighted.shape[0], -1)
