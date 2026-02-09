from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherModel(nn.Module):
    """Step2 教师模型：仅使用客户端不可逆行为特征做离线自监督预训练，之后冻结。"""

    def __init__(self, behavior_dim: int, num_subspaces: int, subspace_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.behavior_dim = int(behavior_dim)
        self.num_subspaces = int(num_subspaces)
        self.subspace_dim = int(subspace_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.behavior_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, self.subspace_dim) for _ in range(self.num_subspaces)])

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        # x: [B, behavior_dim]
        h = self.encoder(x)  # [B, hidden]
        outs = []
        for head in self.heads:
            y = head(h)
            if normalize:
                y = F.normalize(y, dim=-1)
            outs.append(y)
        return torch.stack(outs, dim=1)  # [B, M, d_s]