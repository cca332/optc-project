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

    def augment(self, x: torch.Tensor, dropout_p: float = 0.2, jitter_sigma: float = 0.1) -> torch.Tensor:
        """A3: Random Augmentation (Masking/Dropout/Jitter)"""
        # 1. Dropout (Masking)
        x_aug = F.dropout(x, p=dropout_p, training=True)
        # 2. Jitter (Noise)
        noise = torch.randn_like(x_aug) * jitter_sigma
        return x_aug + noise

    def forward_contrastive(self, x: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
        """A3: Self-Supervised InfoNCE Training Step."""
        B = x.shape[0]
        # 1. Augment twice
        x1 = self.augment(x)
        x2 = self.augment(x)
        # 2. Encode to subspaces
        z1 = self.forward(x1).view(B, -1)
        z2 = self.forward(x2).view(B, -1)
        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        # 3. InfoNCE Loss
        logits = torch.matmul(z1, z2.T) / temp
        labels = torch.arange(B, device=x.device)
        return F.cross_entropy(logits, labels)

    def augment(self, x: torch.Tensor, dropout_p: float = 0.2, jitter_sigma: float = 0.1) -> torch.Tensor:
        """A3: Random Augmentation (Masking/Dropout/Jitter)"""
        # 1. Dropout (Masking)
        x_aug = F.dropout(x, p=dropout_p, training=True)
        # 2. Jitter (Noise)
        noise = torch.randn_like(x_aug) * jitter_sigma
        return x_aug + noise

    def forward_contrastive(self, x: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
        """A3: Self-Supervised InfoNCE Training Step.
        
        Returns: loss
        """
        B = x.shape[0]
        
        # 1. Augment twice
        x1 = self.augment(x)
        x2 = self.augment(x)
        
        # 2. Encode to subspaces
        # Output: [B, M, d_s] -> flatten to [B, M*d_s] for contrastive
        z1 = self.forward(x1).view(B, -1)
        z2 = self.forward(x2).view(B, -1)
        
        # Normalize for cosine similarity
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # 3. InfoNCE Loss
        # Positive pairs: (z1[i], z2[i])
        # Negative pairs: (z1[i], z2[j]) and (z1[i], z1[j]) etc.
        # Simplified implementation: standard InfoNCE
        
        logits = torch.matmul(z1, z2.T) / temp # [B, B]
        labels = torch.arange(B, device=x.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss

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