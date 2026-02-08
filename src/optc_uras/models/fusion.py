from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x,y: [B,d]
    x0 = x - x.mean(dim=-1, keepdim=True)
    y0 = y - y.mean(dim=-1, keepdim=True)
    num = (x0 * y0).mean(dim=-1)
    den = torch.sqrt((x0**2).mean(dim=-1) * (y0**2).mean(dim=-1) + eps)
    return num / (den + eps)  # [B]


class SingleHeadViewAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, view_vecs: torch.Tensor) -> torch.Tensor:
        # view_vecs: [B, V, d]
        Q = self.q(view_vecs)
        K = self.k(view_vecs)
        V = self.v(view_vecs)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-1, -2)) * self.scale, dim=-1)  # [B,V,V]
        out = torch.matmul(attn, V)  # [B,V,d]
        return out


class GatedFusion(nn.Module):
    def __init__(self, dim: int, gamma: float = 0.15, mode: str = "hard", beta: float = 10.0, use_interaction: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.gamma = float(gamma)
        self.mode = str(mode)
        self.beta = float(beta)
        self.use_interaction = bool(use_interaction)
        self.attn = SingleHeadViewAttention(dim) if self.use_interaction else None

    def forward(self, view_vecs: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """C1-C3：门控融合

        view_vecs: [B,V,d]  对齐后 view-level
        w: [B,V]            可靠性权重
        returns: z [B,d], gate [B], rho_bar [B]
        """
        B, V, d = view_vecs.shape
        # pairwise pearson corr
        rhos = []
        for i in range(V):
            for j in range(i + 1, V):
                rhos.append(pearson_corr(view_vecs[:, i, :], view_vecs[:, j, :]).abs())
        rho_bar = torch.stack(rhos, dim=0).mean(dim=0) if rhos else torch.zeros((B,), device=view_vecs.device)

        if self.mode == "hard":
            gate = (rho_bar > self.gamma).float()
        elif self.mode == "soft":
            gate = torch.sigmoid((rho_bar - self.gamma) * self.beta)
        else:
            raise ValueError(f"unknown gate mode={self.mode}")

        base = torch.einsum("bv,bvd->bd", w, view_vecs)  # weighted sum
        if self.use_interaction and self.attn is not None:
            inter = self.attn(view_vecs)                 # [B,V,d]
            inter_fused = torch.einsum("bv,bvd->bd", w, inter) # [B,d]
        else:
            inter_fused = base

        z = (1.0 - gate).unsqueeze(-1) * base + gate.unsqueeze(-1) * inter_fused
        return z, gate, rho_bar
