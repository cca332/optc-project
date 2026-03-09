from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Routing Component (from routing.py)
# =============================================================================

class RouterMLP(nn.Module):
    def __init__(self, in_dim: int, num_subspaces: int, hidden_dims: Sequence[int] = (256, 128), dropout: float = 0.1, num_views: int = 0):
        super().__init__()
        dims = [int(in_dim)] + [int(h) for h in hidden_dims]
        self.num_views = num_views
        
        if num_views > 1:
            # View-Specific Input Layers (for C2 Adaptive Clipping)
            assert in_dim % num_views == 0, "in_dim must be divisible by num_views"
            dim_per_view = in_dim // num_views
            self.view_layers = nn.ModuleList([
                nn.Linear(dim_per_view, dims[1]) for _ in range(num_views)
            ])
            
            layers: List[nn.Module] = []
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
                
            # Remaining MLP
            for a, b in zip(dims[1:-1], dims[2:]):
                layers.append(nn.Linear(a, b))
                layers.append(nn.ReLU())
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
            layers.append(nn.Linear(dims[-1], int(num_subspaces)))
            self.rest = nn.Sequential(*layers)
        else:
            # Monolithic MLP
            layers: List[nn.Module] = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers.append(nn.Linear(a, b))
                layers.append(nn.ReLU())
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
            layers.append(nn.Linear(dims[-1], int(num_subspaces)))
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        if self.num_views > 1:
            # Split input and apply per-view layers
            # x is flattened [B, V*d]. We need to reshape to [B, V, d] or chunk it.
            chunks = x.chunk(self.num_views, dim=1)
            
            # Sum the outputs of the first layer (Fusion)
            # h = Sum(Linear_v(x_v))
            h = 0
            for i, layer in enumerate(self.view_layers):
                h = h + layer(chunks[i])
            
            logits = self.rest(h)
        else:
            logits = self.net(x)
        return F.softmax(logits, dim=-1)


# =============================================================================
# 2. Alignment Component (from alignment.py)
# =============================================================================

class AlignmentBases(nn.Module):
    """B2：共享对齐基矩阵（对齐基混合）。"""

    def __init__(self, num_bases: int, dim: int):
        super().__init__()
        self.num_bases = int(num_bases)
        self.dim = int(dim)
        # [M, d, d]
        self.bases = nn.Parameter(torch.randn(self.num_bases, self.dim, self.dim) * 0.02)

    def forward(self, route_p: torch.Tensor) -> torch.Tensor:
        # route_p: [B, M]
        # 输出：对齐算子 A: [B, d, d] = sum_m p_m * B_m
        A = torch.einsum("bm,mij->bij", route_p, self.bases)
        return A

    def apply(self, A: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        # slots: [B, K, V, d]
        # A: [B, d, d]
        return torch.einsum("bij,bkvj->bkvi", A, slots)


# =============================================================================
# 3. Fusion Component (from fusion.py)
# =============================================================================

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
