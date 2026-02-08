from __future__ import annotations

import torch
import torch.nn as nn


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
