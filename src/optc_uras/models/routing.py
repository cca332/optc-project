from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


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
