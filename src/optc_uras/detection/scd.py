from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCDHead(nn.Module):
    """SCD：风格-内容解耦头（可选）。只训练这个小头，backbone 冻结。"""

    def __init__(self, in_dim: int, style_dim: int, content_dim: int):
        super().__init__()
        self.style = nn.Sequential(nn.Linear(in_dim, style_dim), nn.ReLU(), nn.Linear(style_dim, style_dim))
        self.content = nn.Sequential(nn.Linear(in_dim, content_dim), nn.ReLU(), nn.Linear(content_dim, content_dim))
        self.decoder = nn.Sequential(nn.Linear(style_dim + content_dim, in_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.style(x)
        c = self.content(x)
        recon = self.decoder(torch.cat([s, c], dim=-1))
        return s, c, recon


def decorr_loss(s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # batch cross-covariance
    s0 = s - s.mean(dim=0, keepdim=True)
    c0 = c - c.mean(dim=0, keepdim=True)
    cov = (s0.t() @ c0) / max(s.shape[0] - 1, 1)
    return (cov ** 2).mean()


def variance_floor_loss(x: torch.Tensor, v_min: float) -> torch.Tensor:
    v = x.var(dim=0)  # [d]
    return F.relu(float(v_min) - v).mean()


@dataclass
class SCDTrainConfig:
    epochs: int
    lr: float
    lambda_decorr: float
    lambda_recon: float
    lambda_var: float
    variance_floor: float


def train_scd(head: SCDHead, xs: torch.Tensor, cfg: SCDTrainConfig) -> Dict[str, float]:
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr)
    losses = []
    for _ in range(int(cfg.epochs)):
        s, c, recon = head(xs)
        l_dec = decorr_loss(s, c)
        l_rec = F.mse_loss(recon, xs)
        l_var = variance_floor_loss(s, cfg.variance_floor) + variance_floor_loss(c, cfg.variance_floor)
        loss = float(cfg.lambda_decorr) * l_dec + float(cfg.lambda_recon) * l_rec + float(cfg.lambda_var) * l_var
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return {"scd_loss": sum(losses) / max(len(losses), 1)}
