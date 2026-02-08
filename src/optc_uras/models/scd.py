from __future__ import annotations

import torch
import torch.nn as nn


class SCDProjector(nn.Module):
    """
    Step 3 Module A: Style-Content Disentanglement (SCD).
    
    A1: 风格/内容投影 (Style/Content Projection)
    定义线性或小型 MLP 投影头，将 Step 2 的 URAS 表示分解为
    '风格' (style, 稳定, 用于异常检测) 和 '内容' (content, 保留区分性) 分量。
    
    s(x) = P_s(u^S(x))
    c(x) = P_c(u^S(x))
    """

    def __init__(self, in_dim: int, style_dim: int, content_dim: int, hidden_dim: int = 0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.style_dim = int(style_dim)
        self.content_dim = int(content_dim)
        
        # P_s: Style Projector (风格投影头)
        if hidden_dim > 0:
            self.p_s = nn.Sequential(
                nn.Linear(self.in_dim, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), self.style_dim)
            )
            # P_c: Content Projector (内容投影头)
            self.p_c = nn.Sequential(
                nn.Linear(self.in_dim, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), self.content_dim)
            )
        else:
            # Linear projection (default as per simple implementation)
            self.p_s = nn.Linear(self.in_dim, self.style_dim)
            self.p_c = nn.Linear(self.in_dim, self.content_dim)
            
        # A3 (A): Lightweight Reconstruction Decoder D(.)
        # u_hat = D([s || c])
        # Linear or Small MLP
        decoder_in = self.style_dim + self.content_dim
        if hidden_dim > 0:
             self.decoder = nn.Sequential(
                nn.Linear(decoder_in, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), self.in_dim)
            )
        else:
            self.decoder = nn.Linear(decoder_in, self.in_dim)

    def forward(self, uras: torch.Tensor, center_batch: bool = False, return_rec: bool = False) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            uras: [B, in_dim] Step 2 产生的 URAS 向量 (u^S)。
            center_batch: 是否进行 Batch 中心化 (仅在训练时稳定用)。
                          s <- s - E_B[s], c <- c - E_B[c]
            return_rec: 是否返回重构结果 u_hat (用于 A3 训练)。
            
        Returns:
            (s, c) if return_rec is False
            (s, c, u_hat) if return_rec is True
        """
        s = self.p_s(uras)
        c = self.p_c(uras)
        
        if center_batch and s.shape[0] > 1:
            # 批量去均值，降低正常样本的漂移干扰
            s = s - s.mean(dim=0, keepdim=True)
            c = c - c.mean(dim=0, keepdim=True)
            
        if return_rec:
            # Reconstruct u_hat from [s || c]
            sc_cat = torch.cat([s, c], dim=-1)
            u_hat = self.decoder(sc_cat)
            return s, c, u_hat
            
        return s, c