from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherModel(nn.Module):
    """Step2 教师模型 (Teacher Model)
    
    功能：
    1. 离线自监督预训练 (Offline Self-Supervised Pretraining)。
    2. 仅使用客户端的“行为特征” (Behavioral Features) 作为输入，不依赖标签。
    3. 核心算法：A3 InfoNCE (Contrastive Learning)。
       - 数据增强：Masking (Dropout) + Jitter (Gaussian Noise)。
       - 损失函数：InfoNCE Loss，最大化同一样本不同增强视图的相似度。
    4. 训练完成后冻结 (Frozen)，作为 Step 2 联邦学生模型的蒸馏指导。
    """

    def __init__(
        self,
        behavior_dim: int,
        num_subspaces: int,
        subspace_dim: int,
        hidden_dim: int = 256,
        backbone: str = "mlp",
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behavior_dim = int(behavior_dim)
        self.num_subspaces = int(num_subspaces)
        self.subspace_dim = int(subspace_dim)
        self.hidden_dim = int(hidden_dim)
        self.backbone = str(backbone)
        if self.backbone == "mlp":
            self.input_proj = None
            self.cls_token = None
            self.pos_emb = None
            self.encoder = nn.Sequential(
                nn.Linear(self.behavior_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        elif self.backbone == "transformer":
            self.input_proj = nn.Linear(self.behavior_dim, hidden_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.pos_emb = nn.Parameter(torch.randn(1, 1025, hidden_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=int(num_heads),
                dim_feedforward=hidden_dim * 4,
                dropout=float(dropout),
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        else:
            raise ValueError(f"Unsupported teacher backbone: {self.backbone}")
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, self.subspace_dim) for _ in range(self.num_subspaces)])

    def encode(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.backbone == "mlp":
            if x.dim() > 2:
                x = x.mean(dim=1)
            return self.encoder(x)

        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if x.shape[1] > self.pos_emb.shape[1]:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds teacher max positional length {self.pos_emb.shape[1]}")
        x = x + self.pos_emb[:, : x.shape[1]]
        if padding_mask is not None:
            cls_mask = torch.zeros((padding_mask.shape[0], 1), dtype=padding_mask.dtype, device=padding_mask.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1).bool()
        h = self.encoder(x, src_key_padding_mask=padding_mask)
        return h[:, 0]

    def forward(self, x: torch.Tensor, normalize: bool = True, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.encode(x, padding_mask=padding_mask)
        outs = []
        for head in self.heads:
            y = head(h)
            if normalize:
                y = F.normalize(y, dim=-1)
            outs.append(y)
        return torch.stack(outs, dim=1)  # [B, M, d_s]

    def augment(self, x: torch.Tensor, mask_p: float = 0.2, noise_std: float = 0.01) -> torch.Tensor:
        """随机增强：Masking + Jitter
        
        Args:
            x: [B, D] input features
            mask_p: probability of masking a feature (Dropout)
            noise_std: standard deviation of Gaussian noise
        """
        # 1. Random Masking (Dropout)
        mask = torch.rand_like(x) > mask_p
        x_aug = x * mask.float()
        
        # 2. Random Jitter (Gaussian Noise)
        noise = torch.randn_like(x) * noise_std
        x_aug = x_aug + noise
        
        return x_aug

    def forward_contrastive(
        self,
        x: torch.Tensor,
        temp: float = 0.1,
        mask_p: float = 0.2,
        noise_std: float = 0.01,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """A3 InfoNCE Loss Calculation
        
        Formula:
          u1 = F(aug1(x))
          u2 = F(aug2(x))
          L = InfoNCE(u1, u2)
        """
        # 1. Augmentation
        x1 = self.augment(x, mask_p=mask_p, noise_std=noise_std)
        x2 = self.augment(x, mask_p=mask_p, noise_std=noise_std)
        
        # 2. Forward & Normalize
        # output: [B, M, d_s] -> Flatten to [B, D_out] for global contrast
        # Or contrast per subspace? Usually global contrast for Teacher.
        # Let's flatten all heads to get a single vector per sample.
        u1 = self.forward(x1, normalize=True, padding_mask=padding_mask).reshape(x.size(0), -1)
        u2 = self.forward(x2, normalize=True, padding_mask=padding_mask).reshape(x.size(0), -1)
        
        # Re-normalize after flattening
        u1 = F.normalize(u1, dim=-1)
        u2 = F.normalize(u2, dim=-1)
        
        # 3. InfoNCE Loss
        # Cosine similarity: (B, B)
        sim_12 = torch.mm(u1, u2.t()) / temp
        sim_21 = torch.mm(u2, u1.t()) / temp
        
        # Labels: diagonal
        labels = torch.arange(x.size(0), device=x.device)
        
        loss_1 = F.cross_entropy(sim_12, labels)
        loss_2 = F.cross_entropy(sim_21, labels)
        
        return (loss_1 + loss_2) / 2.0
