
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DeepReconstructionDetector(nn.Module):
    """
    4-token reconstruction detector:
    - 3 view tokens: process / file / network
    - 1 global token: projected student-side global feature
    """

    def __init__(
        self,
        view_dim: int,
        num_views: int,
        global_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        mask_ratio: float = 0.25,
    ):
        super().__init__()
        self.view_dim = int(view_dim)
        self.num_views = int(num_views)
        self.global_dim = int(global_dim)
        self.num_tokens = self.num_views + 1
        self.mask_ratio = float(mask_ratio)

        self.global_projector = nn.Sequential(
            nn.Linear(self.global_dim, self.view_dim),
            nn.LayerNorm(self.view_dim),
        )
        self.token_type = nn.Parameter(torch.randn(self.num_tokens, self.view_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.view_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.view_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reconstructor = nn.Sequential(
            nn.LayerNorm(self.view_dim),
            nn.Linear(self.view_dim, self.view_dim),
        )

        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("mean_error", torch.tensor(0.0))
        self.register_buffer("std_error", torch.tensor(1.0))
        self.register_buffer("view_mean_error", torch.zeros(self.num_views))
        self.register_buffer("view_std_error", torch.ones(self.num_views))
        self.register_buffer("global_mean_error", torch.tensor(0.0))
        self.register_buffer("global_std_error", torch.tensor(1.0))
        default_weights = torch.ones(self.num_tokens) / float(self.num_tokens)
        self.register_buffer("score_weights", default_weights)

    def build_tokens(self, view_tokens: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        global_token = self.global_projector(global_features).unsqueeze(1)
        tokens = torch.cat([view_tokens, global_token], dim=1)
        return tokens + self.token_type.unsqueeze(0)

    def make_mask(self, batch_size: int, device: torch.device, force_global: bool = False) -> torch.Tensor:
        mask = torch.rand(batch_size, self.num_tokens, device=device) < self.mask_ratio
        if force_global:
            mask[:, -1] = True
        empty_rows = mask.sum(dim=1) == 0
        if empty_rows.any():
            mask[empty_rows, -1] = True
        return mask

    def forward(
        self,
        view_tokens: torch.Tensor,
        global_features: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        tokens = self.build_tokens(view_tokens, global_features)
        if token_mask is None:
            token_mask = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        masked_tokens = torch.where(token_mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        latent = self.transformer(masked_tokens)
        recon = self.reconstructor(latent)
        return {"tokens": tokens, "recon": recon, "mask": token_mask}

    def reconstruction_loss(self, view_tokens: torch.Tensor, global_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask = self.make_mask(view_tokens.shape[0], view_tokens.device, force_global=True)
        out = self.forward(view_tokens, global_features, token_mask=mask)
        token_mse = F.mse_loss(out["recon"], out["tokens"], reduction="none").mean(dim=-1)

        masked_loss = (token_mse * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        unmasked_loss = (token_mse * (~mask).float()).sum() / (~mask).float().sum().clamp_min(1.0)
        total_loss = masked_loss + 0.25 * unmasked_loss
        return {
            "loss": total_loss,
            "masked_loss": masked_loss,
            "unmasked_loss": unmasked_loss,
            "token_mse": token_mse,
        }

    def compute_score(self, view_tokens: torch.Tensor, global_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.forward(view_tokens, global_features)
        token_mse = F.mse_loss(out["recon"], out["tokens"], reduction="none").mean(dim=-1)
        per_view_mse = token_mse[:, :self.num_views]
        global_mse = token_mse[:, -1]
        view_z = (per_view_mse - self.view_mean_error.unsqueeze(0)) / (self.view_std_error.unsqueeze(0) + 1e-6)
        global_z = (global_mse - self.global_mean_error) / (self.global_std_error + 1e-6)
        token_z = torch.cat([view_z, global_z.unsqueeze(1)], dim=1)
        view_mean_score = per_view_mse.mean(dim=-1)
        norm_score = (view_mean_score - self.mean_error) / (self.std_error + 1e-6)
        return {
            "score": view_mean_score,
            "norm_score": norm_score,
            "per_view_score": per_view_mse,
            "global_score": global_mse,
            "token_score": token_mse,
            "per_view_z": view_z,
            "global_z": global_z,
            "token_z": token_z,
            "recon": out["recon"],
        }

    def fit_stats(self, train_stats: Dict[str, torch.Tensor], quantile: float = 0.99):
        per_view_mse = train_stats["per_view_score"]
        global_mse = train_stats["global_score"]
        scores = train_stats["score"]

        self.view_mean_error = per_view_mse.mean(dim=0)
        self.view_std_error = per_view_mse.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.global_mean_error = global_mse.mean()
        self.global_std_error = global_mse.std(unbiased=False).clamp_min(1e-6)

        view_z = (per_view_mse - self.view_mean_error.unsqueeze(0)) / (self.view_std_error.unsqueeze(0) + 1e-6)
        global_z = (global_mse - self.global_mean_error) / (self.global_std_error + 1e-6)
        token_z = torch.cat([view_z, global_z.unsqueeze(1)], dim=1)
        token_strength = torch.relu(token_z).mean(dim=0) + 1e-6
        self.score_weights = token_strength / token_strength.sum()

        self.mean_error = scores.mean()
        self.std_error = scores.std(unbiased=False).clamp_min(1e-6)
        self.threshold = torch.quantile(scores, quantile)
        print(
            f"[DeepDetector] Stats Fitted: Mean={self.mean_error:.4f}, "
            f"Std={self.std_error:.4f}, Threshold({quantile})={self.threshold:.4f}, "
            f"Weights={self.score_weights.detach().cpu().tolist()}"
        )
