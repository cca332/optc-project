from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventMaskedReconstructionDetector(nn.Module):
    """Event-level masked reconstruction detector."""

    def __init__(
        self,
        event_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        mask_ratio: float = 0.2,
        topk_fraction: float = 0.1,
        chunk_size: int = 128,
        inference_mask_batch: int = 32,
    ):
        super().__init__()
        self.event_dim = int(event_dim)
        self.mask_ratio = float(mask_ratio)
        self.topk_fraction = float(topk_fraction)
        self.chunk_size = int(chunk_size)
        self.inference_mask_batch = int(inference_mask_batch)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.event_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.event_dim,
            nhead=int(num_heads),
            dim_feedforward=self.event_dim * 4,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.reconstructor = nn.Sequential(
            nn.LayerNorm(self.event_dim),
            nn.Linear(self.event_dim, self.event_dim),
        )

        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("mean_score", torch.tensor(0.0))
        self.register_buffer("std_score", torch.tensor(1.0))

    def make_random_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        rand = torch.rand_like(valid_mask.float())
        mask = (rand < self.mask_ratio) & valid_mask
        empty_rows = mask.sum(dim=1) == 0
        if empty_rows.any():
            first_valid = valid_mask.float().argmax(dim=1)
            for b in torch.nonzero(empty_rows, as_tuple=False).flatten().tolist():
                if valid_mask[b].any():
                    mask[b, int(first_valid[b].item())] = True
        return mask

    def _forward_chunk(
        self,
        event_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if mask_positions is None:
            mask_positions = torch.zeros_like(valid_mask)
        masked_inputs = torch.where(mask_positions.unsqueeze(-1), self.mask_token.expand_as(event_embeddings), event_embeddings)
        latent = self.encoder(masked_inputs, src_key_padding_mask=~valid_mask)
        recon = self.reconstructor(latent)
        return {"recon": recon, "mask_positions": mask_positions}

    def forward(
        self,
        event_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        seq_len = event_embeddings.shape[1]
        if seq_len <= self.chunk_size:
            return self._forward_chunk(event_embeddings, valid_mask, mask_positions=mask_positions)

        if mask_positions is None:
            mask_positions = torch.zeros_like(valid_mask)
        recon_chunks = []
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            out = self._forward_chunk(
                event_embeddings[:, start:end],
                valid_mask[:, start:end],
                mask_positions=mask_positions[:, start:end],
            )
            recon_chunks.append(out["recon"])
        return {"recon": torch.cat(recon_chunks, dim=1), "mask_positions": mask_positions}

    def reconstruction_loss(self, event_embeddings: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask_positions = self.make_random_mask(valid_mask)
        out = self.forward(event_embeddings, valid_mask, mask_positions=mask_positions)
        event_mse = F.mse_loss(out["recon"], event_embeddings, reduction="none").mean(dim=-1)

        masked_loss = (event_mse * mask_positions.float()).sum() / mask_positions.float().sum().clamp_min(1.0)
        visible = valid_mask & (~mask_positions)
        visible_loss = (event_mse * visible.float()).sum() / visible.float().sum().clamp_min(1.0)
        total_loss = masked_loss + 0.1 * visible_loss
        return {
            "loss": total_loss,
            "masked_loss": masked_loss,
            "visible_loss": visible_loss,
            "event_mse": event_mse,
        }

    def compute_event_scores(
        self,
        event_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        leave_one_out: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = event_embeddings.shape
        per_event_score = torch.zeros((batch_size, seq_len), device=event_embeddings.device)

        if leave_one_out:
            for start in range(0, seq_len, self.chunk_size):
                end = min(start + self.chunk_size, seq_len)
                chunk_embeddings = event_embeddings[:, start:end]
                chunk_valid = valid_mask[:, start:end]
                chunk_len = end - start
                for b in range(batch_size):
                    valid_positions = torch.nonzero(chunk_valid[b], as_tuple=False).flatten().tolist()
                    if not valid_positions:
                        continue
                    for mask_start in range(0, len(valid_positions), self.inference_mask_batch):
                        pos_group = valid_positions[mask_start: mask_start + self.inference_mask_batch]
                        emb_batch = chunk_embeddings[b:b+1].repeat(len(pos_group), 1, 1)
                        valid_batch = chunk_valid[b:b+1].repeat(len(pos_group), 1)
                        mask_batch = torch.zeros_like(valid_batch)
                        for row_idx, pos in enumerate(pos_group):
                            mask_batch[row_idx, pos] = True
                        out = self._forward_chunk(emb_batch, valid_batch, mask_positions=mask_batch)
                        event_mse = F.mse_loss(out["recon"], emb_batch, reduction="none").mean(dim=-1)
                        for row_idx, pos in enumerate(pos_group):
                            per_event_score[b, start + pos] = event_mse[row_idx, pos]
        else:
            out = self.forward(event_embeddings, valid_mask)
            per_event_score = F.mse_loss(out["recon"], event_embeddings, reduction="none").mean(dim=-1)

        per_event_score = per_event_score * valid_mask.float()
        window_scores = []
        for b in range(batch_size):
            valid_scores = per_event_score[b][valid_mask[b]]
            if valid_scores.numel() == 0:
                window_scores.append(torch.tensor(0.0, device=event_embeddings.device))
                continue
            k = max(1, int(valid_scores.numel() * self.topk_fraction))
            topk_scores = torch.topk(valid_scores, k=min(k, valid_scores.numel())).values
            window_scores.append(topk_scores.mean())

        window_score = torch.stack(window_scores, dim=0)
        norm_score = (window_score - self.mean_score) / (self.std_score + 1e-6)
        return {
            "score": window_score,
            "norm_score": norm_score,
            "per_event_score": per_event_score,
        }

    def fit_stats(self, scores: torch.Tensor, quantile: float = 0.99) -> None:
        self.mean_score = scores.mean()
        self.std_score = scores.std(unbiased=False).clamp_min(1e-6)
        self.threshold = torch.quantile(scores, quantile)
