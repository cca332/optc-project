
import torch
import torch.nn as nn
from typing import Optional

class SlotSemanticAggregator(nn.Module):
    """
    Step1 A3: Slot 级融合与时序编码
    
    A3.1: Slot 内语义聚合 (Attention Pooling)
    A3.2: 语义-统计融合 (Linear + LayerNorm)
    A3.3: Slot 序列时序编码 (Masked Temporal Transformer)
    """
    def __init__(self, 
                 semantic_dim: int, 
                 stat_dim: int,
                 fusion_dim: int,
                 num_slots: int = 10,
                 num_heads: int = 4,
                 num_layers: int = 1,
                 dropout: int = 0.1):
        super().__init__()
        self.num_slots = num_slots
        self.fusion_dim = fusion_dim
        
        # A3.1 Attention Pooling Params (q, W, b)
        # alpha_{k,j} = softmax( q^T * tanh(W * h_{k,j} + b) )
        self.attn_W = nn.Linear(semantic_dim, semantic_dim)
        self.attn_q = nn.Linear(semantic_dim, 1, bias=False)
        
        # A3.2 Fusion
        # s_k = LN( W_f * [u_sem || u_stat] + b_f )
        self.fusion_proj = nn.Linear(semantic_dim + stat_dim, fusion_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        
        # A3.3 Temporal Transformer
        # Positional Encoding p_k
        self.pos_emb = nn.Parameter(torch.randn(1, num_slots, fusion_dim) * 0.02)
        
        # Empty Slot Embedding e_empty (learnable)
        self.empty_emb = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, 
                h_sem: torch.Tensor,       # [B*K, L, D_sem] - Event embeddings
                h_stat: torch.Tensor,      # [B*K, D_stat] - Statistical embeddings
                event_mask: torch.Tensor,  # [B*K, L] - Valid event mask (1=valid, 0=pad)
                slot_mask: torch.Tensor    # [B, K] - Valid slot mask (1=non-empty, 0=empty)
               ) -> torch.Tensor:
        
        batch_size_k = h_sem.shape[0]
        # Recover B and K from B*K (assuming input is flattened)
        B = slot_mask.shape[0]
        K = self.num_slots
        assert batch_size_k == B * K
        
        # --- A3.1 Slot Internal Aggregation (Attention Pooling) ---
        # h_sem: [BK, L, D_sem]
        # W * h + b
        attn_scores = self.attn_W(h_sem) # [BK, L, D_sem]
        attn_scores = torch.tanh(attn_scores)
        # q^T * ...
        attn_scores = self.attn_q(attn_scores).squeeze(-1) # [BK, L]
        
        # Mask padding events (-inf)
        attn_scores = attn_scores.masked_fill(~event_mask.bool(), -1e9)
        
        # Softmax over L
        attn_weights = torch.softmax(attn_scores, dim=1) # [BK, L]
        
        # Weighted Sum
        # u_sem = sum( alpha * h )
        u_sem = torch.bmm(attn_weights.unsqueeze(1), h_sem).squeeze(1) # [BK, D_sem]
        
        # Handle case where all events are padding (empty slot) -> u_sem should be 0
        # event_mask.sum(1) == 0
        is_empty_slot_internal = (event_mask.sum(dim=1) == 0) # [BK]
        u_sem[is_empty_slot_internal] = 0.0
        
        # --- A3.2 Fusion (Semantic + Stat) ---
        # [BK, D_sem + D_stat]
        concat = torch.cat([u_sem, h_stat], dim=-1)
        s_k = self.fusion_proj(concat)
        s_k = self.fusion_norm(s_k) # [BK, D_fusion]
        
        # Reshape to [B, K, D]
        s_seq = s_k.view(B, K, -1)
        
        # Apply Empty Slot Embedding for truly empty slots (based on slot_mask)
        # if m_k == 0, s_k = e_empty
        # slot_mask: 1=valid, 0=empty
        # expand empty_emb to [B, K, D]
        empty_expanded = self.empty_emb.expand(B, K, -1)
        
        # Select: valid -> s_seq, empty -> empty_emb
        s_seq = torch.where(slot_mask.unsqueeze(-1).bool(), s_seq, empty_expanded)
        
        # --- A3.3 Temporal Transformer ---
        # Add Positional Encoding
        s_seq = s_seq + self.pos_emb # [B, K, D]
        
        # Temporal Encoding
        # No causal mask needed (bidirectional context is fine for offline detection), 
        # or causal mask if we want strict online simulation. 
        # Document says "masked temporal Transformer", usually implies padding mask.
        # Here we mask attention where slot_mask is 0? 
        # Actually, empty slots HAVE embeddings (e_empty), so they participate in attention.
        # But we might want to mask "future" if online?
        # Assuming offline detection (Step1 typically processes whole window), bidirectional is better.
        # But to be safe and robust to variable window length (if any), we can use src_key_padding_mask
        # Note: We keep empty slots in attention so context flows through gaps.
        
        s_encoded = self.temporal_encoder(s_seq) # [B, K, D]
        
        return s_encoded
