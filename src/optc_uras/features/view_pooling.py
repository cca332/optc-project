
import torch
import torch.nn as nn

class MaskedAttentionPooling(nn.Module):
    """
    Step1 A4: 视图摘要向量 (View-level Masked Attention Pooling)
    
    公式:
    a_k^(v) = (m_k * exp(u^T tanh(W s_k + b))) / sum(m_u * exp(...))
    s_bar^(v) = sum(a_k * s_k)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # Attention Params (u, W, b)
        self.attn_W = nn.Linear(input_dim, input_dim)
        self.attn_u = nn.Linear(input_dim, 1, bias=False)
        
        # Empty View Embedding (e_{view, empty})
        self.empty_view_emb = nn.Parameter(torch.zeros(1, input_dim))
        
    def forward(self, 
                slot_seqs: torch.Tensor,   # [B, K, D] - Contextual slot embeddings (from A3)
                slot_mask: torch.Tensor    # [B, K] - Valid slot mask (1=non-empty, 0=empty)
               ) -> torch.Tensor:
        """
        Args:
            slot_seqs: [B, K, D]
            slot_mask: [B, K]
        Returns:
            view_summary: [B, D]
        """
        # W * s + b
        attn_scores = self.attn_W(slot_seqs) # [B, K, D]
        attn_scores = torch.tanh(attn_scores)
        # u^T * ...
        attn_scores = self.attn_u(attn_scores).squeeze(-1) # [B, K]
        
        # Apply Mask: m_k * exp(...)
        # We implement this by setting masked scores to -inf before softmax
        # Note: slot_mask is 1 for non-empty slots.
        # If a slot is empty (mask=0), it should NOT contribute to view summary 
        # (unlike Temporal Transformer where it acts as a bridge).
        attn_scores = attn_scores.masked_fill(~slot_mask.bool(), -1e9)
        
        # Softmax over K
        attn_weights = torch.softmax(attn_scores, dim=1) # [B, K]
        
        # Weighted Sum
        # s_bar = sum( a_k * s_k )
        view_summary = torch.bmm(attn_weights.unsqueeze(1), slot_seqs).squeeze(1) # [B, D]
        
        # Handle completely empty view (sum(m_k) == 0)
        # If all slots are empty, view_summary should be e_{view, empty}
        is_empty_view = (slot_mask.sum(dim=1) == 0) # [B]
        
        # Expand empty embedding to batch
        empty_emb_expanded = self.empty_view_emb.expand(view_summary.shape[0], -1)
        
        # Select
        view_summary = torch.where(is_empty_view.unsqueeze(-1), empty_emb_expanded, view_summary)
        
        return view_summary
