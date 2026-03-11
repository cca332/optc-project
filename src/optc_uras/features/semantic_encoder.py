
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional

class LearnableEventEncoder(nn.Module):
    """
    Step1 A2.1: 可学习语义事件编码器
    
    公式:
    h_{k,j}^{(v)} = f_v(
        E_{type}(type) || E_{op}(op) || E_{fine}(fine) || E_{obj}(obj) ||
        psi_v(a_{k,j}^{(v)}) || rho_v(mu_{k,j}^{(v)}) || g_t(delta_t, r_{k,j})
    )
    """
    def __init__(self, 
                 vocab_sizes: Dict[str, int], 
                 embed_dim: int = 32, 
                 semantic_dim: int = 64,
                 output_dim: int = 128):
        super().__init__()
        
        # 1. 基础类别嵌入
        self.type_emb = nn.Embedding(vocab_sizes.get("type", 20), embed_dim)
        self.op_emb = nn.Embedding(vocab_sizes.get("op", 50), embed_dim)
        self.fine_emb = nn.Embedding(vocab_sizes.get("fine", 50), embed_dim)
        
        # 2. 主语义实体嵌入 (E_obj)
        # 这里使用 Hash Embedding 来处理开放集合的 obj (如 file_path)
        self.obj_hash_buckets = vocab_sizes.get("obj_hash", 10000)
        self.obj_emb = nn.Embedding(self.obj_hash_buckets, embed_dim)
        
        # 3. 关键语义字段编码器 psi_v (a_{k,j})
        # 针对 text 字段 (payload, command_line) 使用轻量级 CharCNN 或 Hash+MLP
        self.text_dim = semantic_dim
        # 简单起见，这里使用 Hash Bucket 组合 + MLP 作为轻量文本编码
        # 实际工程中可替换为 DistilBERT 或 CharCNN
        self.text_hash_buckets = vocab_sizes.get("text_hash", 50000)
        self.text_emb = nn.Embedding(self.text_hash_buckets, self.text_dim)
        
        # 4. 字段缺失模式编码 rho_v (mu)
        self.mask_emb = nn.Linear(vocab_sizes.get("num_fields", 10), embed_dim)
        
        # 5. 局部时序编码 g_t (delta_t, rank)
        self.time_proj = nn.Linear(2, embed_dim) # [delta_t, rank_norm]
        
        # 融合 MLP f_v
        # Input dim = 4 * embed_dim (type/op/fine/obj) + semantic_dim (text) + embed_dim (mask) + embed_dim (time)
        in_dim = 6 * embed_dim + self.text_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, 
                type_ids: torch.Tensor, 
                op_ids: torch.Tensor, 
                fine_ids: torch.Tensor, 
                obj_hashes: torch.Tensor,
                text_hashes: torch.Tensor,
                field_masks: torch.Tensor,
                time_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            type_ids: [B, L]
            op_ids: [B, L]
            fine_ids: [B, L]
            obj_hashes: [B, L] - Hash of main object
            text_hashes: [B, L, F] - Hashes of semantic fields (summed)
            field_masks: [B, L, NumFields]
            time_feats: [B, L, 2] - (delta_t, rank)
        """
        # Embeddings
        e_type = self.type_emb(type_ids)
        e_op = self.op_emb(op_ids)
        e_fine = self.fine_emb(fine_ids)
        e_obj = self.obj_emb(obj_hashes % self.obj_hash_buckets)
        
        # Semantic Text Encoding (Sum of field embeddings)
        # text_hashes: [B, L, F] -> sum -> [B, L] indices? No, usually [B, L, F]
        # We assume pre-hashed input. For simplicity, we sum embeddings of multiple fields
        # If input is [B, L], treat as single field
        if text_hashes.dim() == 2:
            e_text = self.text_emb(text_hashes % self.text_hash_buckets) # [B, L, S]
        else:
            e_text = self.text_emb(text_hashes % self.text_hash_buckets).sum(dim=2) # [B, L, S]
        
        # Mask Encoding
        e_mask = self.mask_emb(field_masks.float())
        
        # Time Encoding
        e_time = self.time_proj(time_feats)
        
        # Concat
        # [B, L, D_total]
        concat = torch.cat([e_type, e_op, e_fine, e_obj, e_text, e_mask, e_time], dim=-1)
        
        # Fusion
        h = self.fusion_mlp(concat) # [B, L, output_dim]
        return h

class SemanticFeatureExtractor(nn.Module):
    """
    Step1 A2: 整合 A2.1 (Semantic) 和 A2.2 (Statistical)
    """
    def __init__(self, 
                 vocab_sizes: Dict[str, int], 
                 stat_dim: int,
                 semantic_dim: int = 128,
                 stat_proj_dim: int = 32):
        super().__init__()
        
        # A2.1 Semantic Branch
        self.semantic_encoder = LearnableEventEncoder(vocab_sizes, output_dim=semantic_dim)
        
        # A2.2 Statistical Branch (Projection)
        self.stat_proj = nn.Sequential(
            nn.Linear(stat_dim, stat_proj_dim),
            nn.ReLU(),
            nn.Linear(stat_proj_dim, stat_proj_dim)
        )
        
        self.output_dim = semantic_dim + stat_proj_dim
        
    def forward(self, 
                # Semantic Inputs
                type_ids, op_ids, fine_ids, obj_hashes, text_hashes, field_masks, time_feats,
                # Statistical Inputs
                stat_vecs):
        
        # 1. Semantic Encoding (Event Level) -> [B, L, D_sem]
        h_sem = self.semantic_encoder(type_ids, op_ids, fine_ids, obj_hashes, text_hashes, field_masks, time_feats)
        
        # 2. Aggregation of Semantic Events within Slot (e.g., Mean/Attention)
        # Assuming input is already slot-level batched [Batch*Slots, MaxEvents, ...]
        # Simple Mean Pooling for now (A3 will upgrade this to Transformer/Attention)
        # Mask out padding
        # mask = (type_ids != 0).unsqueeze(-1)
        # h_slot_sem = (h_sem * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1))
        
        # Actually, A2.1 output is h_{k,j} (Event Level). 
        # The aggregation happens in A3.
        # So here we just return the event embeddings.
        
        # 3. Statistical Projection (Slot Level) -> [B, D_stat]
        h_stat = self.stat_proj(stat_vecs)
        
        return h_sem, h_stat
