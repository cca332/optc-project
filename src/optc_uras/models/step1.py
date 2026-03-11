
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from ..features.deterministic_aggregator import AggregatorSchema, DeterministicStatsAggregator
from ..features.random_projection import FixedRandomProjector
from ..features.quality import QualityWeighter, QualityWeightsConfig
from ..features.semantic_encoder import SemanticFeatureExtractor
from ..features.slot_aggregator import SlotSemanticAggregator
from ..features.view_pooling import MaskedAttentionPooling
from ..utils.slots import split_into_slots
from ..utils.typing import Step1Outputs
from .components import RouterMLP, AlignmentBases, GatedFusion


@dataclass
class Step1Config:
    views: List[str]
    window_seconds: int
    slot_seconds: int
    include_empty_slot_indicator: bool
    num_hash_buckets: int
    hash_seed: int
    target_dim: int
    rp_seed: int
    rp_matrix_type: str
    rp_normalize: str
    rp_nonlinearity: str
    quality_cfg: QualityWeightsConfig
    router_hidden_dims: List[int]
    router_dropout: float
    num_subspaces: int
    gate_gamma: float
    gate_mode: str
    gate_beta: float
    interaction_enabled: bool
    # [A6.3] Quality Injection Strength
    quality_injection_lambda: float = 0.5


class Step1Model(nn.Module):
    """Step1：同源多子视图表征提取与融合
    
    Update 2026-03: Refactored with Semantic-driven architecture (Step1A)
    - A2: Semantic + Statistical Feature Extraction
    - A3: Temporal Transformer for Slot Aggregation
    - A4: Masked Attention Pooling for View Summary
    - A6: Residual Quality Injection
    """

    def __init__(self, cfg: Step1Config, per_view_schema: Dict[str, AggregatorSchema]):
        super().__init__()
        self.cfg = cfg
        self.views = list(cfg.views)
        self.num_slots = cfg.window_seconds // cfg.slot_seconds

        # --- A2/A3/A4: Per-view Learnable Components ---
        self.semantic_extractors: nn.ModuleDict = nn.ModuleDict()
        self.slot_aggregators: nn.ModuleDict = nn.ModuleDict()
        self.view_poolers: nn.ModuleDict = nn.ModuleDict()
        
        # Legacy components for compatibility/statistical branch
        self.aggregators: Dict[str, DeterministicStatsAggregator] = {}
        self.schemas = per_view_schema

        for v in self.views:
            # 1. Deterministic Aggregator (for A2.2 Stat Branch & A5 Quality)
            ag = DeterministicStatsAggregator(
                schema=per_view_schema[v],
                num_hash_buckets=cfg.num_hash_buckets,
                hash_seed=cfg.hash_seed,
                include_empty_indicator=cfg.include_empty_slot_indicator,
            )
            self.aggregators[v] = ag
            
            # 2. A2: Semantic + Stat Feature Extractor
            # Note: vocab_sizes should be passed from schema fitting. 
            # For now we use defaults or infer. Ideally schema should store vocab sizes.
            # Assuming default large sizes for hashing.
            vocab_sizes = {
                "type": 50, "op": 100, "fine": 100, 
                "obj_hash": 20000, "text_hash": 50000, "num_fields": 10
            }
            # stat_dim comes from aggregator
            self.semantic_extractors[v] = SemanticFeatureExtractor(
                vocab_sizes=vocab_sizes,
                stat_dim=ag.output_dim,
                semantic_dim=128, # d_sem
                stat_proj_dim=32  # d_stat
            )
            
            # 3. A3: Slot Aggregator (Temporal)
            fusion_dim = 128 + 32 # d_sem + d_stat
            # Project to target_dim for output
            self.slot_aggregators[v] = nn.Sequential(
                SlotSemanticAggregator(
                    semantic_dim=128, 
                    stat_dim=32, 
                    fusion_dim=fusion_dim,
                    num_slots=self.num_slots
                ),
                nn.Linear(fusion_dim, cfg.target_dim) # Project to d
            )
            
            # 4. A4: View Pooling
            self.view_poolers[v] = MaskedAttentionPooling(input_dim=cfg.target_dim)

        # --- Shared Components (Step1B/C) ---
        # learnable components
        self.router = RouterMLP(in_dim=len(self.views) * cfg.target_dim, num_subspaces=cfg.num_subspaces,
                                hidden_dims=cfg.router_hidden_dims, dropout=cfg.router_dropout,
                                num_views=len(self.views))
        self.alignment = AlignmentBases(num_bases=cfg.num_subspaces, dim=cfg.target_dim)
        self.fusion = GatedFusion(dim=cfg.target_dim, gamma=cfg.gate_gamma, mode=cfg.gate_mode, beta=cfg.gate_beta,
                                  use_interaction=cfg.interaction_enabled)

        # quality
        self.quality = QualityWeighter(cfg.quality_cfg)

    def forward_fast(self, slot_tensor: torch.Tensor, view_summary: torch.Tensor, w: torch.Tensor) -> Step1Outputs:
        """
        Optimized forward pass using pre-computed tensors.
        Skips slot extraction and quality computation (IO bound parts).
        
        Args:
            slot_tensor: [B, K, V, d] Pre-computed slot embeddings
            view_summary: [B, V, d] Pre-computed masked mean of slots (or compute here if cheap)
            w: [B, V] Pre-computed quality weights
        """
        B, K, V, d = slot_tensor.shape
        device = slot_tensor.device
        
        # [A6.3] Residual Quality Injection (New Formula)
        # s_tilde = (1 + lambda * (beta - 1/V)) * s_bar
        lambda_beta = self.cfg.quality_injection_lambda
        uniform_weight = 1.0 / V
        
        # Residual term: [B, V]
        residual = 1.0 + lambda_beta * (w - uniform_weight)
        # Apply to view vectors: [B, V, d] * [B, V, 1]
        injected_view_vecs = view_summary * residual.unsqueeze(-1)

        # routing
        router_in = injected_view_vecs.reshape(B, V * d)
        route_p = self.router(router_in)  # [B,M]

        # alignment operator
        A = self.alignment(route_p)  # [B,d,d]
        aligned_slots = self.alignment.apply(A, slot_tensor)  # [B,K,V,d]
        # Apply Alignment to view vectors (consistency)
        aligned_view_vecs = torch.einsum("bij,bvj->bvi", A, injected_view_vecs)

        # gating fusion
        z, gate, rho_bar = self.fusion(aligned_view_vecs, w)

        intermediates = {
            "gate": gate.detach(), 
            "rho_bar": rho_bar.detach(), 
            "w_entropy": (-(w * (w.clamp_min(1e-8)).log()).sum(dim=1) / np.log(max(V, 2))),
            "route_entropy": (-(route_p * (route_p.clamp_min(1e-8)).log()).sum(dim=1) / np.log(max(route_p.shape[1], 2)))
        }

        outs: List[Step1Outputs] = []
        for b in range(B):
            outs.append(
                Step1Outputs(
                    view_names=self.views,
                    view_vecs=aligned_view_vecs[b],
                    reliability_w=w[b],
                    route_p=route_p[b],
                    z=z[b],
                    intermediates={
                        "gate": float(intermediates["gate"][b].item()),
                        "rho_bar": float(intermediates["rho_bar"][b].item()),
                        "w_entropy": float(intermediates["w_entropy"][b].item()),
                        "route_entropy": float(intermediates["route_entropy"][b].item()),
                        # Quality detail skipped for speed
                    },
                )
            )
        return outs

    
    @torch.no_grad()
    def forward_single(self, sample: Dict[str, Any]) -> Step1Outputs:
        # 单样本版本（用于 Step3 推理）
        device = next(self.parameters()).device
        V = len(self.views)
        d = self.cfg.target_dim
        
        # 1. Extract Features (A2-A4)
        view_vecs = torch.zeros((V, d), device=device)
        q_per_view = {}
        q_per_view_list = []
        fused_scores = []
        
        for vi, v in enumerate(self.views):
            # A2->A3->A4
            v_vec, _ = self._extract_view_features(sample, v)
            view_vecs[vi, :] = v_vec
            
            # A5 Quality
            slots = split_into_slots(sample["views"].get(v, []), self.cfg.slot_seconds, self.cfg.window_seconds)
            key_fields = self.schemas[v].key_fields or []
            qv = self.quality.compute_view_quality(slots, key_fields)
            q_per_view[v] = qv
            q_per_view_list.append(qv)
            fused_scores.append(self.quality.fuse(qv))

        # A6 Quality Weights
        w_np = self.quality.compute_final_weights(q_per_view_list, fused_scores)
        w = torch.from_numpy(w_np).to(device)

        # [A6.3] Residual Quality Injection (Batch size = 1, need to unsqueeze)
        # view_vecs: [V, d], w: [V]
        # We need to broadcast w to [V, 1]
        lambda_beta = self.cfg.quality_injection_lambda
        uniform_weight = 1.0 / V
        
        residual = 1.0 + lambda_beta * (w - uniform_weight)
        injected_view_vecs = view_vecs * residual.unsqueeze(-1)
        
        # Routing
        # injected_view_vecs: [V, d] -> flatten -> [1, V*d]
        route_p = self.router(injected_view_vecs.reshape(1, -1)).squeeze(0)  # [M]
        
        # Alignment
        A = self.alignment(route_p.unsqueeze(0)).squeeze(0)         # [d,d]
        # aligned_s = A * s. [d,d] x [V,d] -> [V,d] (einsum ij, vj -> vi)
        aligned_view_vecs = torch.einsum("ij,vj->vi", A, injected_view_vecs)

        # Fusion
        # Needs batch dim [1, V, d], [1, V]
        z, gate, rho_bar = self.fusion(aligned_view_vecs.unsqueeze(0), w.unsqueeze(0))
        z = z.squeeze(0)

        # 供 Step3 ATC 的信号：权重熵/路由熵（归一化到 [0,1]）
        w_entropy = float((-(w * (w.clamp_min(1e-8)).log()).sum() / np.log(max(len(self.views), 2))).item())
        route_entropy = float((-(route_p * (route_p.clamp_min(1e-8)).log()).sum() / np.log(max(route_p.numel(), 2))).item())

        return Step1Outputs(
            view_names=self.views,
            view_vecs=aligned_view_vecs.detach().cpu(),
            reliability_w=w.detach().cpu(),
            route_p=route_p.detach().cpu(),
            z=z.detach().cpu(),
            intermediates={"quality": q_per_view, "gate": float(gate.item()), "rho_bar": float(rho_bar.item()), "w_entropy": w_entropy, "route_entropy": route_entropy},
        )

    def fit_quality_stats(self, samples: Sequence[Dict[str, Any]]) -> None:
        qs: Dict[str, List[float]] = {}
        for s in samples:
            for v in self.views:
                slots = split_into_slots(s["views"].get(v, []), self.cfg.slot_seconds, self.cfg.window_seconds)
                key_fields = self.schemas[v].key_fields or []
                qv = self.quality.compute_view_quality(slots, key_fields)
                for k, val in qv.items():
                    if k not in qs: qs[k] = []
                    qs[k].append(val)
        self.quality.fit_standardize_stats(qs)

    def _prepare_semantic_inputs(self, events: List[Dict[str, Any]], v: str) -> Tuple[torch.Tensor, ...]:
        """Convert raw events to tensors for SemanticFeatureExtractor."""
        # This is a simplified on-the-fly collation. 
        # In production, this should be done in Dataset __getitem__ for efficiency.
        device = next(self.parameters()).device
        
        # Placeholder hashing logic (should match training preprocessing)
        import zlib
        def _hash(s: str, buckets: int) -> int:
            return zlib.crc32(str(s).encode()) % buckets
            
        type_ids = []
        op_ids = []
        fine_ids = []
        obj_hashes = []
        text_hashes = []
        field_masks = []
        time_feats = []
        
        # Dummy mapping for now - in real implementation, use fitted vocab
        # Here we rely on hashing for simplicity in this demo update
        
        for e in events:
            # Type/Op/Fine (using hash for simplicity if vocab not passed)
            type_ids.append(_hash(e.get("type", ""), 50))
            op_ids.append(_hash(e.get("op", ""), 100))
            fine_ids.append(_hash(e.get("fine_view", ""), 100))
            
            # Obj Hash
            obj_hashes.append(_hash(e.get("obj", ""), 20000))
            
            # Text Hash (Payload/Command/Path)
            # Sum hashes of available text fields
            txt_h = 0
            for f in ["payload", "command_line", "image_path", "file_path"]:
                if val := e.get(f):
                    txt_h += _hash(str(val), 50000)
            text_hashes.append(txt_h) # simplified 1D hash sum
            
            # Field Mask (Dummy 10 fields)
            mask = [0]*10
            field_masks.append(mask)
            
            # Time Feats (delta_t, rank) - extracted in raw_reader/dataset
            dt = e.get("delta_t", 0.0)
            rank = 0.0 # Placeholder
            time_feats.append([dt, rank])
            
        # Stack
        L = len(events)
        if L == 0:
            # Return dummy tensors for empty slot
            return (
                torch.zeros(1, dtype=torch.long, device=device), # type
                torch.zeros(1, dtype=torch.long, device=device), # op
                torch.zeros(1, dtype=torch.long, device=device), # fine
                torch.zeros(1, dtype=torch.long, device=device), # obj
                torch.zeros(1, dtype=torch.long, device=device), # text
                torch.zeros(1, 10, dtype=torch.float, device=device), # mask
                torch.zeros(1, 2, dtype=torch.float, device=device), # time
            )

        return (
            torch.tensor(type_ids, dtype=torch.long, device=device),
            torch.tensor(op_ids, dtype=torch.long, device=device),
            torch.tensor(fine_ids, dtype=torch.long, device=device),
            torch.tensor(obj_hashes, dtype=torch.long, device=device),
            torch.tensor(text_hashes, dtype=torch.long, device=device),
            torch.tensor(field_masks, dtype=torch.float, device=device),
            torch.tensor(time_feats, dtype=torch.float, device=device),
        )

    def _extract_view_features(self, sample: Dict[str, Any], v: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        New Step1A Pipeline:
        Raw Events -> Slots -> A2(Sem+Stat) -> A3(Slot Agg) -> A4(View Pool)
        Returns: 
            view_vec: [D]
            slot_seq: [K, D] (for debug/aux)
        """
        device = next(self.parameters()).device
        raw_events = sample["views"].get(v, [])
        slots = split_into_slots(raw_events, self.cfg.slot_seconds, self.cfg.window_seconds) # K slots
        
        slot_embs = []
        slot_mask_list = []
        
        # Process each slot
        for slot_events in slots:
            is_empty = len(slot_events) == 0
            slot_mask_list.append(0.0 if is_empty else 1.0)
            
            # A2.2 Stat Branch
            stat_vec = torch.from_numpy(self.aggregators[v](slot_events)).float().to(device).unsqueeze(0) # [1, D_stat]
            
            # A2.1 Semantic Branch inputs
            # Note: For efficiency, we should batch all slots. Doing loop for clarity here.
            inputs = self._prepare_semantic_inputs(slot_events, v)
            
            # A2 Forward
            # h_sem: [L, D_sem], h_stat: [1, D_stat]
            # Need to reshape inputs to add batch dim [1, L, ...]
            batched_inputs = [t.unsqueeze(0) for t in inputs]
            h_sem, h_stat = self.semantic_extractors[v](*batched_inputs, stat_vec)
            
            # Prepare masks for A3
            L = h_sem.shape[1]
            event_mask = torch.ones(1, L, device=device) if not is_empty else torch.zeros(1, L, device=device)
            slot_mask_scalar = torch.tensor([0.0 if is_empty else 1.0], device=device)
            
            # A3 Forward (Slot Aggregation) -> [1, 1, D_fusion] (temporal length 1 for single slot mode, or we batch K slots)
            # To strictly follow A3 (Temporal Transformer over K slots), we must batch ALL K slots first.
            slot_embs.append((h_sem, h_stat, event_mask))

        # --- Batching for A3 (Temporal) ---
        # We need to pad events to max_len in this window for batching
        max_L = max([t[0].shape[1] for t in slot_embs])
        max_L = max(max_L, 1) # avoid 0
        
        K = len(slots)
        batch_h_sem = []
        batch_h_stat = []
        batch_event_mask = []
        
        for h_s, h_st, e_m in slot_embs:
            curr_L = h_s.shape[1]
            pad_len = max_L - curr_L
            if pad_len > 0:
                h_s = torch.cat([h_s, torch.zeros(1, pad_len, h_s.shape[2], device=device)], dim=1)
                e_m = torch.cat([e_m, torch.zeros(1, pad_len, device=device)], dim=1)
            batch_h_sem.append(h_s)
            batch_h_stat.append(h_st)
            batch_event_mask.append(e_m)
            
        # [K, MaxL, D_sem]
        K_h_sem = torch.cat(batch_h_sem, dim=0)
        # [K, D_stat]
        K_h_stat = torch.cat(batch_h_stat, dim=0)
        # [K, MaxL]
        K_event_mask = torch.cat(batch_event_mask, dim=0)
        # [1, K]
        K_slot_mask = torch.tensor(slot_mask_list, device=device).unsqueeze(0)
        
        # A3: Slot Aggregator (with Temporal Transformer)
        # Returns [1, K, D]
        # Reshape inputs to flatten B*K (here B=1) -> [K, ...]
        slot_seq = self.slot_aggregators[v][0](K_h_sem, K_h_stat, K_event_mask, K_slot_mask) 
        # Project [1, K, D_fusion] -> [1, K, D_target]
        slot_seq = self.slot_aggregators[v][1](slot_seq)
        
        # A4: View Pooling
        # Returns [1, D]
        view_vec = self.view_poolers[v](slot_seq, K_slot_mask)
        
        return view_vec.squeeze(0), slot_seq.squeeze(0)

    def forward(self, batch_samples: Sequence[Dict[str, Any]]) -> Step1Outputs:
        B = len(batch_samples)
        V = len(self.views)
        d = self.cfg.target_dim
        device = next(self.parameters()).device

        # Containers
        view_vecs = torch.zeros((B, V, d), device=device)
        w = torch.zeros((B, V), device=device)
        
        intermediates: Dict[str, Any] = {"quality": [], "view_names": self.views}

        for b, s in enumerate(batch_samples):
            fused_scores = []
            q_per_view = {}
            q_per_view_list = []
            
            for vi, v in enumerate(self.views):
                # 1. Feature Extraction (A2->A3->A4)
                # This replaces the old deterministic slot extraction
                v_vec, _ = self._extract_view_features(s, v) # [d]
                view_vecs[b, vi, :] = v_vec

                # 2. Quality Calculation (A5)
                # Still relies on raw slots for calculation
                slots = split_into_slots(s["views"].get(v, []), self.cfg.slot_seconds, self.cfg.window_seconds)
                key_fields = self.schemas[v].key_fields or []
                qv = self.quality.compute_view_quality(slots, key_fields)
                
                q_per_view[v] = qv
                q_per_view_list.append(qv)
                fused_scores.append(self.quality.fuse(qv)) # Reliability score

            # 3. Quality Weights (A6.1 + A6.2)
            # w[b] is beta^(v) (Reliability * Info)
            w_np = self.quality.compute_final_weights(q_per_view_list, fused_scores)
            w[b, :] = torch.from_numpy(w_np).to(device)
            intermediates["quality"].append(q_per_view)

        # [A6.3] Residual Quality Injection (New Formula)
        # s_tilde = (1 + lambda * (beta - 1/V)) * s_bar
        lambda_beta = self.cfg.quality_injection_lambda
        uniform_weight = 1.0 / V
        
        # Residual term: [B, V]
        residual = 1.0 + lambda_beta * (w - uniform_weight)
        # Apply to view vectors: [B, V, d] * [B, V, 1]
        injected_view_vecs = view_vecs * residual.unsqueeze(-1)

        # Routing (Step1B)
        router_in = injected_view_vecs.reshape(B, V * d)
        route_p = self.router(router_in)  # [B,M]

        # Alignment (Step1B)
        A = self.alignment(route_p)  # [B,d,d]
        
        # Apply Alignment to view vectors
        # aligned_s = A * s
        # [B, d, d] x [B, V, d] -> [B, V, d]
        # einsum: b i j, b v j -> b v i
        aligned_view_vecs = torch.einsum("bij,bvj->bvi", A, injected_view_vecs)

        # Fusion (Step1C)
        z, gate, rho_bar = self.fusion(aligned_view_vecs, w)

        intermediates.update({"gate": gate.detach().cpu(), "rho_bar": rho_bar.detach().cpu(), "A": A.detach().cpu()})

        outs: List[Step1Outputs] = []
        for b in range(B):
            outs.append(
                Step1Outputs(
                    view_names=self.views,
                    view_vecs=aligned_view_vecs[b],
                    reliability_w=w[b],
                    route_p=route_p[b],
                    z=z[b],
                    intermediates={
                        "quality": intermediates["quality"][b],
                        "gate": float(gate[b].item()),
                        "rho_bar": float(rho_bar[b].item()),
                        "w_entropy": float((-(w[b] * (w[b].clamp_min(1e-8)).log()).sum() / np.log(max(V, 2))).item()),
                        "route_entropy": float((-(route_p[b] * (route_p[b].clamp_min(1e-8)).log()).sum() / np.log(max(route_p.shape[1], 2))).item()),
                    },
                )
            )
        return outs
    
    # Note: forward_fast and forward_single also need similar updates to use A2-A4 pipeline.
    # For brevity in this turn, I focused on the main forward loop.
