from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..features.deterministic_aggregator import AggregatorSchema, DeterministicStatsAggregator
from ..features.random_projection import FixedRandomProjector
from ..features.quality import QualityWeighter, QualityWeightsConfig
from ..features.slots import split_into_slots
from ..typing import Step1Outputs
from .routing import RouterMLP
from .alignment import AlignmentBases
from .fusion import GatedFusion


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


class Step1Model(nn.Module):
    """Step1：同源多子视图表征提取与融合

    确定性部分：
      - slots 划分
      - 确定性聚合器
      - 固定随机投影

    可学习部分（Step2 联邦训练会更新）：
      - 路由器
      - 对齐基矩阵
      - 注意力交互（可选）
    """

    def __init__(self, cfg: Step1Config, per_view_schema: Dict[str, AggregatorSchema]):
        super().__init__()
        self.cfg = cfg
        self.views = list(cfg.views)

        # per-view deterministic components
        self.aggregators: Dict[str, DeterministicStatsAggregator] = {}
        self.projectors: Dict[str, FixedRandomProjector] = {}
        self.schemas = per_view_schema

        for v in self.views:
            ag = DeterministicStatsAggregator(
                schema=per_view_schema[v],
                num_hash_buckets=cfg.num_hash_buckets,
                hash_seed=cfg.hash_seed,
                include_empty_indicator=cfg.include_empty_slot_indicator,
            )
            self.aggregators[v] = ag
            # projector input dim 取决于 vocab，要在 fit_vocab 后初始化
        self._projectors_ready = False

        # learnable components
        self.router = RouterMLP(in_dim=len(self.views) * cfg.target_dim, num_subspaces=cfg.num_subspaces,
                                hidden_dims=cfg.router_hidden_dims, dropout=cfg.router_dropout,
                                num_views=len(self.views))
        self.alignment = AlignmentBases(num_bases=cfg.num_subspaces, dim=cfg.target_dim)
        self.fusion = GatedFusion(dim=cfg.target_dim, gamma=cfg.gate_gamma, mode=cfg.gate_mode, beta=cfg.gate_beta,
                                  use_interaction=cfg.interaction_enabled)

        # quality
        self.quality = QualityWeighter(cfg.quality_cfg)

    def fit_vocabs_and_init_projectors(self, samples: Sequence[Dict[str, Any]], max_types: int = 128) -> None:
        # 扫描每个 view 的事件类型 vocab，并初始化固定随机投影
        for v in self.views:
            all_events = []
            for s in samples:
                all_events.extend(s["views"].get(v, []))
            ag = self.aggregators[v]
            if not ag.event_type_vocab:
                ag.fit_event_type_vocab(all_events, max_types=max_types)
            self.projectors[v] = FixedRandomProjector(
                in_dim=ag.output_dim,
                out_dim=self.cfg.target_dim,
                seed=self.cfg.rp_seed + (abs(hash(v)) % 997),
                matrix_type=self.cfg.rp_matrix_type,
                normalize_mode=self.cfg.rp_normalize,
                nonlinearity=self.cfg.rp_nonlinearity,
            )
        self._projectors_ready = True

    def fit_quality_stats(self, samples: Sequence[Dict[str, Any]]) -> None:
        # 仅用于 quality.standardize=train_stats 的离线统计
        qs: Dict[str, List[float]] = {}
        
        for s in samples:
            for v in self.views:
                slots = split_into_slots(s["views"].get(v, []), self.cfg.slot_seconds, self.cfg.window_seconds)
                key_fields = self.schemas[v].key_fields or []
                qv = self.quality.compute_view_quality(slots, key_fields)
                
                # Dynamically collect all keys present in qv
                for k, val in qv.items():
                    if k not in qs:
                        qs[k] = []
                    qs[k].append(val)
                    
        self.quality.fit_standardize_stats(qs)

    def _extract_view_slots(self, sample: Dict[str, Any], v: str) -> np.ndarray:
        assert self._projectors_ready, "call fit_vocabs_and_init_projectors() first"
        slots = split_into_slots(sample["views"].get(v, []), self.cfg.slot_seconds, self.cfg.window_seconds)  # K
        ag = self.aggregators[v]
        proj = self.projectors[v]
        slot_vecs = np.stack([ag(se) for se in slots], axis=0)  # [K, dv]
        slot_emb = proj(slot_vecs)  # [K, d]
        return slot_emb.astype(np.float32), slots

    def forward(self, batch_samples: Sequence[Dict[str, Any]]) -> Step1Outputs:
        B = len(batch_samples)
        V = len(self.views)
        d = self.cfg.target_dim
        K = self.cfg.window_seconds // self.cfg.slot_seconds

        # deterministic extraction: slots -> embeddings
        slot_tensor = torch.zeros((B, K, V, d), dtype=torch.float32, device=next(self.parameters()).device)
        view_summary = torch.zeros((B, V, d), dtype=torch.float32, device=slot_tensor.device)
        w = torch.zeros((B, V), dtype=torch.float32, device=slot_tensor.device)

        intermediates: Dict[str, Any] = {"quality": [], "view_names": self.views}

        for b, s in enumerate(batch_samples):
            fused_scores = []
            q_per_view = {}
            q_per_view_list = []
            for vi, v in enumerate(self.views):
                slot_emb, slots_raw = self._extract_view_slots(s, v)  # [K,d]
                slot_tensor[b, :, vi, :] = torch.from_numpy(slot_emb).to(slot_tensor.device)

                # [FIX] Masked Mean Pooling: Ignore empty slots to prevent feature collapse
                # If events are truncated (e.g. first 20k), later slots become empty.
                # Since slots are L2-normalized, empty slots become constant unit vectors.
                # Averaging them dilutes the real signal. We must mask them out.
                has_events = [len(se) > 0 for se in slots_raw]
                if any(has_events):
                    mask = torch.tensor(has_events, dtype=torch.float32, device=slot_tensor.device).unsqueeze(1)  # [K, 1]
                    view_summary[b, vi, :] = (slot_tensor[b, :, vi, :] * mask).sum(dim=0) / mask.sum()
                else:
                    # Fallback for completely empty view
                    view_summary[b, vi, :] = slot_tensor[b, :, vi, :].mean(dim=0)

                # quality (A5/A6)
                key_fields = self.schemas[v].key_fields or []
                qv = self.quality.compute_view_quality(slots_raw, key_fields)
                q_per_view[v] = qv
                q_per_view_list.append(qv)
                fused_scores.append(self.quality.fuse(qv))

            w_np = self.quality.compute_final_weights(q_per_view_list, fused_scores)
            w[b, :] = torch.from_numpy(w_np).to(slot_tensor.device)
            intermediates["quality"].append(q_per_view)

        # inject reliability into view vecs (scale)
        view_vecs = view_summary * w.unsqueeze(-1)

        # routing
        router_in = view_vecs.reshape(B, V * d)
        route_p = self.router(router_in)  # [B,M]

        # alignment operator
        A = self.alignment(route_p)  # [B,d,d]
        aligned_slots = self.alignment.apply(A, slot_tensor)  # [B,K,V,d]
        aligned_view_vecs = aligned_slots.mean(dim=1)  # [B,V,d]

        # gating fusion
        z, gate, rho_bar = self.fusion(aligned_view_vecs, w)

        intermediates.update({"gate": gate.detach().cpu(), "rho_bar": rho_bar.detach().cpu(), "A": A.detach().cpu()})

        # 返回 batch 内每个样本的 Step1Outputs（逐样本契约）
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

    @torch.no_grad()
    def forward_single(self, sample: Dict[str, Any]) -> Step1Outputs:
        # 单样本版本（用于 Step3 推理）
        device = next(self.parameters()).device
        V = len(self.views)
        d = self.cfg.target_dim
        K = self.cfg.window_seconds // self.cfg.slot_seconds

        slots_emb = []
        fused_scores = []
        q_per_view = {}
        q_per_view_list = []
        for v in self.views:
            slot_emb, slots_raw = self._extract_view_slots(sample, v)  # [K,d]
            slots_emb.append(torch.from_numpy(slot_emb))
            key_fields = self.schemas[v].key_fields or []
            qv = self.quality.compute_view_quality(slots_raw, key_fields)
            q_per_view[v] = qv
            q_per_view_list.append(qv)
            fused_scores.append(self.quality.fuse(qv))

        w_np = self.quality.compute_final_weights(q_per_view_list, fused_scores)
        w = torch.from_numpy(w_np).to(device)

        slots_tensor = torch.stack(slots_emb, dim=1).to(device)  # [K,V,d]
        
        # [FIX] Masked Mean Pooling for Inference (Align with forward)
        # Calculate mask per view based on raw events
        view_vecs_list = []
        for vi, v in enumerate(self.views):
            # Re-check empty slots from raw extraction
            _, slots_raw = self._extract_view_slots(sample, v)
            has_events = [len(se) > 0 for se in slots_raw]
            
            # Get slots for this view: [K, d]
            v_slots = slots_tensor[:, vi, :]
            
            if any(has_events):
                mask = torch.tensor(has_events, dtype=torch.float32, device=device).unsqueeze(1) # [K, 1]
                # Weighted sum by mask / count of valid slots
                v_vec = (v_slots * mask).sum(dim=0) / mask.sum()
            else:
                # Fallback: simple mean if all empty
                v_vec = v_slots.mean(dim=0)
            view_vecs_list.append(v_vec)
            
        view_vecs0 = torch.stack(view_vecs_list, dim=0) # [V, d]
        view_vecs = view_vecs0 * w.unsqueeze(-1)

        route_p = self.router(view_vecs.reshape(1, -1)).squeeze(0)  # [M]
        A = self.alignment(route_p.unsqueeze(0)).squeeze(0)         # [d,d]
        aligned_slots = torch.einsum("ij,kvj->kvi", A, slots_tensor)  # [K,V,d]
        aligned_view_vecs = aligned_slots.mean(dim=0)  # [V,d]

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