from __future__ import annotations

import zlib
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventWindowEncoder(nn.Module):
    """Encode full event sequences for each host-window sample."""

    def __init__(
        self,
        event_dim: int = 128,
        num_subspaces: int = 4,
        max_events: int = 512,
        type_buckets: int = 128,
        op_buckets: int = 256,
        fine_buckets: int = 256,
        obj_buckets: int = 50000,
        text_buckets: int = 50000,
        host_buckets: int = 4096,
        field_count: int = 12,
        num_views: int = 3,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.event_dim = int(event_dim)
        self.num_subspaces = int(num_subspaces)
        self.max_events = int(max_events)
        self.type_buckets = int(type_buckets)
        self.op_buckets = int(op_buckets)
        self.fine_buckets = int(fine_buckets)
        self.obj_buckets = int(obj_buckets)
        self.text_buckets = int(text_buckets)
        self.host_buckets = int(host_buckets)
        self.field_count = int(field_count)
        self.num_views = int(num_views)

        small_dim = max(16, self.event_dim // 4)
        text_dim = max(32, self.event_dim // 2)
        self.type_emb = nn.Embedding(self.type_buckets, small_dim)
        self.op_emb = nn.Embedding(self.op_buckets, small_dim)
        self.fine_emb = nn.Embedding(self.fine_buckets, small_dim)
        self.obj_emb = nn.Embedding(self.obj_buckets, small_dim)
        self.text_emb = nn.Embedding(self.text_buckets, text_dim)
        self.view_emb = nn.Embedding(self.num_views, small_dim)
        self.host_emb = nn.Embedding(self.host_buckets, small_dim)
        self.field_mask_proj = nn.Linear(self.field_count, small_dim)
        self.time_proj = nn.Linear(3, small_dim)

        in_dim = small_dim * 8 + text_dim
        self.event_proj = nn.Sequential(
            nn.Linear(in_dim, self.event_dim * 2),
            nn.LayerNorm(self.event_dim * 2),
            nn.GELU(),
            nn.Linear(self.event_dim * 2, self.event_dim),
            nn.LayerNorm(self.event_dim),
        )

        self.route_feature_dim = 8
        self.router = nn.Sequential(
            nn.Linear(self.num_views * self.route_feature_dim, self.event_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.event_dim, self.num_subspaces),
        )
        self.alignment_bases = nn.Parameter(torch.randn(self.num_subspaces, self.event_dim, self.event_dim) * 0.02)

        self.pos_emb = nn.Parameter(torch.randn(1, self.max_events, self.event_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.event_dim,
            nhead=int(num_heads),
            dim_feedforward=self.event_dim * 4,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))

    @staticmethod
    def _hash_token(value: Any, buckets: int) -> int:
        return zlib.crc32(str(value).encode("utf-8")) % int(buckets)

    def _event_to_features(self, event: Dict[str, Any], view_id: int, host: str, rank: int, prev_ts: float) -> Dict[str, Any]:
        ts = float(event.get("ts", 0.0) or 0.0)
        delta_t = float(event.get("delta_t", ts - prev_ts if rank > 0 else ts) or 0.0)
        text_parts = [
            event.get("image_path", ""),
            event.get("parent_image_path", ""),
            event.get("command_line", ""),
            event.get("payload", ""),
            event.get("file_path", ""),
            event.get("src_ip", ""),
            event.get("dest_ip", ""),
            event.get("direction", ""),
            event.get("l4protocol", ""),
            event.get("hostname", ""),
        ]
        text_hash = 0
        for part in text_parts:
            if part:
                text_hash = (text_hash + self._hash_token(part, self.text_buckets)) % self.text_buckets

        field_mask = [
            float(bool(event.get("type"))),
            float(bool(event.get("op"))),
            float(bool(event.get("obj"))),
            float(bool(event.get("fine_view"))),
            float(bool(event.get("image_path"))),
            float(bool(event.get("parent_image_path"))),
            float(bool(event.get("command_line"))),
            float(bool(event.get("payload"))),
            float(bool(event.get("file_path"))),
            float(bool(event.get("src_ip") or event.get("dest_ip"))),
            float(bool(event.get("src_port") or event.get("dest_port"))),
            float(bool(event.get("valid", 1))),
        ]

        return {
            "type_id": self._hash_token(event.get("type", ""), self.type_buckets),
            "op_id": self._hash_token(event.get("op", ""), self.op_buckets),
            "fine_id": self._hash_token(event.get("fine_view", event.get("view", "")), self.fine_buckets),
            "obj_id": self._hash_token(event.get("obj", ""), self.obj_buckets),
            "text_id": text_hash,
            "view_id": view_id,
            "host_id": self._hash_token(host, self.host_buckets),
            "time": [ts / 300.0, delta_t / 300.0, float(rank) / max(1.0, float(self.max_events - 1))],
            "field_mask": field_mask[: self.field_count],
            "meta": {
                "view_id": view_id,
                "view_name": ["process", "file", "network"][view_id],
                "ts": ts,
                "type": event.get("type", ""),
                "op": event.get("op", ""),
                "obj": event.get("obj", ""),
            },
        }

    def _build_route_stats(self, sample: Dict[str, Any], events: List[Dict[str, Any]]) -> torch.Tensor:
        route_stats = torch.zeros(self.num_views, self.route_feature_dim, dtype=torch.float32)
        for view_id in range(self.num_views):
            view_events = [e for e in events if e["view_id"] == view_id]
            if view_events:
                ts_vals = torch.tensor([e["meta"]["ts"] for e in view_events], dtype=torch.float32)
                valid_vals = torch.tensor([e["field_mask"][-1] for e in view_events], dtype=torch.float32)
                route_stats[view_id, 0] = float(len(view_events)) / max(1.0, float(len(events)))
                route_stats[view_id, 1] = valid_vals.mean()
                route_stats[view_id, 2] = ts_vals.mean() / 300.0
                route_stats[view_id, 3] = ts_vals.std(unbiased=False) / 300.0 if len(view_events) > 1 else 0.0

        cached_stat = sample.get("_cached_stat_vecs")
        cached_quality = sample.get("_cached_quality")
        if cached_stat is not None:
            stat_tensor = cached_stat if torch.is_tensor(cached_stat) else torch.tensor(cached_stat)
            stat_tensor = stat_tensor.float()
            route_stats[:, 4] = stat_tensor.abs().mean(dim=(1, 2))
            route_stats[:, 5] = stat_tensor.std(dim=(1, 2), unbiased=False)
        if cached_quality is not None:
            quality_tensor = cached_quality if torch.is_tensor(cached_quality) else torch.tensor(cached_quality)
            quality_tensor = quality_tensor.float()
            route_stats[:, 6] = quality_tensor[:, 0]
            route_stats[:, 7] = quality_tensor[:, 1]
        return route_stats

    def _collect_cached_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        cache = sample["_event_cache"]
        if cache.get("raw_events"):
            host = str(sample.get("host", cache.get("metadata", {}).get("host", "unknown")))
            events = []
            prev_ts_by_view = {0: 0.0, 1: 0.0, 2: 0.0}
            for rank, raw_event in enumerate(cache["raw_events"]):
                view_name = raw_event.get("_view_name", raw_event.get("view", "process"))
                view_id = {"process": 0, "file": 1, "network": 2}.get(view_name, 0)
                feat = self._event_to_features(raw_event, view_id, host, rank, prev_ts_by_view[view_id])
                prev_ts_by_view[view_id] = float(raw_event.get("ts", 0.0) or 0.0)
                events.append(feat)
            route_stats = cache.get("route_stats")
            if route_stats is None:
                route_stats = self._build_route_stats(sample, events)
            else:
                route_stats = route_stats.float()
            return {"events": events, "view_stats": route_stats}

        events = []
        for i, meta in enumerate(cache["event_meta"]):
            events.append(
                {
                    "type_id": int(cache["type_ids"][i]),
                    "op_id": int(cache["op_ids"][i]),
                    "fine_id": int(cache["fine_ids"][i]),
                    "obj_id": int(cache["obj_ids"][i]),
                    "text_id": int(cache["text_ids"][i]),
                    "view_id": int(cache["view_ids"][i]),
                    "host_id": int(cache["host_ids"][i]),
                    "time": cache["time_feats"][i].tolist(),
                    "field_mask": cache["field_masks"][i].tolist(),
                    "meta": meta,
                }
            )
        route_stats = cache.get("route_stats")
        if route_stats is None:
            route_stats = self._build_route_stats(sample, events)
        else:
            route_stats = route_stats.float()
        return {"events": events, "view_stats": route_stats}

    def _collect_sample_events(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "_event_cache" in sample:
            return self._collect_cached_sample(sample)
        host = str(sample.get("host", "unknown"))
        events: List[Dict[str, Any]] = []
        view_order = {"process": 0, "file": 1, "network": 2}

        for view_name in ("process", "file", "network"):
            raw_events = sample.get("views", {}).get(view_name, [])
            raw_events = sorted(raw_events, key=lambda e: float(e.get("ts", 0.0) or 0.0))
            prev_ts = 0.0
            for rank, event in enumerate(raw_events):
                feat = self._event_to_features(event, view_order[view_name], host, rank, prev_ts)
                prev_ts = float(event.get("ts", 0.0) or 0.0)
                events.append(feat)

        events.sort(key=lambda e: (e["meta"]["ts"], e["view_id"], e["meta"]["obj"]))
        return {"events": events, "view_stats": self._build_route_stats(sample, events)}

    def forward(self, batch_samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        sample_cache = [self._collect_sample_events(sample) for sample in batch_samples]
        batch_size = len(sample_cache)
        max_len = max(1, max(len(item["events"]) for item in sample_cache))
        device = next(self.parameters()).device

        type_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        op_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        fine_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        obj_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        text_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        view_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        host_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        time_feats = torch.zeros((batch_size, max_len, 3), dtype=torch.float32, device=device)
        field_masks = torch.zeros((batch_size, max_len, self.field_count), dtype=torch.float32, device=device)
        valid_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        route_stats = torch.stack([item["view_stats"] for item in sample_cache], dim=0).to(device)
        event_meta: List[List[Dict[str, Any]]] = []

        for b, item in enumerate(sample_cache):
            meta_row = []
            for i, event in enumerate(item["events"][:max_len]):
                type_ids[b, i] = event["type_id"]
                op_ids[b, i] = event["op_id"]
                fine_ids[b, i] = event["fine_id"]
                obj_ids[b, i] = event["obj_id"]
                text_ids[b, i] = event["text_id"]
                view_ids[b, i] = event["view_id"]
                host_ids[b, i] = event["host_id"]
                time_feats[b, i] = torch.tensor(event["time"], dtype=torch.float32, device=device)
                field_masks[b, i] = torch.tensor(event["field_mask"], dtype=torch.float32, device=device)
                valid_mask[b, i] = True
                meta_row.append(event["meta"])
            event_meta.append(meta_row)

        base = torch.cat(
            [
                self.type_emb(type_ids),
                self.op_emb(op_ids),
                self.fine_emb(fine_ids),
                self.obj_emb(obj_ids),
                self.text_emb(text_ids),
                self.view_emb(view_ids),
                self.host_emb(host_ids),
                self.field_mask_proj(field_masks),
                self.time_proj(time_feats),
            ],
            dim=-1,
        )
        event_tokens = self.event_proj(base)
        route_p = F.softmax(self.router(route_stats.reshape(batch_size, -1)), dim=-1)
        align_op = torch.einsum("bm,mij->bij", route_p, self.alignment_bases)
        aligned = torch.einsum("bij,blj->bli", align_op, event_tokens)

        contextual_chunks = []
        for start in range(0, max_len, self.max_events):
            end = min(start + self.max_events, max_len)
            chunk = aligned[:, start:end]
            chunk_mask = valid_mask[:, start:end]
            chunk = chunk + self.pos_emb[:, : end - start]
            contextual_chunks.append(self.sequence_encoder(chunk, src_key_padding_mask=~chunk_mask))
        contextual = torch.cat(contextual_chunks, dim=1)
        pooled = []
        for b in range(batch_size):
            valid_tokens = contextual[b][valid_mask[b]]
            pooled.append(valid_tokens.mean(dim=0) if valid_tokens.numel() else torch.zeros(self.event_dim, device=device))

        return {
            "event_embeddings": contextual,
            "event_mask": valid_mask,
            "window_embedding": torch.stack(pooled, dim=0),
            "route_p": route_p,
            "event_meta": event_meta,
        }
