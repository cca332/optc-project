import os
import sys
import argparse
import zlib
import pickle
import glob

import torch
import yaml
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from optc_uras.data.dataset import OpTCEcarDataset


def deep_update(base, override):
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def hash_token(value, buckets):
    return zlib.crc32(str(value).encode("utf-8")) % int(buckets)


def build_route_stats(events, cached_stat=None, cached_quality=None, num_views=3):
    route_feature_dim = 8
    route_stats = torch.zeros(num_views, route_feature_dim, dtype=torch.float32)
    for view_id in range(num_views):
        view_events = [e for e in events if e["meta"]["view_id"] == view_id]
        if view_events:
            ts_vals = torch.tensor([e["meta"]["ts"] for e in view_events], dtype=torch.float32)
            valid_vals = torch.tensor([e["field_mask"][-1] for e in view_events], dtype=torch.float32)
            route_stats[view_id, 0] = float(len(view_events)) / max(1.0, float(len(events)))
            route_stats[view_id, 1] = valid_vals.mean()
            route_stats[view_id, 2] = ts_vals.mean() / 300.0
            route_stats[view_id, 3] = ts_vals.std(unbiased=False) / 300.0 if len(view_events) > 1 else 0.0
    if cached_stat is not None:
        stat_tensor = cached_stat.float()
        route_stats[:, 4] = stat_tensor.abs().mean(dim=(1, 2))
        route_stats[:, 5] = stat_tensor.std(dim=(1, 2), unbiased=False)
    if cached_quality is not None:
        quality_tensor = cached_quality.float()
        route_stats[:, 6] = quality_tensor[:, 0]
        route_stats[:, 7] = quality_tensor[:, 1]
    return route_stats


def flush_shard(cache_dir, split, shard_idx, shard_buffer, index_entries):
    if not shard_buffer:
        return index_entries, shard_idx
    shard_path = os.path.join(cache_dir, f"event_{split}_part{shard_idx:03d}.pt")
    torch.save(shard_buffer, shard_path)
    for local_idx in range(len(shard_buffer)):
        index_entries.append((shard_path, local_idx))
    shard_buffer = []
    shard_idx += 1
    return index_entries, shard_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_production.yaml")
    parser.add_argument("--event_config", default="configs/event_method.yaml")
    parser.add_argument("--splits", default="detector,test", help="Comma-separated splits to cache")
    parser.add_argument("--samples_per_shard", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if os.path.exists(args.event_config):
        with open(args.event_config, "r", encoding="utf-8") as f:
            config = deep_update(config, yaml.safe_load(f) or {})

    optc_cfg = config["data"]["optc"]
    model_cfg = config["model"]
    event_cfg = config.get("event_method", {})
    cache_dir = optc_cfg["cache_dir"]
    event_cache_cfg = event_cfg.get("cache", {})
    max_events = int(event_cache_cfg.get("max_events", model_cfg.get("event_cache_max_events", 0)) or 0)
    samples_per_shard = int(args.samples_per_shard if args.samples_per_shard is not None else event_cache_cfg.get("samples_per_shard", 64))

    aux_path = os.path.join(cache_dir, "optimized_data.pt")
    aux_map = {}
    if os.path.exists(aux_path):
        aux = torch.load(aux_path, map_location="cpu", weights_only=False)
        for i, meta in enumerate(aux.get("metadata", [])):
            key = (meta.get("host", "unknown"), int(meta.get("t0", 0)), int(meta.get("label", 0)))
            aux_map[key] = {"stat_vecs": aux["stat_vecs"][i], "quality": aux["quality"][i]}

    host_buckets = 4096
    type_buckets = 128
    op_buckets = 256
    fine_buckets = 256
    obj_buckets = 50000
    text_buckets = 50000
    view_order = {"process": 0, "file": 1, "network": 2}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        for stale_path in glob.glob(os.path.join(cache_dir, f"event_{split}_part*.pt")):
            os.remove(stale_path)
        stale_index = os.path.join(cache_dir, f"event_index_cache_{split}.pkl")
        if os.path.exists(stale_index):
            os.remove(stale_index)

        ds = OpTCEcarDataset(cache_dir, split=split, preload=False)
        shard_buffer = []
        index_entries = []
        shard_idx = 0
        for idx in tqdm(range(len(ds)), desc=f"Building event cache ({split})"):
            sample = ds[idx]
            meta = {"host": sample.get("host", "unknown"), "t0": int(sample.get("t0", 0)), "label": int(sample.get("label", 0))}
            key = (meta["host"], meta["t0"], meta["label"])
            cached_aux = aux_map.get(key, {})
            sample_events = []
            for view_name in ("process", "file", "network"):
                raw_events = sorted(sample.get("views", {}).get(view_name, []), key=lambda e: float(e.get("ts", 0.0) or 0.0))
                prev_ts = 0.0
                for rank, event in enumerate(raw_events):
                    raw_event = dict(event)
                    raw_event["_view_name"] = view_name
                    ts = float(event.get("ts", 0.0) or 0.0)
                    delta_t = float(event.get("delta_t", ts - prev_ts if rank > 0 else ts) or 0.0)
                    prev_ts = ts
                    text_hash = 0
                    for part in [
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
                    ]:
                        if part:
                            text_hash = (text_hash + hash_token(part, text_buckets)) % text_buckets
                    sample_events.append(
                        {
                            "raw_event": raw_event,
                            "type_id": hash_token(event.get("type", ""), type_buckets),
                            "op_id": hash_token(event.get("op", ""), op_buckets),
                            "fine_id": hash_token(event.get("fine_view", event.get("view", "")), fine_buckets),
                            "obj_id": hash_token(event.get("obj", ""), obj_buckets),
                            "text_id": text_hash,
                            "view_id": view_order[view_name],
                            "host_id": hash_token(meta["host"], host_buckets),
                            "time": [ts / 300.0, delta_t / 300.0, 0.0],
                            "field_mask": [
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
                            ],
                            "meta": {
                                "view_id": view_order[view_name],
                                "view_name": view_name,
                                "ts": ts,
                                "type": event.get("type", ""),
                                "op": event.get("op", ""),
                                "obj": event.get("obj", ""),
                            },
                        }
                    )
            sample_events.sort(key=lambda e: (e["meta"]["ts"], e["meta"]["view_id"], e["meta"]["obj"]))
            if max_events > 0:
                sample_events = sample_events[:max_events]
            for rank, event in enumerate(sample_events):
                event["time"][2] = float(rank) / max(1.0, float(max(len(sample_events), 1) - 1))

            shard_buffer.append(
                {
                    "metadata": meta,
                    "raw_events": [e["raw_event"] for e in sample_events],
                    "type_ids": torch.tensor([e["type_id"] for e in sample_events], dtype=torch.long),
                    "op_ids": torch.tensor([e["op_id"] for e in sample_events], dtype=torch.long),
                    "fine_ids": torch.tensor([e["fine_id"] for e in sample_events], dtype=torch.long),
                    "obj_ids": torch.tensor([e["obj_id"] for e in sample_events], dtype=torch.long),
                    "text_ids": torch.tensor([e["text_id"] for e in sample_events], dtype=torch.long),
                    "view_ids": torch.tensor([e["view_id"] for e in sample_events], dtype=torch.long),
                    "host_ids": torch.tensor([e["host_id"] for e in sample_events], dtype=torch.long),
                    "time_feats": torch.tensor([e["time"] for e in sample_events], dtype=torch.float32),
                    "field_masks": torch.tensor([e["field_mask"] for e in sample_events], dtype=torch.float32),
                    "event_meta": [e["meta"] for e in sample_events],
                    "route_stats": build_route_stats(sample_events, cached_aux.get("stat_vecs"), cached_aux.get("quality")),
                }
            )
            if len(shard_buffer) >= samples_per_shard:
                index_entries, shard_idx = flush_shard(cache_dir, split, shard_idx, shard_buffer, index_entries)
                shard_buffer = []

        index_entries, shard_idx = flush_shard(cache_dir, split, shard_idx, shard_buffer, index_entries)
        index_path = os.path.join(cache_dir, f"event_index_cache_{split}.pkl")
        with open(index_path, "wb") as f:
            pickle.dump(index_entries, f)
        print(f"Saved event cache index to {index_path} ({len(index_entries)} samples)")


if __name__ == "__main__":
    main()
