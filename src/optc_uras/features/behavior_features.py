from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np

# Step2 A2：客户端不可逆行为特征（统计/结构摘要拼接）
# 这里给一个“固定维度”的最小骨架实现：
#   - 前 2*V 维：每个 view 的 [event_count, unique_obj_count]
#   - 剩余维：hash-bucket 计数（不可逆），再做归一化
#
# 真实工程可把进程树深度分布等结构摘要拼进来，只要保证不可逆、维度固定即可。


def behavior_features_from_sample(
    sample: Dict[str, Any],
    views: Sequence[str],
    out_dim: int,
    seed: int = 13,
) -> np.ndarray:
    import zlib

    V = len(views)
    # Per-view features:
    # 0: event_count
    # 1: unique_obj_count
    # 2: type_entropy (distribution summary)
    # 3: max_obj_freq_ratio (structural concentration)
    feats_per_view = 4
    base_dim = feats_per_view * V
    
    if out_dim < base_dim + 8:
        raise ValueError(f"out_dim too small: need >= {base_dim + 8}, got {out_dim}")
    num_buckets = out_dim - base_dim

    def stable_hash(text: str) -> int:
        data = (str(seed) + ':' + text).encode('utf-8', errors='ignore')
        return zlib.crc32(data) & 0xFFFFFFFF

    def compute_entropy(counts: List[int]) -> float:
        if not counts:
            return 0.0
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts]
        return -sum(p * np.log(p + 1e-9) for p in probs)

    vec = np.zeros((out_dim,), dtype=np.float32)

    # per-view scalars
    for i, v in enumerate(views):
        evs = sample["views"].get(v, [])
        count = float(len(evs))
        
        # 1. Basic Counts
        vec[feats_per_view * i] = count
        
        objs = [str(e.get("obj", "")) for e in evs]
        unique_objs = set(objs)
        vec[feats_per_view * i + 1] = float(len(unique_objs))

        # 2. Structural/Distribution Summaries
        # Type Entropy
        types = [str(e.get("type", "")) for e in evs]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        vec[feats_per_view * i + 2] = float(compute_entropy(list(type_counts.values())))

        # Max Object Frequency Ratio (Concentration)
        if count > 0:
            obj_counts = {}
            for o in objs:
                obj_counts[o] = obj_counts.get(o, 0) + 1
            max_freq = max(obj_counts.values()) if obj_counts else 0
            vec[feats_per_view * i + 3] = float(max_freq / count)
        else:
            vec[feats_per_view * i + 3] = 0.0

        # hash buckets
        for e in evs:
            key = f"{v}|{e.get('type','')}|{e.get('op','')}|{e.get('obj','')}"
            b = stable_hash(key) % num_buckets
            vec[base_dim + b] += 1.0

    # normalize hash part
    s = vec[base_dim:].sum()
    if s > 0:
        vec[base_dim:] = vec[base_dim:] / s
    return vec
