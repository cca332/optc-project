from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zlib
import numpy as np


def _stable_hash(text: str, seed: int) -> int:
    data = (str(seed) + ':' + text).encode('utf-8', errors='ignore')
    return zlib.crc32(data) & 0xFFFFFFFF


@dataclass
class AggregatorSchema:
    event_type_vocab: Optional[List[str]] = None
    key_fields: Optional[List[str]] = None


class DeterministicStatsAggregator:
    """确定性聚合器（可复现、不可逆）

    输出 slot 向量由以下拼接构成（可按 view 裁剪）：
      - 事件类型计数直方图
      - hash-bucket 计数（对 obj/op/type 等做稳定 hash）
      - 简单强度统计（事件数、unique obj 数）
      - 可选空槽指示
    """

    def __init__(self, schema: AggregatorSchema, num_hash_buckets: int, hash_seed: int, include_empty_indicator: bool = True):
        self.schema = schema
        self.num_hash_buckets = int(num_hash_buckets)
        self.hash_seed = int(hash_seed)
        self.include_empty_indicator = bool(include_empty_indicator)

        self.event_type_vocab = schema.event_type_vocab or []
        self._type_to_idx = {t: i for i, t in enumerate(self.event_type_vocab)}

    def fit_event_type_vocab(self, events: Sequence[Dict[str, Any]], max_types: int = 128) -> None:
        # 扫描构建 vocab（可复现：排序 + 截断）
        types = sorted({str(e.get("type", "")) for e in events if e.get("type") is not None})
        self.event_type_vocab = types[:max_types]
        self._type_to_idx = {t: i for i, t in enumerate(self.event_type_vocab)}

    @property
    def output_dim(self) -> int:
        dim = len(self.event_type_vocab) + self.num_hash_buckets + 2
        if self.include_empty_indicator:
            dim += 1
        return dim

    def __call__(self, slot_events: Sequence[Dict[str, Any]]) -> np.ndarray:
        hist = np.zeros((len(self.event_type_vocab),), dtype=np.float32)
        buckets = np.zeros((self.num_hash_buckets,), dtype=np.float32)

        objs = set()
        for e in slot_events:
            et = str(e.get("type", ""))
            if et in self._type_to_idx:
                hist[self._type_to_idx[et]] += 1.0
            obj = str(e.get("obj", ""))
            op = str(e.get("op", ""))
            key = f"{et}|{op}|{obj}"
            b = _stable_hash(key, self.hash_seed) % self.num_hash_buckets
            buckets[b] += 1.0
            objs.add(obj)

        count = float(len(slot_events))
        uniq = float(len(objs))
        stats = np.asarray([count, uniq], dtype=np.float32)

        parts = [hist, buckets, stats]
        if self.include_empty_indicator:
            parts.append(np.asarray([1.0 if len(slot_events) == 0 else 0.0], dtype=np.float32))
        return np.concatenate(parts, axis=0)
