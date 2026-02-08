from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


def split_into_slots(events: Sequence[Dict[str, Any]], slot_seconds: int, window_seconds: int) -> List[List[Dict[str, Any]]]:
    """把单个 view 的事件按固定 slot 划分。

    events: event_dict 列表（需要 'ts' 为相对窗口起点的秒）
    返回：长度 K=window_seconds/slot_seconds 的 list，每个元素是该槽的事件列表
    """
    assert window_seconds % slot_seconds == 0
    K = window_seconds // slot_seconds
    slots: List[List[Dict[str, Any]]] = [[] for _ in range(K)]
    for e in events:
        t = float(e.get("ts", 0.0))
        if t < 0 or t >= window_seconds:
            continue
        k = int(t // slot_seconds)
        k = min(max(k, 0), K - 1)
        slots[k].append(e)
    return slots
