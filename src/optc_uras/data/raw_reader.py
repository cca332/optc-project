from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional


class RawReader(ABC):
    """把你们的原始/预处理数据读成统一 Sample 结构的抽象类。

    你需要根据 PIDSMaker 输出格式实现：
      - iterate_samples(): 逐条产出 Sample（host×15min）
      - 只要保证 Sample 的 'views'->每个 view 是 event_dict 列表即可
    """

    @abstractmethod
    def iterate_samples(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError


class ToyReader(RawReader):
    """用于骨架自检的 toy 数据生成器（不代表真实 OpTC 分布）。"""

    def __init__(self, num_samples: int = 200, seed: int = 42):
        import random
        self.num_samples = num_samples
        self.rng = random.Random(seed)

    def _rand_events(self, n: int, view: str) -> List[Dict[str, Any]]:
        types = {
            "process": ["exec", "fork", "exit", "clone"],
            "file": ["open", "read", "write", "close"],
            "network": ["connect", "send", "recv", "close"],
        }[view]
        out = []
        for _ in range(n):
            t = self.rng.random() * 900.0
            et = self.rng.choice(types)
            obj = f"{view}_obj_{self.rng.randint(0, 200)}"
            op = et
            ok = 1
            out.append({"ts": t, "type": et, "obj": obj, "op": op, "ok": ok, "extra": {}})
        return out

    def iterate_samples(self) -> Iterator[Dict[str, Any]]:
        for i in range(self.num_samples):
            host = f"H{i % 10}"
            # 用 label 仅做评测；训练/校准请过滤 benign-only
            label = 1 if (i % 20 == 0) else 0
            # 让异常样本在 network view 上更密集一点（仅 toy）
            n_net = 120 if label == 1 else 40
            sample = {
                "host": host,
                "t0": 0,
                "label": label,
                "views": {
                    "process": self._rand_events(30, "process"),
                    "file": self._rand_events(50, "file"),
                    "network": self._rand_events(n_net, "network"),
                },
            }
            yield sample
