from __future__ import annotations

from typing import Any, Dict, List, Iterator, Tuple, Optional
from optc_uras.data.raw_reader import RawReader

class ToyReader(RawReader):
    """用于骨架自检的 toy 数据生成器（不代表真实 OpTC 分布）。"""

    def __init__(self, num_samples: int = 200, seed: int = 42, file_path: str = "toy.jsonl"):
        import random
        self.num_samples = num_samples
        self.rng = random.Random(seed)
        self.file_path = file_path
        self.window_minutes = 15
        self.time_range = None
        self.host_filter = None

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
        from optc_uras.data.ecar import EcarJsonlReader, WindowAggregator
        import random
        
        # 1. 初始化流式读取器
        reader = EcarJsonlReader(
            self.file_path, 
            time_range=self.time_range,
            host_filter=self.host_filter
        )
        
        # 2. 初始化窗口聚合器 (设置较大的 max_events 以避免过早截断，确保后续全局采样的随机性)
        aggregator = WindowAggregator(window_minutes=self.window_minutes, max_events_per_window=100000)
        
        # 3. 流式处理并应用动态采样
        for raw_sample in aggregator.process(reader):
            # 统计该样本所有 view 的总事件数
            total_events = sum(len(evs) for evs in raw_sample["views"].values())
            
            # 阶梯式采样策略 (最终确认版 - 适配 4090)
            # 目标：温和扩充，避免数据爆炸，同时利用 8000 长度优势
            # < 1w: 1次 (全量)
            # 1w-3w: 3次
            # 3w-6w: 5次
            # > 6w: 10次
            if total_events < 10000:
                num_augmentations = 1
            elif total_events < 30000:
                num_augmentations = 3
            elif total_events < 60000:
                num_augmentations = 5
            else:
                num_augmentations = 10
                
            # 目标采样大小：8000 (全局限制，而非分视图限制)
            GLOBAL_MAX_EVENTS = 8000
            
            # 1. 先合并所有视图的事件，确保全局随机性
            all_events = []
            for v_name, events in raw_sample["views"].items():
                for evt in events:
                    all_events.append((v_name, evt))
            
            for i in range(num_augmentations):
                aug_sample = {
                    "host": raw_sample["host"],
                    "t0": raw_sample["t0"],
                    "t1": raw_sample.get("t1"),
                    "label": raw_sample["label"],
                    "aug_id": i, # 标记这是第几次增强
                    "views": {v: [] for v in raw_sample["views"].keys()}
                }
                
                # 2. 全局随机采样
                if len(all_events) > GLOBAL_MAX_EVENTS:
                    sampled_pairs = random.sample(all_events, GLOBAL_MAX_EVENTS)
                else:
                    sampled_pairs = list(all_events) # copy
                
                # 3. 重新分配回视图
                for v_name, evt in sampled_pairs:
                    aug_sample["views"][v_name].append(evt)
                
                yield aug_sample