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


class OptcRawReader(RawReader):
    """OpTC 真实数据读取器。
    
    使用 EcarJsonlReader 读取原始 JSONL，并使用 WindowAggregator 聚合为 15min 样本。
    """

    def __init__(self, file_path: str, window_minutes: int = 15, 
                 host_filter: Optional[List[str]] = None,
                 time_range: Optional[Tuple[str, str]] = None):
        self.file_path = file_path
        self.window_minutes = window_minutes
        self.host_filter = host_filter
        self.time_range = time_range

    def iterate_samples(self) -> Iterator[Dict[str, Any]]:
        from optc_uras.data.ecar import EcarJsonlReader, WindowAggregator
        import random
        
        # Check if file_path is a list (comma separated string or list object)
        paths = []
        if isinstance(self.file_path, list):
            paths = self.file_path
        elif isinstance(self.file_path, str):
            if "," in self.file_path:
                paths = [p.strip() for p in self.file_path.split(",")]
            else:
                paths = [self.file_path]
        
        # [ADDED] Shuffle files to ensure randomness if reading multiple files
        random.shuffle(paths)
        
        for p in paths:
            # 1. 初始化流式读取器
            reader = EcarJsonlReader(
                p, 
                time_range=self.time_range,
                host_filter=self.host_filter
            )
            
            # 2. 初始化窗口聚合器
            aggregator = WindowAggregator(window_minutes=self.window_minutes, max_events_per_window=None, augment=False)
            
            # 3. 流式处理并产出样本
            for raw_sample in aggregator.process(reader):
                yield raw_sample
