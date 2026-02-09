from __future__ import annotations

import os
import pickle
import logging
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class OpTCEcarDataset(Dataset):
    """OpTC eCAR Dataset
    
    Loads preprocessed samples (aggregated windows) from cache directory.
    Each sample is a dict: {host, t0, t1, views: {process:[], file:[], network:[]}, label}
    """

    def __init__(self, cache_dir: str, split: str = "train"):
        """
        Args:
            cache_dir: Directory containing preprocessed .pkl files
            split: 'train' or 'test' (or 'all'). 
                   Files should be named like 'train_partX.pkl' or 'test_partX.pkl'.
        """
        self.cache_dir = cache_dir
        self.split = split
        # Lazy Loading Index: map global_idx -> (file_path, local_idx_in_file)
        self.index_map = []
        
        self._load_index()

    def _load_index(self):
        if not os.path.exists(self.cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        files = sorted([f for f in os.listdir(self.cache_dir) if f.endswith(".pt") or f.endswith(".pkl")])
        loaded_count = 0
        
        logger.info(f"Scanning {len(files)} files for indexing (Lazy Loading)...")
        
        for f in files:
            # Simple filtering based on filename
            if self.split != "all" and self.split not in f:
                continue
                
            path = os.path.join(self.cache_dir, f)
            try:
                # Support both .pt and .pkl
                if f.endswith(".pt"):
                    chunk = torch.load(path, map_location="cpu")
                else:
                    with open(path, "rb") as fp:
                        chunk = pickle.load(fp)
                        
                if isinstance(chunk, list):
                    num_samples = len(chunk)
                    for i in range(num_samples):
                        self.index_map.append((path, i))
                    loaded_count += 1
                del chunk
            except Exception as e:
                logger.warning(f"Failed to scan cache file {f}: {e}")

        if loaded_count == 0:
            logger.warning(f"No cache files loaded for split '{self.split}' in {self.cache_dir}")
        else:
            logger.info(f"Loaded {len(self.index_map)} samples index from {loaded_count} files for split '{self.split}'")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Lazy load from disk
        fpath, local_idx = self.index_map[idx]
        
        # This is slow (IO every time), but memory safe.
        if fpath.endswith(".pt"):
            chunk = torch.load(fpath, map_location="cpu")
            return chunk[local_idx]
        else:
            with open(fpath, "rb") as f:
                chunk = pickle.load(f)
                return chunk[local_idx]

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


Sample = Dict[str, Any]
# 统一 sample 结构约定（骨架版）：
# {
#   "host": str,
#   "t0": int,   # window start timestamp (unix seconds) or 0 in toy
#   "label": Optional[int],  # 仅用于 test 评测，不用于训练
#   "views": {
#       "process": List[event_dict],
#       "file": List[event_dict],
#       "network": List[event_dict],
#   }
# }
#
# event_dict 约定字段（可扩展）：
#   ts: float (seconds since t0, in [0, 15*60))
#   type: str
#   obj: str  (不可逆/已脱敏也可)
#   op: str
#   ok: int (0/1)
#   extra: dict (任意扩展字段)


class ProcessedDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def collate_samples(batch: Sequence[Sample]) -> List[Sample]:
    # Step1 在样本级做特征抽取，这里保持 list 形式即可
    return list(batch)
