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

    def __init__(self, cache_dir: str, split: str = "train", preload: bool = False):
        """
        Args:
            cache_dir: Directory containing preprocessed .pkl files
            split: 'train' or 'test' (or 'all'). 
                   Files should be named like 'train_partX.pkl' or 'test_partX.pkl'.
            preload: If True, load all data into RAM for faster access (may use significant RAM)
        """
        self.cache_dir = cache_dir
        self.split = split
        self.index_cache_file = os.path.join(cache_dir, f"index_cache_{split}.pkl")
        self.preload = preload
        
        # Lazy Loading Index: map global_idx -> (file_path, local_idx_in_file, t0, host)
        self.index_map = []
        
        # Runtime Cache for sequential access
        self._last_fpath = None
        self._last_chunk = None
        
        # Preloaded Data Cache: map file_path -> list[Sample]
        self._preloaded_data = {}
        
        self._load_index()
        
        if self.preload:
            self._do_preload()

    def _do_preload(self):
        """Load all data from index_map into RAM"""
        unique_files = sorted(list(set([m[0] for m in self.index_map])))
        logger.info(f"Preloading {len(unique_files)} files into RAM for split '{self.split}'...")
        from tqdm import tqdm
        for fpath in tqdm(unique_files, desc=f"Preloading {self.split}"):
            try:
                if fpath.endswith(".pt"):
                    chunk = torch.load(fpath, map_location="cpu")
                else:
                    with open(fpath, "rb") as f:
                        chunk = pickle.load(f)
                self._preloaded_data[fpath] = chunk
            except Exception as e:
                logger.error(f"Failed to preload {fpath}: {e}")
        logger.info(f"Preload complete. Using RAM for {self.split} data.")

    def _load_index(self):
        if not os.path.exists(self.cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        # Try load from cache
        if os.path.exists(self.index_cache_file):
            try:
                with open(self.index_cache_file, "rb") as f:
                    self.index_map = pickle.load(f)
                logger.info(f"Loaded index from cache: {self.index_cache_file} ({len(self.index_map)} samples)")
                return
            except Exception as e:
                logger.warning(f"Failed to load index cache: {e}")

        files = sorted([f for f in os.listdir(self.cache_dir) if f.endswith(".pt") or f.endswith(".pkl")])
        loaded_count = 0
        
        logger.info(f"Scanning {len(files)} files for indexing (Lazy Loading)...")
        
        # Parallelize file scanning
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_file(f):
            # Simple filtering based on filename
            if "index_cache" in f: return None
            if self.split != "all" and self.split not in f:
                return None
                
            path = os.path.join(self.cache_dir, f)
            try:
                # Support both .pt and .pkl
                if f.endswith(".pt"):
                    chunk = torch.load(path, map_location="cpu")
                else:
                    with open(path, "rb") as fp:
                        chunk = pickle.load(fp)
                        
                local_entries = []
                if isinstance(chunk, list):
                    num_samples = len(chunk)
                    for i in range(num_samples):
                        s = chunk[i]
                        # Store metadata for fast access
                        t0 = s.get("t0", 0)
                        host = s.get("host", "unknown")
                        local_entries.append((path, i, t0, host))
                del chunk
                return local_entries
            except Exception as e:
                logger.warning(f"Failed to scan cache file {f}: {e}")
                return None

        # Use threads for IO bound task
        entries = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(process_file, f) for f in files]
            for future in as_completed(futures):
                res = future.result()
                if res:
                    entries.extend(res)
                    loaded_count += 1
        
        # Sort to maintain deterministic order (file path + index)
        entries.sort(key=lambda x: (x[0], x[1]))
        self.index_map = entries

        if loaded_count == 0:
            logger.warning(f"No cache files loaded for split '{self.split}' in {self.cache_dir}")
        else:
            logger.info(f"Loaded {len(self.index_map)} samples index from {loaded_count} files for split '{self.split}'")
            
            # Save index cache
            try:
                with open(self.index_cache_file, "wb") as f:
                    pickle.dump(self.index_map, f)
                logger.info(f"Saved index cache to {self.index_cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save index cache: {e}")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Lazy load from disk
        fpath, local_idx, _, _ = self.index_map[idx]
        
        # Check preloaded cache
        if fpath in self._preloaded_data:
            return self._preloaded_data[fpath][local_idx]
            
        # Check runtime cache
        if self._last_fpath == fpath and self._last_chunk is not None:
            return self._last_chunk[local_idx]
            
        # This is slow (IO every time), but memory safe.
        if fpath.endswith(".pt"):
            chunk = torch.load(fpath, map_location="cpu")
        else:
            with open(fpath, "rb") as f:
                chunk = pickle.load(f)
        
        # Update runtime cache
        self._last_fpath = fpath
        self._last_chunk = chunk
        
        return chunk[local_idx]

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Fast metadata access without loading full sample"""
        _, _, t0, host = self.index_map[idx]
        return {"t0": t0, "host": host}

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


class TensorDataset(Dataset):
    """
    Dataset wrapping pre-computed semantic inputs and stats in RAM.
    Returns: (indices, behavior, c_type, c_op, c_fine, c_obj, c_text, c_masks, c_times, stat_vecs, quality)
    """
    def __init__(self, data_dict: Dict[str, torch.Tensor], indices: List[int]):
        self.data = data_dict
        self.indices = indices
        # Expose sliced tensors as attributes for convenience (used in main.py)
        # Slicing ensures that sub-indexing (e.g. via Subset) works correctly.
        self.behavior = data_dict.get("behavior")[indices] if data_dict.get("behavior") is not None else None
        self.quality = data_dict.get("quality")[indices] if data_dict.get("quality") is not None else None
        self.c_type = data_dict.get("c_type")[indices] if data_dict.get("c_type") is not None else None
        self.c_op = data_dict.get("c_op")[indices] if data_dict.get("c_op") is not None else None
        self.c_fine = data_dict.get("c_fine")[indices] if data_dict.get("c_fine") is not None else None
        self.c_obj = data_dict.get("c_obj")[indices] if data_dict.get("c_obj") is not None else None
        self.c_text = data_dict.get("c_text")[indices] if data_dict.get("c_text") is not None else None
        self.c_masks = data_dict.get("c_masks")[indices] if data_dict.get("c_masks") is not None else None
        self.c_times = data_dict.get("c_times")[indices] if data_dict.get("c_times") is not None else None
        self.stat_vecs = data_dict.get("stat_vecs")[indices] if data_dict.get("stat_vecs") is not None else None
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        # Since attributes are already sliced, we can just index them directly if we wanted,
        # but to keep it consistent with the 11-tuple return:
        return (
            self.indices[idx], 
            self.behavior[idx],
            self.c_type[idx],
            self.c_op[idx],
            self.c_fine[idx],
            self.c_obj[idx],
            self.c_text[idx],
            self.c_masks[idx],
            self.c_times[idx],
            self.stat_vecs[idx],
            self.quality[idx]
        )

def tensor_collate(batch):
    # batch is list of tuples
    out = []
    for i in range(len(batch[0])):
        if isinstance(batch[0][i], int):
            out.append([b[i] for b in batch])
        else:
            out.append(torch.stack([b[i] for b in batch]))
    return tuple(out)


class FeatureCollate:
    """
    Collate function that pre-computes behavior features in parallel workers.
    Returns:
        samples: List[Sample] (original raw samples for Step1)
        behavior_features: torch.Tensor [B, behavior_dim]
    """
    def __init__(self, views: List[str], behavior_dim: int, feature_dp_config: Optional[Any] = None):
        self.views = views
        self.behavior_dim = behavior_dim
        self.feature_dp_config = feature_dp_config
        # Avoid circular import
        from ..features.behavior_features import behavior_features_from_sample
        self._extractor = behavior_features_from_sample

    def __call__(self, batch: Sequence[Sample]) -> Tuple[List[Sample], torch.Tensor]:
        import torch
        import numpy as np
        
        # 1. Compute behavior features (CPU heavy part)
        feats_list = []
        for s in batch:
            f = self._extractor(s, self.views, out_dim=self.behavior_dim)
            feats_list.append(f)
        
        behavior_features = torch.from_numpy(np.stack(feats_list)).float()
        
        # 2. Apply DP if configured (can be done here or in model, here is fine too but typically DP noise is per-step on device)
        # Usually DP noise is added on GPU to be fast. We just return clean features here.
        
        return list(batch), behavior_features
