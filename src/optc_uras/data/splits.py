from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import random


def make_splits(samples: List[Dict[str, Any]], seed: int, val_ratio: float = 0.1, test_ratio: float = 0.2) -> Dict[str, List[int]]:
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    n = len(idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = idx[:n_test]
    val = idx[n_test:n_test+n_val]
    train = idx[n_test+n_val:]
    return {"train": train, "val": val, "test": test}


def save_splits(splits: Dict[str, List[int]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)


def load_splits(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
