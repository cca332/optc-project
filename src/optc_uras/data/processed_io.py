from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch


def save_samples_pt(samples: List[Dict[str, Any]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, out_path)


def load_samples_pt(path: str) -> List[Dict[str, Any]]:
    return torch.load(path, map_location="cpu")


def processed_path(processed_dir: str, split: str) -> str:
    # split: train/val/test/all
    return str(Path(processed_dir) / f"samples_{split}.pt")
