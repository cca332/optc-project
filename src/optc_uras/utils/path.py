from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import datetime


def make_run_dir(run_root: str, exp_name: str, add_timestamp: bool = True) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
    name = f"{exp_name}_{stamp}" if stamp else exp_name
    p = Path(run_root) / name
    p.mkdir(parents=True, exist_ok=True)
    (p / "checkpoints").mkdir(exist_ok=True)
    (p / "logs").mkdir(exist_ok=True)
    (p / "artifacts").mkdir(exist_ok=True)
    return p
