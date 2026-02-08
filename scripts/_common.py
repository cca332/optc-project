from __future__ import annotations

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'src'))


import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from optc_uras.config import load_config, save_config_snapshot
from optc_uras.utils.seed import set_global_seed
from optc_uras.utils.path import make_run_dir


def parse_args(desc: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", type=str, required=True, help="Path to configs/default.yaml or a merged config file")
    p.add_argument("--override", type=str, action="append", default=[], help="Dotlist override, e.g. step3.scd.enabled=false")
    return p.parse_args()


def setup(cfg_path: str, overrides: List[str]) -> tuple[Dict[str, Any], Path]:
    cfg = load_config(cfg_path, overrides=overrides)
    set_global_seed(int(cfg["project"]["seed"]), deterministic=bool(cfg["project"]["deterministic"]))
    run_dir = make_run_dir(cfg["paths"]["run_root"], cfg["paths"]["exp_name"], add_timestamp=True)
    if bool(cfg["io"]["save_config_snapshot"]):
        save_config_snapshot(cfg, str(run_dir / "config_snapshot.yaml"))
    return cfg, run_dir
