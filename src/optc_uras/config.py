from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from omegaconf import OmegaConf


def _register_resolvers() -> None:
    # ${env:VAR,default}
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda k, default=None: os.environ.get(k, default))


def load_config(default_path: str, overrides: Optional[list[str]] = None) -> Any:
    """加载 YAML config，并支持 CLI 覆盖。

    - default_path: configs/default.yaml
    - overrides: 例如 ['step3.scd.enabled=false', 'paths.exp_name=exp1']
    """
    _register_resolvers()
    cfg = OmegaConf.load(default_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def save_config_snapshot(cfg: Any, path: str) -> None:
    _register_resolvers()
    oc = OmegaConf.create(cfg)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=oc, f=path)
