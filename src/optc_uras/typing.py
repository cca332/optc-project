from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class Step1Outputs:
    """Step1 输出契约（Step2/Step3 共用）。

    约束：
      - 最小样本粒度是 host×15min
      - slots 仅内部使用，不作为对外定位输出
    """

    view_names: List[str]
    view_vecs: torch.Tensor                 # [V, d] 对齐后 view-level 表示
    reliability_w: torch.Tensor             # [V] 视图可靠性权重 w
    route_p: torch.Tensor                   # [M] 路由分布 p
    z: torch.Tensor                         # [d] 融合后的样本级表示
    intermediates: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PretrainSignals:
    confidence: torch.Tensor                # [B] 视图权重集中度置信
    route_uncertainty: torch.Tensor         # [B] 路由不确定性（熵）
    distill_error: torch.Tensor             # [B] 蒸馏误差（师生差异）
    privacy_risk: torch.Tensor              # [B] 隐私风险信号（无则 0）


@dataclass
class DetectorArtifacts:
    mu: torch.Tensor
    cov: torch.Tensor
    base_threshold: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionOutputs:
    score: float
    threshold: float
    pred: int
    evidence: Optional[List[Tuple[str, float]]] = None  # [(view_name, score)] 排名
