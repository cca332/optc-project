from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..typing import DetectorArtifacts, DetectionOutputs, Step1Outputs
from ..models.student import StudentHeads, uras_from_subspaces
from ..models.step1 import Step1Model
from .scd import SCDHead, SCDTrainConfig, train_scd
from .mahalanobis import CovConfig, fit_gaussian, mahalanobis_score
from .thresholding import quantile_threshold
from .atc import ATCConfig, atc_threshold
from .drift import DriftConfig, maybe_update
from .explain import ExplainConfig, ViewStyleProjectors, evidence_ranking


@dataclass
class Step3Config:
    use_scd: bool
    style_dim: int
    content_dim: int
    scd_train: SCDTrainConfig
    cov: CovConfig
    q: float
    use_squared: bool
    atc: ATCConfig
    drift: DriftConfig
    explain: ExplainConfig


class DetectorPipeline:
    def __init__(self, step1: Step1Model, student_heads: StudentHeads, cfg: Step3Config, device: str):
        self.step1 = step1
        self.student_heads = student_heads
        self.cfg = cfg
        self.device = device

        self.scd_head: Optional[SCDHead] = None
        self.explain_proj: Optional[ViewStyleProjectors] = None
        self.artifacts: Optional[DetectorArtifacts] = None

    def _repr(self, step1_out: Step1Outputs) -> torch.Tensor:
        # representation source fixed to URAS in this skeleton
        z = step1_out.z.to(self.device).unsqueeze(0)  # [1,d]
        p = step1_out.route_p.to(self.device).unsqueeze(0)  # [1,M]
        sub = self.student_heads(z, normalize=True)  # [1,M,d_s]
        uras = uras_from_subspaces(sub, p)  # [1, M*d_s]
        uras = torch.nn.functional.normalize(uras, dim=-1)
        return uras.squeeze(0)

    def fit(self, benign_samples: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        self.step1.eval()
        self.student_heads.eval()

        reps = []
        view_vecs = []
        ws = []
        for s in benign_samples:
            out = self.step1.forward_single(s)
            reps.append(self._repr(out))
            view_vecs.append(out.view_vecs)
            ws.append(out.reliability_w)
        X = torch.stack(reps, dim=0).to(self.device)  # [N, D]

        logs: Dict[str, float] = {}

        if self.cfg.use_scd:
            self.scd_head = SCDHead(in_dim=X.shape[1], style_dim=self.cfg.style_dim, content_dim=self.cfg.content_dim).to(self.device)
            logs |= train_scd(self.scd_head, X, self.cfg.scd_train)
            with torch.no_grad():
                style, _, _ = self.scd_head(X)
                X_fit = style
        else:
            X_fit = X

        mu, cov = fit_gaussian(X_fit, self.cfg.cov)
        scores = mahalanobis_score(X_fit, mu, cov, use_squared=self.cfg.use_squared)
        tau0 = quantile_threshold(scores, self.cfg.q)

        self.artifacts = DetectorArtifacts(mu=mu.detach().cpu(), cov=cov.detach().cpu(), base_threshold=float(tau0), extra={})

        # explanation projectors (optional)
        if self.cfg.explain.enabled and self.scd_head is not None:
            self.explain_proj = ViewStyleProjectors(self.step1.views, in_dim=self.step1.cfg.target_dim, style_dim=self.cfg.style_dim).to(self.device)

        logs["base_threshold"] = float(tau0)
        logs["benign_score_mean"] = float(scores.mean().item())
        return logs

    def infer_one(self, sample: Dict[str, Any], label: Optional[int] = None) -> DetectionOutputs:
        assert self.artifacts is not None, "call fit() first"
        out = self.step1.forward_single(sample)
        x = self._repr(out).to(self.device).unsqueeze(0)

        if self.scd_head is not None:
            self.scd_head.eval()
            with torch.no_grad():
                style, _, _ = self.scd_head(x)
                x_fit = style
        else:
            x_fit = x

        mu = self.artifacts.mu.to(self.device)
        cov = self.artifacts.cov.to(self.device)
        score = float(mahalanobis_score(x_fit, mu, cov, use_squared=self.cfg.use_squared).item())

        # signals
        confidence = 1.0 - float(out.intermediates.get("w_entropy", 0.0)) if "w_entropy" in out.intermediates else float(1.0)
        route_unc = float(out.intermediates.get("route_entropy", 0.0)) if "route_entropy" in out.intermediates else float(0.0)
        privacy_risk = 0.0

        tau = atc_threshold(self.artifacts.base_threshold, confidence, route_unc, privacy_risk, self.cfg.atc)
        pred = 1 if score > tau else 0

        # drift update
        if self.cfg.drift.enabled:
            new_mu, new_cov = maybe_update(mu, cov, x_fit.squeeze(0), score, tau, self.cfg.drift)
            self.artifacts.mu = new_mu.detach().cpu()
            self.artifacts.cov = new_cov.detach().cpu()

        evidence = None
        if self.cfg.explain.enabled and self.explain_proj is not None and self.scd_head is not None:
            # delta direction
            with torch.no_grad():
                style, _, _ = self.scd_head(x)
                delta = (style.squeeze(0) - mu)
                evidence = evidence_ranking(out.view_vecs.to(self.device), out.reliability_w.to(self.device), delta, self.explain_proj, self.cfg.explain)

        return DetectionOutputs(score=score, threshold=float(tau), pred=pred, evidence=evidence)
