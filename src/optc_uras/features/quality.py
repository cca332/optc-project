from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np


def validity(events: Sequence[Dict[str, Any]]) -> float:
    """OpTC data is pre-cleaned JSONL, so validity is considered 1.0."""
    return 1.0


def completeness(events: Sequence[Dict[str, Any]], key_fields: Sequence[str]) -> float:
    if not events:
        return 1.0
    if not key_fields:
        return 1.0
    vals = []
    for e in events:
        for k in key_fields:
            # Support nested keys via dot notation (e.g., "properties.image_path")
            # or implicit lookup in "properties" dict
            v = None
            if "." in k:
                # Explicit dot notation
                parts = k.split(".")
                curr = e
                for p in parts:
                    if isinstance(curr, dict):
                        curr = curr.get(p)
                    else:
                        curr = None
                        break
                v = curr
            else:
                # Direct lookup + implicit properties fallback
                v = e.get(k)
                if v is None and "properties" in e and isinstance(e["properties"], dict):
                    v = e["properties"].get(k)
            
            vals.append(1.0 if (v is not None and str(v) != "") else 0.0)
    return float(np.mean(vals)) if vals else 1.0


def entropy_proxy(events: Sequence[Dict[str, Any]]) -> float:
    """A proxy for information entropy (Shannon Entropy) based on event keys."""
    if not events:
        return 0.0
    
    # 1. Count event keys
    keys = [f"{e.get('type','')}|{e.get('op','')}|{e.get('obj','')}" for e in events]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
        
    # 2. Compute probability distribution
    total = len(keys)
    probs = np.array(list(counts.values()), dtype=np.float32) / total
    
    # 3. Compute Shannon Entropy: H = -sum(p * log(p))
    # Add epsilon to avoid log(0)
    ent = -np.sum(probs * np.log(probs + 1e-9))
    
    # 4. Normalize by max possible entropy log(|K|) to get [0, 1]
    # |K| is number of unique keys
    max_ent = np.log(len(counts) + 1e-9)
    if max_ent < 1e-9:
        return 0.0
        
    return float(ent / max_ent)

@dataclass
class QualityWeightsConfig:
    # A6.1 Reliability Weights (val, com)
    reliability_weights: Dict[str, float] = None 
    # A6.2 Information Gain Config
    # Weights for combining Entropy and Intensity (e.g., 0.5/0.5)
    info_weights: Dict[str, float] = None
    info_gain_w_min: float = 0.3
    info_gain_temp: float = 1.0
    # A6.3 Softmax Temp
    softmax_temperature: float = 0.5
    standardize: str = "train_stats"

    def __post_init__(self):
        if self.reliability_weights is None:
            self.reliability_weights = {"validity": 1.0, "completeness": 1.0}
        if self.info_weights is None:
            # Balanced: Diversity (Entropy) and Volume (Intensity) are both important
            self.info_weights = {"entropy": 0.5, "intensity": 0.5}


class QualityWeighter:
    """A5+A6: Two-layer Quality Fusion (Reliability * Information Gain)."""

    def __init__(self, cfg: QualityWeightsConfig):
        self.cfg = cfg
        self._mu: Dict[str, float] = {}
        self._sig: Dict[str, float] = {}

    def fit_standardize_stats(self, qs: Dict[str, List[float]]) -> None:
        for k, arr in qs.items():
            a = np.asarray(arr, dtype=np.float32)
            self._mu[k] = float(a.mean()) if a.size else 0.0
            self._sig[k] = float(a.std() + 1e-6) if a.size else 1.0

    def _z(self, k: str, x: float) -> float:
        if not self._mu:
            return float(x)
        return float((x - self._mu.get(k, 0.0)) / self._sig.get(k, 1.0))
        
    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def compute_view_quality(self, slots: Sequence[Sequence[Dict[str, Any]]], key_fields: Sequence[str]) -> Dict[str, float]:
        # Coverage removed as per user request
        all_events = [e for s in slots for e in s]
        q = {
            "validity": validity(all_events),
            "completeness": completeness(all_events, key_fields),
            "entropy": entropy_proxy(all_events),
            # Intensity: Log-scaled event count to capture "Volume" (e.g., Brute Force)
            "intensity": np.log1p(len(all_events)),
        }
        return q

    def fuse(self, q: Dict[str, float]) -> float:
        """Calculates the RELIABILITY score (q_rel) before softmax."""
        s = 0.0
        # Only fuse reliability metrics (val, com)
        for k, w in self.cfg.reliability_weights.items():
            if k not in q:
                continue
            s += float(w) * self._z(k, q[k])
        return float(s)

    def compute_final_weights(self, q_per_view: List[Dict[str, float]], fused_rel_scores: List[float]) -> np.ndarray:
        """
        Computes final weights: beta_rel * w_info
        """
        V = len(q_per_view)
        tau = float(self.cfg.softmax_temperature)
        
        # 1. Reliability Weights (Softmax) -> beta_rel
        # q_rel already computed by fuse()
        x = np.asarray(fused_rel_scores, dtype=np.float32) / max(tau, 1e-6)
        x = x - x.max()
        e = np.exp(x)
        beta_rel = e / (e.sum() + 1e-8) # [V]
        
        # 2. Information Gain Weights (Sigmoid) -> w_info
        # w_info = w_min + (1 - w_min) * sigmoid( (w_ent*z_ent + w_int*z_int) / tau_e )
        w_info = np.zeros(V, dtype=np.float32)
        w_min = self.cfg.info_gain_w_min
        tau_e = self.cfg.info_gain_temp
        
        # Get weights for info components (default 0.5/0.5)
        w_ent_cfg = self.cfg.info_weights.get("entropy", 0.5)
        w_int_cfg = self.cfg.info_weights.get("intensity", 0.5)
        
        for i, q in enumerate(q_per_view):
            z_ent = self._z("entropy", q.get("entropy", 0.0))
            z_int = self._z("intensity", q.get("intensity", 0.0))
            
            # Combined Z-score for Information Gain
            z_combined = w_ent_cfg * z_ent + w_int_cfg * z_int
            
            sigmoid_val = self._sigmoid(z_combined / max(tau_e, 1e-6))
            w_info[i] = w_min + (1.0 - w_min) * sigmoid_val
            
        raw_w = beta_rel * w_info
        return raw_w / (raw_w.sum() + 1e-8)
        
        return final_w

    # Deprecated: kept for compatibility if needed, but logic moved to compute_final_weights
    def softmax_weights(self, fused_scores: Sequence[float]) -> np.ndarray:
        return self.compute_final_weights([{"entropy": 0.0}] * len(fused_scores), fused_scores)

