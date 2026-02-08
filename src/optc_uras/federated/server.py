from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import random

import torch
import torch.nn as nn

from .secure_agg import SecureAggregator


@dataclass
class ServerConfig:
    rounds: int
    client_fraction: float
    min_clients: int
    server_lr: float
    secure_agg_enabled: bool
    secure_agg_protocol: str


class FederatedServer:
    def __init__(self, clients: List[Any], cfg: ServerConfig, seed: int = 42):
        self.clients = clients
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.aggregator = SecureAggregator(enabled=cfg.secure_agg_enabled, protocol=cfg.secure_agg_protocol)

    def sample_clients(self) -> List[Any]:
        m = max(int(len(self.clients) * self.cfg.client_fraction), self.cfg.min_clients)
        m = min(m, len(self.clients))
        return self.rng.sample(self.clients, m)

    def aggregate_and_apply(self, params: List[torch.nn.Parameter], updates: List[torch.Tensor], ns: List[int], metrics: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        # FedAvg: weighted sum of client updates
        device = torch.nn.utils.parameters_to_vector(params).device
        weights = torch.tensor(ns, dtype=torch.float32, device=device)
        weights = weights / (weights.sum() + 1e-12)

        agg_update = self.aggregator.aggregate([u.to(device) for u in updates], weights)  # [P]
        cur = torch.nn.utils.parameters_to_vector(params).detach()
        new = cur + float(self.cfg.server_lr) * agg_update
        torch.nn.utils.vector_to_parameters(new, params)

        # Aggregate metrics if provided (Simulation Observability)
        agg_metrics = {}
        if metrics:
            # Aggregate beta_bar and omega
            # We assume all clients have same views
            total_beta = None
            total_omega = None
            view_names = []
            
            count = 0
            for m in metrics:
                if "view_stats" in m:
                    vs = m["view_stats"]
                    b = torch.tensor(vs.get("beta_bar", []), device=device)
                    o = torch.tensor(vs.get("omega", []), device=device)
                    if total_beta is None:
                        total_beta = torch.zeros_like(b)
                        total_omega = torch.zeros_like(o)
                        view_names = vs.get("view_names", [])
                    
                    if b.shape == total_beta.shape:
                        total_beta += b
                        total_omega += o
                        count += 1
            
            if count > 0:
                agg_metrics["avg_beta_bar"] = (total_beta / count).cpu().tolist()
                agg_metrics["avg_omega"] = (total_omega / count).cpu().tolist()
                agg_metrics["view_names"] = view_names

        return agg_metrics