from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import torch

from ..models.step1 import Step1Model
from ..models.student import StudentHeads
from ..models.teacher import TeacherModel
from .client import ClientTrainConfig, FederatedClient
from .dp import FeatureDPConfig, GradDPConfig
from .server import FederatedServer, ServerConfig


def build_clients_by_host(samples: List[Dict[str, Any]]) -> List[FederatedClient]:
    buckets = defaultdict(list)
    for s in samples:
        buckets[str(s.get("host", "unknown"))].append(s)
    return [FederatedClient(cid, ss) for cid, ss in buckets.items()]


def run_federated_pretrain(
    samples: List[Dict[str, Any]],
    step1: Step1Model,
    student_heads: StudentHeads,
    teacher: TeacherModel,
    cfg: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    clients = build_clients_by_host(samples)
    server_cfg = ServerConfig(
        rounds=int(cfg["federated"]["rounds"]),
        client_fraction=float(cfg["federated"]["client_fraction"]),
        min_clients=int(cfg["federated"]["min_clients"]),
        server_lr=float(cfg["federated"]["server_lr"]),
        secure_agg_enabled=bool(cfg["dp"]["secure_aggregation"]["enabled"]),
        secure_agg_protocol=str(cfg["dp"]["secure_aggregation"]["protocol"]),
    )
    server = FederatedServer(clients, server_cfg, seed=int(cfg.get("seed", 42)))

    train_cfg = ClientTrainConfig(
        local_epochs=int(cfg["federated"]["local_epochs"]),
        batch_size=int(cfg["federated"]["batch_size"]),
        lr=float(cfg["federated"]["optimizer"]["lr"]),
        weight_decay=float(cfg["federated"]["optimizer"]["weight_decay"]),
        lambda_stats=float(cfg["losses"]["asd"]["lambda_stats"]),
        lambda_infonce=float(cfg["losses"]["at_infonce"]["lambda"]),
        temp_params=dict(cfg["losses"]["temperature_schedule"]["params"]),
        feature_dp=FeatureDPConfig(**cfg["dp"]["feature_dp"]),
        grad_dp=GradDPConfig(**cfg["dp"]["grad_dp"]),
        views=list(cfg.get("views", ["process", "file", "network"])),
        behavior_feature_dim=int(cfg["uras"]["behavior_feature_dim"]),
    )

    params = list(step1.parameters()) + list(student_heads.parameters())

    history = []
    for r in range(server_cfg.rounds):
        selected = server.sample_clients()
        updates, ns, mets = [], [], []
        for c in selected:
            u, m, n = c.local_train(step1, student_heads, teacher, train_cfg, device=device)
            updates.append(u)
            ns.append(n)
            mets.append(m)
        agg_metrics = server.aggregate_and_apply(params, updates, ns, metrics=mets)
        avg_loss = sum(x["client_loss"] for x in mets) / max(len(mets), 1)
        
        round_info = {
            "round": r, 
            "avg_client_loss": avg_loss, 
            "num_clients": len(selected),
            "server_metrics": agg_metrics
        }
        history.append(round_info)
    return {"history": history, "num_clients": len(clients)}