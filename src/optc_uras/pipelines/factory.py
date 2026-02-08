from __future__ import annotations

from typing import Any, Dict, Tuple

from ..features.deterministic_aggregator import AggregatorSchema
from ..features.quality import QualityWeightsConfig
from ..models.step1 import Step1Config, Step1Model
from ..models.student import StudentHeads
from ..models.teacher import TeacherModel


def build_step1(cfg: Dict[str, Any]) -> Step1Model:
    views = list(cfg["data"]["views"])
    window_seconds = int(cfg["data"]["sample_granularity_minutes"]) * 60
    slot_seconds = int(cfg["step1"]["time"]["slot_minutes"]) * 60

    schema = {}
    per_view_schema_cfg = cfg["step1"]["aggregator"]["per_view_schema"]
    for v in views:
        sc = per_view_schema_cfg.get(v, {})
        schema[v] = AggregatorSchema(
            event_type_vocab=sc.get("event_type_vocab", None),
            key_fields=sc.get("key_fields", None),
        )

    qcfg = QualityWeightsConfig(
        weights=dict(cfg["step1"]["quality"]["weights"]),
        softmax_temperature=float(cfg["step1"]["quality"]["softmax_temperature"]),
        standardize=str(cfg["step1"]["quality"]["standardize"]),
    )

    s1cfg = Step1Config(
        views=views,
        window_seconds=window_seconds,
        slot_seconds=slot_seconds,
        include_empty_slot_indicator=bool(cfg["step1"]["time"]["include_empty_slot_indicator"]),
        num_hash_buckets=int(cfg["step1"]["aggregator"]["hash"]["num_buckets"]),
        hash_seed=int(cfg["step1"]["aggregator"]["hash"]["seed"]),
        target_dim=int(cfg["step1"]["random_projection"]["target_dim"]),
        rp_seed=int(cfg["step1"]["random_projection"]["seed"]),
        rp_matrix_type=str(cfg["step1"]["random_projection"]["matrix_type"]),
        rp_normalize=str(cfg["step1"]["random_projection"]["normalize"]),
        rp_nonlinearity=str(cfg["step1"]["random_projection"]["nonlinearity"]),
        quality_cfg=qcfg,
        router_hidden_dims=list(cfg["step1"]["routing_alignment"]["router"]["hidden_dims"]),
        router_dropout=float(cfg["step1"]["routing_alignment"]["router"]["dropout"]),
        num_subspaces=int(cfg["step1"]["routing_alignment"]["num_subspaces"]),
        gate_gamma=float(cfg["step1"]["gating_fusion"]["pearson_gate"]["gamma"]),
        gate_mode=str(cfg["step1"]["gating_fusion"]["pearson_gate"]["mode"]),
        gate_beta=float(cfg["step1"]["gating_fusion"]["pearson_gate"]["beta"]),
        interaction_enabled=bool(cfg["step1"]["gating_fusion"]["interaction"]["enabled"]),
    )
    return Step1Model(s1cfg, schema)


def build_teacher_student(cfg: Dict[str, Any], step1: Step1Model) -> Tuple[TeacherModel, StudentHeads]:
    M = int(cfg["step1"]["routing_alignment"]["num_subspaces"])
    d_s = int(cfg["step2"]["uras"]["subspace_dim"])
    d_b = int(cfg["step2"]["uras"]["behavior_feature_dim"])
    teacher = TeacherModel(behavior_dim=d_b, num_subspaces=M, subspace_dim=d_s, hidden_dim=256)
    student_heads = StudentHeads(in_dim=step1.cfg.target_dim, num_subspaces=M, subspace_dim=d_s, hidden_dim=256)
    return teacher, student_heads
