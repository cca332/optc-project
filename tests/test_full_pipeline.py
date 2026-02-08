import sys
import os
import torch
import torch.nn as nn
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from optc_uras.features.deterministic_aggregator import AggregatorSchema
from optc_uras.features.quality import QualityWeightsConfig
from optc_uras.models.step1 import Step1Config, Step1Model
from optc_uras.models.student import StudentHeads, uras_from_subspaces
from optc_uras.models.detector import AnomalyDetector
from optc_uras.typing import Step1Outputs

def test_full_pipeline_integration():
    """
    Integration Test: Step 1 -> Step 2 -> Step 3
    Verifies data flow, shapes, and type consistency across the full OPTC pipeline.
    """
    print("\n[Test] Starting Full Pipeline Integration...")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    views = ["view_A", "view_B"]
    target_dim = 16          # Step 1 Output Dim
    subspace_dim = 8         # Step 2 Subspace Dim
    num_subspaces = 2        # Step 2 Num Subspaces
    uras_dim = num_subspaces * subspace_dim  # Step 3 Input Dim (16)
    
    # Step 1 Config
    quality_cfg = QualityWeightsConfig(
        weights={"completeness": 1.0}, 
        standardize="minmax"
    )
    step1_cfg = Step1Config(
        views=views,
        window_seconds=10,
        slot_seconds=1,
        include_empty_slot_indicator=True,
        num_hash_buckets=5,
        hash_seed=42,
        target_dim=target_dim,
        rp_seed=123,
        rp_matrix_type="gaussian",
        rp_normalize="l2",
        rp_nonlinearity="relu",
        quality_cfg=quality_cfg,
        router_hidden_dims=[32],
        router_dropout=0.1,
        num_subspaces=num_subspaces,
        gate_gamma=1.0,
        gate_mode="soft",
        gate_beta=1.0,
        interaction_enabled=True
    )
    
    per_view_schema = {
        v: AggregatorSchema(event_type_vocab=["evt1", "evt2"], key_fields=["id"]) 
        for v in views
    }

    print("[Test] Configs initialized.")

    # -------------------------------------------------------------------------
    # 2. Model Initialization
    # -------------------------------------------------------------------------
    # Step 1: Behavior Extraction
    step1_model = Step1Model(step1_cfg, per_view_schema)
    
    # Step 2: Student Heads (Project Step 1 output to Subspaces)
    student_heads = StudentHeads(
        in_dim=target_dim,
        num_subspaces=num_subspaces,
        subspace_dim=subspace_dim,
        hidden_dim=32
    )
    
    # Step 3: Anomaly Detector (A, B, C Modules)
    # Note: feature_dim matches URAS dim, view_dim matches Step 1 target_dim
    detector = AnomalyDetector(
        feature_dim=uras_dim,
        style_dim=8,
        content_dim=8,
        view_names=views,
        view_dim=target_dim  # Explicitly set view input dim
    )
    
    print("[Test] Models initialized.")

    # -------------------------------------------------------------------------
    # 3. Data Simulation & Vocab Fitting
    # -------------------------------------------------------------------------
    mock_samples = [
        {
            "views": {
                "view_A": [{"type": "evt1", "timestamp": 0}, {"type": "evt2", "timestamp": 5}],
                "view_B": [{"type": "evt1", "timestamp": 2}]
            }
        },
        {
            "views": {
                "view_A": [{"type": "evt2", "timestamp": 1}],
                "view_B": []
            }
        }
    ]
    
    # Initialize Step 1 Projectors and Stats
    step1_model.fit_vocabs_and_init_projectors(mock_samples)
    step1_model.fit_quality_stats(mock_samples)
    
    print("[Test] Vocabs and stats fitted.")

    # -------------------------------------------------------------------------
    # 4. Execution Flow
    # -------------------------------------------------------------------------
    
    # --- Step 1 Forward ---
    # Create a batch of 4 samples
    batch_samples = [mock_samples[0], mock_samples[1], mock_samples[0], mock_samples[1]]
    step1_outs: List[Step1Outputs] = step1_model(batch_samples)
    
    assert len(step1_outs) == 4
    print(f"[Test] Step 1 output count: {len(step1_outs)}")
    
    # Stack outputs for batch processing
    z_batch = torch.stack([out.z for out in step1_outs])          # [B, target_dim]
    route_p_batch = torch.stack([out.route_p for out in step1_outs]) # [B, num_subspaces]
    
    assert z_batch.shape == (4, target_dim)
    assert route_p_batch.shape == (4, num_subspaces)
    
    # --- Step 2 Forward ---
    # Generate Subspace Vectors
    subspace_vecs = student_heads(z_batch) # [B, M, d_s]
    assert subspace_vecs.shape == (4, num_subspaces, subspace_dim)
    
    # Generate URAS Features (Weighted Concatenation)
    uras_features = uras_from_subspaces(subspace_vecs, route_p_batch) # [B, M*d_s]
    assert uras_features.shape == (4, uras_dim)
    print(f"[Test] Step 2 URAS features shape: {uras_features.shape}")
    
    # Detach features for Step 3 training (treat as fixed data)
    uras_features_data = uras_features.detach()
    
    # --- Step 3 Module A/B: Detector Training (Benign) ---
    # Train SCD and fit statistics (B1, B2)
    detector.fit(uras_features_data, epochs=2, batch_size=2)
    
    assert detector.style_mu.shape == (8,)
    assert detector.style_inv_cov.shape == (8, 8)
    print("[Test] Step 3 Detector fitted.")
    
    # --- Step 3 Module C: Interpreter Training ---
    # Prepare view inputs from Step 1 outputs
    view_features_list = []
    view_weights_list = []
    for out in step1_outs:
        view_features_list.append(out.view_vecs)      # [V, target_dim]
        view_weights_list.append(out.reliability_w)   # [V]
        
    view_features_batch = torch.stack(view_features_list).detach() # [B, V, target_dim]
    view_weights_batch = torch.stack(view_weights_list).detach()   # [B, V]
    
    # Train view projectors (C1)
    detector.fit_interpreter(view_features_batch, view_weights_batch, uras_features_data, epochs=2, batch_size=2)
    print("[Test] Step 3 Interpreter fitted.")
    
    # --- Step 3 Inference & Interpretation ---
    detector.eval()
    
    # Detect (B4/B5)
    results = detector(uras_features_data, adapt=True)
    assert "score" in results
    assert "prob" in results
    assert results["anomaly"].shape == (4,)
    print(f"[Test] Detection scores: {results['score'].tolist()}")
    
    # Interpret (C2)
    rankings = detector.interpret(view_features_batch, view_weights_batch, uras_features_data, top_k=2)
    assert len(rankings) == 4
    for rank in rankings:
        assert len(rank) <= 2
        assert isinstance(rank[0], tuple) # (view_name, score)
        
    print(f"[Test] Interpretation example: {rankings[0]}")
    print("\n[Test] SUCCESS: Full pipeline integration verified.")

if __name__ == "__main__":
    test_full_pipeline_integration()