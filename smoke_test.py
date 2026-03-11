
import torch
import yaml
import sys
import os
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from optc_uras.models.step1 import Step1Model, Step1Config
from optc_uras.features.quality import QualityWeightsConfig
from optc_uras.features.deterministic_aggregator import AggregatorSchema
from optc_uras.models.detector import AnomalyDetector

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_smoke_test():
    print("=== Starting Smoke Test ===")
    
    # 1. Load Config
    print("[1] Loading Config...")
    cfg_path = "configs/final_production.yaml"
    if not os.path.exists(cfg_path):
        print(f"Error: {cfg_path} not found.")
        return
    
    raw_cfg = load_yaml(cfg_path)
    model_cfg = raw_cfg['model']
    data_cfg = raw_cfg['data']['optc']
    
    # 2. Construct Step1Config
    print("[2] Constructing Step1Config...")
    
    quality_cfg = QualityWeightsConfig(
        reliability_weights={"validity": 1.0, "completeness": 1.0},
        info_weights={"entropy": 0.5, "intensity": 0.5},
        softmax_temperature=model_cfg.get('quality_softmax_temp', 0.5),
        info_gain_w_min=0.3,
        info_gain_temp=1.0
    )
    
    step1_cfg = Step1Config(
        views=['file', 'network', 'process'],
        window_seconds=data_cfg.get('window_minutes', 5) * 60,
        slot_seconds=model_cfg.get('slot_seconds', 30),
        include_empty_slot_indicator=True,
        num_hash_buckets=model_cfg.get('num_hash_buckets', 50),
        hash_seed=42,
        target_dim=model_cfg.get('target_dim', 32),
        rp_seed=42,
        rp_matrix_type='gaussian',
        rp_normalize='l2',
        rp_nonlinearity='relu',
        quality_cfg=quality_cfg,
        router_hidden_dims=model_cfg.get('router_hidden_dims', [64]),
        router_dropout=0.1,
        num_subspaces=model_cfg.get('num_subspaces', 4),
        gate_gamma=model_cfg.get('gate_gamma', 0.5),
        gate_mode='soft', # inferred from beta existence
        gate_beta=model_cfg.get('gate_beta', 5.0),
        interaction_enabled=model_cfg.get('interaction_enabled', True),
        quality_injection_lambda=0.5
    )
    
    # 3. Construct Schemas
    print("[3] Constructing Schemas...")
    per_view_schema = {
        'file': AggregatorSchema(
            key_fields=['image_path', 'file_path'],
            event_type_vocab=['FILE']
        ),
        'network': AggregatorSchema(
            key_fields=['dest_ip', 'dest_port'],
            event_type_vocab=['FLOW']
        ),
        'process': AggregatorSchema(
            key_fields=['command_line', 'user'],
            event_type_vocab=['PROCESS', 'THREAD', 'image_load']
        )
    }
    
    # 4. Initialize Step 1 Model
    print("[4] Initializing Step1Model...")
    step1_model = Step1Model(step1_cfg, per_view_schema)
    step1_model.eval()
    
    # 5. Initialize Anomaly Detector (Step 2/3)
    print("[5] Initializing AnomalyDetector...")
    detector = AnomalyDetector(
        feature_dim=model_cfg.get('target_dim', 32),
        style_dim=model_cfg.get('subspace_dim', 16), # Using subspace_dim as style/content dim
        content_dim=model_cfg.get('subspace_dim', 16),
        alpha_conf=model_cfg['atc'].get('alpha_conf', 0.5),
        alpha_unc=model_cfg['atc'].get('alpha_unc', 0.5),
        alpha_risk=model_cfg['atc'].get('alpha_risk', 0.5),
        view_names=['file', 'network', 'process']
    )
    detector.eval()
    
    # 6. Generate Synthetic Data
    print("[6] Generating Synthetic Data...")
    # Create a batch of 2 samples
    batch = []
    for i in range(2):
        sample = {
            "timestamp": "2025-01-01T12:00:00",
            "host": f"host_{i}",
            "views": {
                "file": [
                    {"type": "FILE", "op": "OPEN", "image_path": "c:\\windows\\calc.exe", "delta_t": 0.1},
                    {"type": "FILE", "op": "WRITE", "file_path": "c:\\temp\\test.txt", "delta_t": 0.5}
                ],
                "network": [
                    {"type": "FLOW", "op": "CONNECT", "dest_ip": "192.168.1.1", "dest_port": 80, "delta_t": 0.2}
                ],
                "process": [] # Empty view
            }
        }
        batch.append(sample)
        
    # 7. Run Pipeline
    print("[7] Running Pipeline...")
    
    # Step 1 Forward
    print("  -> Step 1 Forward...")
    with torch.no_grad():
        step1_outputs = step1_model(batch)
    
    print(f"  <- Step 1 Output Count: {len(step1_outputs)}")
    z_batch = torch.stack([out.z for out in step1_outputs])
    print(f"  <- Step 1 Output Shape (z): {z_batch.shape}") # Should be [2, 32]
    
    # Step 2/3 Detection
    print("  -> Step 2/3 Detection...")
    with torch.no_grad():
        # Fake risk scores for ATC
        conf_scores = torch.rand(2)
        unc_scores = torch.rand(2)
        risk_scores = torch.rand(2)
        
        results = detector(
            uras_features=z_batch,
            conf_scores=conf_scores,
            unc_scores=unc_scores,
            risk_scores=risk_scores
        )
        
    print(f"  <- Detection Results: {results.keys()}")
    print(f"  <- Anomaly Scores: {results['score']}")
    print(f"  <- Probabilities: {results['prob']}")
    print(f"  <- Thresholds: {results['threshold']}")
    print(f"  <- Is Anomaly: {results['anomaly']}")
    
    print("=== Smoke Test Passed ===")

if __name__ == "__main__":
    run_smoke_test()
