import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from optc_uras.data.dataset import OpTCEcarDataset
from optc_uras.features.quality import QualityWeighter, QualityWeightsConfig, completeness, validity, entropy_proxy

def main():
    # [FIX] Load config safely
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config not found at {config_path}, using defaults.")
        config = {"data": {"optc": {"cache_dir": "data/optc"}}}
    
    cache_dir = config.get("data", {}).get("optc", {}).get("cache_dir")
    # Fallback if cache_dir is relative
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", cache_dir))
        
    print(f"Loading dataset from {cache_dir}...")
    
    # Load a small subset of train data
    dataset = OpTCEcarDataset(cache_dir, split="train")
    if len(dataset) == 0:
        print("No data found.")
        return
        
    print(f"Loaded {len(dataset)} samples. Checking first 10...")
    
    # Config for metrics
    key_fields_map = {
        "process": ["action", "object", "image_path", "command_line"],
        "file": ["action", "object", "file_path"],
        "network": ["action", "object", "dest_ip", "dest_port"],
    }
    
    # Initialize QualityWeighter
    print("\nInitializing QualityWeighter...")
    cfg = QualityWeightsConfig(
        info_gain_w_min=0.3,
        info_gain_temp=0.1,  # Sharp sigmoid for demonstration
        softmax_temperature=0.5
    )
    weighter = QualityWeighter(cfg)
    
    # 1. Collect statistics from first 50 samples to calculate standardization parameters
    print("Collecting statistics from first 50 samples to fit Standardizer...")
    stats_data = {"validity": [], "completeness": [], "entropy": [], "intensity": []}
    
    samples_to_check = min(50, len(dataset))
    views = ["process", "file", "network"]
    
    for i in range(samples_to_check):
        sample = dataset[i]
        for v in views:
            events = sample["views"].get(v, [])
            kf = key_fields_map.get(v, [])
            # Directly invoke the actual QualityWeighter logic
            q = weighter.compute_view_quality([events], kf) 
            stats_data["validity"].append(q["validity"])
            stats_data["completeness"].append(q["completeness"])
            stats_data["entropy"].append(q["entropy"])
            stats_data["intensity"].append(q["intensity"])
            
    weighter.fit_standardize_stats(stats_data)
    print(f"Stats fitted (Real Data): Mean/Std = {weighter._mu} / {weighter._sig}")
    
    # 2. Detailed Verification on first 5 samples
    print("\n" + "="*80)
    print("VERIFICATION OF REAL ALGORITHM WEIGHT DIFFERENTIATION")
    print("="*80)
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        host = sample.get("host")
        print(f"\nSample {i} (Host: {host}):")
        
        q_per_view_list = []
        fused_scores_list = []
        
        # Header
        print(f"  {'View':<10} | {'Val':<6} {'Com':<6} {'Ent':<6} {'Int':<6} | {'Rel_Score':<9} | {'Info_W':<6} | {'FINAL_W':<7}")
        print(f"  {'-'*10} | {'-'*6} {'-'*6} {'-'*6} {'-'*6} | {'-'*9} | {'-'*6} | {'-'*7}")

        # Compute metrics per view
        for v in views:
            events = sample["views"].get(v, [])
            kf = key_fields_map.get(v, [])
            
            # Note: compute_view_quality expects slots (list of list), here we pass the full event list as one slot
            q = weighter.compute_view_quality([events], kf)
            
            # Reliability Score (Calculated by QualityWeighter.fuse)
            rel_score = weighter.fuse(q)
            
            q_per_view_list.append(q)
            fused_scores_list.append(rel_score)
            
        # Compute Final Weights (Calculated by QualityWeighter.compute_final_weights)
        final_weights = weighter.compute_final_weights(q_per_view_list, fused_scores_list)
        
        # Display
        for idx, v in enumerate(views):
            q = q_per_view_list[idx]
            fw = final_weights[idx]
            rel_s = fused_scores_list[idx]
            
            # Re-calculate Info Weight for display (it's internal to compute_final_weights)
            z_ent = weighter._z("entropy", q["entropy"])
            z_int = weighter._z("intensity", q["intensity"])
            # Use default 0.5/0.5 weights
            z_combined = 0.5 * z_ent + 0.5 * z_int
            info_w = cfg.info_gain_w_min + (1 - cfg.info_gain_w_min) * weighter._sigmoid(z_combined / cfg.info_gain_temp)
            
            print(f"  {v:<10} | {q['validity']:.4f} {q['completeness']:.4f} {q['entropy']:.4f} {q['intensity']:.4f} | {rel_s:.4f}    | {info_w:.4f} | {fw:.4f}")
            
        print(f"  Total Weight Sum: {final_weights.sum():.4f}")
        
        # Analysis
        max_w = final_weights.max()
        min_w = final_weights.min()
        if max_w - min_w > 0.1:
            print("  -> SIGNIFICANT Differentiation Detected (Delta > 0.1)")
        else:
            print("  -> Low Differentiation (Balanced Views)")

if __name__ == "__main__":
    main()