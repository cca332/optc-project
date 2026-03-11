
import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from optc_uras.data.dataset import OpTCEcarDataset
from optc_uras.models.step1 import Step1Config, Step1Model
from optc_uras.features.deterministic_aggregator import AggregatorSchema
from optc_uras.features.quality import QualityWeightsConfig
from optc_uras.features.behavior_features import behavior_features_from_sample
from optc_uras.features.random_projection import FixedRandomProjector

# Global worker state
worker_dataset = None
worker_step1 = None
worker_views = None
worker_behavior_dim = None
worker_schema = None

def worker_init(cache_dir, step1_cfg, per_view_schema, views, behavior_dim):
    """Initialize worker state: Load Dataset Index & Step1 Model"""
    global worker_dataset, worker_step1, worker_views, worker_behavior_dim, worker_schema
    
    # 1. Load Dataset (Index only, fast)
    # Each worker gets its own dataset instance reading from disk
    worker_dataset = OpTCEcarDataset(cache_dir, split="all", preload=False)
    
    # 2. Init Step1 Model (CPU)
    # We only need the deterministic parts (Aggregators, Projectors)
    worker_step1 = Step1Model(step1_cfg, per_view_schema)
    worker_views = views
    worker_behavior_dim = behavior_dim
    worker_schema = per_view_schema
    
    # 3. Manually Init Projectors (Deterministic)
    for v in views:
        ag = worker_step1.aggregators[v]
        if not ag.event_type_vocab:
            ag.event_type_vocab = per_view_schema[v].event_type_vocab
            
        worker_step1.projectors[v] = FixedRandomProjector(
            in_dim=ag.output_dim,
            out_dim=step1_cfg.target_dim,
            seed=step1_cfg.rp_seed + (abs(hash(v)) % 997),
            matrix_type=step1_cfg.rp_matrix_type,
            normalize_mode=step1_cfg.rp_normalize,
            nonlinearity=step1_cfg.rp_nonlinearity,
        )
    worker_step1._projectors_ready = True

def process_indices(indices):
    """Worker function: Read data -> Compute Features"""
    global worker_dataset, worker_step1, worker_views, worker_behavior_dim, worker_schema
    
    batch_samples = []
    metadata = []
    
    # 1. IO: Read from disk (Parallel IO)
    for idx in indices:
        try:
            s = worker_dataset[idx]
            batch_samples.append(s)
            metadata.append({
                "host": s.get("host", "unknown"),
                "t0": s.get("t0", 0),
                "label": s.get("label", 0)
            })
        except Exception as e:
            print(f"Error reading index {idx}: {e}")
            continue
            
    if not batch_samples:
        return None
        
    # 2. Compute: Behavior Features
    b_feats = []
    for s in batch_samples:
        f = behavior_features_from_sample(s, worker_views, out_dim=worker_behavior_dim)
        b_feats.append(f)
    b_tensor = np.stack(b_feats)
    
    # 3. Compute: Step1 Slots & Quality
    slot_list = []
    quality_list = []
    
    for s in batch_samples:
        v_slots = []
        q_metrics = []
        for v in worker_views:
            slot_emb, slots_raw = worker_step1._extract_view_slots(s, v)
            v_slots.append(slot_emb)
            
            key_fields = worker_schema[v].key_fields or []
            qv = worker_step1.quality.compute_view_quality(slots_raw, key_fields)
            # [FIX] Save all 4 metrics: validity, completeness, entropy, intensity
            q_vec = [
                qv.get("validity", 0.0), 
                qv.get("completeness", 0.0), 
                qv.get("entropy", 0.0), 
                qv.get("intensity", 0.0)
            ]
            q_metrics.append(q_vec)
            
        slot_list.append(np.stack(v_slots, axis=1))
        quality_list.append(np.stack(q_metrics, axis=0))
        
    return b_tensor, np.stack(slot_list), np.stack(quality_list), metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_production.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    optc_cfg = config["data"]["optc"]
    model_cfg = config["model"]
    cache_dir = optc_cfg["cache_dir"]
    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Fit/Load Schema (Main Process)
    print("Preparing Schema...")
    vocab_path = os.path.join(output_dir, "vocab_schema.pt")
    
    # [UPDATED] Reverted to 3 Views
    views = ["process", "file", "network"]
    
    temp_dataset = OpTCEcarDataset(cache_dir, split="all", preload=False)
    total_samples = len(temp_dataset)
    print(f"Total samples to process: {total_samples}")
    
    if os.path.exists(vocab_path):
        print(f"Loading existing schema from {vocab_path}")
        per_view_schema = torch.load(vocab_path, weights_only=False)
    else:
        print("Fitting new schema (vocab)...")
        # indices = np.random.choice(total_samples, min(2000, total_samples), replace=False)
        # Use sequential indices to avoid massive I/O during schema fitting
        indices = np.arange(min(200, total_samples))
        samples = [temp_dataset[i] for i in indices]
        
        per_view_schema = {v: AggregatorSchema(event_type_vocab=[], key_fields=["type", "op", "obj"]) for v in views}
        
        # [UPDATED] Key fields for quality metrics based on new extracted fields (Merged for 3 views)
        key_fields_map = {
            "process": ["type", "op", "obj", "image_path", "command_line", "module_path", "payload"], 
            "file": ["type", "op", "obj", "file_path", "key", "value"], 
            "network": ["type", "op", "obj", "dest_ip", "dest_port"],
        }
        for v, schema in per_view_schema.items():
            if v in key_fields_map: schema.key_fields = key_fields_map[v]
        
        for v in views:
            all_events = []
            for s in samples:
                all_events.extend(s["views"].get(v, []))
            from optc_uras.features.deterministic_aggregator import DeterministicStatsAggregator
            ag = DeterministicStatsAggregator(per_view_schema[v], 50, 42, True)
            ag.fit_event_type_vocab(all_events, max_types=model_cfg.get("max_vocab_types", 500))
        
        torch.save(per_view_schema, vocab_path)
        print(f"Saved schema to {vocab_path}")
    
    del temp_dataset
    
    # 2. Config
    quality_cfg = QualityWeightsConfig(
        reliability_weights={"validity": 1.0, "completeness": 1.0}, 
        info_gain_w_min=0.3, 
        info_gain_temp=1.0, 
        standardize="minmax", 
        softmax_temperature=model_cfg.get("quality_softmax_temp", 0.5)
    )
    
    s1_cfg = Step1Config(
        views=views, 
        window_seconds=optc_cfg.get("window_minutes", 5)*60, 
        slot_seconds=model_cfg.get("slot_seconds", 60),
        include_empty_slot_indicator=True, 
        num_hash_buckets=model_cfg.get("num_hash_buckets", 50), 
        hash_seed=42, 
        target_dim=model_cfg.get("target_dim", 32),
        rp_seed=123, 
        rp_matrix_type="gaussian", 
        rp_normalize="l2", 
        rp_nonlinearity="relu", 
        quality_cfg=quality_cfg,
        router_hidden_dims=model_cfg.get("router_hidden_dims", [64]), 
        router_dropout=0.1, 
        num_subspaces=model_cfg.get("num_subspaces", 4),
        gate_gamma=model_cfg.get("gate_gamma", 0.5), 
        gate_mode="soft", 
        gate_beta=model_cfg.get("gate_beta", 5.0), 
        interaction_enabled=True
    )
    
    # 3. Parallel Processing
    num_workers = 4 # Use 4 workers
    batch_size = 500 
    all_indices = list(range(total_samples))
    batches = [all_indices[i:i+batch_size] for i in range(0, len(all_indices), batch_size)]
    
    print(f"Starting Parallel Processing: {num_workers} workers, {len(batches)} batches...")
    
    # Map start_index -> result
    results_map = {}
    
    with ProcessPoolExecutor(max_workers=num_workers, 
                             initializer=worker_init, 
                             initargs=(cache_dir, s1_cfg, per_view_schema, views, model_cfg.get("behavior_dim", 128))) as executor:
        
        futures = {executor.submit(process_indices, batch): batch[0] for batch in batches}
        
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            start_idx = futures[f]
            try:
                res = f.result()
                if res is not None:
                    results_map[start_idx] = res
            except Exception as e:
                print(f"Task failed for batch starting at {start_idx}: {e}")
                import traceback
                traceback.print_exc()
                
    # 4. Reassemble in order
    print("Reassembling results in order...")
    behavior_results = []
    slot_results = []
    quality_results = []
    metadata_results = []
    
    # Iterate through batches in original order
    for batch in batches:
        start_idx = batch[0]
        if start_idx in results_map:
            b, s, q, m = results_map[start_idx]
            behavior_results.append(b)
            slot_results.append(s)
            quality_results.append(q)
            metadata_results.extend(m)
        else:
            print(f"Warning: Missing batch starting at {start_idx}")

    # 5. Save
    print("Concatenating results...")
    if not behavior_results:
        print("Error: No data processed!")
        return

    X_behavior = torch.from_numpy(np.concatenate(behavior_results, axis=0)).float()
    X_slots = torch.from_numpy(np.concatenate(slot_results, axis=0)).float()
    X_quality = torch.from_numpy(np.concatenate(quality_results, axis=0)).float()
    
    save_path = os.path.join(cache_dir, "optimized_data.pt")
    
    print(f"Saving optimized cache to {save_path}...")
    torch.save({
        "behavior": X_behavior,
        "slots": X_slots,
        "quality": X_quality, # [N, V, 4]
        "metadata": metadata_results,
        "cfg": s1_cfg
    }, save_path)
    
    print(f"Success! Saved {X_behavior.shape[0]} samples.")

if __name__ == "__main__":
    main()
