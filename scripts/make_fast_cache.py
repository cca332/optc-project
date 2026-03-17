
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import collections
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
from optc_uras.utils.slots import split_into_slots

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
    # Each worker gets its own model instance
    worker_step1 = Step1Model(step1_cfg, per_view_schema)
    worker_views = views
    worker_behavior_dim = behavior_dim
    worker_schema = per_view_schema

def process_indices(indices):
    """Worker function: Read data -> Compute Semantic Inputs & Stats"""
    global worker_dataset, worker_step1, worker_views, worker_behavior_dim, worker_schema
    
    batch_samples = []
    metadata = []
    
    for idx in indices:
        try:
            s = worker_dataset[idx]
            batch_samples.append(s)
            metadata.append({"host": s.get("host", "unknown"), "t0": s.get("t0", 0), "label": s.get("label", 0)})
        except Exception as e:
            print(f"Error reading index {idx}: {e}"); continue
            
    if not batch_samples: return None
        
    # 1. Behavior Features (Teacher input)
    b_feats = []
    for s in batch_samples:
        b_feats.append(behavior_features_from_sample(s, worker_views, out_dim=worker_behavior_dim))
    b_tensor = np.stack(b_feats)
    
    # 2. Semantic Inputs & Stats & Quality
    # We need to pack semantic inputs to ensure 100% logic identity with Step1Model._prepare_semantic_inputs
    N = len(batch_samples)
    V = len(worker_views)
    K = worker_step1.num_slots
    L = 20 # Max events per slot
    
    # Pre-allocate full matrices
    c_type = np.zeros((N, V, K, L), dtype=np.int32)
    c_op = np.zeros((N, V, K, L), dtype=np.int32)
    c_fine = np.zeros((N, V, K, L), dtype=np.int32)
    c_obj = np.zeros((N, V, K, L), dtype=np.int32)
    c_text = np.zeros((N, V, K, L), dtype=np.int32)
    c_masks = np.zeros((N, V, K, L, 10), dtype=np.float32)
    c_times = np.zeros((N, V, K, L, 2), dtype=np.float32)
    
    stat_vecs = np.zeros((N, V, K, worker_step1.aggregators[worker_views[0]].output_dim), dtype=np.float32)
    quality_metrics = np.zeros((N, V, 4), dtype=np.float32)
    
    for i, s in enumerate(batch_samples):
        for vi, v in enumerate(worker_views):
            raw_events = s["views"].get(v, [])
            slots = split_into_slots(raw_events, worker_step1.cfg.slot_seconds, worker_step1.cfg.window_seconds)
            
            # Quality (A5)
            key_fields = worker_schema[v].key_fields or []
            qv = worker_step1.quality.compute_view_quality(slots, key_fields)
            quality_metrics[i, vi] = [qv.get("validity", 0.0), qv.get("completeness", 0.0), qv.get("entropy", 0.0), qv.get("intensity", 0.0)]
            
            for ki, slot_events in enumerate(slots):
                if ki >= K: break
                # Stats (A2.2)
                stat_vecs[i, vi, ki] = worker_step1.aggregators[v](slot_events)
                
                # Tokens (A2.1) - Exact logic from _prepare_semantic_inputs
                if len(slot_events) > 0:
                    # worker_step1._prepare_semantic_inputs is a bit complex, let's replicate its core logic 
                    # but ensure it matches the 7-tuple return
                    res = worker_step1._prepare_semantic_inputs(slot_events[:L], v)
                    # res is (type, op, fine, obj, text, mask, time)
                    l_actual = res[0].shape[0]
                    c_type[i, vi, ki, :l_actual] = res[0].cpu().numpy()
                    c_op[i, vi, ki, :l_actual] = res[1].cpu().numpy()
                    c_fine[i, vi, ki, :l_actual] = res[2].cpu().numpy()
                    c_obj[i, vi, ki, :l_actual] = res[3].cpu().numpy()
                    c_text[i, vi, ki, :l_actual] = res[4].cpu().numpy()
                    c_masks[i, vi, ki, :l_actual] = res[5].cpu().numpy()
                    c_times[i, vi, ki, :l_actual] = res[6].cpu().numpy()
                    
    return b_tensor, c_type, c_op, c_fine, c_obj, c_text, c_masks, c_times, stat_vecs, quality_metrics, metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_production.yaml")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of parallel workers. Set to 0 for sequential processing (recommended on Windows).")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
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
    
    # 3. Parallel or Sequential Processing
    num_workers = args.num_workers
    batch_size = 500 if num_workers > 0 else 100
    all_indices = list(range(total_samples))
    batches = [all_indices[i:i+batch_size] for i in range(0, len(all_indices), batch_size)]
    
    results_map = {}
    
    if num_workers > 0:
        print(f"Starting Parallel Processing: {num_workers} workers, {len(batches)} batches...")
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
    else:
        print(f"Starting Sequential Processing: {len(batches)} batches...")
        # Initialize worker state in main process
        worker_init(cache_dir, s1_cfg, per_view_schema, views, model_cfg.get("behavior_dim", 128))
        
        for batch in tqdm(batches, desc="Processing"):
            start_idx = batch[0]
            try:
                res = process_indices(batch)
                if res is not None:
                    results_map[start_idx] = res
            except Exception as e:
                print(f"Task failed for batch starting at {start_idx}: {e}")
                import traceback
                traceback.print_exc()
                
    # 4. Reassemble in order
    print("Reassembling results in order...")
    behavior_results = []
    c_type_list = []
    c_op_list = []
    c_fine_list = []
    c_obj_list = []
    c_text_list = []
    c_masks_list = []
    c_times_list = []
    stat_results = []
    quality_results = []
    metadata_results = []
    
    for batch in batches:
        start_idx = batch[0]
        if start_idx in results_map:
            res = results_map[start_idx]
            behavior_results.append(res[0])
            c_type_list.append(res[1])
            c_op_list.append(res[2])
            c_fine_list.append(res[3])
            c_obj_list.append(res[4])
            c_text_list.append(res[5])
            c_masks_list.append(res[6])
            c_times_list.append(res[7])
            stat_results.append(res[8])
            quality_results.append(res[9])
            metadata_results.extend(res[10])
        else:
            print(f"Warning: Missing batch starting at {start_idx}")

    # 5. Save and Aggregate per Client
    print("Concatenating results...")
    if not behavior_results:
        print("Error: No data processed!")
        return

    # Client-level aggregation for omega_init
    # We aggregate acc_w_init and count_w_init per host
    host_to_acc_w = collections.defaultdict(lambda: torch.zeros(len(worker_views)))
    host_to_count = collections.defaultdict(int)
    
    # Also save individual quality vectors for training use
    X_quality = torch.from_numpy(np.concatenate(quality_results, axis=0)).float() # [N, V, 4]
    
    print("Aggregating metrics per Host for fast omega calculation...")
    for i, meta in enumerate(metadata_results):
        host = meta.get("host", "unknown")
        # q_vec: [V, 4]
        q_vec = X_quality[i]
        
        # We need to simulate how step1.forward_from_cache calculates reliability_w
        # For simplicity and correctness, let's just store the aggregated reliability_w 
        # using the INITIAL model state's quality standardizers (fit from a small subset).
        # client.py wants: acc_w_init += w_batch.sum(dim=0)
        
        # Calculate reliability_w for this sample using worker_step1.quality
        # qv is a dict: validity, completeness, entropy, intensity
        q_list = []
        f_list = []
        for vi in range(len(worker_views)):
            qv = {
                "validity": q_vec[vi, 0].item(),
                "completeness": q_vec[vi, 1].item(),
                "entropy": q_vec[vi, 2].item(),
                "intensity": q_vec[vi, 3].item()
            }
            q_list.append(qv)
            f_list.append(worker_step1.quality.fuse(qv))
        
        # final_w: [V]
        final_w = torch.from_numpy(worker_step1.quality.compute_final_weights(q_list, f_list))
        
        host_to_acc_w[host] += final_w
        host_to_count[host] += 1

    # Convert defaultdicts to regular dicts for saving
    client_profiles = {}
    for host in host_to_acc_w:
        client_profiles[host] = {
            "acc_w_init": host_to_acc_w[host],
            "count_w_init": host_to_count[host]
        }

    save_dict = {
        "behavior": torch.from_numpy(np.concatenate(behavior_results, axis=0)).float(),
        "c_type": torch.from_numpy(np.concatenate(c_type_list, axis=0)).int(),
        "c_op": torch.from_numpy(np.concatenate(c_op_list, axis=0)).int(),
        "c_fine": torch.from_numpy(np.concatenate(c_fine_list, axis=0)).int(),
        "c_obj": torch.from_numpy(np.concatenate(c_obj_list, axis=0)).int(),
        "c_text": torch.from_numpy(np.concatenate(c_text_list, axis=0)).int(),
        "c_masks": torch.from_numpy(np.concatenate(c_masks_list, axis=0)).float(),
        "c_times": torch.from_numpy(np.concatenate(c_times_list, axis=0)).float(),
        "stat_vecs": torch.from_numpy(np.concatenate(stat_results, axis=0)).float(),
        "quality": X_quality,
        "metadata": metadata_results,
        "client_profiles": client_profiles, # [HostID] -> {acc_w_init, count_w_init}
        "cfg": s1_cfg
    }
    
    save_path = os.path.join(cache_dir, "optimized_data.pt")
    print(f"Saving optimized cache to {save_path}...")
    torch.save(save_dict, save_path)
    
    print(f"Success! Saved {save_dict['behavior'].shape[0]} samples across {len(client_profiles)} hosts.")

if __name__ == "__main__":
    main()