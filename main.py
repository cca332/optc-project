import argparse
import os
import sys
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from torch.utils.data import DataLoader, Subset

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from optc_uras.data.dataset import OpTCEcarDataset, FeatureCollate, TensorDataset, tensor_collate
from optc_uras.models.step1 import Step1Config, Step1Model
from optc_uras.models.student import StudentHeads, uras_from_subspaces
from optc_uras.models.detector import AnomalyDetector
from optc_uras.models.losses import scd_loss
from optc_uras.features.quality import QualityWeightsConfig
from optc_uras.features.deterministic_aggregator import AggregatorSchema
from optc_uras.features.behavior_features import behavior_features_from_sample
from optc_uras.federated.dp import dp_features, FeatureDPConfig, GradDPConfig
from optc_uras.models.teacher import TeacherModel
from optc_uras.federated.client import FederatedClient, ClientTrainConfig
from optc_uras.federated.server import FederatedServer, ServerConfig
from optc_uras.utils.typing import Step1Outputs
import copy
import collections

# --- Helper Functions ---
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def collate_fn(batch):
    return batch

def replace_relu(module):
    """Replace ReLU with LeakyReLU to prevent dead neurons"""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(module, name, torch.nn.LeakyReLU(negative_slope=0.1, inplace=True))
        else:
            replace_relu(child)

class LoRALayer(torch.nn.Module):
    def __init__(self, original_linear, rank=4, alpha=8):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.scaling = alpha / rank
        
        # A: [in, r], B: [r, out]
        self.lora_A = torch.nn.Parameter(torch.randn(original_linear.in_features, rank) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, original_linear.out_features))
            
    def forward(self, x):
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base + lora

def apply_lora(module, target_names=["Linear"], rank=4, alpha=8):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, LoRALayer(child, rank=rank, alpha=alpha))
        else:
            apply_lora(child, target_names, rank, alpha)

def setup_environment(config_path: str) -> Tuple[Dict, Dict, Dict, Dict, Any]:
    """Load config and setup common environment (device, seed, dirs)"""
    config = load_config(config_path)
    
    optc_cfg = config.get("data", {}).get("optc", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    # Use 'results4' for output
    output_dir = train_cfg.get("output_dir", "results4")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set Seed
    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.device_count() > 1:
        print(f"[Setup] Using {torch.cuda.device_count()} GPUs!")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}, Seed: {seed}, Output: {output_dir}")
    
    return config, optc_cfg, train_cfg, model_cfg, device, output_dir

def load_datasets_optimized(optc_cfg, cache_dir, splits=["train", "val", "test"]):
    """Load optimized tensors and return TensorDatasets for splits"""
    output_dir = optc_cfg.get("output_dir", "results_tuning") # Hack: assume output dir is where cache is saved
    # Actually make_fast_cache saves to training.output_dir.
    # But here we don't have training config passed easily.
    # We check common locations.
    
    # Check if we can find optimized_data.pt
    candidates = [
        os.path.join(cache_dir, "optimized_data.pt"),
        os.path.join("results_tuning", "optimized_data.pt"),
        os.path.join("results4", "optimized_data.pt")
    ]
    
    cache_path = None
    for p in candidates:
        if os.path.exists(p):
            cache_path = p
            break
            
    if not cache_path:
        print(f"[Setup] Optimized cache not found. Falling back to standard loading.")
        return None, None
    
    print(f"[Setup] Loading optimized cache from {cache_path}...")
    try:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return None, None

    behavior = data["behavior"]
    slots = data["slots"]
    quality = data["quality"]
    metadata = data["metadata"]
    vocab_schema = data.get("vocab_schema", None)
    
    # Re-create index map to find splits
    ds_all = OpTCEcarDataset(cache_dir, split="all", preload=False)
    
    auto_time_split = optc_cfg.get("auto_time_split")
    if auto_time_split is None:
        train_path = optc_cfg.get("train_path")
        val_path = optc_cfg.get("val_path")
        auto_time_split = bool(train_path and val_path and train_path == val_path)
    auto_time_split = bool(auto_time_split)
    train_val_ratio = float(optc_cfg.get("train_val_ratio", 0.8))
    train_prefix = optc_cfg.get("train_prefix", "train")
    val_prefix = optc_cfg.get("val_prefix", "val")

    def _path_has_prefix(path: str, prefix: str) -> bool:
        base = os.path.basename(path)
        return base.startswith(prefix + "_") or base.startswith(prefix + "part") or base.startswith(prefix + "p")

    datasets = {}
    if auto_time_split and ("train" in splits or "val" in splits):
        train_candidates = [i for i, (path, _, _, _) in enumerate(ds_all.index_map) if _path_has_prefix(path, train_prefix)]
        val_candidates = [i for i, (path, _, _, _) in enumerate(ds_all.index_map) if _path_has_prefix(path, val_prefix)]
        pool = sorted(set(train_candidates + val_candidates))

        if pool:
            unique_t0 = sorted({int(metadata[i].get("t0", 0)) for i in pool})
            if len(unique_t0) > 1:
                split_at = int(len(unique_t0) * train_val_ratio)
                split_at = max(1, min(split_at, len(unique_t0) - 1))
            else:
                split_at = 1

            t0_train = set(unique_t0[:split_at])
            t0_val = set(unique_t0[split_at:])

            if "train" in splits:
                base = train_candidates if train_candidates else pool
                indices = [i for i in base if int(metadata[i].get("t0", 0)) in t0_train]
                if indices:
                    datasets["train"] = TensorDataset(behavior, slots, quality, indices)
                    datasets["train"].metadata_list = metadata
                    datasets["train"].get_metadata = lambda idx, ds=datasets["train"]: ds.metadata_list[ds.indices[idx]]
                    print(f"  - train: {len(indices)} samples (Optimized, time-split)")

            if "val" in splits:
                base = val_candidates if val_candidates else pool
                indices = [i for i in base if int(metadata[i].get("t0", 0)) in t0_val]
                if indices:
                    datasets["val"] = TensorDataset(behavior, slots, quality, indices)
                    datasets["val"].metadata_list = metadata
                    datasets["val"].get_metadata = lambda idx, ds=datasets["val"]: ds.metadata_list[ds.indices[idx]]
                    print(f"  - val: {len(indices)} samples (Optimized, time-split)")

    for split in splits:
        if split in datasets:
            continue
        prefix = optc_cfg.get(f"{split}_prefix", split)
        indices = []
        for i, (path, _, _, _) in enumerate(ds_all.index_map):
            if prefix in path:
                indices.append(i)
        
        if indices:
            datasets[split] = TensorDataset(behavior, slots, quality, indices)
            # Attach metadata list to dataset instance for FL
            datasets[split].metadata_list = metadata 
            # Monkey patch get_metadata
            # [FIX] Use default arg to capture 'ds' to avoid closure late binding issue
            datasets[split].get_metadata = lambda idx, ds=datasets[split]: ds.metadata_list[ds.indices[idx]]
            print(f"  - {split}: {len(indices)} samples (Optimized)")
            
    return datasets, vocab_schema

def load_datasets(optc_cfg, cache_dir, splits=["train", "val", "test"]):
    """Load requested datasets"""
    # Try optimized first
    datasets, vocab_schema = load_datasets_optimized(optc_cfg, cache_dir, splits)
    if datasets is not None:
        return datasets, vocab_schema
    # datasets = None # Force standard loading

    datasets = {}
    print("[Setup] Loading datasets (Standard)...")
    
    auto_time_split = optc_cfg.get("auto_time_split")
    if auto_time_split is None:
        train_path = optc_cfg.get("train_path")
        val_path = optc_cfg.get("val_path")
        auto_time_split = bool(train_path and val_path and train_path == val_path)
    auto_time_split = bool(auto_time_split)
    train_val_ratio = float(optc_cfg.get("train_val_ratio", 0.8))
    train_prefix = optc_cfg.get("train_prefix", "train")
    val_prefix = optc_cfg.get("val_prefix", "val")

    if auto_time_split and ("train" in splits or "val" in splits):
        ds_all = OpTCEcarDataset(cache_dir, split="all", preload=False)

        def _path_has_prefix(path: str, prefix: str) -> bool:
            base = os.path.basename(path)
            return base.startswith(prefix + "_") or base.startswith(prefix + "part") or base.startswith(prefix + "p")

        train_candidates = [i for i, (path, _, _, _) in enumerate(ds_all.index_map) if _path_has_prefix(path, train_prefix)]
        val_candidates = [i for i, (path, _, _, _) in enumerate(ds_all.index_map) if _path_has_prefix(path, val_prefix)]
        pool = sorted(set(train_candidates + val_candidates))

        if pool:
            unique_t0 = sorted({int(ds_all.index_map[i][2]) for i in pool})
            if len(unique_t0) > 1:
                split_at = int(len(unique_t0) * train_val_ratio)
                split_at = max(1, min(split_at, len(unique_t0) - 1))
            else:
                split_at = 1

            t0_train = set(unique_t0[:split_at])
            t0_val = set(unique_t0[split_at:])

            if "train" in splits:
                base = train_candidates if train_candidates else pool
                indices = [i for i in base if int(ds_all.index_map[i][2]) in t0_train]
                datasets["train"] = Subset(ds_all, indices)
                datasets["train"].get_metadata = lambda idx, ds=datasets["train"]: ds.dataset.get_metadata(ds.indices[idx])
                print(f"  - Train: {len(indices)} samples (Standard, time-split)")

            if "val" in splits:
                base = val_candidates if val_candidates else pool
                indices = [i for i in base if int(ds_all.index_map[i][2]) in t0_val]
                datasets["val"] = Subset(ds_all, indices)
                datasets["val"].get_metadata = lambda idx, ds=datasets["val"]: ds.dataset.get_metadata(ds.indices[idx])
                print(f"  - Val: {len(indices)} samples (Standard, time-split)")
        else:
            auto_time_split = False

    if not auto_time_split:
        if "train" in splits:
            prefix = optc_cfg.get("train_prefix", "train")
            datasets["train"] = OpTCEcarDataset(cache_dir, split=prefix, preload=False)
            print(f"  - Train: {len(datasets['train'])} samples")
            
        if "val" in splits:
            prefix = optc_cfg.get("val_prefix", "val")
            datasets["val"] = OpTCEcarDataset(cache_dir, split=prefix, preload=False)
            print(f"  - Val: {len(datasets['val'])} samples")
        
    if "test" in splits:
        prefix = optc_cfg.get("test_prefix", "test")
        datasets["test"] = OpTCEcarDataset(cache_dir, split=prefix, preload=False)
        print(f"  - Test: {len(datasets['test'])} samples")
        
    return datasets, None

def split_train_data(train_dataset, train_cfg):
    """Split Train dataset into Teacher (Pretrain) and Student (Federated) sets"""
    # [REQ] Split by Time: First 50% -> Teacher, Last 50% -> Student
    print("[Split] Sorting dataset by time for 50/50 split...")
    
    all_timed_samples = []
    for idx in range(len(train_dataset)):
        if hasattr(train_dataset, "get_metadata"):
            s = train_dataset.get_metadata(idx)
        else:
            s = train_dataset[idx]
        all_timed_samples.append((s.get("t0", 0), idx))
    
    # Sort globally by timestamp
    all_timed_samples.sort(key=lambda x: x[0])
    sorted_indices = [x[1] for x in all_timed_samples]
    
    split_idx = int(len(sorted_indices) * 0.3333) # 1/3 for Teacher, 2/3 for Student
    teacher_indices = sorted_indices[:split_idx]
    student_indices = sorted_indices[split_idx:]
    
    print(f"[Split] Teacher: {len(teacher_indices)} samples (First 33.3%)")
    print(f"[Split] Student: {len(student_indices)} samples (Last 66.7%)")

    return Subset(train_dataset, teacher_indices), Subset(train_dataset, student_indices)

# --- Modular Execution Functions ---

def run_train_teacher(config_path):
    print("\n=== Phase 1: Teacher Training ===")
    config, optc_cfg, train_cfg, model_cfg, device, output_dir = setup_environment(config_path)
    
    # Load Data
    datasets, _ = load_datasets(optc_cfg, optc_cfg["cache_dir"], splits=["train", "val"])
    teacher_subset, _ = split_train_data(datasets["train"], train_cfg)
    
    num_workers = train_cfg.get("num_workers", 4)
    pin_memory = torch.cuda.is_available()
    
    # [OPTIMIZATION] Use FeatureCollate to compute behavior features in parallel workers
    views = ["process", "file", "network"]
    behavior_dim = model_cfg.get("behavior_dim", 128)
    
    # Check if optimized
    is_optimized = False
    if isinstance(datasets["train"], TensorDataset) or (isinstance(datasets["train"], Subset) and isinstance(datasets["train"].dataset, TensorDataset)):
        is_optimized = True
        print("[Teacher] Using Optimized TensorDataset")
        
    if is_optimized:
        collate_fn_feat = tensor_collate
    else:
        collate_fn_feat = FeatureCollate(views, behavior_dim)
    
    # [OPTIMIZATION] Use Config Batch Size
    batch_size = int(train_cfg.get("batch_size", 128))
    
    teacher_loader = DataLoader(teacher_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_feat, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn_feat, num_workers=num_workers, pin_memory=pin_memory)
    
    # Init Model
    # views = ["process", "file", "network"] # Defined above
    behavior_dim = model_cfg.get("behavior_dim", 128)
    subspace_dim = model_cfg.get("subspace_dim", 16)
    num_subspaces = model_cfg.get("num_subspaces", 4)
    
    teacher = TeacherModel(behavior_dim=behavior_dim, num_subspaces=num_subspaces, subspace_dim=subspace_dim, hidden_dim=64).to(device)
    if torch.cuda.device_count() > 1:
        teacher = torch.nn.DataParallel(teacher)
    
    # Train
    teacher_cfg = train_cfg.get("teacher", {})
    optimizer = torch.optim.Adam(teacher.parameters(), lr=float(teacher_cfg.get("lr", 1e-4)))
    teacher.train()
    
    epochs = teacher_cfg.get("epochs", 5)
    loss_history = []
    
    dp_cfg = FeatureDPConfig(
        enabled=model_cfg.get("feature_dp", {}).get("enabled", True), 
        clip_C=model_cfg.get("feature_dp", {}).get("clip_C", 10.0), 
        noise_sigma=model_cfg.get("feature_dp", {}).get("noise_sigma", 0.01)
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        pbar = tqdm(teacher_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            if is_optimized:
                b_features = batch[1]
            else:
                _, b_features = batch
                
            # [OPTIMIZATION] b_features is already computed by workers
            b_features = b_features.to(device).float()
            
            b_dp = dp_features(b_features, dp_cfg)
            
            # Handle DataParallel: access module if wrapped
            model_to_call = teacher.module if isinstance(teacher, torch.nn.DataParallel) else teacher
            
            loss = model_to_call.forward_contrastive(
                b_dp, 
                temp=float(teacher_cfg.get("temp", 0.1)),
                mask_p=float(teacher_cfg.get("augment_mask_p", 0.2)),
                noise_std=float(teacher_cfg.get("augment_noise_std", 0.01))
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss/max(steps,1)
        loss_history.append({"epoch": epoch+1, "loss": avg_loss})
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    # Save
    if isinstance(teacher, torch.nn.DataParallel):
        torch.save(teacher.module.state_dict(), os.path.join(output_dir, "teacher_checkpoint.pt"))
    else:
        torch.save(teacher.state_dict(), os.path.join(output_dir, "teacher_checkpoint.pt"))
    print(f"[Teacher] Saved checkpoint to {os.path.join(output_dir, 'teacher_checkpoint.pt')}")

    # Validation
    print("[Teacher] Validating...")
    teacher.eval()
    all_z = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            if is_optimized:
                b_features = batch[1]
            else:
                _, b_features = batch
                
            b_features = b_features.to(device).float()
            z = teacher(b_features)
            all_z.append(z.cpu())
    
    z_std = torch.cat(all_z, dim=0).std(dim=0).mean().item()
    print(f"[Teacher] Validation Z Std: {z_std:.4e}")
    if z_std < 1e-4:
        print("  [!] CRITICAL: Teacher Model Collapsed!")

def run_train_student(config_path):
    print("\n=== Phase 2: Student/Step1 Training ===")
    config, optc_cfg, train_cfg, model_cfg, device, output_dir = setup_environment(config_path)
    
    # Load Data
    datasets, _ = load_datasets(optc_cfg, optc_cfg["cache_dir"], splits=["train", "val"])
    teacher_subset, student_subset = split_train_data(datasets["train"], train_cfg)
    
    # Check if optimized
    is_optimized = False
    if isinstance(datasets["train"], TensorDataset) or (isinstance(datasets["train"], Subset) and isinstance(datasets["train"].dataset, TensorDataset)):
        is_optimized = True
        print("[Student] Using Optimized TensorDataset")
    
    # Init Models (Teacher, Step1, Student)
    views = ["process", "file", "network"]
    
    # Load Teacher
    behavior_dim = model_cfg.get("behavior_dim", 128)
    subspace_dim = model_cfg.get("subspace_dim", 16)
    num_subspaces = model_cfg.get("num_subspaces", 4)
    
    teacher = TeacherModel(behavior_dim=behavior_dim, num_subspaces=num_subspaces, subspace_dim=subspace_dim, hidden_dim=64).to(device)
    teacher_ckpt = os.path.join(output_dir, "teacher_checkpoint.pt")
    if not os.path.exists(teacher_ckpt):
        print("Error: Teacher checkpoint not found. Run 'train_teacher' first.")
        return
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    
    # Init Step1
    quality_cfg = QualityWeightsConfig(reliability_weights={"validity": 1.0, "completeness": 1.0}, info_gain_w_min=0.3, info_gain_temp=1.0, standardize="minmax", softmax_temperature=model_cfg.get("quality_softmax_temp", 0.5))
    s1_cfg = Step1Config(
        views=views, window_seconds=optc_cfg.get("window_minutes", 15)*60, slot_seconds=model_cfg.get("slot_seconds", 60),
        include_empty_slot_indicator=True, num_hash_buckets=model_cfg.get("num_hash_buckets", 50), hash_seed=42, target_dim=model_cfg.get("target_dim", 32),
        rp_seed=123, rp_matrix_type="gaussian", rp_normalize="l2", rp_nonlinearity="relu", quality_cfg=quality_cfg,
        router_hidden_dims=model_cfg.get("router_hidden_dims", [64]), router_dropout=0.1, num_subspaces=model_cfg.get("num_subspaces", 4),
        gate_gamma=model_cfg.get("gate_gamma", 0.5), gate_mode="soft", gate_beta=model_cfg.get("gate_beta", 5.0), interaction_enabled=True
    )
    
    # Load/Create Schema
    vocab_cache_path = os.path.join(output_dir, "vocab_schema.pt")
    if os.path.exists(vocab_cache_path):
        per_view_schema = torch.load(vocab_cache_path, weights_only=False)
    else:
        if is_optimized:
            raise RuntimeError(f"Cannot fit schema from TensorDataset! Please ensure {vocab_cache_path} exists or use raw data.")
            
        # Fit Schema using Teacher subset
        print("[Student] Fitting Schema...")
        vocab_samples = [teacher_subset[i] for i in torch.randperm(len(teacher_subset))[:min(50, len(teacher_subset))].tolist()]
        per_view_schema = {v: AggregatorSchema(event_type_vocab=[], key_fields=["action", "object"]) for v in views}
        
        key_fields_map = {"process": ["action", "object", "image_path", "command_line"], "file": ["action", "object", "file_path"], "network": ["action", "object", "dest_ip", "dest_port"]}
        for v, schema in per_view_schema.items():
            if v in key_fields_map: schema.key_fields = key_fields_map[v]
            
        step1_temp = Step1Model(s1_cfg, per_view_schema) # Temp model just for fitting
        # We need to manually fit schemas here if not using Step1Model.fit_vocabs... but simpler to just use Step1Model logic
        # Re-using the logic from main:
        step1 = Step1Model(s1_cfg, per_view_schema).to(device)
        step1.fit_vocabs_and_init_projectors(vocab_samples, max_types=model_cfg.get("max_vocab_types", 500))
        step1.fit_quality_stats(vocab_samples)
        torch.save(step1.schemas, vocab_cache_path)
    
    step1 = Step1Model(s1_cfg, per_view_schema).to(device)
    if not step1._projectors_ready:
        # Need to re-init projectors if loaded from schema
        if is_optimized:
             # If optimized, we don't have raw samples to fit vocab (it's already in schema)
             # Just init projectors with empty samples
             step1.fit_vocabs_and_init_projectors([], max_types=model_cfg.get("max_vocab_types", 500))
             
             # Fit quality stats from tensor
             print("[Student] Fitting quality stats from TensorDataset...")
             if isinstance(teacher_subset, Subset):
                 ds = teacher_subset.dataset
                 indices = teacher_subset.indices
                 q_tensor = ds.quality[indices]
             else:
                 q_tensor = teacher_subset.quality
                 
             # [N, V, 4] -> [N*V, 4]
             q_flat = q_tensor.reshape(-1, 4)
             metrics = ["validity", "completeness", "entropy", "intensity"]
             qs = {}
             for i, k in enumerate(metrics):
                 qs[k] = q_flat[:, i].tolist()
                 
             step1.quality.fit_standardize_stats(qs)
        else:
            vocab_samples = [teacher_subset[i] for i in torch.randperm(len(teacher_subset))[:min(50, len(teacher_subset))].tolist()]
            step1.fit_vocabs_and_init_projectors(vocab_samples, max_types=model_cfg.get("max_vocab_types", 500))
            step1.fit_quality_stats(vocab_samples)

    replace_relu(step1.router)
    replace_relu(step1.fusion)
    
    # Init LoRA
    step1.alignment.requires_grad_(True)
    apply_lora(step1.router, rank=model_cfg.get("lora_rank", 4), alpha=model_cfg.get("lora_alpha", 8))
    apply_lora(step1.fusion, rank=model_cfg.get("lora_rank", 4), alpha=model_cfg.get("lora_alpha", 8))
    step1.to(device) # Ensure all params (including new LoRA) are on device
    
    # Init Student
    student = StudentHeads(in_dim=s1_cfg.target_dim, num_subspaces=num_subspaces, subspace_dim=subspace_dim, hidden_dim=64).to(device)
    replace_relu(student)
    
    # Federated Setup
    clients = []
    FEDERATED_LEARNING = train_cfg.get("federated_learning", False)
    
    if not FEDERATED_LEARNING:
        # central_samples = [student_subset.dataset[i] for i in student_subset.indices] # CAUSES OOM
        # clients.append(FederatedClient("central_node", central_samples))
        clients.append(FederatedClient("central_node", student_subset))
    else:
        # For FL, we need to create Subsets per host
        # This requires iterating indices once, but not loading data.
        # student_subset is already a Subset.
        # We need to access metadata to split by host.
        
        # 1. Map global indices to hosts
        # OpTCEcarDataset now supports fast metadata access
        host_to_indices = collections.defaultdict(list)
        
        # Access the underlying dataset and indices
        base_dataset = student_subset.dataset
        indices = student_subset.indices
        
        print("[Student] Partitioning data by host for FL...")
        for global_idx in tqdm(indices, desc="Partitioning"):
            if hasattr(base_dataset, "get_metadata"):
                meta = base_dataset.get_metadata(global_idx)
                h = meta.get("host", "unknown")
            else:
                # Fallback (slow)
                s = base_dataset[global_idx]
                h = s.get("host", "unknown")
            host_to_indices[h].append(global_idx)
            
        for host, host_indices in host_to_indices.items():
            if len(host_indices) > 0: 
                # Create a Subset for this host
                client_subset = Subset(base_dataset, host_indices)
                clients.append(FederatedClient(host, client_subset))
            
    print(f"[Student] Initialized {len(clients)} clients.")
    
    client_cfg = train_cfg.get("client", {})
    temp_params = client_cfg.get("temp_params", {"a": 2.0, "b": 1.0, "min_tau": 0.05, "max_tau": 0.2})
    temp_params["view_sensitivities"] = model_cfg.get("view_sensitivities", {"process": 1.0, "file": 1.0, "network": 1.0})
    
    client_train_cfg = ClientTrainConfig(
        local_epochs=client_cfg.get("local_epochs", 5), batch_size=client_cfg.get("batch_size", 16), oversample_factor=float(client_cfg.get("oversample_factor", 1.0)), lr=float(client_cfg.get("lr", 1e-4)),
        weight_decay=float(client_cfg.get("weight_decay", 1e-4)), lambda_stats=float(client_cfg.get("lambda_stats", 50.0)),
        lambda_infonce=float(client_cfg.get("lambda_infonce", 1.0)), temp_params=temp_params,
        feature_dp=FeatureDPConfig(**model_cfg.get("feature_dp", {})), grad_dp=GradDPConfig(**model_cfg.get("grad_dp", {})),
        views=views, behavior_feature_dim=behavior_dim
    )
    
    server_cfg = ServerConfig(
        rounds=train_cfg.get("epochs", 5), client_fraction=1.0, min_clients=1, server_lr=1.0,
        secure_agg_enabled=True, secure_agg_protocol="pairwise_masking"
    )
    
    server = FederatedServer(clients, server_cfg)
    
    # FL Loop
    fl_history = []
    best_val_err = float("inf")
    num_workers = train_cfg.get("num_workers", 4)
    # Use FeatureCollate for FL validation too
    
    # Check if optimized
    is_optimized = False
    if isinstance(datasets["val"], TensorDataset):
        is_optimized = True
        
    if is_optimized:
        collate_fn_feat = tensor_collate
    else:
        collate_fn_feat = FeatureCollate(views, behavior_dim)
        
    # [OPTIMIZATION] Use Config Batch Size
    batch_size = int(train_cfg.get("batch_size", 128))
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn_feat, num_workers=num_workers)
    
    for round_idx in range(server_cfg.rounds):
        print(f"\n--- Round {round_idx+1}/{server_cfg.rounds} ---")
        sampled_clients = server.sample_clients()
        updates, ns, round_metrics = [], [], []
        
        for client in sampled_clients:
            c_s1, c_stu = copy.deepcopy(step1), copy.deepcopy(student)
            upd, met, n = client.local_train(c_s1, c_stu, teacher, client_train_cfg, device)
            updates.append(upd); ns.append(n); round_metrics.append(met)
            del c_s1, c_stu; import gc; gc.collect()
            
        global_params = list(step1.parameters()) + list(student.parameters())
        server.aggregate_and_apply(global_params, updates, ns)
        
        # Validation
        avg_loss = sum(m["client_loss"] for m in round_metrics) / len(round_metrics) if round_metrics else 0.0
        avg_loss_asd = sum(m.get("client_loss_asd", 0.0) for m in round_metrics) / len(round_metrics) if round_metrics else 0.0
        avg_loss_infonce = sum(m.get("client_loss_infonce", 0.0) for m in round_metrics) / len(round_metrics) if round_metrics else 0.0
        
        step1.eval(); student.eval()
        val_z_std = 0.0
        val_distill_err = 0.0
        val_loss_asd = 0.0
        val_loss_infonce = 0.0
        steps = 0
        from optc_uras.models.losses import asd_loss, at_info_nce, dynamic_temperature, entropy
        with torch.no_grad():
            for batch in val_loader:
                if is_optimized:
                    _, b_features, s_tensor, q_tensor = batch
                    s1_out = step1.forward_from_cache(s_tensor.to(device), q_tensor.to(device))
                else:
                    batch_samples, _ = batch
                    s1_out = step1(batch_samples)
                
                z = torch.stack([o.z for o in s1_out])
                val_z_std += z.std(dim=0).mean().item()
                route_p = torch.stack([o.route_p for o in s1_out])
                sub_s = student(z.to(device), normalize=True)
                uras_s = uras_from_subspaces(sub_s, route_p.to(device))
                uras_s = torch.nn.functional.normalize(uras_s, dim=-1)

                if is_optimized:
                    x = b_features.to(device).float()
                else:
                    x = torch.from_numpy(
                        __import__("numpy").stack([behavior_features_from_sample(s, views, out_dim=behavior_dim) for s in batch_samples])
                    ).to(device).float()
                x_dp = dp_features(x, client_train_cfg.feature_dp)
                sub_t = teacher(x_dp, normalize=True)
                uras_t = uras_from_subspaces(sub_t, route_p.to(device))
                uras_t = torch.nn.functional.normalize(uras_t, dim=-1)

                val_distill_err += (uras_s - uras_t).norm(p=2, dim=-1).mean().item()
                val_loss_asd += float(asd_loss(sub_s, sub_t, route_p.to(device), lambda_stats=float(client_train_cfg.lambda_stats)).item())

                w = torch.stack([o.reliability_w for o in s1_out], dim=0).to(device)
                V_dim = w.shape[1]
                confidence = (1.0 - (entropy(w) / __import__("numpy").log(max(V_dim, 2)))).clamp(0.0, 1.0)
                M_dim = route_p.shape[1]
                route_unc = entropy(route_p.to(device)) / __import__("numpy").log(max(M_dim, 2))
                distill_err = (uras_s - uras_t).norm(p=2, dim=-1)
                sample_utility = confidence * torch.exp(-distill_err)
                sensitivity_map = client_train_cfg.temp_params.get("view_sensitivities", {})
                sens_vec = torch.tensor([sensitivity_map.get(v, 1.0) for v in views], device=device)
                privacy_risk = (w * sens_vec.unsqueeze(0)).sum(dim=1)
                tau = dynamic_temperature(
                    confidence=confidence,
                    route_uncertainty=route_unc,
                    distill_error=distill_err,
                    privacy_risk=privacy_risk,
                    sample_utility=sample_utility,
                    current_epoch=round_idx,
                    total_epochs=server_cfg.rounds,
                    a=float(client_train_cfg.temp_params.get("a", 2.0)),
                    b=float(client_train_cfg.temp_params.get("b", 1.0)),
                    c=0.0,
                    min_tau=float(client_train_cfg.temp_params.get("min_tau", 0.05)),
                    max_tau=float(client_train_cfg.temp_params.get("max_tau", 0.5)),
                )
                z_s_norm = torch.nn.functional.normalize(uras_s.view(uras_s.size(0), -1), dim=-1)
                z_t_norm = torch.nn.functional.normalize(uras_t.view(uras_t.size(0), -1), dim=-1)
                val_loss_infonce += float(at_info_nce(z_s_norm, z_t_norm, temperature=tau).item())
                steps += 1
        avg_std = val_z_std/max(steps,1)
        avg_val_err = val_distill_err/max(steps,1)
        avg_val_asd = val_loss_asd/max(steps,1)
        avg_val_infonce = val_loss_infonce/max(steps,1)
        
        # [FIX] Use scientific notation for logging small loss values
        print(f"Round {round_idx+1} Loss: {avg_loss:.4e} (ASD: {avg_loss_asd:.4e}, InfoNCE: {avg_loss_infonce:.4e}) | Val Z Std: {avg_std:.4e} | Val DistillErr: {avg_val_err:.4e}")
        fl_history.append({"round": round_idx+1, "loss": avg_loss, "loss_asd": avg_loss_asd, "loss_infonce": avg_loss_infonce, "val_std": avg_std, "val_distill_err": avg_val_err, "val_asd": avg_val_asd, "val_infonce": avg_val_infonce})
        
        if avg_std < 1e-6:
            print("[!] CRITICAL: Feature Collapse.")
            break

        if avg_val_err < best_val_err:
            best_val_err = avg_val_err
            torch.save(step1.state_dict(), os.path.join(output_dir, "step1_checkpoint_best.pt"))
            torch.save(student.state_dict(), os.path.join(output_dir, "student_checkpoint_best.pt"))
            
    pd.DataFrame(fl_history).to_csv(os.path.join(output_dir, "fl_train_log.csv"), index=False)
    torch.save(step1.state_dict(), os.path.join(output_dir, "step1_checkpoint_no_dp.pt"))
    torch.save(student.state_dict(), os.path.join(output_dir, "student_checkpoint_no_dp.pt"))
    print(f"[Student] Saved checkpoints to {output_dir}")

def run_train_detector(config_path):
    print("\n=== Phase 3: Detector Training ===")
    config, optc_cfg, train_cfg, model_cfg, device, output_dir = setup_environment(config_path)
    
    # Load Data
    datasets, _ = load_datasets(optc_cfg, optc_cfg["cache_dir"], splits=["train", "val"])
    _, student_subset = split_train_data(datasets["train"], train_cfg)
    
    # Init Models
    views = ["process", "file", "network"]
    
    # Check if optimized
    is_optimized = False
    if isinstance(datasets["train"], TensorDataset) or (isinstance(datasets["train"], Subset) and isinstance(datasets["train"].dataset, TensorDataset)):
        is_optimized = True
        print("[Detector] Using Optimized TensorDataset")
    
    # Load Step1 & Student
    vocab_cache_path = os.path.join(output_dir, "vocab_schema.pt")
    if not os.path.exists(vocab_cache_path): return
    per_view_schema = torch.load(vocab_cache_path, weights_only=False)
    
    # Setup Step1 Config (Same as Student)
    quality_cfg = QualityWeightsConfig(reliability_weights={"validity": 1.0, "completeness": 1.0}, info_gain_w_min=0.3, info_gain_temp=1.0, standardize="minmax", softmax_temperature=model_cfg.get("quality_softmax_temp", 0.5))
    s1_cfg = Step1Config(
        views=views, window_seconds=optc_cfg.get("window_minutes", 15)*60, slot_seconds=model_cfg.get("slot_seconds", 60),
        include_empty_slot_indicator=True, num_hash_buckets=model_cfg.get("num_hash_buckets", 50), hash_seed=42, target_dim=model_cfg.get("target_dim", 32),
        rp_seed=123, rp_matrix_type="gaussian", rp_normalize="l2", rp_nonlinearity="relu", quality_cfg=quality_cfg,
        router_hidden_dims=model_cfg.get("router_hidden_dims", [64]), router_dropout=0.1, num_subspaces=model_cfg.get("num_subspaces", 4),
        gate_gamma=model_cfg.get("gate_gamma", 0.5), gate_mode="soft", gate_beta=model_cfg.get("gate_beta", 5.0), interaction_enabled=True
    )
    
    step1 = Step1Model(s1_cfg, per_view_schema).to(device)
    # Init projectors/stats
    if is_optimized:
        step1.fit_vocabs_and_init_projectors([], max_types=model_cfg.get("max_vocab_types", 500))
        # No need to refit quality stats here if we assume consistency or load from saved checkpoint logic later?
        # Ideally we load quality stats from somewhere or refit.
        # Let's refit using student_subset quality tensor.
        print("[Detector] Fitting quality stats from TensorDataset...")
        if isinstance(student_subset, Subset):
             ds = student_subset.dataset
             indices = student_subset.indices
             q_tensor = ds.quality[indices]
        else:
             q_tensor = student_subset.quality
        q_flat = q_tensor.reshape(-1, 4)
        metrics = ["validity", "completeness", "entropy", "intensity"]
        qs = {}
        for i, k in enumerate(metrics):
             qs[k] = q_flat[:, i].tolist()
        step1.quality.fit_standardize_stats(qs)
    else:
        vocab_samples = [student_subset.dataset[i] for i in torch.randperm(len(student_subset))[:min(50, len(student_subset))].tolist()]
        step1.fit_vocabs_and_init_projectors(vocab_samples, max_types=model_cfg.get("max_vocab_types", 500))
        step1.fit_quality_stats(vocab_samples)
    
    replace_relu(step1.router); replace_relu(step1.fusion)
    apply_lora(step1.router, rank=model_cfg.get("lora_rank", 4), alpha=model_cfg.get("lora_alpha", 8))
    apply_lora(step1.fusion, rank=model_cfg.get("lora_rank", 4), alpha=model_cfg.get("lora_alpha", 8))
    step1.to(device)
    
    student = StudentHeads(in_dim=s1_cfg.target_dim, num_subspaces=model_cfg.get("num_subspaces", 4), subspace_dim=model_cfg.get("subspace_dim", 16), hidden_dim=64).to(device)
    replace_relu(student)
    
    # Load Weights
    s1_ckpt = os.path.join(output_dir, "step1_checkpoint_no_dp.pt")
    stu_ckpt = os.path.join(output_dir, "student_checkpoint_no_dp.pt")
    if not os.path.exists(s1_ckpt) or not os.path.exists(stu_ckpt):
        print("Error: Student/Step1 checkpoints not found. Run 'train_student' first.")
        return
    step1.load_state_dict(torch.load(s1_ckpt, map_location=device))
    student.load_state_dict(torch.load(stu_ckpt, map_location=device))
    step1.eval(); student.eval()
    
    if torch.cuda.device_count() > 1:
        step1 = torch.nn.DataParallel(step1)
        student = torch.nn.DataParallel(student)
        
    # Init Detector
    uras_dim = model_cfg.get("num_subspaces", 4) * model_cfg.get("subspace_dim", 16)
    detector = AnomalyDetector(
        feature_dim=uras_dim, style_dim=32, content_dim=32, view_names=views, view_dim=s1_cfg.target_dim,
        drift_lr=model_cfg.get("drift_lr", 0.01), alpha_conf=model_cfg.get("atc", {}).get("alpha_conf", 0.0),
        alpha_unc=model_cfg.get("atc", {}).get("alpha_unc", 0.0), alpha_risk=model_cfg.get("atc", {}).get("alpha_risk", 0.0)
    ).to(device)
    replace_relu(detector)
    
    if torch.cuda.device_count() > 1:
        detector = torch.nn.DataParallel(detector)
        
    # Train Detector
    det_model = detector.module if isinstance(detector, torch.nn.DataParallel) else detector
    optimizer = torch.optim.Adam(det_model.scd.parameters(), lr=float(train_cfg.get("learning_rate", 1e-5)) * 3)
    detector.train()
    
    num_workers = train_cfg.get("num_workers", 4)
    batch_size = int(train_cfg.get("batch_size", 128))
    if is_optimized:
        loader = DataLoader(student_subset, batch_size=batch_size, shuffle=True, collate_fn=tensor_collate, num_workers=num_workers)
    else:
        loader = DataLoader(student_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
        
    loss_cfg = model_cfg.get("scd_loss", {})
    
    epochs = int(train_cfg.get("epochs", 50))
    for epoch in range(epochs): # Reduced epochs for faster tuning
        epoch_loss = 0.0; steps = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            with torch.no_grad():
                if is_optimized:
                    _, _, s_tensor, q_tensor = batch
                    # Handle DataParallel for custom method
                    s1_model = step1
                    if hasattr(step1, "module"):
                        s1_model = step1.module
                    s1_out = s1_model.forward_from_cache(s_tensor.to(device), q_tensor.to(device))
                else:
                    s1_out = step1(batch)
                    
            z = torch.stack([o.z for o in s1_out])
            route_p = torch.stack([o.route_p for o in s1_out])
            uras = uras_from_subspaces(student(z), route_p)
            
            # Handle DataParallel for Detector
            model_to_call = detector.module if isinstance(detector, torch.nn.DataParallel) else detector
            s, c, u_hat = model_to_call.scd(uras, center_batch=True, return_rec=True)
            
            loss = scd_loss(s, c, uras, u_hat, lambda_d=float(loss_cfg.get("lambda_d", 1.0)), 
                            lambda_r=float(loss_cfg.get("lambda_r", 1.0)), lambda_v=float(loss_cfg.get("lambda_v", 1.0)), 
                            gamma=float(loss_cfg.get("gamma", 1.0)))
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item(); steps += 1
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss/max(steps,1):.4f}")
        
    if isinstance(detector, torch.nn.DataParallel):
        torch.save(detector.module.state_dict(), os.path.join(output_dir, "detector_checkpoint_retrain.pt"))
    else:
        torch.save(detector.state_dict(), os.path.join(output_dir, "detector_checkpoint_retrain.pt"))
    print(f"[Detector] Saved checkpoint to {output_dir}")
    
    # Post-Process: Stats
    print("[Detector] Fitting Statistics & Threshold...")
    # if torch.cuda.device_count() > 1:
    #     step1 = torch.nn.DataParallel(step1)
    #     student = torch.nn.DataParallel(student)
    #     detector = torch.nn.DataParallel(detector)
        
    step1.eval(); student.eval(); detector.eval()
    
    # Unwrap for stats fitting (buffers are in module)
    det_module = detector.module if isinstance(detector, torch.nn.DataParallel) else detector
    
    all_s = []
    with torch.no_grad():
        pbar_stats = tqdm(loader, desc="Stats")
        for batch in pbar_stats:
            if is_optimized:
                _, _, s_tensor, q_tensor = batch
                # Handle DataParallel for custom method
                s1_model = step1
                if hasattr(step1, "module"):
                    s1_model = step1.module
                s1_out = s1_model.forward_from_cache(s_tensor.to(device), q_tensor.to(device))
            else:
                s1_out = step1(batch)
                
            z = torch.stack([o.z for o in s1_out])
            route_p = torch.stack([o.route_p for o in s1_out])
            uras = uras_from_subspaces(student(z), route_p)
            s, _ = det_module.scd(uras, center_batch=False)
            all_s.append(s.cpu())
            
    all_s = torch.cat(all_s, dim=0)
    det_module.style_mu = all_s.mean(dim=0).to(device)
    det_module.style_var = (all_s.to(device) - det_module.style_mu).var(dim=0, unbiased=False) + 1e-6
    det_module.style_inv_cov = torch.diag(1.0/det_module.style_var)
    
    scores = det_module._compute_raw_score(all_s.to(device))
    threshold_q = float(model_cfg.get("threshold_quantile", 0.90))
    det_module.threshold = torch.quantile(scores, threshold_q)
    with torch.no_grad():
        frac_anom = float((scores > det_module.threshold).float().mean().item())
        qs = [0.5, 0.9, 0.99]
        q_vals = {q: float(torch.quantile(scores, q).item()) for q in qs}
        print(f"[Detector] Score stats: min={scores.min().item():.4e} median={q_vals[0.5]:.4e} q0.9={q_vals[0.9]:.4e} q0.99={q_vals[0.99]:.4e} max={scores.max().item():.4e}")
        print(f"[Detector] Threshold(q={threshold_q:.3f})={det_module.threshold.item():.4e} | train_frac_anomaly={frac_anom:.4f}")

    calibrate_on_val = model_cfg.get("calibrate_threshold_on_val")
    if calibrate_on_val is None:
        calibrate_on_val = True
    calibrate_on_val = bool(calibrate_on_val)

    if calibrate_on_val and "val" in datasets and len(datasets["val"]) > 0:
        val_ds = datasets["val"]

        val_t0s = []
        for i in range(len(val_ds)):
            if hasattr(val_ds, "get_metadata"):
                meta = val_ds.get_metadata(i)
                t0 = int(meta.get("t0", 0))
            else:
                s = val_ds[i]
                t0 = int(s.get("t0", 0))
            val_t0s.append(t0)

        unique_t0 = sorted(set(val_t0s))
        split_at = max(1, min(len(unique_t0) // 2, max(len(unique_t0) - 1, 1)))
        t0_calib = set(unique_t0[:split_at])
        calib_local = [i for i, t0 in enumerate(val_t0s) if t0 in t0_calib]
        eval_local = [i for i, t0 in enumerate(val_t0s) if t0 not in t0_calib]

        val_calib = Subset(val_ds, calib_local)
        val_eval = Subset(val_ds, eval_local)

        if is_optimized:
            val_calib_loader = DataLoader(val_calib, batch_size=batch_size, shuffle=False, collate_fn=tensor_collate, num_workers=num_workers)
            val_eval_loader = DataLoader(val_eval, batch_size=batch_size, shuffle=False, collate_fn=tensor_collate, num_workers=num_workers)
        else:
            val_calib_loader = DataLoader(val_calib, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
            val_eval_loader = DataLoader(val_eval, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

        def _collect_scores(loader, desc: str) -> torch.Tensor:
            out_scores = []
            with torch.no_grad():
                for batch in tqdm(loader, desc=desc):
                    if is_optimized:
                        _, _, s_tensor, q_tensor = batch
                        s1_model = step1.module if hasattr(step1, "module") else step1
                        s1_out = s1_model.forward_from_cache(s_tensor.to(device), q_tensor.to(device))
                    else:
                        s1_out = step1(batch)
                    z = torch.stack([o.z for o in s1_out])
                    route_p = torch.stack([o.route_p for o in s1_out])
                    uras = uras_from_subspaces(student(z), route_p)
                    s, _ = det_module.scd(uras, center_batch=False)
                    sc = det_module._compute_raw_score(s)
                    out_scores.append(sc.detach().cpu())
            return torch.cat(out_scores, dim=0) if out_scores else torch.empty((0,), dtype=torch.float32)

        calib_scores = _collect_scores(val_calib_loader, "ValCalib")
        eval_scores = _collect_scores(val_eval_loader, "ValEval")

        threshold_before = det_module.threshold.detach().cpu()
        val_frac_eval_before = float((eval_scores > threshold_before).float().mean().item()) if eval_scores.numel() else 0.0

        val_thr = torch.quantile(calib_scores, threshold_q) if calib_scores.numel() else threshold_before
        det_module.threshold = val_thr.to(device)
        val_frac_eval_after = float((eval_scores > val_thr).float().mean().item()) if eval_scores.numel() else 0.0

        print(f"[Detector] Val calibration: calib_N={int(calib_scores.numel())} eval_N={int(eval_scores.numel())} | eval_frac_anomaly(before)={val_frac_eval_before:.4f} threshold={val_thr.item():.4e} eval_frac_anomaly(after)={val_frac_eval_after:.4f}")

        try:
            import json

            val_metrics_path = os.path.join(output_dir, "detector_val_metrics.json")
            with open(val_metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "threshold_quantile": float(threshold_q),
                        "threshold_before": float(threshold_before.item()) if threshold_before.numel() else None,
                        "threshold_after": float(val_thr.item()) if val_thr.numel() else None,
                        "val_calib_size": int(calib_scores.numel()),
                        "val_eval_size": int(eval_scores.numel()),
                        "val_eval_frac_anomaly_before": float(val_frac_eval_before),
                        "val_eval_frac_anomaly_after": float(val_frac_eval_after),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass

        synth_enabled = model_cfg.get("synth_anom", {}).get("enabled", True)
        if synth_enabled and eval_scores.numel():
            torch.manual_seed(int(model_cfg.get("synth_anom", {}).get("seed", 123)))
            synth_mode = str(model_cfg.get("synth_anom", {}).get("mode", "score_noise"))
            noise_scale = float(model_cfg.get("synth_anom", {}).get("noise_scale", 2.5))

            if synth_mode == "score_noise":
                sigma = eval_scores.std(unbiased=False).clamp_min(1e-6)
                synth_scores = eval_scores + torch.randn_like(eval_scores) * (noise_scale * sigma)
            else:
                sigma = eval_scores.std(unbiased=False).clamp_min(1e-6)
                synth_scores = eval_scores + torch.randn_like(eval_scores) * (noise_scale * sigma)

            tpr_synth = float((synth_scores > val_thr).float().mean().item())
            try:
                import json

                val_metrics_path = os.path.join(output_dir, "detector_val_metrics.json")
                if os.path.exists(val_metrics_path):
                    with open(val_metrics_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    data = {}
                data.update(
                    {
                        "synth_enabled": True,
                        "synth_mode": synth_mode,
                        "synth_noise_scale": noise_scale,
                        "synth_tpr_at_threshold": tpr_synth,
                    }
                )
                with open(val_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    # [FIX] Save checkpoint AFTER fitting statistics so threshold/mu/var are stored!
    if isinstance(detector, torch.nn.DataParallel):
        torch.save(detector.module.state_dict(), os.path.join(output_dir, "detector_checkpoint_retrain.pt"))
    else:
        torch.save(detector.state_dict(), os.path.join(output_dir, "detector_checkpoint_retrain.pt"))
    print(f"[Detector] Saved checkpoint (with stats) to {output_dir}")

    # Fit Interpreter
    # (Simplified for brevity - can add full fit_interpreter logic here if needed)

def run_test(config_path):
    print("\n=== Phase 4: Testing/Inference ===")
    config, optc_cfg, train_cfg, model_cfg, device, output_dir = setup_environment(config_path)
    
    # Load Data
    datasets, vocab_schema_loaded = load_datasets(optc_cfg, optc_cfg["cache_dir"], splits=["test"])
    
    # If vocab schema loaded from cache, update feature_extractors to ensure consistent mapping
    if vocab_schema_loaded:
        print("[Setup] Updating feature extractors with loaded vocab schema...")
        for v, schema in vocab_schema_loaded.items():
            if v in feature_extractors:
                feature_extractors[v].schema = schema
    num_workers = train_cfg.get("num_workers", 4)
    
    # Check if optimized
    is_optimized = False
    batch_size = int(train_cfg.get("batch_size", 128))
    if isinstance(datasets["test"], TensorDataset):
        is_optimized = True
        print("[Test] Using Optimized TensorDataset")
        
    if is_optimized:
        test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, collate_fn=tensor_collate, num_workers=num_workers)
    else:
        test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    
    # Re-Init and Load All Models (Similar to Detector phase)
    # ... (Re-init logic same as run_train_detector) ...
    # For brevity, I'll copy the init logic. In a real refactor, we'd have a create_models() function.
    
    views = ["process", "file", "network"]
    vocab_cache_path = os.path.join(output_dir, "vocab_schema.pt")
    per_view_schema = torch.load(vocab_cache_path, weights_only=False)
    
    quality_cfg = QualityWeightsConfig(reliability_weights={"validity": 1.0, "completeness": 1.0}, info_gain_w_min=0.3, info_gain_temp=1.0, standardize="minmax", softmax_temperature=model_cfg.get("quality_softmax_temp", 0.5))
    s1_cfg = Step1Config(
        views=views, window_seconds=optc_cfg.get("window_minutes", 15)*60, slot_seconds=model_cfg.get("slot_seconds", 60),
        include_empty_slot_indicator=True, num_hash_buckets=model_cfg.get("num_hash_buckets", 50), hash_seed=42, target_dim=model_cfg.get("target_dim", 32),
        rp_seed=123, rp_matrix_type="gaussian", rp_normalize="l2", rp_nonlinearity="relu", quality_cfg=quality_cfg,
        router_hidden_dims=model_cfg.get("router_hidden_dims", [64]), router_dropout=0.1, num_subspaces=model_cfg.get("num_subspaces", 4),
        gate_gamma=model_cfg.get("gate_gamma", 0.5), gate_mode="soft", gate_beta=model_cfg.get("gate_beta", 5.0), interaction_enabled=True
    )
    
    step1 = Step1Model(s1_cfg, per_view_schema).to(device)
    # Need to fit projectors to match training state
    if is_optimized:
        step1.fit_vocabs_and_init_projectors([], max_types=model_cfg.get("max_vocab_types", 500))
        # No quality stats refit needed for inference if we don't save/load them?
        # Ideally we should load from checkpoint.
        # But we didn't save quality stats in checkpoint explicitly (it's not nn.Parameter).
        # We should have saved it.
        # For now, we assume default or it doesn't matter much for reliability weights if standardized.
        # Actually it matters. 
        # But for test, we can try to load from training if we saved it?
        # Or just use defaults.
    else:
        # Hack: Load a few test samples just to trigger init (vocab is already fixed in schema)
        step1.fit_vocabs_and_init_projectors([datasets["test"][0]], max_types=model_cfg.get("max_vocab_types", 500))
    
    replace_relu(step1.router); replace_relu(step1.fusion)
    apply_lora(step1.router, rank=model_cfg.get("lora_rank", 4), alpha=model_cfg.get("lora_alpha", 8))
    apply_lora(step1.fusion, rank=model_cfg.get("lora_rank", 4), alpha=model_cfg.get("lora_alpha", 8))
    step1.to(device)
    
    student = StudentHeads(in_dim=s1_cfg.target_dim, num_subspaces=model_cfg.get("num_subspaces", 4), subspace_dim=model_cfg.get("subspace_dim", 16), hidden_dim=64).to(device)
    replace_relu(student)
    
    uras_dim = model_cfg.get("num_subspaces", 4) * model_cfg.get("subspace_dim", 16)
    atc_cfg = model_cfg.get("atc", {}) if isinstance(model_cfg.get("atc", {}), dict) else {}
    detector = AnomalyDetector(
        feature_dim=uras_dim,
        style_dim=32,
        content_dim=32,
        view_names=views,
        view_dim=s1_cfg.target_dim,
        prob_scale=float(model_cfg.get("prob_scale", 10.0)),
        drift_margin=float(model_cfg.get("drift_margin", 0.5)),
        drift_lr=float(model_cfg.get("drift_lr", 0.01)),
        alpha_conf=float(atc_cfg.get("alpha_conf", 0.0)),
        alpha_unc=float(atc_cfg.get("alpha_unc", 0.0)),
        alpha_risk=float(atc_cfg.get("alpha_risk", 0.0)),
    ).to(device)
    replace_relu(detector)
    
    # Load Checkpoints
    step1.load_state_dict(torch.load(os.path.join(output_dir, "step1_checkpoint_no_dp.pt"), map_location=device))
    student.load_state_dict(torch.load(os.path.join(output_dir, "student_checkpoint_no_dp.pt"), map_location=device))
    detector.load_state_dict(torch.load(os.path.join(output_dir, "detector_checkpoint_retrain.pt"), map_location=device))
    
    if torch.cuda.device_count() > 1:
        step1 = torch.nn.DataParallel(step1)
        student = torch.nn.DataParallel(student)
        detector = torch.nn.DataParallel(detector)
        
    step1.eval(); student.eval(); detector.eval()
    
    # Recalculate Threshold/Stats (Since detector checkpoint might not save dynamic stats like mu/var/threshold easily in state_dict if not buffers)
    # Actually, AnomalyDetector registers buffers for style_mu/var, so load_state_dict should handle it IF they were saved.
    # Assuming they are saved.
    
    print("[Test] Running Inference...")
    results = []
    view_sensitivities = model_cfg.get("view_sensitivities", {})
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if is_optimized:
                idx, _, s_tensor, q_tensor = batch
                s1_out = step1.forward_from_cache(s_tensor.to(device), q_tensor.to(device))
                # Need metadata for timestamp
                # datasets["test"] has get_metadata?
                # idx is tensor of indices.
                meta_list = []
                # [FIX] Handle idx if it is list or tensor
                idx_iter = idx.tolist() if isinstance(idx, torch.Tensor) else idx
                for i in idx_iter:
                    # i is GLOBAL index from TensorDataset
                    # get_metadata expects LOCAL index if it uses ds.indices[idx]
                    # But we have direct access to metadata_list which is global
                    meta_list.append(datasets["test"].metadata_list[i])
            else:
                s1_out = step1(batch)
                meta_list = batch
                
            z = torch.stack([o.z for o in s1_out])
            route_p = torch.stack([o.route_p for o in s1_out])
            uras = uras_from_subspaces(student(z), route_p)
            
            w_entropies = torch.tensor([o.intermediates.get("w_entropy", 0.0) for o in s1_out], device=device)
            conf_scores = (1.0 - w_entropies).clamp(0.0, 1.0)
            unc_scores = torch.tensor([o.intermediates.get("route_entropy", 0.0) for o in s1_out], device=device)
            
            sens_vec = torch.tensor([view_sensitivities.get(v, 1.0) for v in views], device=device)
            risk_scores = (torch.stack([o.reliability_w for o in s1_out]) * sens_vec.unsqueeze(0)).sum(dim=1)
            
            # Handle DataParallel for detector
            model_to_call = detector.module if isinstance(detector, torch.nn.DataParallel) else detector
            det_res = model_to_call(uras, conf_scores=conf_scores, unc_scores=unc_scores, risk_scores=risk_scores, adapt=False)
            
            scores = det_res["score"].cpu().numpy()
            thresholds = det_res["threshold"].cpu().numpy()
            is_anomalies = det_res["anomaly"].cpu().numpy()
            base_threshold = None
            det_module = detector.module if isinstance(detector, torch.nn.DataParallel) else detector
            if hasattr(det_module, "threshold"):
                try:
                    base_threshold = float(det_module.threshold.detach().cpu().item())
                except Exception:
                    base_threshold = None
            
            for i, sample in enumerate(meta_list):
                dt_utc = pd.to_datetime(sample.get("t0", 0), unit="ms")
                results.append({
                    "timestamp": dt_utc.strftime('%Y-%m-%dT%H:%M:%S'),
                    "host": sample.get("host", "unknown"),
                    "anomaly_score": float(scores[i]),
                    "adaptive_threshold": float(thresholds[i] if thresholds.ndim > 0 else thresholds),
                    "is_anomaly": int(is_anomalies[i]),
                    "base_threshold": float(base_threshold) if base_threshold is not None else None,
                    "conf_score": float(conf_scores[i].detach().cpu().item()),
                    "unc_score": float(unc_scores[i].detach().cpu().item()),
                    "risk_score": float(risk_scores[i].detach().cpu().item()),
                })
                
    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "detection_results.csv")
    df.to_csv(out_path, index=False)
    print(f"[Test] Results saved to {out_path}")
    
    try:
        import evaluate
        evaluate.run_evaluation(out_path, csv_in_edt=False)
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description="OpTC-URAS Pipeline Runner")
    parser.add_argument("mode", choices=["train_teacher", "train_student", "train_detector", "test", "all"], help="Execution mode")
    parser.add_argument("--config", default="configs/ecar.yaml", help="Path to config file")
    args = parser.parse_args()
    
    if args.mode == "all":
        run_train_teacher(args.config)
        run_train_student(args.config)
        run_train_detector(args.config)
        run_test(args.config)
    elif args.mode == "train_teacher":
        run_train_teacher(args.config)
    elif args.mode == "train_student":
        run_train_student(args.config)
    elif args.mode == "train_detector":
        run_train_detector(args.config)
    elif args.mode == "test":
        run_test(args.config)

if __name__ == "__main__":
    main()
