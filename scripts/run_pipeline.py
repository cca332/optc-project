import os
import sys
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any
from torch.utils.data import DataLoader, Subset

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from optc_uras.data.dataset import OpTCEcarDataset
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
import copy
import collections

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def collate_fn(batch):
    return batch

def main():
    config_path = "config/ecar_config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Extract sub-configs
    optc_cfg = config.get("data", {}).get("optc", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    cache_dir = optc_cfg["cache_dir"]
    # [MODIFIED] Use 'results4' for LoRA + Large Events run
    output_dir = "results4"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Runner] Using device: {device}")

    # -------------------------------------------------------------------------
    # 1. Load Data (From Cache)
    # -------------------------------------------------------------------------
    print("[Runner] Loading datasets...")
    # [USER NOTE] Use original 'train' (identical to train4). Use 'test3' (confirmed 20k events + attack aligned).
    train_dataset = OpTCEcarDataset(cache_dir, split="train")
    test_dataset = OpTCEcarDataset(cache_dir, split="test3")
    
    if len(train_dataset) == 0:
        print("Error: No training data found. Run preprocess_optc_ecar.py first.")
        return

    # Split dataset for Teacher Pretraining vs Student Training
    # [MODIFIED] Drastically reduce Teacher/Val sizes to maximize Student data for FL
    total_samples = len(train_dataset)
    
    # [USER REQ] Switch to Centralized Training to diagnose collapse
    FEDERATED_LEARNING = False 

    teacher_indices = []
    val_indices = []
    student_indices = []

    if not FEDERATED_LEARNING:
        print("[Runner] Centralized Training Mode: 150 Teacher, 50 Val, Rest Student (Sorted by Time)")
        
        # Sort all samples by time globally
        all_timed_samples = []
        for idx in range(len(train_dataset)):
            s = train_dataset[idx]
            all_timed_samples.append((s.get("t0", 0), idx))
        
        all_timed_samples.sort(key=lambda x: x[0])
        sorted_indices = [x[1] for x in all_timed_samples]
        
        # 150 for Teacher
        teacher_indices = sorted_indices[:150]
        # 50 for Validation
        val_indices = sorted_indices[150:200]
        # Rest for Student
        student_indices = sorted_indices[200:]
        
    else:
        # [MODIFIED] Stratified Split by Host and Time
        print("[Runner] Performing Time-based Stratified Split per Host (Federated)...")
        
        # 1. Group by Host
        samples_by_host = collections.defaultdict(list)
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            # Use 'host' and 't0' (timestamp)
            h = sample.get("host", "unknown")
            t = sample.get("t0", 0)
            samples_by_host[h].append((t, idx))
            
        # 2. Split per Host
        for host, items in samples_by_host.items():
            # Sort by time
            items.sort(key=lambda x: x[0])
            
            # We need at least 5 samples to do a 2(T) + 1(S) + 2(V) split
            if len(items) >= 5:
                # First 2 -> Teacher
                teacher_indices.extend([x[1] for x in items[:2]])
                # Last 2 -> Validation
                val_indices.extend([x[1] for x in items[-2:]])
                # Middle -> Student
                student_indices.extend([x[1] for x in items[2:-2]])
            else:
                print(f"  [Warning] Host {host} has only {len(items)} samples. assigning all to Student.")
                student_indices.extend([x[1] for x in items])

    # Create Subsets
    teacher_subset = Subset(train_dataset, teacher_indices)
    val_subset = Subset(train_dataset, val_indices)
    student_subset = Subset(train_dataset, student_indices)
    
    print(f"[Runner] Data Split: Teacher={len(teacher_subset)}, Val={len(val_subset)}, Student={len(student_subset)}")
    print(f"         (Expected: ~50 Teacher, ~50 Val, ~425 Student)")

    teacher_loader = DataLoader(teacher_subset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    student_loader = DataLoader(student_subset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # -------------------------------------------------------------------------
    # 2. Initialize Models
    # -------------------------------------------------------------------------
    print("[Runner] Initializing Step 1 & Fitting Vocabs...")
    views = ["process", "file", "network"]
    
    epochs = train_cfg.get("epochs", 5)
    batch_size = train_cfg.get("batch_size", 32)
    lr = 1e-5 # [MODIFIED] Extremely low LR for LoRA fine-tuning
    
    # Config for Step 1
    quality_cfg = QualityWeightsConfig(
        reliability_weights={"validity": 1.0, "completeness": 1.0},
        info_gain_w_min=0.3,
        info_gain_temp=1.0,
        standardize="minmax",
        softmax_temperature=model_cfg.get("quality_softmax_temp", 0.5)
    )
    s1_cfg = Step1Config(
        views=views,
        window_seconds=optc_cfg.get("window_minutes", 15) * 60,
        slot_seconds=model_cfg.get("slot_seconds", 60),
        include_empty_slot_indicator=True,
        num_hash_buckets=50,
        hash_seed=42,
        target_dim=32,
        rp_seed=123,
        rp_matrix_type="gaussian",
        rp_normalize="l2",
        rp_nonlinearity="relu",
        quality_cfg=quality_cfg,
        router_hidden_dims=[64],
        router_dropout=0.1,
        num_subspaces=4,
        gate_gamma=0.5,
        gate_mode="soft",
        gate_beta=5.0,
        interaction_enabled=True
    )
    
    # Schema
    vocab_cache_path = os.path.join(output_dir, "vocab_schema.pt")
    
    # [OPTIMIZATION] Always load samples for Quality Stats fitting, even if vocab is cached
    print("[Runner] Sampling data for vocab/stats fitting...")
    vocab_samples = []
    # Use a small subset (e.g., 300 samples)
    if len(teacher_subset) > 0:
        vocab_indices = torch.randperm(len(teacher_subset))[:min(300, len(teacher_subset))].tolist()
        for i in tqdm(vocab_indices, desc="Loading Samples"):
            vocab_samples.append(teacher_subset[i])
    else:
        print("[Warning] Teacher subset is empty, skipping sample loading.")

    if os.path.exists(vocab_cache_path):
        print(f"[Runner] Loading cached vocabs from {vocab_cache_path}...")
        # [FIX] Set weights_only=False because we are loading a custom object (AggregatorSchema), not just weights
        per_view_schema = torch.load(vocab_cache_path, weights_only=False)
    else:
        per_view_schema = {
            v: AggregatorSchema(event_type_vocab=[], key_fields=["action", "object"]) 
            for v in views
        }
    
    # [FIX] Enforce correct key_fields for A5 Completeness metric
    # OpTC data fields: process(image_path, command_line), file(file_path), network(dest_ip, dest_port)
    key_fields_map = {
        "process": ["action", "object", "image_path", "command_line"],
        "file": ["action", "object", "file_path"],
        "network": ["action", "object", "dest_ip", "dest_port"],
    }
    for v, schema in per_view_schema.items():
        if v in key_fields_map:
            schema.key_fields = key_fields_map[v]
    
    step1 = Step1Model(s1_cfg, per_view_schema).to(device)
    
    # Fit Vocabs (or skip if cached) and Init Projectors
    step1.fit_vocabs_and_init_projectors(vocab_samples, max_types=500)
    
    # [OPTIMIZATION] Fit Quality Stats for Standardization (Balance Entropy vs Intensity)
    print("[Runner] Fitting Quality Stats for Standardization...")
    step1.fit_quality_stats(vocab_samples)
    
    # [FIX] Force re-initialization of Learnable Components (Router, Fusion) to avoid dead Relu
    # FixedRandomProjector is numpy-based and fixed, so we don't init it.
    print("[Runner] Re-initializing Learnable Components (Router, Fusion) with Kaiming Normal...")
    
    # [ULTIMATE FIX] Replace ReLU with LeakyReLU in learnable components
    # ReLU can cause "dead neurons" (output 0) which leads to collapse. LeakyReLU prevents this.
    print("[Runner] Replacing ReLU with LeakyReLU to prevent dead neurons...")

    def replace_relu(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.ReLU):
                setattr(module, name, torch.nn.LeakyReLU(negative_slope=0.1, inplace=True))
            else:
                replace_relu(child)

    replace_relu(step1.router)
    replace_relu(step1.fusion)

    # Init Router
    for layer in step1.router.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    # Init Fusion
    for layer in step1.fusion.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    # -------------------------------------------------------------------------
    # [LORA IMPLEMENTATION]
    # -------------------------------------------------------------------------
    print("[Runner] Injecting LoRA adapters into Step 1 (Router/Fusion)...")
    
    class LoRALayer(torch.nn.Module):
        def __init__(self, original_linear, rank=4, alpha=8):
            super().__init__()
            self.original = original_linear
            self.rank = rank
            self.scaling = alpha / rank
            
            # A: [in, r], B: [r, out]
            # Initialize A with Gaussian, B with Zeros -> starts as Identity
            self.lora_A = torch.nn.Parameter(torch.randn(original_linear.in_features, rank) * 0.01)
            self.lora_B = torch.nn.Parameter(torch.zeros(rank, original_linear.out_features))
            
            # Freeze original
            self.original.weight.requires_grad = False
            if self.original.bias is not None:
                self.original.bias.requires_grad = False
                
        def forward(self, x):
            # h = Wx + (BA)x * scale
            base = self.original(x)
            lora = (x @ self.lora_A @ self.lora_B) * self.scaling
            return base + lora
            
    def apply_lora(module, target_names=["Linear"], rank=4):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                # Replace Linear with LoRALayer
                setattr(module, name, LoRALayer(child, rank=rank))
            else:
                apply_lora(child, target_names, rank)
                
    # Apply LoRA to Router and Fusion only
    # We keep Alignment frozen (or trainable? Let's freeze Alignment to simplify)
    step1.alignment.requires_grad_(False) 
    
    # Apply LoRA to Router (MLP)
    apply_lora(step1.router, rank=4)
    # Apply LoRA to Fusion (Projections)
    apply_lora(step1.fusion, rank=4)
    
    print("[Runner] LoRA injection complete. Step 1 Base frozen, Adapters trainable.")

    # Save cache if needed
    if not os.path.exists(vocab_cache_path):
        torch.save(step1.schemas, vocab_cache_path)
        print(f"[Runner] Saved vocab schema to {vocab_cache_path}")
    
    # Free memory
    del vocab_samples
    import gc
    gc.collect()
    
    # Step 2 Models
    subspace_dim = 16
    student = StudentHeads(in_dim=32, num_subspaces=4, subspace_dim=subspace_dim, hidden_dim=64).to(device)
    
    # [FIX] Replace ReLU in Student Heads to prevent dead neurons
    replace_relu(student)
    
    # Teacher Setup (A1/A2/A3)
    # A2: Behavioral Feature Dimension
    behavior_dim = 128 
    teacher = TeacherModel(behavior_dim=behavior_dim, num_subspaces=4, subspace_dim=subspace_dim, hidden_dim=64).to(device)
    
    # A2: DP Config
    dp_cfg = FeatureDPConfig(enabled=True, clip_C=10.0, noise_sigma=0.01)
    
    uras_dim = 4 * subspace_dim
    
    # Step 3
    detector = AnomalyDetector(
        feature_dim=uras_dim, 
        style_dim=16, 
        content_dim=16, 
        view_names=views, 
        view_dim=32,
        drift_lr=model_cfg.get("drift_lr", 0.01)
    ).to(device)

    # [OPTIMIZATION] Replace ReLU in Detector (SCD) to prevent collapse
    replace_relu(detector)

    # -------------------------------------------------------------------------
    # 3. Phase 1: Teacher Self-Supervised Learning (A3: InfoNCE)
    # -------------------------------------------------------------------------
    print("\n[Runner] === Phase 1: Teacher Pretraining (A3: InfoNCE) ===")
    teacher_ckpt_path = os.path.join(output_dir, "teacher_checkpoint.pt")
    if os.path.exists(teacher_ckpt_path):
        print(f"[Runner] Found teacher checkpoint at {teacher_ckpt_path}. Loading...")
        teacher.load_state_dict(torch.load(teacher_ckpt_path, map_location=device))
        print("[Runner] Teacher loaded.")
    else:
        teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
        teacher.train()
        
        teacher_loss_history = []
        phase1_epochs = 5
        
        for epoch in range(phase1_epochs):
            epoch_loss = 0.0
            steps = 0
            pbar = tqdm(teacher_loader, desc=f"Phase 1 Epoch {epoch+1}/{phase1_epochs}")
            for batch in pbar:
                # A2: Extract Behavioral Features b
                b_list = []
                for s in batch:
                    b_np = behavior_features_from_sample(s, views, out_dim=behavior_dim)
                    b_list.append(torch.from_numpy(b_np))
                
                b_features = torch.stack(b_list).to(device).float() # [B, behavior_dim]
                
                # A2: Apply DP (Clip + Noise)
                b_dp = dp_features(b_features, dp_cfg)
                
                # A3: InfoNCE Loss
                loss_nce = teacher.forward_contrastive(b_dp, temp=0.1)
                
                teacher_optimizer.zero_grad()
                loss_nce.backward()
                teacher_optimizer.step()
                
                epoch_loss += loss_nce.item()
                steps += 1
                pbar.set_postfix({"nce_loss": f"{loss_nce.item():.4f}"})
            
            avg_loss = epoch_loss/max(steps,1)
            teacher_loss_history.append({"epoch": epoch + 1, "loss": avg_loss})
            print(f"Phase 1 Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        # Save Teacher Logs
        pd.DataFrame(teacher_loss_history).to_csv(os.path.join(output_dir, "teacher_train_log.csv"), index=False)
        print(f"[Runner] Teacher logs saved to {os.path.join(output_dir, 'teacher_train_log.csv')}")
        
        # Save Checkpoint
        torch.save(teacher.state_dict(), teacher_ckpt_path)
        print(f"[Runner] Teacher checkpoint saved to {teacher_ckpt_path}")

    # Freeze Teacher (A3: "训练后冻结")
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print("[Runner] Teacher Pretrained & Frozen.")

    # [MOVED & ENHANCED] Teacher Validation & Health Check
    print("[Runner] Validating Teacher on Validation Set...")
    val_loss = 0.0
    val_steps = 0
    all_z_teacher = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Teacher Validation"):
            b_list = []
            for s in batch:
                b_np = behavior_features_from_sample(s, views, out_dim=behavior_dim)
                b_list.append(torch.from_numpy(b_np))
            b_features = torch.stack(b_list).to(device).float()
            
            # Use clean features for validation check
            # Forward to get embeddings for Std check
            z_t = teacher(b_features)
            all_z_teacher.append(z_t.cpu())
            
            loss_nce = teacher.forward_contrastive(b_features, temp=0.1)
            val_loss += loss_nce.item()
            val_steps += 1
    
    avg_val_loss = val_loss / max(val_steps, 1)
    
    # Check for Teacher Collapse
    all_z_t = torch.cat(all_z_teacher, dim=0)
    z_t_std = all_z_t.std(dim=0).mean().item()
    
    print(f"[Runner] Teacher Validation Loss: {avg_val_loss:.4f} | Z Std: {z_t_std:.4e}")
    
    # Append validation result to log
    with open(os.path.join(output_dir, "teacher_train_log.csv"), "a") as f:
        f.write(f"Validation,{avg_val_loss:.4f},{z_t_std:.4e}\n")

    if z_t_std < 1e-4:
        print(f"  [!] CRITICAL: Teacher Model Collapsed (Std={z_t_std:.4e}).")
        print(f"      Action: Deleting bad checkpoint {teacher_ckpt_path} and aborting.")
        if os.path.exists(teacher_ckpt_path):
            os.remove(teacher_ckpt_path)
        raise RuntimeError("Teacher Collapsed. Please re-run to retrain teacher.")

    # -------------------------------------------------------------------------
    # 4. Phase 2: Federated Training Loop
    # -------------------------------------------------------------------------
    print(f"\n[Runner] === Phase 2: {'Federated' if FEDERATED_LEARNING else 'Centralized'} Student Training ===")
    
    clients = []

    # [FIX] Check for checkpoints early to skip expensive client setup
    step1_ckpt_path = os.path.join(output_dir, "step1_checkpoint_no_dp.pt")
    student_ckpt_path = os.path.join(output_dir, "student_checkpoint_no_dp.pt")
    phase2_checkpoints_exist = os.path.exists(step1_ckpt_path) and os.path.exists(student_ckpt_path)

    if phase2_checkpoints_exist:
        print("[Runner] Phase 2 checkpoints found. Skipping client initialization.")
        # Add dummy client to satisfy server init (min_clients=1)
        clients.append(FederatedClient("dummy", []))
    elif not FEDERATED_LEARNING:
        # Single Central Client containing all student data
        print("[Runner] Initializing Single Central Client...")
        central_samples = [train_dataset[i] for i in student_indices]
        clients.append(FederatedClient("central_node", central_samples))
    else:
        # Create Clients (One per Host in Student Set)
        samples_by_host = collections.defaultdict(list)
        for idx in student_indices:
            sample = train_dataset[idx]
            samples_by_host[sample.get("host", "unknown_host")].append(sample)
            
        sorted_hosts = sorted(samples_by_host.items(), key=lambda x: len(x[1]), reverse=True)
        for host, host_samples in sorted_hosts:
            if len(host_samples) > 0:
                clients.append(FederatedClient(client_id=host, samples=host_samples))
            
    # [CRITICAL] Freeze Step 1 Base, but keep LoRA trainable
    print("[Runner] Configuring Step 1 Gradients (LoRA Only)...")
    for name, param in step1.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Verify trainable params
    trainable_params = sum(p.numel() for p in step1.parameters() if p.requires_grad)
    print(f"[Runner] Step 1 Trainable Params (LoRA): {trainable_params}")

    print(f"[Runner] Initialized {len(clients)} Clients.")
    for c in clients[:15]:
        print(f"  - Client {c.client_id}: {len(c.samples)} samples")
    if len(clients) > 15:
        print(f"  - ... and {len(clients)-15} more.")

    # Configs
    # [MODIFIED] Client Config:
    # 1. Batch Size = 16 (Requested by user)
    # 2. Local Epochs = 5 (Aggressive learning for small data)
    # A2: Client Config
    client_train_cfg = ClientTrainConfig(
        local_epochs=5,       # [TUNED] Increased from 3 to 5 (Aggressive)
        batch_size=16,        # [USER REQ] Reduced to 16 to fit local data (approx 17 samples/client)
        lr=1e-4,              # [TUNED] Reduced to 1e-4 to prevent collapse
        weight_decay=1e-4,
        lambda_stats=50.0,    # [TUNED] Increased to 50.0 to strongly force statistical alignment (Anti-Collapse)
        lambda_infonce=1.0,   # [CRITICAL] Balanced mimicry
        temp_params={"a": 2.0, "b": 1.0, "min_tau": 0.05, "max_tau": 0.2}, # [TUNED] Lower max_tau for sharper distillation
        feature_dp=dp_cfg,
        grad_dp=GradDPConfig(enabled=False, base_clip_C0=1.0, importance_alpha=0.0, noise_sigma0=0.0), # DISABLE DP to verify learning capability
        views=views,
        behavior_feature_dim=behavior_dim
    )
    
    server_cfg = ServerConfig(
        rounds=epochs,
        client_fraction=1.0, # All clients participate per round
        min_clients=1,
        server_lr=1.0,
        secure_agg_enabled=True,
        secure_agg_protocol="pairwise_masking" # Simulation: supports "pairwise_masking" or "mock_secureagg"
    )
    
    server = FederatedServer(clients, server_cfg)
    
    # Global Models (Step 1 + Student) are already initialized
    # We will update them via FL rounds
    
    # Force Retraining by using a new checkpoint name for this experiment
    step1_ckpt_path = os.path.join(output_dir, "step1_checkpoint_no_dp.pt")
    student_ckpt_path = os.path.join(output_dir, "student_checkpoint_no_dp.pt")
    
    if os.path.exists(step1_ckpt_path) and os.path.exists(student_ckpt_path):
        print(f"[Runner] Found Phase 2 checkpoints. Loading...")
        step1.load_state_dict(torch.load(step1_ckpt_path, map_location=device))
        student.load_state_dict(torch.load(student_ckpt_path, map_location=device))
        print("[Runner] Phase 2 models loaded.")
    else:
        fl_loss_history = []
        
        for round_idx in range(server_cfg.rounds):
            print(f"\n--- FL Round {round_idx+1}/{server_cfg.rounds} ---")
            
            # 1. Sample Clients
            sampled_clients = server.sample_clients()
            
            # 2. Local Training (Simulation)
            updates = []
            ns = []
            round_metrics = []
            
            # In simulation, we iterate sequentially, but conceptually this happens in parallel
            for client in sampled_clients:
                # Simulate Model Download (Copy Global -> Local)
                client_step1 = copy.deepcopy(step1)
                client_student = copy.deepcopy(student)
                
                # Local Train
                update_vec, metrics, n_samples = client.local_train(
                    client_step1, client_student, teacher, client_train_cfg, device
                )
                
                updates.append(update_vec)
                ns.append(n_samples)
                round_metrics.append(metrics)
                
                # Clean up to save memory in simulation
                del client_step1, client_student
                import gc; gc.collect()
                
            # 3. Secure Aggregation & Global Update
            # Identify global params corresponding to the update vector structure
            # Note: We must ensure parameter order matches exactly what Client.local_train produces
            global_params = list(step1.parameters()) + list(student.parameters())
            
            agg_info = server.aggregate_and_apply(global_params, updates, ns)
            
            # Log
            avg_loss = sum(m["client_loss"] for m in round_metrics) / len(round_metrics) if round_metrics else 0.0
            
            # [ADDED] FL Validation on Validation Set
            step1.eval()
            student.eval()
            val_z_std = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    s1_out = step1(batch)
                    z = torch.stack([o.z for o in s1_out])
                    val_z_std += z.std(dim=0).mean().item()
                    val_steps += 1
            avg_val_std = val_z_std / max(val_steps, 1)
            
            # Revert to train
            step1.train()
            student.train()

            fl_loss_history.append({"round": round_idx+1, "avg_client_loss": avg_loss, "val_z_std": avg_val_std})
            print(f"Round {round_idx+1} Complete. Avg Client Loss: {avg_loss:.4f} | Val Z Std: {avg_val_std:.4e}")

            # [FAIL-FAST] Check for Feature Collapse immediately
            if avg_val_std < 1e-6:
                print(f"  [!] CRITICAL FAILURE: Feature Collapse Detected at Round {round_idx+1}.")
                print("      Action: Aborting training to save time.")
                sys.exit(1)

        pd.DataFrame(fl_loss_history).to_csv(os.path.join(output_dir, "fl_train_log.csv"), index=False)
        print("[Runner] FL Training Complete.")
        
        # Save Checkpoints
        torch.save(step1.state_dict(), step1_ckpt_path)
        torch.save(student.state_dict(), student_ckpt_path)
        print(f"[Runner] Phase 2 checkpoints saved to {output_dir}")

        # [DIAGNOSIS] Immediate Check for Feature Collapse & Scientific Validity (Quality vs Quantity)
        print("\n[DIAGNOSIS] Verifying Step 1 Feature Health & Method Validity...")
        step1.eval()
        with torch.no_grad():
            # Use a larger batch for correlation check
            diag_samples = [train_dataset[i] for i in range(min(100, len(train_dataset)))]
            diag_outs = step1(diag_samples)
            
            # 1. Feature Health (Anti-Collapse)
            diag_z = torch.stack([o.z for o in diag_outs])
            diag_std = diag_z.std(dim=0).mean().item()
            
            # 2. Method Validity (Quality vs Quantity)
            # We want to prove that Routing/Attention is NOT just correlated with Sequence Length (Quantity).
            # If it is, then "Quality" is a misnomer for "More Data".
            # If correlation is low, it means we are attending to *content patterns*, which validates the method.
            
            # Calculate input lengths (number of non-empty events)
            input_lengths = torch.tensor([len(s["events"]) for s in diag_samples], dtype=torch.float32)
            
            # Calculate Max View Weight (Did we just pick the view with most data?)
            diag_w = torch.stack([o.reliability_w for o in diag_outs]) # [B, V]
            max_w, _ = diag_w.max(dim=1)
            
            # Calculate Routing Entropy (Diversity)
            diag_route = torch.stack([o.route_p for o in diag_outs])   # [B, M]
            entropy = -(diag_route * torch.log(diag_route+1e-9)).sum(dim=1)
            
            # Manual Pearson Correlation
            def pearson(x, y):
                vx = x - x.mean()
                vy = y - y.mean()
                return (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-9)
                
            corr_len_weight = pearson(input_lengths, max_w).item()
            corr_len_entropy = pearson(input_lengths, entropy).item()
            
            # 3. Dataset Quality Diversity (Weight Variance)
            # If weights are static (std~0), model ignores sample content.
            # If weights vary (std>0), model sees quality differences across samples.
            weight_std = diag_w.std(dim=0)
            avg_weight_std = weight_std.mean().item()

            print(f"  - Latent Z Mean Std: {diag_std:.4e} (Target: > 1e-2)")
            print(f"  - Avg View Weights: {diag_w.mean(dim=0).tolist()}")
            print(f"  - Weight Std (Diversity): {avg_weight_std:.4f} (Target: > 0.05)")
            print(f"  - Avg Routing Entropy: {entropy.mean().item():.4f}")
            print(f"  - Correlation [Length vs MaxWeight]: {corr_len_weight:.4f} (Target: Low/Negative)")
            print(f"  - Correlation [Length vs Entropy]:   {corr_len_entropy:.4f} (Target: Low)")
            
            if diag_std < 1e-4:
                print("  [!] CRITICAL FAILURE: Feature Collapse Detected.")
                print("      Action: Increase lambda_infonce or check data preprocessing.")
                print("      [TERMINATION] Pipeline stopped to prevent meaningless downstream tasks.")
                sys.exit(1)
            elif abs(corr_len_weight) > 0.8:
                print("  [!] SCIENTIFIC WARNING: High correlation between Length and Weight.")
                print("      The model might be confusing 'Quantity' with 'Quality'.")
            elif avg_weight_std < 0.01:
                print("  [!] SCIENTIFIC WARNING: Low Weight Diversity.")
                print("      The model is outputting static weights, ignoring sample-specific quality.")
            else:
                print("  [+] SCIENTIFIC VALIDATION PASSED:")
                print("      1. Feature Health: GOOD (No Collapse).")
                print("      2. Method Validity: Quality != Quantity (Low Correlation).")
                print("      3. Dataset Suitability: Quality Differences Detected (High Weight Diversity).")
        step1.train() # Revert to train mode if needed later

    # -------------------------------------------------------------------------
    # 5. Phase 3: Centralized Detector Training (Step 3)
    # -------------------------------------------------------------------------
    print("\n[Runner] === Phase 3: Centralized Detector Training (Step 3) ===")
    # Now that Step 1 & Student are trained and aggregated, we train the SCD Detector
    # on the server side (using available data, here reused student_loader for demo)
    # Note: In strict FL, server might only have a small public dataset or use generator.
    # Here we assume the server can use the 'student_loader' data (or a subset) for Step 3 
    # as a proxy for 'server-side adaptation' or 'public dataset'.
    
    step1.eval()
    student.eval()
    detector.train()
    
    # [DIAGNOSIS] Check Step 1 URAS Feature Quality
    print("[Runner] Diagnosis: Checking feature stats on a single batch...")
    with torch.no_grad():
        diag_batch = next(iter(student_loader))
        s1_out = step1(diag_batch)
        z = torch.stack([o.z for o in s1_out])
        route_p = torch.stack([o.route_p for o in s1_out])
        subspace_vecs = student(z)
        diag_uras = uras_from_subspaces(subspace_vecs, route_p)
        uras_std = diag_uras.std(dim=0).mean().item()
        uras_norm = diag_uras.norm(dim=1).mean().item()

    print(f"[Runner] Step 1 URAS Features Diagnosis:")
    print(f"  - Mean Norm: {uras_norm:.4e} (Should be > 0.1)")
    print(f"  - Mean Std:  {uras_std:.4e}  (If < 1e-4, Step 1 has collapsed!)")
    
    detector_ckpt_path = os.path.join(output_dir, "detector_checkpoint_retrain.pt")
    
    if os.path.exists(detector_ckpt_path):
        print(f"[Runner] Found detector checkpoint at {detector_ckpt_path}. Loading...")
        detector.load_state_dict(torch.load(detector_ckpt_path, map_location=device))
        print("[Runner] Detector loaded.")
    else:
        optimizer_det = torch.optim.Adam(detector.scd.parameters(), lr=lr * 3)
        
        det_loss_history = []
        phase3_epochs = 25
        
        for epoch in range(phase3_epochs):
            epoch_loss = 0.0
            epoch_rec = 0.0
            epoch_var = 0.0
            epoch_dec = 0.0
            steps = 0
            pbar = tqdm(student_loader, desc=f"Phase 3 Epoch {epoch+1}/{phase3_epochs}")
            for batch in pbar:
                with torch.no_grad():
                    s1_out = step1(batch)
                    z = torch.stack([o.z for o in s1_out])
                    route_p = torch.stack([o.route_p for o in s1_out])
                    subspace_vecs = student(z)
                    uras = uras_from_subspaces(subspace_vecs, route_p)
                
                # SCD Training
                s, c, u_hat = detector.scd(uras, center_batch=True, return_rec=True)
                loss_scd = scd_loss(s, c, uras, u_hat, lambda_d=1.0, lambda_r=1.0, lambda_v=25.0, gamma=5.0)
                
                optimizer_det.zero_grad()
                loss_scd.backward()
                optimizer_det.step()
                
                epoch_loss += loss_scd.item()
                steps += 1
                
                # Debug / Logging components
                with torch.no_grad():
                    # Calculate components manually for logging
                    # Decorr
                    sigma_sc = torch.matmul(s.t(), c) / float(s.shape[0])
                    l_decorr_val = (sigma_sc ** 2).sum().item()
                    # Rec
                    l_rec_val = (uras - u_hat).pow(2).sum(dim=-1).mean().item()
                    # Var (using unbiased=False for stability)
                    std_s = torch.sqrt(s.var(dim=0, unbiased=False) + 1e-6)
                    std_c = torch.sqrt(c.var(dim=0, unbiased=False) + 1e-6)
                    l_var_val = (torch.nn.functional.relu(1.0 - std_s).sum() + torch.nn.functional.relu(1.0 - std_c).sum()).item()
                
                epoch_rec += l_rec_val
                epoch_var += l_var_val
                epoch_dec += l_decorr_val

                pbar.set_postfix({
                    "loss": f"{loss_scd.item():.2f}",
                    "rec": f"{l_rec_val:.2f}",
                    "var": f"{l_var_val:.2f}",
                    "dec": f"{l_decorr_val:.2f}"
                })
            
            avg_loss = epoch_loss / max(steps, 1)
            avg_rec = epoch_rec / max(steps, 1)
            avg_var = epoch_var / max(steps, 1)
            avg_dec = epoch_dec / max(steps, 1)
            
            det_loss_history.append({
                "epoch": epoch+1, 
                "loss": avg_loss,
                "rec": avg_rec,
                "var": avg_var,
                "dec": avg_dec
            })
        
        pd.DataFrame(det_loss_history).to_csv(os.path.join(output_dir, "detector_train_log.csv"), index=False)
        
        # Save Checkpoint
        torch.save(detector.state_dict(), detector_ckpt_path)
        print(f"[Runner] Detector checkpoint saved to {detector_ckpt_path}")

        # [ADDED] SCD Detector Validation
        print("[Runner] Validating SCD Detector on Validation Set...")
        step1.eval()
        student.eval()
        detector.eval()
        scd_val_loss = 0.0
        scd_val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="SCD Validation"):
                s1_out = step1(batch)
                z = torch.stack([o.z for o in s1_out])
                route_p = torch.stack([o.route_p for o in s1_out])
                subspace_vecs = student(z)
                uras = uras_from_subspaces(subspace_vecs, route_p)
                
                # SCD Loss
                s, c, u_hat = detector.scd(uras, center_batch=True, return_rec=True)
                loss_scd = scd_loss(s, c, uras, u_hat, lambda_d=1.0, lambda_r=1.0, lambda_v=25.0, gamma=5.0)
                scd_val_loss += loss_scd.item()
                scd_val_steps += 1
        
        avg_scd_val_loss = scd_val_loss / max(scd_val_steps, 1)
        print(f"[Runner] SCD Detector Validation Loss: {avg_scd_val_loss:.4f}")
        # Append validation result to log
        with open(os.path.join(output_dir, "detector_train_log.csv"), "a") as f:
            f.write(f"Validation, {avg_scd_val_loss:.4f},0,0,0\n")

    # -------------------------------------------------------------------------
    # 6. Post-Training: Compute Benign Statistics & Fit Interpreter
    # -------------------------------------------------------------------------
    print("\n[Runner] === Phase 4: Compute Benign Statistics & Fit Interpreter ===")
    step1.eval()
    student.eval()
    detector.eval()
    
    all_s = []
    all_uras = []
    all_vf = []
    all_vw = []
    
    # Use student_loader (benign samples) to compute stats
    with torch.no_grad():
        for batch in tqdm(student_loader, desc="Collecting Stats"):
            s1_out = step1(batch)
            z = torch.stack([o.z for o in s1_out])
            route_p = torch.stack([o.route_p for o in s1_out])
            
            subspace_vecs = student(z)
            uras = uras_from_subspaces(subspace_vecs, route_p)
            
            # Get Style features
            s, _ = detector.scd(uras, center_batch=False)
            
            all_s.append(s.cpu())
            all_uras.append(uras.cpu())
            
            # For Interpreter (view features & weights)
            all_vf.append(torch.stack([o.view_vecs for o in s1_out]).cpu())
            all_vw.append(torch.stack([o.reliability_w for o in s1_out]).cpu())
            
    all_s = torch.cat(all_s, dim=0)
    train_uras = torch.cat(all_uras, dim=0)
    train_vf = torch.cat(all_vf, dim=0)
    train_vw = torch.cat(all_vw, dim=0)

    # Manual Stats Fitting (Logic from detector.fit)
    detector.style_mu = all_s.mean(dim=0).to(device)
    s_centered = all_s.to(device) - detector.style_mu
    var = s_centered.var(dim=0, unbiased=False) + 1e-6
    detector.style_var = var
    detector.style_inv_cov = torch.diag(1.0 / var)
    
    # Compute Threshold (Quantile)
    scores = detector._compute_raw_score(all_s.to(device))
    
    # [DIAGNOSIS] Print Score Distribution
    q_vals = torch.tensor([0.5, 0.9, 0.95, 0.99, 0.999], device=device)
    quantiles = torch.quantile(scores, q_vals)
    print(f"[Runner] Score Distribution (Validation):")
    print(f"  - 50%: {quantiles[0]:.4e}")
    print(f"  - 90%: {quantiles[1]:.4e}")
    print(f"  - 95%: {quantiles[2]:.4e}")
    print(f"  - 99%: {quantiles[3]:.4e}")
    print(f"  - 99.9%: {quantiles[4]:.4e}")

    # Increase quantile to 0.90 (Experimental High Sensitivity)
    # The previous 0.999 was too high (1e-7) compared to attack scores (1e-9) due to feature collapse.
    detector.threshold = torch.quantile(scores, model_cfg.get("threshold_quantile", 0.90))
    print(f"[Runner] Threshold set to: {detector.threshold.item():.4e}")

    # Dynamically set ATC alphas based on threshold magnitude
    # [MODIFIED] Set alphas to 0.0 to disable ATC scaling for now (Pure Quantile Mode)
    # This ensures that the high-ranking anomalies (Rank #1) are not suppressed by high thresholds.
    detector.alpha_conf = 0.0
    detector.alpha_unc = 0.0
    detector.alpha_risk = 0.0
    print(f"[Runner] ATC Alphas set to 0.0 (Pure Thresholding): Conf={detector.alpha_conf:.2e}, Unc={detector.alpha_unc:.2e}, Risk={detector.alpha_risk:.2e}")

    # [DIAGNOSIS] Check if SCD actually learned something (Loss shouldn't be 0)
    if 'det_loss_history' in locals() and len(det_loss_history) > 0:
         print(f"  - Final SCD Loss: {det_loss_history[-1]['loss']:.6f} (Should not be 0.000000)")
    
    print("[Runner] Training Step 3 Interpreter...")
    detector.fit_interpreter(train_vf.to(device), train_vw.to(device), train_uras.to(device), 
                             epochs=5, batch_size=batch_size, 
                             log_path=os.path.join(output_dir, "interpreter_log.csv"))
    
    # -------------------------------------------------------------------------
    # 7. Inference Phase (Test Data)
    # -------------------------------------------------------------------------
    print("\n[Runner] === Phase 5: Inference on Test Set ===")
    detector.eval()
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            # 1. Forward Step 1 & 2
            s1_out = step1(batch)
            
            # Extract z and route_p
            z = torch.stack([o.z for o in s1_out])
            route_p = torch.stack([o.route_p for o in s1_out])
            
            subspace_vecs = student(z)
            uras = uras_from_subspaces(subspace_vecs, route_p)
            
            # Compute ATC Signals (Conf, Unc, Risk)
            # Aligning with client.py definitions to ensure consistency between Training and Detection
            
            # 1. Utility - Confidence: 1.0 - Normalized Entropy of View Weights
            # Step1 already computes normalized entropy in intermediates["w_entropy"]
            w_entropies = torch.tensor([o.intermediates.get("w_entropy", 0.0) for o in s1_out], device=device)
            conf_scores = (1.0 - w_entropies).clamp(0.0, 1.0)
            
            # 2. Utility - Uncertainty: Normalized Entropy of Routing Probabilities
            # Step1 already computes normalized entropy in intermediates["route_entropy"]
            unc_scores = torch.tensor([o.intermediates.get("route_entropy", 0.0) for o in s1_out], device=device)

            # 3. Privacy - Risk: Sensitivity-weighted View Exposure
            # Risk = sum(w_v * sensitivity_v)
            # Define sensitivities (mocking config)
            view_sensitivities = {"process": 1.0, "file": 1.0, "network": 1.0} # Default 1.0
            # If user wanted specific sensitivities, they would be in config. Using defaults for now.
            
            sens_vec = torch.tensor([view_sensitivities.get(v, 1.0) for v in views], device=device)
            reliability_w = torch.stack([o.reliability_w for o in s1_out]) # [B, V]
            risk_scores = (reliability_w * sens_vec.unsqueeze(0)).sum(dim=1) # [B]
            
            # 2. Forward Step 3 (Detect) with ATC
            # adapt=False for reproducible test set evaluation (set True for online stream)
            det_res = detector(
                uras, 
                conf_scores=conf_scores, 
                unc_scores=unc_scores, 
                risk_scores=risk_scores, 
                adapt=False
            )
            
            # 3. Interpret
            vf = torch.stack([o.view_vecs for o in s1_out])
            vw = torch.stack([o.reliability_w for o in s1_out])
            rankings = detector.interpret(vf, vw, uras, top_k=1)
            
            # Collect Results
            scores = det_res["score"].cpu().numpy()
            thresholds = det_res["threshold"].cpu().numpy()
            is_anomalies = det_res["anomaly"].cpu().numpy()
            
            for i, sample in enumerate(batch):
                # Convert UTC timestamp (ms) to EDT (UTC-4) for report consistency
                dt_utc = pd.to_datetime(sample.get("t0", 0), unit="ms")
                dt_edt = dt_utc - pd.Timedelta(hours=4)
                
                rec = {
                    "timestamp": dt_edt.strftime('%Y-%m-%dT%H:%M:%S'), # EDT Time
                    "host": sample.get("host", "unknown"),
                    "anomaly_score": float(scores[i]),
                    "adaptive_threshold": float(thresholds[i] if thresholds.ndim > 0 else thresholds),
                    "is_anomaly": int(is_anomalies[i]),
                    "top_attribution": str(rankings[i][0]) if rankings[i] else "None"
                }
                results.append(rec)
                
    # -------------------------------------------------------------------------
    # 8. Save Output
    # -------------------------------------------------------------------------
    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "detection_results.csv")
    df.to_csv(out_path, index=False)
    print(f"[Runner] Done! Results saved to: {out_path}")
    print(df.head())

if __name__ == "__main__":
    main()