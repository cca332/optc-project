
import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse

# Add root to path to import from main.py
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from main import load_datasets, split_train_data
from optc_uras.data.dataset import OpTCEcarDataset, TensorDataset, tensor_collate
from optc_uras.models.step1 import Step1Config, Step1Model
from optc_uras.models.student import StudentHeads, uras_from_subspaces
from optc_uras.models.deep_detector import DeepReconstructionDetector
from optc_uras.features.quality import QualityWeightsConfig
from optc_uras.features.deterministic_aggregator import AggregatorSchema
from optc_uras.utils.misc import replace_relu, apply_lora

VIEWS = ["process", "file", "network"]

def load_cached_split(cache_dir, split_prefix):
    cache_path = os.path.join(cache_dir, "optimized_data.pt")
    if not os.path.exists(cache_path):
        return None

    try:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[DeepDetector] Failed to load optimized cache '{cache_path}': {exc}")
        return None

    metadata = data.get("metadata")
    if metadata is None:
        return None

    ds_all = OpTCEcarDataset(cache_dir, split="all", preload=False)
    if len(metadata) != len(ds_all.index_map):
        raise RuntimeError(
            f"Optimized cache '{cache_path}' is stale: metadata has {len(metadata)} samples, "
            f"but current cache index has {len(ds_all.index_map)} samples. "
            f"Please rebuild optimized_data.pt before training the deep detector."
        )

    def _path_has_prefix(path: str, prefix: str) -> bool:
        base = os.path.basename(path)
        return base.startswith(prefix + "_") or base.startswith(prefix + "part") or base.startswith(prefix + "p")

    indices = [i for i, (path, _, _, _) in enumerate(ds_all.index_map) if _path_has_prefix(path, split_prefix)]
    if not indices:
        raw_split = OpTCEcarDataset(cache_dir, split=split_prefix, preload=False)
        if len(raw_split) > 0:
            raise RuntimeError(
                f"Optimized cache '{cache_path}' does not include split '{split_prefix}'. "
                f"Please rebuild optimized_data.pt after preprocessing the '{split_prefix}' cache."
            )
        return None

    if max(indices) >= len(metadata):
        raise RuntimeError(
            f"Optimized cache '{cache_path}' is stale: split '{split_prefix}' references indices "
            f"up to {max(indices)}, but optimized metadata only has {len(metadata)} samples. "
            f"Please rebuild optimized_data.pt."
        )

    dataset = TensorDataset(data, indices)
    dataset.metadata_list = metadata
    dataset.get_metadata = lambda idx, ds=dataset: ds.metadata_list[ds.indices[idx]]
    return dataset

def setup_environment(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    optc_cfg = config["data"]["optc"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    output_dir = train_cfg["output_dir"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    return config, optc_cfg, train_cfg, model_cfg, device, output_dir

def build_step1_config(optc_cfg, model_cfg):
    quality_cfg = QualityWeightsConfig(
        reliability_weights={"validity": 1.0, "completeness": 1.0},
        info_gain_w_min=0.3,
        info_gain_temp=1.0,
        standardize="minmax",
        softmax_temperature=model_cfg.get("quality_softmax_temp", 0.5),
    )
    return Step1Config(
        views=VIEWS,
        window_seconds=optc_cfg.get("window_minutes", 5) * 60,
        slot_seconds=model_cfg.get("slot_seconds", 30),
        include_empty_slot_indicator=True,
        num_hash_buckets=model_cfg.get("num_hash_buckets", 50),
        hash_seed=42,
        target_dim=model_cfg["target_dim"],
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
        interaction_enabled=True,
        quality_injection_lambda=model_cfg.get("quality_injection_lambda", 0.5),
        semantic_dim=model_cfg.get("semantic_dim", 128),
        stat_proj_dim=model_cfg.get("stat_proj_dim", 32),
        vocab_sizes=model_cfg.get("vocab_sizes"),
    )

def get_dataset_metadata(dataset, global_idx):
    if isinstance(dataset, Subset):
        base = dataset.dataset
        if hasattr(base, "data") and isinstance(base.data, dict):
            return base.data["metadata"][int(global_idx)]
        if hasattr(base, "get_metadata"):
            try:
                return base.get_metadata(int(global_idx))
            except Exception:
                pass
    if hasattr(dataset, "data") and isinstance(dataset.data, dict):
        return dataset.data["metadata"][int(global_idx)]
    if hasattr(dataset, "get_metadata"):
        return dataset.get_metadata(int(global_idx))
    raise AttributeError("Dataset does not expose metadata access.")

def get_token_features(loader, step1, student, device):
    step1.eval()
    student.eval()
    all_view_vecs = []
    all_global_vecs = []
    all_labels = []
    all_meta = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Features"):
            s1_model = step1.module if hasattr(step1, "module") else step1
            if isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], dict):
                idx = None
                batch_samples = batch
                s1_out = s1_model(batch_samples)
            else:
                idx, _, *sem_batch, st_tensor, q_tensor = batch
                s1_out = s1_model.forward_from_cache(*[t.to(device) for t in sem_batch], st_tensor.to(device), q_tensor.to(device))
            
            # view_vecs: [B, V, d]
            v_vecs = torch.stack([o.view_vecs for o in s1_out])
            # Normalize each view independently (alignment with training)
            v_vecs = torch.nn.functional.normalize(v_vecs, dim=-1)
            z = torch.stack([o.z for o in s1_out])
            route_p = torch.stack([o.route_p for o in s1_out])
            sub_s = student(z, normalize=True)
            u_s = uras_from_subspaces(sub_s, route_p)
            u_s = torch.nn.functional.normalize(u_s, dim=-1)
            
            all_view_vecs.append(v_vecs.cpu())
            all_global_vecs.append(u_s.cpu())
            
            # Get metadata/labels
            if idx is None:
                for sample in batch_samples:
                    all_labels.append(sample.get("label", 0))
                    all_meta.append(sample)
            else:
                for i in idx:
                    m = get_dataset_metadata(loader.dataset, i)
                    all_labels.append(m.get("label", 0))
                    all_meta.append(m)
                
    return torch.cat(all_view_vecs, dim=0), torch.cat(all_global_vecs, dim=0), np.array(all_labels), all_meta

def build_results_dataframe(test_meta, scores, threshold, per_view_scores, global_scores):
    results = []
    token_names = list(VIEWS) + ["global"]
    preds = (scores > threshold).astype(int)
    for i, meta in enumerate(test_meta):
        dt_utc = pd.to_datetime(meta.get("t0", 0), unit="ms")
        token_scores = np.concatenate([per_view_scores[i], [global_scores[i]]], axis=0)
        top_view_idx = int(np.argmax(token_scores))
        top_view_name = token_names[top_view_idx]
        results.append({
            "timestamp": dt_utc.strftime('%Y-%m-%dT%H:%M:%S'),
            "host": meta.get("host", "unknown"),
            "anomaly_score": float(scores[i]),
            "adaptive_threshold": float(threshold),
            "base_threshold": float(threshold),
            "is_anomaly": int(preds[i]),
            "top_attribution": top_view_name,
            "process_score": float(per_view_scores[i][0]),
            "file_score": float(per_view_scores[i][1]),
            "network_score": float(per_view_scores[i][2]),
            "global_score": float(global_scores[i]),
        })
    return pd.DataFrame(results)

def train_deep_detector(config_path):
    config, optc_cfg, train_cfg, model_cfg, device, output_dir = setup_environment(config_path)
    
    # Load Data using main.py logic
    split_names = ["train", "val", "test"]
    datasets, vocab_schema_loaded, _ = load_datasets(optc_cfg, optc_cfg["cache_dir"], splits=split_names)
    detector_prefix = optc_cfg.get("detector_prefix", "detector")
    detector_ds = load_cached_split(optc_cfg["cache_dir"], detector_prefix)
    if detector_ds is None:
        detector_ds = OpTCEcarDataset(optc_cfg["cache_dir"], split=detector_prefix, preload=False)
    if len(detector_ds) > 0:
        train_ds = detector_ds
        source = "optimized tensor cache" if isinstance(train_ds, TensorDataset) else "raw cache split"
        print(f"[DeepDetector] Using dedicated detector cache split '{detector_prefix}' from {source}: {len(train_ds)} samples")
    else:
        _, student_subset = split_train_data(datasets["train"], train_cfg)
        train_ds = student_subset
    val_ds = datasets.get("val")
    test_ds = datasets["test"]
    
    # Init Models
    vocab_path = os.path.join(output_dir, "vocab_schema.pt")
    per_view_schema = torch.load(vocab_path, map_location="cpu")
    
    s1_cfg = build_step1_config(optc_cfg, model_cfg)
    
    step1 = Step1Model(s1_cfg, per_view_schema).to(device)
    step1.fit_vocabs_and_init_projectors([], max_types=model_cfg.get("max_vocab_types", 500))
    replace_relu(step1.router); replace_relu(step1.fusion)
    apply_lora(step1.router, rank=model_cfg.get("lora_rank", 4), alpha=model_cfg.get("lora_alpha", 8))
    apply_lora(step1.fusion, rank=model_cfg.get("lora_rank", 4), alpha=model_cfg.get("lora_alpha", 8))
    step1.to(device)
    
    student = StudentHeads(in_dim=s1_cfg.target_dim, num_subspaces=model_cfg.get("num_subspaces", 4), subspace_dim=model_cfg.get("subspace_dim", 16), hidden_dim=64).to(device)
    replace_relu(student)
    student.to(device)
    
    # Load Weights
    s1_ckpt = os.path.join(output_dir, "step1_checkpoint_no_dp.pt")
    stu_ckpt = os.path.join(output_dir, "student_checkpoint_no_dp.pt")
    step1.load_state_dict(torch.load(s1_ckpt, map_location=device), strict=False)
    student.load_state_dict(torch.load(stu_ckpt, map_location=device), strict=False)

    # [FIX] Ensure quality stats are fitted if buffers are still zero
    if step1.quality._mu.abs().sum() < 1e-6:
        print("[DeepDetector] Quality buffers are zero, fitting from training data...")
        if isinstance(train_ds, Subset) and isinstance(train_ds.dataset, TensorDataset):
            q_tensor = train_ds.dataset.quality[train_ds.indices]
            q_flat = q_tensor.reshape(-1, 4)
            metrics = ["validity", "completeness", "entropy", "intensity"]
            qs = {k: q_flat[:, i].tolist() for i, k in enumerate(metrics)}
            step1.quality.fit_standardize_stats(qs)
        elif isinstance(train_ds, TensorDataset):
            q_tensor = train_ds.quality
            q_flat = q_tensor.reshape(-1, 4)
            metrics = ["validity", "completeness", "entropy", "intensity"]
            qs = {k: q_flat[:, i].tolist() for i, k in enumerate(metrics)}
            step1.quality.fit_standardize_stats(qs)
        else:
            sample_cap = min(len(train_ds), 500)
            sample_indices = torch.randperm(len(train_ds))[:sample_cap].tolist()
            quality_samples = [train_ds[i] for i in sample_indices]
            step1.fit_quality_stats(quality_samples)
    
    # Extract Features for Training (Benign only)
    train_collate = tensor_collate if isinstance(train_ds, TensorDataset) or (isinstance(train_ds, Subset) and isinstance(train_ds.dataset, TensorDataset)) else (lambda batch: batch)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, collate_fn=train_collate)
    train_view_vecs, train_global_vecs, train_labels, _ = get_token_features(train_loader, step1, student, device)
    
    # Filter benign for training
    benign_train_view_vecs = train_view_vecs[train_labels == 0]
    benign_train_global_vecs = train_global_vecs[train_labels == 0]
    print(f"Training on {len(benign_train_view_vecs)} benign samples.")
    if len(benign_train_view_vecs) == 0:
        raise RuntimeError("No benign training samples found for deep detector training.")
    
    # Init Deep Detector
    deep_detector = DeepReconstructionDetector(
        view_dim=model_cfg["target_dim"],
        num_views=len(s1_cfg.views),
        global_dim=model_cfg.get("num_subspaces", 4) * model_cfg.get("subspace_dim", 16),
        nhead=4,
        num_layers=2,
        mask_ratio=0.35,
    ).to(device)
    
    optimizer = torch.optim.Adam(deep_detector.parameters(), lr=1e-4)
    
    # Training Loop
    epochs = 50
    batch_size = 64
    deep_detector.train()
    
    for epoch in range(epochs):
        perm = torch.randperm(len(benign_train_view_vecs))
        epoch_loss = 0
        for i in range(0, len(benign_train_view_vecs), batch_size):
            indices = perm[i:i+batch_size]
            batch_view = benign_train_view_vecs[indices].to(device)
            batch_global = benign_train_global_vecs[indices].to(device)
            loss_dict = deep_detector.reconstruction_loss(batch_view, batch_global)
            loss = loss_dict["loss"]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(indices)
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(benign_train_view_vecs):.6f}")
            
    # Fit Stats
    deep_detector.eval()
    with torch.no_grad():
        train_stats = deep_detector.compute_score(benign_train_view_vecs.to(device), benign_train_global_vecs.to(device))
        threshold_q = float(train_cfg.get("threshold_quantile", 0.99))
        deep_detector.fit_stats(train_stats, quantile=threshold_q)
        calibrated_train_scores = deep_detector.compute_score(
            benign_train_view_vecs.to(device),
            benign_train_global_vecs.to(device),
        )["score"]
        margin = float(train_cfg.get("threshold_margin", 1.0))
        train_thresholds = {
            "0.95": float(torch.quantile(calibrated_train_scores, 0.95).item()),
            "0.97": float(torch.quantile(calibrated_train_scores, 0.97).item()),
            "0.98": float(torch.quantile(calibrated_train_scores, 0.98).item()),
            "0.99": float(torch.quantile(calibrated_train_scores, 0.99).item()),
        }
        if val_ds is not None:
            val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=tensor_collate)
            val_view_vecs, val_global_vecs, _, _ = get_token_features(val_loader, step1, student, device)
            val_scores = deep_detector.compute_score(val_view_vecs.to(device), val_global_vecs.to(device))["score"]
            for key in list(train_thresholds.keys()):
                q = float(key)
                val_threshold = float(torch.quantile(val_scores, q).item())
                train_thresholds[key] = max(train_thresholds[key], val_threshold)
        for key in list(train_thresholds.keys()):
            train_thresholds[key] = float(train_thresholds[key] * margin)
        preferred_quantile = "0.97" if "0.97" in train_thresholds else f"{threshold_q:.2f}"
        deep_detector.threshold.fill_(train_thresholds[preferred_quantile])
        print(f"[DeepDetector] Threshold sweep: {train_thresholds}")
        print(f"[DeepDetector] Default threshold ({preferred_quantile}, margin={margin}): {deep_detector.threshold.item():.4f}")
        
    # Save
    save_dir = os.path.join("experiments", "deep_ad_v1")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(deep_detector.state_dict(), os.path.join(save_dir, "deep_detector.pt"))
    print(f"Deep Detector saved to {save_dir}")
    
    # Evaluate on Test Set
    print("\n--- Evaluating on Test Set ---")
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=tensor_collate)
    test_view_vecs, test_global_vecs, test_labels, test_meta = get_token_features(test_loader, step1, student, device)
    
    deep_detector.eval()
    with torch.no_grad():
        res = deep_detector.compute_score(test_view_vecs.to(device), test_global_vecs.to(device))
        scores = res["score"].cpu().numpy()
        per_view_scores = res["per_view_score"].cpu().numpy()
        global_scores = res["global_score"].cpu().numpy()
        threshold = deep_detector.threshold.item()
        
    # Save Results in format compatible with evaluate.py
    df = build_results_dataframe(test_meta, scores, threshold, per_view_scores, global_scores)
    out_path = os.path.join(save_dir, "deep_detection_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Deep Detection Results saved to {out_path}")
    try:
        import evaluate
        default_metrics = evaluate.run_evaluation(out_path, csv_in_edt=False)
        sweep_metrics = {"default_quantile": preferred_quantile}
        for q_key, q_threshold in train_thresholds.items():
            sweep_path = os.path.join(save_dir, f"deep_detection_results_q{q_key.replace('.', '')}.csv")
            sweep_df = build_results_dataframe(test_meta, scores, q_threshold, per_view_scores, global_scores)
            sweep_df.to_csv(sweep_path, index=False)
            sweep_metrics[q_key] = evaluate.run_evaluation(sweep_path, csv_in_edt=False)
        with open(os.path.join(save_dir, "threshold_sweep_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(sweep_metrics, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[DeepDetector] evaluate.py auto-run skipped: {exc}")
    
    # Final Metrics
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
    preds = (scores > threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(test_labels, preds, average='binary', zero_division=0)
    auc = roc_auc_score(test_labels, scores) if len(set(test_labels)) > 1 else 0.5
    ap = average_precision_score(test_labels, scores) if len(set(test_labels)) > 1 else 0
    
    print(f"\n=== Deep Detector Metrics ===")
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"F1: {f:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"AUPRC: {ap:.4f}")
    
    # View Contribution Analysis
    per_view_scores = res["per_view_score"].cpu().numpy()
    avg_contrib = per_view_scores.mean(axis=0)
    avg_global = float(res["global_score"].mean().cpu().item())
    print("\nAvg Token Contribution (MSE):")
    for i, v in enumerate(s1_cfg.views):
        print(f"  - {v}: {avg_contrib[i]:.6f}")
    print(f"  - global: {avg_global:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_production.yaml")
    args = parser.parse_args()
    train_deep_detector(args.config)
