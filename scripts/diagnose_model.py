
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optc_uras.data.dataset import TensorDataset, tensor_collate
from optc_uras.models.step1 import Step1Config, Step1Model
from optc_uras.models.teacher import TeacherModel
from optc_uras.models.student import StudentHeads, uras_from_subspaces
from optc_uras.models.detector import AnomalyDetector
from optc_uras.features.quality import QualityWeightsConfig
from optc_uras.utils.misc import replace_relu, apply_lora
from evaluate import GROUND_TRUTH_UTC, normalize_host


def infer_labels_from_metadata(metadata):
    labels = []
    for m in metadata:
        raw_label = m.get("label", None)
        if raw_label is not None and int(raw_label) > 0:
            labels.append(int(raw_label))
            continue

        host_id = normalize_host(m.get("host"))
        if host_id is None:
            labels.append(0)
            continue

        try:
            dt_utc = pd.to_datetime(int(m.get("t0", 0)), unit="ms")
            dt_utc = dt_utc.floor("5min")
            key = (
                dt_utc.strftime("%Y-%m-%d"),
                dt_utc.strftime("%H:%M"),
                host_id,
            )
            labels.append(1 if key in GROUND_TRUTH_UTC else 0)
        except Exception:
            labels.append(0)
    return np.array(labels, dtype=np.int64)

def setup_diagnosis(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    optc_cfg = config["data"]["optc"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    output_dir = train_cfg["output_dir"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    cache_path = os.path.join(optc_cfg["cache_dir"], "optimized_data.pt")
    print(f"Loading cache from {cache_path}...")
    data = torch.load(cache_path, map_location="cpu")
    
    # 2. Load Models
    # Teacher
    teacher = TeacherModel(
        behavior_dim=model_cfg["behavior_dim"],
        num_subspaces=model_cfg["num_subspaces"],
        subspace_dim=model_cfg["subspace_dim"],
        hidden_dim=64
    ).to(device)
    teacher_ckpt = os.path.join(output_dir, "teacher_checkpoint.pt")
    if os.path.exists(teacher_ckpt):
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        print("Teacher model loaded.")
    
    # Step1
    vocab_path = os.path.join(output_dir, "vocab_schema.pt")
    per_view_schema = torch.load(vocab_path, map_location="cpu")
    
    quality_cfg = QualityWeightsConfig(
        reliability_weights={"validity": 1.0, "completeness": 1.0},
        standardize="minmax",
        softmax_temperature=model_cfg.get("quality_softmax_temp", 0.5)
    )
    s1_cfg = Step1Config(
        views=["process", "file", "network"],
        window_seconds=optc_cfg.get("window_minutes", 5) * 60,
        slot_seconds=model_cfg.get("slot_seconds", 30),
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
        interaction_enabled=True,
        quality_injection_lambda=model_cfg.get("quality_injection_lambda", 0.5),
        semantic_dim=model_cfg.get("semantic_dim", 128),
        stat_proj_dim=model_cfg.get("stat_proj_dim", 32),
        vocab_sizes=model_cfg.get("vocab_sizes")
    )
    step1 = Step1Model(s1_cfg, per_view_schema).to(device)
    
    # Apply LoRA to match checkpoint
    apply_lora(step1.router, rank=model_cfg.get("lora_rank", 4), alpha=model_cfg.get("lora_alpha", 8))
    apply_lora(step1.fusion, rank=model_cfg.get("lora_rank", 4), alpha=model_cfg.get("lora_alpha", 8))
    
    # Ensure LoRA parameters are on the correct device
    step1.to(device)
    
    s1_ckpt = os.path.join(output_dir, "step1_checkpoint_no_dp.pt")
    if os.path.exists(s1_ckpt):
        step1.load_state_dict(torch.load(s1_ckpt, map_location=device))
        print("Step1 model loaded.")
        
    # Student
    student = StudentHeads(
        in_dim=model_cfg["target_dim"],
        num_subspaces=model_cfg["num_subspaces"],
        subspace_dim=model_cfg["subspace_dim"],
        hidden_dim=64
    ).to(device)
    stu_ckpt = os.path.join(output_dir, "student_checkpoint_no_dp.pt")
    if os.path.exists(stu_ckpt):
        student.load_state_dict(torch.load(stu_ckpt, map_location=device))
        print("Student heads loaded.")
        
    # Detector
    uras_dim = model_cfg["num_subspaces"] * model_cfg["subspace_dim"]
    detector = AnomalyDetector(
        feature_dim=uras_dim, 
        style_dim=32, 
        content_dim=32,
        view_names=["process", "file", "network"],
        view_dim=model_cfg["target_dim"]
    ).to(device)
    det_ckpt = os.path.join(output_dir, "detector_checkpoint_retrain.pt")
    if os.path.exists(det_ckpt):
        detector.load_state_dict(torch.load(det_ckpt, map_location=device))
        print("Detector loaded.")
        
    return data, teacher, step1, student, detector, device, output_dir

def run_diagnosis(config_path):
    data, teacher, step1, student, detector, device, output_dir = setup_diagnosis(config_path)
    
    diag_dir = os.path.join("experiments", "diagnosis_results")
    os.makedirs(diag_dir, exist_ok=True)
    
    teacher.eval(); step1.eval(); student.eval(); detector.eval()
    
    # Sample a mix of data (Normal and Attack if possible)
    metadata = data["metadata"]
    labels = np.array([m.get("label", 0) for m in metadata])
    if np.sum(labels > 0) == 0:
        labels = infer_labels_from_metadata(metadata)
        print(f"[Diagnosis] No positive labels found in cache metadata, inferred {int(labels.sum())} attacks from evaluate.py ground truth.")
    hosts = np.array([m.get("host", "unknown") for m in metadata])
    
    # Take a subset for visualization (e.g., 1000 samples)
    indices = np.arange(len(metadata))
    if len(indices) > 1000:
        # Try to keep all attacks
        attack_idx = np.where(labels > 0)[0]
        normal_idx = np.where(labels == 0)[0]
        np.random.shuffle(normal_idx)
        keep_normal = 1000 - len(attack_idx)
        indices = np.concatenate([attack_idx, normal_idx[:max(0, keep_normal)]])
    
    ds = TensorDataset(data, indices.tolist())
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=tensor_collate)
    
    all_ut = []
    all_us = []
    all_style = []
    all_content = []
    all_scores = []
    all_labels = labels[indices]
    all_hosts = hosts[indices]
    
    print("Extracting features...")
    with torch.no_grad():
        for batch in tqdm(loader):
            # Unpack 11 fields
            _, b_features, *sem_batch, st_tensor, q_tensor = batch
            b_features = b_features.to(device).float()
            
            # 1. Teacher URAS
            sub_t = teacher(b_features, normalize=True)
            
            # 2. Student Step1
            s1_out = step1.forward_from_cache(*[t.to(device) for t in sem_batch], st_tensor.to(device), q_tensor.to(device))
            zs = torch.stack([o.z for o in s1_out])
            rp = torch.stack([o.route_p for o in s1_out])
            
            # 3. Student URAS
            sub_s = student(zs, normalize=True)
            u_s = uras_from_subspaces(sub_s, rp)
            u_s = torch.nn.functional.normalize(u_s, dim=-1)
            
            # Teacher URAS needs the student's route_p for alignment
            u_t = uras_from_subspaces(sub_t, rp)
            u_t = torch.nn.functional.normalize(u_t, dim=-1)
            
            # 4. SCD
            s, c = detector.scd(u_s)
            
            # 5. Score
            scores = detector._compute_raw_score(s)
            
            all_ut.append(u_t.cpu().numpy())
            all_us.append(u_s.cpu().numpy())
            all_style.append(s.cpu().numpy())
            all_content.append(c.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
            
    all_ut = np.concatenate(all_ut, axis=0)
    all_us = np.concatenate(all_us, axis=0)
    all_style = np.concatenate(all_style, axis=0)
    all_content = np.concatenate(all_content, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    
    # --- 1. Distillation Alignment Analysis ---
    mse = np.mean((all_ut - all_us)**2)
    cos_sim = np.mean([np.dot(all_ut[i], all_us[i])/(np.linalg.norm(all_ut[i])*np.linalg.norm(all_us[i])+1e-8) for i in range(len(all_ut))])
    
    print(f"\n[Distillation Metrics]")
    print(f"MSE (Teacher vs Student URAS): {mse:.6f}")
    print(f"Cosine Similarity: {cos_sim:.6f}")
    
    # --- 2. Feature Variance (Collapse Check) ---
    us_std = np.std(all_us, axis=0).mean()
    style_std = np.std(all_style, axis=0).mean()
    
    print(f"\n[Variance Metrics]")
    print(f"Student URAS Avg Std: {us_std:.6f}")
    print(f"Style Avg Std: {style_std:.6f}")
    
    if us_std < 1e-4:
        print("WARNING: Feature collapse detected in Student Model!")
    
    # --- 3. Anomaly Score Distribution ---
    plt.figure(figsize=(10, 6))
    sns.histplot(x=all_scores, hue=all_labels, bins=50, kde=True, palette="viridis")
    plt.title("Anomaly Score Distribution (Normal vs Attack)")
    plt.xlabel("Score (Mahalanobis Distance)")
    plt.savefig(os.path.join(diag_dir, "score_distribution.png"))
    print(f"Saved score distribution to {diag_dir}")
    
    # --- 4. T-SNE Visualization ---
    def plot_tsne(features, labels, title, filename):
        print(f"Running T-SNE for {title}...")
        tsne = TSNE(n_components=2, random_state=42)
        emb = tsne.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=labels, style=labels, palette="Set1", alpha=0.7)
        plt.title(title)
        plt.savefig(os.path.join(diag_dir, filename))
        plt.close()

    plot_tsne(all_ut, all_labels, "Teacher URAS Features (U^T)", "tsne_teacher.png")
    plot_tsne(all_us, all_labels, "Student URAS Features (U^S)", "tsne_student.png")
    plot_tsne(all_style, all_labels, "SCD Style Features (s)", "tsne_style.png")
    
    # --- 5. Host Analysis ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=all_hosts, y=all_scores)
    plt.xticks(rotation=90)
    plt.title("Anomaly Scores per Host")
    plt.savefig(os.path.join(diag_dir, "scores_per_host.png"))
    
    # --- 5. Host Analysis ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=all_hosts, y=all_scores)
    plt.xticks(rotation=90)
    plt.title("Anomaly Scores per Host")
    plt.savefig(os.path.join(diag_dir, "scores_per_host.png"))
    plt.close()
    
    # Generate Report
    avg_normal = np.mean(all_scores[all_labels==0])
    avg_attack = np.mean(all_scores[all_labels>0]) if any(all_labels>0) else "N/A"
    pred_at_threshold = (all_scores > float(detector.threshold.item())).astype(int)

    auc = "N/A"
    auprc = "N/A"
    precision = "N/A"
    recall = "N/A"
    f1 = "N/A"
    if len(np.unique(all_labels)) > 1:
        auc = f"{roc_auc_score(all_labels, all_scores):.4f}"
        auprc = f"{average_precision_score(all_labels, all_scores):.4f}"
        p, r, f, _ = precision_recall_fscore_support(all_labels, pred_at_threshold, average="binary", zero_division=0)
        precision, recall, f1 = f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"
    
    report = f"""
Anomaly Detection Diagnosis Report
==================================
MSE (Teacher vs Student URAS): {mse:.6f}
Cosine Similarity: {cos_sim:.6f}

Average Standard Deviation (Check for Collapse):
- Student URAS: {us_std:.6f}
- Style: {style_std:.6f}

Score Statistics:
- Normal Mean: {avg_normal:.4f}
- Attack Mean: {avg_attack if isinstance(avg_attack, str) else f"{avg_attack:.4f}"}
- Max Score: {np.max(all_scores):.4f}
- Threshold (from model): {detector.threshold.item():.4f}

Detection Metrics:
- Positive Samples: {int((all_labels > 0).sum())}
- Precision@Threshold: {precision}
- Recall@Threshold: {recall}
- F1@Threshold: {f1}
- AUC: {auc}
- AUPRC: {auprc}
    """
    with open(os.path.join(diag_dir, "diagnosis_report.txt"), "w") as f:
        f.write(report)
    print(f"Diagnosis report saved to {os.path.join(diag_dir, 'diagnosis_report.txt')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_production.yaml")
    args = parser.parse_args()
    run_diagnosis(args.config)
