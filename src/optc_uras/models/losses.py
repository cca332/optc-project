from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


def entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def asd_loss(sub_s: torch.Tensor, sub_t: torch.Tensor, route_p: torch.Tensor, lambda_stats: float = 0.1) -> torch.Tensor:
    """ASD：样本级对齐（路由加权） + batch 均值对齐（统计对齐）。
    
    Args:
        sub_s: Student subspaces [B, M, d]
        sub_t: Teacher subspaces [B, M, d]
        route_p: Routing weights [B, M]
        lambda_stats: Weight for statistical alignment
    """
    # 1. Instance-level alignment (weighted by routing probability)
    # L_ins = sum_r pi_r * ||r_s - r_t||^2
    diff_sq = (sub_s - sub_t).pow(2).sum(dim=-1)  # [B, M]
    l_ins_per_sample = (diff_sq * route_p).sum(dim=1)  # [B]
    l_ins = l_ins_per_sample.mean()

    # 2. Batch mean & variance alignment (Statistical Alignment)
    # mu_r = mean(r_r) over batch
    mu_s = sub_s.mean(dim=0)  # [M, d]
    mu_t = sub_t.mean(dim=0)  # [M, d]
    loss_mu = (mu_s - mu_t).pow(2).sum()  # sum over subspaces and dimensions
    
    # std_r = std(r_r) over batch
    # Add epsilon to prevent NaN gradient at 0
    std_s = torch.sqrt(sub_s.var(dim=0, unbiased=False) + 1e-6)
    std_t = torch.sqrt(sub_t.var(dim=0, unbiased=False) + 1e-6)
    
    # [Anti-Collapse] Alignment with Teacher's Std
    # We rely on the high lambda_stats (e.g., 50.0) to force the student 
    # to match the teacher's healthy variance (approx 0.045), preventing collapse naturally.
    loss_std_align = (std_s - std_t).pow(2).sum()
    
    # l_stats includes mean alignment and std alignment
    l_stats = loss_mu + loss_std_align
    
    return l_ins + float(lambda_stats) * l_stats


def at_info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """Step 2 B4: AT-InfoNCE (Adaptive Temperature InfoNCE).
    
    Formula:
      L_AT = - sum_i log ( exp(sim(i,i)/tau_i) / sum_j exp(sim(i,j)/tau_i) )
      
    Args:
        z1, z2: Normalized features [B, d]
        temperature: Dynamic temperature per sample [B]
    """
    # z1,z2: [B,d], assume normalized
    B = z1.shape[0]
    sim = (z1 @ z2.t())  # [B,B]
    
    # per-sample temperature: tau_i applies to the i-th row (anchor i)
    # temperature is [B], expand to [B, 1] for broadcasting across columns
    tau = temperature.view(B, 1).clamp_min(1e-6)
    
    logits = sim / tau
    labels = torch.arange(B, device=z1.device)
    
    return F.cross_entropy(logits, labels)


def dynamic_temperature(confidence: torch.Tensor, route_uncertainty: torch.Tensor, distill_error: torch.Tensor, 
                        privacy_risk: torch.Tensor, sample_utility: torch.Tensor,
                        current_epoch: int, total_epochs: int,
                        a: float, b: float, c: float, min_tau: float = 0.05, max_tau: float = 0.5) -> torch.Tensor:
    """Step 2 B3: Dynamic Temperature tau(x, t_e)
    
    Formula: tau = tau_min + (tau_max - tau_min) * sigmoid(a*R(x) - b*U(x)) * s(t_e)
    
    Interpretation:
    - High Risk (R) or Low Utility (U) -> Higher Tau (Weaker Distillation)
    - Low Risk and High Utility -> Lower Tau (Stronger Distillation)
    - s(t_e): Scheduler (e.g., linear decay) to tighten distillation over time.
    """
    # Normalize inputs (assumed to be roughly [0,1] or positive)
    # Risk and Utility are already calculated in client.py
    
    # 1. Base control signal: a*R(x) - b*U(x)
    # Using 'c' as the 'b' parameter from doc for Utility to avoid signature conflict, 
    # or just use a/b/c as weights. Let's map to doc:
    # Doc: a*R(x) - b*U(x)
    # Code signature has a, b, c. Let's use:
    # alpha * Risk - beta * Utility. 
    # We'll use 'a' for Risk weight, 'b' for Utility weight.
    
    x = float(a) * privacy_risk - float(b) * sample_utility
    
    # 2. Sigmoid modulation
    sigma = torch.sigmoid(x)
    
    # 3. Epoch scheduler s(t_e)
    # Linear decay from 1.0 to 0.5 over epochs (example)
    progress = float(current_epoch) / max(total_epochs, 1)
    s_te = max(1.0 - 0.5 * progress, 0.5)
    
    # 4. Final calculation
    # tau = min + (max - min) * sigma * s
    delta = max_tau - min_tau
    tau = min_tau + delta * sigma * s_te
    
    return tau


def scd_loss(
    s: torch.Tensor, 
    c: torch.Tensor, 
    u: torch.Tensor, 
    u_hat: torch.Tensor,
    lambda_d: float = 1.0, 
    lambda_r: float = 1.0, 
    lambda_v: float = 1.0,
    gamma: float = 1.0
) -> torch.Tensor:
    """Step 3 A4: SCD 总损失 (benign-only).
    
    Includes:
    - A2: L_decorr (Decorrelation)
    - A3(A): L_rec (Reconstruction)
    - A3(B): L_var (Variance Hinge)
    
    Formula:
        L_SCD = lambda_d * L_decorr + lambda_r * L_rec + lambda_v * (L_var(s) + L_var(c))
    
    Args:
        s: Style vectors [B, d_s] (centered)
        c: Content vectors [B, d_c] (centered)
        u: Original URAS [B, d_u]
        u_hat: Reconstructed URAS [B, d_u]
        lambda_d: Weight for decorrelation
        lambda_r: Weight for reconstruction
        lambda_v: Weight for variance hinge
        gamma: Target variance threshold (VICReg style)
        
    Returns:
        total_loss: scalar
    """
    B = s.shape[0]
    
    # --- A2: Decorrelation (L_decorr) ---
    # Sigma_sc = 1/|B| * Sum(s * c^T)
    sigma_sc = torch.matmul(s.t(), c) / float(B)
    l_decorr = (sigma_sc ** 2).sum()
    
    # --- A3 (A): Reconstruction (L_rec) ---
    # L_rec = || u - u_hat ||_2^2
    # Mean over batch for stability
    l_rec = (u - u_hat).pow(2).sum(dim=-1).mean()
    
    # --- A3 (B): Variance Hinge (L_var) ---
    # L_var(z) = Sum_j max(0, gamma - Std(z_j))
    # std over batch dimension
    # Use unbiased=False to avoid NaN when batch size is 1 or small
    std_s = torch.sqrt(s.var(dim=0, unbiased=False) + 1e-6)
    std_c = torch.sqrt(c.var(dim=0, unbiased=False) + 1e-6)
    
    l_var_s = F.relu(gamma - std_s).sum()
    l_var_c = F.relu(gamma - std_c).sum()
    l_var = l_var_s + l_var_c
    
    # --- A4: Total Loss ---
    loss = (
        float(lambda_d) * l_decorr + 
        float(lambda_r) * l_rec + 
        float(lambda_v) * l_var
    )
    
    return loss
