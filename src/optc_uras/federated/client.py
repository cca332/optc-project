from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.dataset import ProcessedDataset, collate_samples
from ..features.behavior_features import behavior_features_from_sample
from ..models.losses import asd_loss, dynamic_temperature, entropy, at_info_nce
from ..models.student import StudentHeads, uras_from_subspaces
from ..models.teacher import TeacherModel
from ..models.step1 import Step1Model
from .dp import FeatureDPConfig, GradDPConfig, dp_features, dp_gradients, clip_by_l2_norm_, add_gaussian_noise_


@dataclass
class ClientTrainConfig:
    local_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    lambda_stats: float
    lambda_infonce: float
    temp_params: Dict[str, float]
    feature_dp: FeatureDPConfig
    grad_dp: GradDPConfig
    views: List[str]
    behavior_feature_dim: int


class FederatedClient:
    def __init__(self, client_id: str, samples: List[Dict[str, Any]]):
        self.client_id = client_id
        self.samples = samples

    def benign_samples(self) -> List[Dict[str, Any]]:
        return [s for s in self.samples if int(s.get("label", 0)) == 0]

    def local_train(
        self,
        step1: Step1Model,
        student_heads: StudentHeads,
        teacher: TeacherModel,
        cfg: ClientTrainConfig,
        device: str,
    ) -> Tuple[torch.Tensor, Dict[str, float], int]:
        """本地 benign-only 训练，返回 flattened update 向量和一些日志。"""
        step1.train()
        student_heads.train()
        teacher.eval()

        dataset = ProcessedDataset(self.benign_samples())
        # [FIX] drop_last=True to prevent BatchNorm crash on last batch of size 1
        # [USER REQ] Batch Size 16, typically leaves 1-2 samples remainder
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_samples, drop_last=True)

        # 只优化可学习部分（骨架：step1 + student_heads）
        params = list(step1.parameters()) + list(student_heads.parameters())
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        # snapshot initial
        init_vec = torch.nn.utils.parameters_to_vector(params).detach().clone()

        # C2 Pre-computation: View Importance (Omega)
        # We need this for Adaptive Clipping inside the loop.
        # Estimate omega using initial model state.
        step1.eval()
        acc_w_init = torch.zeros(len(cfg.views), device=device)
        count_w_init = 0
        with torch.no_grad():
            for batch in loader:
                outs = step1(batch)
                w_batch = torch.stack([o.reliability_w for o in outs], dim=0).to(device)
                acc_w_init += w_batch.sum(dim=0)
                count_w_init += w_batch.shape[0]
        
        beta_bar_init = acc_w_init / max(count_w_init, 1)
        omega_init = beta_bar_init / (beta_bar_init.mean() + 1e-12) # [V]

        # C3 State: Gradient Momentum Buffers for Inter-view Projection
        # Map view_index -> flattened momentum vector
        view_momentums: Dict[int, torch.Tensor] = {} 
        eta_m = 0.9 # Momentum factor (default)
        tau_p = 0.5 # Softmax temperature for redundancy weights

        total_loss = 0.0
        total_batches = 0

        for epoch in range(int(cfg.local_epochs)):
            for batch in loader:
                # Step1
                outs = step1(batch)  # list[Step1Outputs]
                z = torch.stack([o.z for o in outs], dim=0).to(device)                 # [B,d]
                route_p = torch.stack([o.route_p for o in outs], dim=0).to(device)     # [B,M]
                w = torch.stack([o.reliability_w for o in outs], dim=0).to(device)     # [B,V]

                # student subspaces -> URAS
                sub_s = student_heads(z, normalize=True)                               # [B,M,d_s]
                uras_s = uras_from_subspaces(sub_s, route_p)                           # [B,M*d_s]
                uras_s = torch.nn.functional.normalize(uras_s, dim=-1)

                # teacher URAS from behavior features (client-side)
                feats = []
                for s in batch:
                    f = behavior_features_from_sample(s, cfg.views, out_dim=int(cfg.behavior_feature_dim))
                    feats.append(torch.from_numpy(f))
                x = torch.stack(feats, dim=0).to(device).float()                       # [B,d_b]
                x_dp = dp_features(x, cfg.feature_dp)

                sub_t = teacher(x_dp, normalize=True)                                  # [B,M,d_s]
                uras_t = uras_from_subspaces(sub_t, route_p)                           # [B,M*d_s]
                uras_t = torch.nn.functional.normalize(uras_t, dim=-1)

                # signals (Step 2 B2: Quality/Utility/Risk Signals)
                # (1) Confidence (View Weight Concentration)
                V_dim = w.shape[1]
                H_w = entropy(w) / np.log(max(V_dim, 2))
                confidence = (1.0 - H_w).clamp(0.0, 1.0)

                # (2) Route Uncertainty
                M_dim = route_p.shape[1]
                route_unc = entropy(route_p) / np.log(max(M_dim, 2))

                # (3) Distillation Error (Utility Proxy)
                # Doc: Err(x) = || u^S - u^T ||_2
                distill_err = (uras_s - uras_t).norm(p=2, dim=-1).detach()

                # (4) Sample Utility Signal (Extensible)
                # Doc: U(x) = Conf(x) * exp(-Err(x))
                # Meaning: High confidence & consistent samples are more "useful"
                sample_utility = confidence * torch.exp(-distill_err)

                # (5) Privacy Risk Signal (Sensitivity-weighted View Exposure)
                # Risk = (sum_v sensitivity_v * beta_v) * (1 + MIA_score)
                # Assuming MIA_score=0 for now (placeholder), focusing on view sensitivity exposure.
                # Default sensitivity = 1.0 for all views if not configured.
                sensitivity_map = cfg.temp_params.get("view_sensitivities", {})
                sens_vec = torch.tensor([sensitivity_map.get(v, 1.0) for v in cfg.views], device=device)
                
                # w is [B, V], sens_vec is [V] -> risk is [B]
                # High weight on sensitive views -> High Privacy Risk
                privacy_risk = (w * sens_vec.unsqueeze(0)).sum(dim=1)

                tau = dynamic_temperature(
                    confidence=confidence,
                    route_uncertainty=route_unc,
                    distill_error=distill_err,
                    privacy_risk=privacy_risk,
                    sample_utility=sample_utility,
                    current_epoch=epoch,
                    total_epochs=cfg.local_epochs,
                    a=float(cfg.temp_params.get("a", 2.0)), # Risk weight
                    b=float(cfg.temp_params.get("b", 1.0)), # Utility weight
                    c=0.0, # Unused
                    min_tau=float(cfg.temp_params.get("min_tau", 0.05)),
                    max_tau=float(cfg.temp_params.get("max_tau", 0.5)),
                ).detach()

                # losses
                # Step2 B1: Strict ASD implementation
                loss_asd = asd_loss(sub_s, sub_t, route_p, lambda_stats=cfg.lambda_stats)
                
                # Step2 B4: AT-InfoNCE (Adaptive Temperature)
                loss_infonce = at_info_nce(uras_s, uras_t, temperature=tau)
                
                # Step2 B5: Local Training Objective (No Task Loss)
                # L_local = lambda_asd * L_asd + lambda_at * L_at
                # L_task = 0 explicitly
                loss = loss_asd + float(cfg.lambda_infonce) * loss_infonce

                opt.zero_grad(set_to_none=True)
                loss.backward()

                # C3: Inter-view Gradient Projection (Redundancy Reduction)
                # Only applicable if we have split view layers in RouterMLP
                if hasattr(step1.router, "view_layers"):
                    with torch.no_grad():
                        # 1. Collect current gradients G^(c,v)
                        curr_grads = {} # idx -> tensor
                        for v_idx, layer in enumerate(step1.router.view_layers):
                            # Flatten all params for this view layer (weight + bias)
                            grads = [p.grad for p in layer.parameters() if p.grad is not None]
                            if grads:
                                curr_grads[v_idx] = torch.cat([g.flatten() for g in grads])
                        
                        # Only proceed if we have gradients
                        if curr_grads:
                            # 2. Update Momentum m^(v)
                            for v_idx, g in curr_grads.items():
                                if v_idx not in view_momentums:
                                    view_momentums[v_idx] = g
                                else:
                                    view_momentums[v_idx] = eta_m * view_momentums[v_idx] + (1 - eta_m) * g
                            
                            # 3. Compute View Correlations s_uv
                            # Use batch-averaged aligned view vectors
                            # view_vecs: [B, V, d]
                            # outs is a list of Step1Outputs, we need to stack them
                            batch_view_vecs = torch.stack([o.view_vecs for o in outs], dim=0) # [B, V, d]
                            s_v = batch_view_vecs.mean(dim=0) # [V, d]
                            # Normalize
                            s_v_norm = torch.nn.functional.normalize(s_v, dim=1)
                            # Correlation matrix S: [V, V]
                            sim_matrix = torch.mm(s_v_norm, s_v_norm.t())

                            # 4. Project and Deduct
                            # For each view u, subtract projection onto other views v
                            for u_idx, layer in enumerate(step1.router.view_layers):
                                if u_idx not in curr_grads: continue
                                
                                g_u = curr_grads[u_idx].clone()
                                adjustment = torch.zeros_like(g_u)
                                
                                # Compute alpha weights for u
                                # s_uv for v != u
                                sims = sim_matrix[u_idx].clone()
                                sims[u_idx] = -float('inf') # mask self
                                alphas = torch.nn.functional.softmax(sims / tau_p, dim=0)
                                
                                for v_idx in curr_grads:
                                    if u_idx == v_idx: continue
                                    
                                    m_v = view_momentums[v_idx]
                                    alpha = alphas[v_idx]
                                    
                                    if alpha > 1e-4: # optimization threshold
                                        # Proj_mv(gu) = (mv.T gu / mv.T mv) * mv
                                        dot_mg = torch.dot(m_v, g_u)
                                        dot_mm = torch.dot(m_v, m_v) + 1e-8
                                        proj = (dot_mg / dot_mm) * m_v
                                        adjustment += alpha * proj
                                
                                # Apply adjustment
                                g_u_new = g_u - adjustment
                                
                                # Assign back to parameters
                                offset = 0
                                for p in layer.parameters():
                                    if p.grad is not None:
                                        numel = p.grad.numel()
                                        p.grad.copy_(g_u_new[offset:offset+numel].reshape(p.grad.shape))
                                        offset += numel

                # gradient DP: C2 Adaptive Clipping + C3 Projection + C4 Risk-Adaptive Noise
                if cfg.grad_dp.enabled:
                    C0 = float(cfg.grad_dp.base_clip_C0)
                    kappa = float(cfg.grad_dp.importance_alpha) # Reuse alpha as kappa
                    sigma0 = float(cfg.grad_dp.noise_sigma0)
                    gamma_r = 1.0 # Risk multiplier factor (could be in config)
                    
                    # Identify View-Specific Params
                    view_param_ids = set()
                    if hasattr(step1.router, "view_layers"):
                        for v_idx, layer in enumerate(step1.router.view_layers):
                            # C2: Adaptive Clipping
                            # C^(c,v) = C0 * (1 + kappa * omega^(c,v))
                            omega_v = float(omega_init[v_idx].item())
                            C_v = C0 * (1.0 + kappa * omega_v)
                            
                            # C4: Risk-Adaptive Noise
                            # R(v) approx. sum of weights on sensitive view v
                            # Simplified: R(v) = beta_bar[v] * sensitivity[v]
                            # Or strictly per doc: sigma^(c,v) = sigma0 * (1 + gamma_r * R(x))
                            # But gradients are aggregated over batch, so we use batch/global stats.
                            # Let's use the pre-computed beta_bar_init as a stable proxy for risk exposure.
                            
                            # Get sensitivity for this view name
                            v_name = cfg.views[v_idx]
                            sens_v = float(cfg.temp_params.get("view_sensitivities", {}).get(v_name, 1.0))
                            
                            # R(v) = beta_bar^(c,v) * sens_v (Exposure to this sensitive view)
                            R_v = float(beta_bar_init[v_idx].item()) * sens_v
                            
                            sigma_v = sigma0 * (1.0 + gamma_r * R_v)
                            
                            for p in layer.parameters():
                                view_param_ids.add(id(p))
                                if p.grad is not None:
                                    # Clip
                                    clip_by_l2_norm_(p.grad, C_v)
                                    # Noise: std = sigma_v * C_v
                                    # add_gaussian_noise_ takes std (sigma)
                                    noise_std = sigma_v * C_v
                                    add_gaussian_noise_(p.grad, noise_std)
                    
                    # Global Params (Standard DP-SGD)
                    for p in params:
                        if p.grad is not None and id(p) not in view_param_ids:
                             clip_by_l2_norm_(p.grad, C0)
                             add_gaussian_noise_(p.grad, sigma0 * C0)

                opt.step()

                total_loss += float(loss.item())
                total_batches += 1

        # Step 2 C1: Client View Statistics (Importance/Utility/Risk Proxy)
        # Calculate view importance weights over the dataset
        # beta_bar = mean(w) over all samples (approximated by last batch or accumulated if possible)
        # Here we do a quick forward pass on full dataset or use accumulated stats from training loop if we tracked them.
        # For efficiency, let's use the statistics from the last batch as a proxy, 
        # OR better, accumulate during the loop. Let's accumulate during the loop.
        # But wait, w depends on model which changes.
        # Standard approach: Compute once at the end of local training on a representative subset.
        
        step1.eval()
        acc_w = torch.zeros(len(cfg.views), device=device)
        count_w = 0
        with torch.no_grad():
            for batch in loader:
                outs = step1(batch)
                w_batch = torch.stack([o.reliability_w for o in outs], dim=0).to(device) # [B, V]
                acc_w += w_batch.sum(dim=0)
                count_w += w_batch.shape[0]
        
        beta_bar = acc_w / max(count_w, 1) # [V]
        
        # Normalize importance factor (omega)
        # omega^(c,v) = beta_bar^(c,v) / (mean(beta_bar^(c)) + epsilon)
        # Doc: omega = beta_bar / (1/V * sum(beta_bar) + eps)
        # Meaning: omega > 1 means this view is more important than average for this client.
        V = len(cfg.views)
        omega = beta_bar / (beta_bar.mean() + 1e-12)
        
        # Pack into metrics for server
        # We need to send this to server for "Personalized Aggregation" or similar logic if needed.
        # The prompt asks for C1 implementation.
        view_stats = {
            "beta_bar": beta_bar.cpu().tolist(),
            "omega": omega.cpu().tolist(),
            "view_names": cfg.views
        }

        # update vector
        new_vec = torch.nn.utils.parameters_to_vector(params).detach().clone()
        update = new_vec - init_vec

        metrics = {
            "client_loss": total_loss / max(total_batches, 1),
            "num_batches": float(total_batches),
            "view_stats": view_stats, # Added C1 stats
        }
        n = len(dataset)
        return update, metrics, n
