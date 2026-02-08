from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np

from .scd import SCDProjector
from .losses import scd_loss


class AnomalyDetector(nn.Module):
    """
    Step 3: Benign-Only Anomaly Detector.
    
    Core Components:
    1. SCD Projector (A1): Disentangles URAS into Style (s) and Content (c).
    2. Anomaly Scoring (D1): Based on Style deviation.
    3. Adaptive Threshold (D2): Dynamic thresholding based on benign statistics.
    """
    
    def __init__(self, feature_dim: int, style_dim: int = 32, content_dim: int = 32,
                 alpha_conf: float = 0.0, alpha_unc: float = 0.0, alpha_risk: float = 0.0,
                 prob_scale: float = 10.0, drift_margin: float = 0.5, drift_lr: float = 0.01,
                 view_names: Optional[List[str]] = None, view_dim: Optional[int] = None):
        super().__init__()
        self.scd = SCDProjector(feature_dim, style_dim, content_dim)
        
        # B3: ATC Hyperparameters
        self.alpha_conf = alpha_conf
        self.alpha_unc = alpha_unc
        self.alpha_risk = alpha_risk

        # B4/B5: Probability & Drift Hyperparameters
        self.prob_scale = prob_scale      # kappa
        self.drift_margin = drift_margin  # m
        self.drift_lr = drift_lr          # eta
        
        # C1: View Contribution Interpretation
        self.view_names = view_names
        self.view_projectors = nn.ModuleDict()
        view_in_dim = view_dim if view_dim is not None else feature_dim
        if view_names:
            for v in view_names:
                # P^(v): z^(v) -> s^(v)
                self.view_projectors[v] = nn.Linear(view_in_dim, style_dim)

        # Statistics for scoring (computed during fit)
        self.register_buffer("style_mu", torch.zeros(style_dim))
        self.register_buffer("style_var", torch.ones(style_dim)) # Store variance for B5 update
        # B1: Mahalanobis Covariance Matrix (or diagonal)
        # Store inverse covariance for speed
        self.register_buffer("style_inv_cov", torch.eye(style_dim))
        self.register_buffer("threshold", torch.tensor(0.0))
        
    def fit(self, uras_features: torch.Tensor, epochs: int = 10, lr: float = 1e-3, batch_size: int = 256, 
            use_diagonal: bool = True, quantile: float = 0.99,
            lambda_d: float = 1.0, lambda_r: float = 1.0, lambda_v: float = 1.0, gamma: float = 1.0) -> List[float]:
        """
        Train SCD projector on benign URAS features and compute statistics.
        Args:
            gamma: Variance regularization weight (higher = more decorrelation/anti-collapse).
            lambda_v: Variance loss weight.
        Returns:
            List[float]: Average loss per epoch.
        """
        self.train()
        optimizer = torch.optim.Adam(self.scd.parameters(), lr=lr)
        
        loss_history = []
        
        # 1. Train SCD Projector
        dataset = torch.utils.data.TensorDataset(uras_features)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            steps = 0
            for (batch_x,) in loader:
                # A3/A4 Training Flow
                # return_rec=True -> (s, c, u_hat)
                s, c, u_hat = self.scd(batch_x, center_batch=True, return_rec=True)
                
                # Full SCD Loss
                loss = scd_loss(
                    s=s, c=c, u=batch_x, u_hat=u_hat,
                    lambda_d=lambda_d, lambda_r=lambda_r, lambda_v=lambda_v, gamma=gamma
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                steps += 1
            
            avg_loss = epoch_loss / max(steps, 1)
            loss_history.append(avg_loss)
            
        # 2. Compute Benign Statistics (for Scoring)
        self.eval()
        with torch.no_grad():
            all_s = []
            for (batch_x,) in loader:
                s, _ = self.scd(batch_x, center_batch=False)
                all_s.append(s)
            
            all_s = torch.cat(all_s, dim=0)
            self.style_mu = all_s.mean(dim=0)
            
            # B1: Compute Covariance Sigma_c
            # Center data
            s_centered = all_s - self.style_mu
            N = s_centered.shape[0]
            
            if use_diagonal:
                # Diagonal approximation: Sigma_c = diag(var) + lambda*I
                var = s_centered.var(dim=0, unbiased=False) + 1e-6
                self.style_var = var
                self.style_inv_cov = torch.diag(1.0 / var)
            else:
                # Full Covariance: Sigma_c = 1/N * S^T S + lambda*I
                cov = torch.matmul(s_centered.t(), s_centered) / N
                cov = cov + torch.eye(self.scd.style_dim, device=cov.device) * 1e-6
                # For non-diagonal, we approximate var diagonal for B5 or just store diag part
                self.style_var = torch.diag(cov) 
                try:
                    self.style_inv_cov = torch.linalg.inv(cov)
                except RuntimeError:
                    # Fallback to diagonal if singular
                    var = s_centered.var(dim=0, unbiased=False) + 1e-6
                    self.style_inv_cov = torch.diag(1.0 / var)
            
            # 3. Determine Threshold (B2: Quantile-based)
            scores = self._compute_raw_score(all_s)
            
            # Calculate q-th quantile of benign scores
            # torch.quantile requires float tensor
            self.threshold = torch.quantile(scores, quantile)
            
        return loss_history

    def _compute_raw_score(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute raw anomaly score: Mahalanobis Distance Squared.
        S(x) = (s - mu)^T * Sigma^-1 * (s - mu)
        """
        diff = s - self.style_mu
        # diff: [B, d], inv_cov: [d, d]
        # left = (s - mu)^T * Sigma^-1 -> [B, d]
        # We want diag(diff @ inv_cov @ diff.T)
        # Efficiently: sum( (diff @ inv_cov) * diff, dim=1 )
        
        term1 = torch.matmul(diff, self.style_inv_cov)
        dist_sq = (term1 * diff).sum(dim=-1)
        
        return dist_sq

    def fit_interpreter(self, 
                        view_features: torch.Tensor, 
                        view_weights: torch.Tensor, 
                        uras_features: torch.Tensor, 
                        epochs: int = 10, lr: float = 1e-3, batch_size: int = 256,
                        log_path: Optional[str] = None):
        """
        Step 3 C1: Train View-to-Style Projections (Benign-Only).
        
        Args:
            view_features: [N, V, d] Step 1 view representations z^(v).
            view_weights: [N, V] Step 1 view reliability weights beta^(v).
            uras_features: [N, d] Step 2 global URAS vectors u^S(x).
            log_path: Optional path to save loss history CSV.
        """
        if not self.view_projectors:
            raise ValueError("view_names must be provided in __init__ to use interpreter")
            
        # Freeze main SCD model
        self.eval()
        # Unfreeze view projectors
        for p in self.view_projectors.values():
            p.train()
            
        optimizer = torch.optim.Adam(self.view_projectors.parameters(), lr=lr)
        
        dataset = torch.utils.data.TensorDataset(view_features, view_weights, uras_features)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            steps = 0
            for batch_z, batch_w, batch_u in loader:
                # batch_z: [B, V, d]
                # batch_w: [B, V]
                # batch_u: [B, d]
                
                with torch.no_grad():
                    # Get ground truth global style from SCD
                    target_s, _ = self.scd(batch_u, center_batch=False)
                    
                # Reconstruct style from views
                # s_hat(x) = sum(beta^(v) * P^(v)(z^(v)))
                recon_s = torch.zeros_like(target_s)
                
                for i, v_name in enumerate(self.view_names):
                    z_v = batch_z[:, i, :]       # [B, d]
                    w_v = batch_w[:, i].unsqueeze(1) # [B, 1]
                    
                    # Project view to style space
                    s_v = self.view_projectors[v_name](z_v) # [B, style_dim]
                    recon_s = recon_s + w_v * s_v
                    
                # Loss: MSE(s_hat, s)
                loss = (recon_s - target_s).pow(2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                steps += 1
            
            avg_loss = epoch_loss / max(steps, 1)
            loss_history.append({"epoch": epoch + 1, "loss": avg_loss})
            
        if log_path:
            import pandas as pd
            pd.DataFrame(loss_history).to_csv(log_path, index=False)
                
        self.view_projectors.eval()

    def interpret(self, 
                  view_features: torch.Tensor, 
                  view_weights: torch.Tensor,
                  uras_features: torch.Tensor,
                  top_k: int = 3) -> List[List[Tuple[str, float]]]:
        """
        Step 3 C2: Evidence Ranking / Explanation.
        
        Args:
            view_features: [B, V, d] View representations z^(v).
            view_weights: [B, V] View reliability weights beta^(v).
            uras_features: [B, d] Global URAS vectors u^S(x).
            top_k: Number of top evidence views to return per sample.
            
        Returns:
            List of length B, each containing a list of (view_name, score) tuples sorted by score.
        """
        if not self.view_projectors:
            raise ValueError("Interpreter not initialized or trained")
            
        self.eval()
        batch_size = uras_features.shape[0]
        results = []
        
        with torch.no_grad():
            # 1. Compute Global Anomaly Direction d(x)
            # s(x) = P_s(u^S(x))
            s_global, _ = self.scd(uras_features, center_batch=False)
            
            # d(x) = (s(x) - mu) / ||s(x) - mu||
            diff = s_global - self.style_mu
            norm_diff = torch.norm(diff, p=2, dim=1, keepdim=True) + 1e-6
            d_x = diff / norm_diff # [B, style_dim]
            
            # Pre-compute all view scores
            view_scores = torch.zeros((batch_size, len(self.view_names)), device=uras_features.device)
            
            for i, v_name in enumerate(self.view_names):
                z_v = view_features[:, i, :] # [B, d]
                beta_v = view_weights[:, i]  # [B]
                
                # 2. Map to Style Space: s_hat^(v)
                s_hat_v = self.view_projectors[v_name](z_v) # [B, style_dim]
                
                # Normalize projected style
                norm_s_hat = torch.norm(s_hat_v, p=2, dim=1) + 1e-6
                s_hat_v_norm = s_hat_v / norm_s_hat.unsqueeze(1)
                
                # 3. Compute Evidence Score E^(v)(x)
                # Cosine Similarity: sim = <s_hat_v_norm, d_x>
                sim = (s_hat_v_norm * d_x).sum(dim=1) # [B]
                
                # E^(v) = beta^(v) * max(0, sim)
                e_v = beta_v * torch.relu(sim)
                view_scores[:, i] = e_v
                
            # 4. Sort and Rank (Top-K)
            # values: [B, V], indices: [B, V]
            sorted_scores, sorted_indices = torch.sort(view_scores, dim=1, descending=True)
            
            for b in range(batch_size):
                sample_ranking = []
                for k in range(min(top_k, len(self.view_names))):
                    idx = sorted_indices[b, k].item()
                    score = sorted_scores[b, k].item()
                    v_name = self.view_names[idx]
                    sample_ranking.append((v_name, float(score)))
                results.append(sample_ranking)
                
        return results

    def forward(self, uras_features: torch.Tensor, 
                conf_scores: Optional[torch.Tensor] = None,
                unc_scores: Optional[torch.Tensor] = None,
                risk_scores: Optional[torch.Tensor] = None,
                adapt: bool = False) -> Dict[str, torch.Tensor]:
        """
        Inference / Detection.
        
        Args:
            uras_features: [B, d] URAS feature vectors.
            conf_scores: [B] View weight concentration (Step 1 Conf).
            unc_scores: [B] Routing uncertainty (Step 1 Unc).
            risk_scores: [B] Privacy risk signal (Step 2/Eng).
            adapt: If True, perform B5 online drift adaptation on benign samples.
        
        Returns:
            Dict containing:
            - score: Anomaly score S(x)
            - prob: Anomaly probability p(x) (B4)
            - threshold: Adaptive threshold theta(x)
            - anomaly: Binary prediction (1=Anomaly, 0=Benign)
            - style: Extracted style vector
            - content: Extracted content vector
        """
        self.eval()
        with torch.no_grad():
            s, c = self.scd(uras_features, center_batch=False)
            score = self._compute_raw_score(s)
            
            # B3: Sample Adaptive Threshold Correction (ATC)
            # theta(x) = theta_c - alpha_conf * Conf + alpha_unc * Unc + alpha_risk * R
            threshold = self.threshold.clone()
            if threshold.ndim == 0:
                threshold = threshold.expand(score.shape[0])
            
            if conf_scores is not None:
                threshold = threshold - self.alpha_conf * conf_scores
            if unc_scores is not None:
                threshold = threshold + self.alpha_unc * unc_scores
            if risk_scores is not None:
                threshold = threshold + self.alpha_risk * risk_scores
                
            is_anomaly = (score > threshold).float()

            # B4: Probability Mapping (Unsupervised)
            # p(x) = 1 - exp(-S(x) / kappa)
            prob = 1.0 - torch.exp(-score / self.prob_scale)

            # B5: Online Drift Adaptation
            # Only update when sample is "clearly benign" (S(x) <= theta_c - m)
            if adapt:
                # Use scalar baseline threshold for drift check
                is_clearly_benign = score <= (self.threshold - self.drift_margin)
                if is_clearly_benign.any():
                    s_benign = s[is_clearly_benign]
                    # EMA Update for Mean
                    # mu <- (1 - eta) * mu + eta * mean(s_benign)
                    current_mean = s_benign.mean(dim=0)
                    self.style_mu = (1 - self.drift_lr) * self.style_mu + self.drift_lr * current_mean
                    
                    # EMA Update for Variance (Diagonal)
                    # var <- (1 - eta) * var + eta * var(s_benign) or simplified mean-diff
                    # Using simple second moment update approximation for stability
                    current_var = s_benign.var(dim=0, unbiased=False) + 1e-6
                    self.style_var = (1 - self.drift_lr) * self.style_var + self.drift_lr * current_var
                    
                    # Recompute Inverse Covariance (Diagonal approximation for online speed)
                    self.style_inv_cov = torch.diag(1.0 / (self.style_var + 1e-6))
            
        return {
            "score": score,
            "prob": prob,
            "threshold": threshold,
            "anomaly": is_anomaly,
            "style": s,
            "content": c
        }