from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


class SecureAggregator:
    """安全聚合 (Secure Aggregation) 的科研仿真实现。

    本模块模拟了真实联邦学习环境下的安全聚合协议。虽然在单机模拟中不需要真实的加密通信，
    但我们在数学上严格模拟了协议的行为和数值特性。

    支持模式：
      - enabled=False: 明文聚合 (Plaintext Aggregation)。
      - enabled=True & protocol='pairwise_masking': 成对掩码模拟。
        模拟 Bonawitz et al. (2017) 协议的核心数学属性：
        1. 客户端生成掩码 M_c，使得所有客户端掩码之和为 0 (Sum(M_c) = 0)。
        2. 服务器只能看到聚合后的结果 Sum(g_c + M_c) = Sum(g_c)。
        3. 服务器无法解析出单个客户端的梯度 g_c。
      - enabled=True & protocol='mock_secureagg': 简化模拟（无噪声/掩码为0），仅用于快速调试。
    """

    def __init__(self, enabled: bool = False, protocol: str = "mock_secureagg"):
        self.enabled = bool(enabled)
        self.protocol = str(protocol)

    def aggregate(self, client_updates: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        # client_updates: list of [P] flattened updates
        # weights: [C] sum to 1
        stacked = torch.stack(client_updates, dim=0)  # [C,P]
        
        # In FedAvg + SecAgg, clients typically weight their updates by sample count (n_k).
        # Here we simulate that by applying weights before masking.
        # Effectively, we simulate clients sending y_k = w_k * u_k + m_k
        weighted_stack = weights.unsqueeze(-1) * stacked

        if not self.enabled:
            return weighted_stack.sum(dim=0)

        if self.protocol not in ["mock_secureagg", "pairwise_masking"]:
            raise ValueError(f"unknown protocol={self.protocol}")

        # 模拟：客户端加掩码，服务器只看到加权和（掩码在聚合后抵消）
        # Simulation: Clients add masks, Server sees weighted sum.
        # Masks must sum to zero across clients for the result to be correct.
        
        if self.protocol == "pairwise_masking":
            # Generate masks that strictly sum to zero (Simulate Pairwise Masking)
            n_clients, n_params = stacked.shape
            
            # Constructive approach for simulation:
            # Generate N random vectors. Subtract their mean to ensure sum is zero.
            # sum(x_i - mean(x)) = sum(x_i) - N*mean(x) = 0.
            raw_masks = torch.randn_like(stacked)
            mean_mask = raw_masks.mean(dim=0, keepdim=True)
            masks = raw_masks - mean_mask
            
        else:
            # mock_secureagg: default to 0 noise for efficiency
            masks = torch.zeros_like(stacked)

        masked = weighted_stack + masks
        agg = masked.sum(dim=0)
        
        return agg