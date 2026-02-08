from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import ProcessedDataset, collate_samples
from ..federated.dp import FeatureDPConfig, dp_features
from ..features.behavior_features import behavior_features_from_sample


def _augment(x: torch.Tensor, aug: str, rng: torch.Generator) -> torch.Tensor:
    if aug == "mask":
        # random feature masking
        m = (torch.rand(x.shape, generator=rng, device=x.device) > 0.1).float()
        return x * m
    if aug == "jitter":
        return x + torch.randn(x.shape, generator=rng, device=x.device) * 0.01
    if aug == "dropout":
        return F.dropout(x, p=0.1, training=True)
    return x


def pretrain_teacher(
    teacher: torch.nn.Module,
    benign_samples: List[Dict[str, Any]],
    views: Sequence[str],
    behavior_dim: int,
    device: str,
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 0.1,
    augmentations: Sequence[str] = ("mask", "jitter", "dropout"),
    dp_config: Optional[FeatureDPConfig] = None,
    seed: int = 42,
) -> Dict[str, float]:
    teacher.train()
    ds = ProcessedDataset(benign_samples)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_samples)

    opt = torch.optim.AdamW(teacher.parameters(), lr=lr)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    losses = []
    for _ in range(int(epochs)):
        for batch in dl:
            feats = []
            for s in batch:
                f = behavior_features_from_sample(s, views, out_dim=int(behavior_dim))
                feats.append(torch.from_numpy(f))
            x = torch.stack(feats, dim=0).float().to(device)  # [B,d_b]

            # Apply DP noise (Step 2 A2 -> A3: Teacher sees noised features)
            if dp_config is not None:
                x = dp_features(x, dp_config)

            x1 = x
            x2 = x
            for aug in augmentations:
                x1 = _augment(x1, aug, g)
                x2 = _augment(x2, aug, g)

            z1 = teacher(x1, normalize=True)  # [B,M,d_s]
            z2 = teacher(x2, normalize=True)
            # average subspaces for teacher pretrain objective (simplify)
            a1 = F.normalize(z1.mean(dim=1), dim=-1)  # [B,d_s]
            a2 = F.normalize(z2.mean(dim=1), dim=-1)

            sim = a1 @ a2.t()
            logits = sim / float(temperature)
            labels = torch.arange(sim.shape[0], device=device)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
    teacher.eval()
    return {"teacher_pretrain_loss": sum(losses) / max(len(losses), 1)}
