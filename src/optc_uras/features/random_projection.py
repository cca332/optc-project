from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


def make_random_projection_matrix(in_dim: int, out_dim: int, seed: int, matrix_type: str = "gaussian") -> np.ndarray:
    rng = np.random.default_rng(seed)
    if matrix_type == "gaussian":
        mat = rng.standard_normal((out_dim, in_dim), dtype=np.float32) / np.sqrt(out_dim)
    elif matrix_type == "achlioptas_sparse":
        # Achlioptas: entries in {-1,0,+1} with probs {1/6,2/3,1/6}
        r = rng.random((out_dim, in_dim), dtype=np.float32)
        mat = np.zeros((out_dim, in_dim), dtype=np.float32)
        mat[r < (1/6)] = 1.0
        mat[r > (5/6)] = -1.0
        mat = mat / np.sqrt(out_dim)
    else:
        raise ValueError(f"unknown matrix_type={matrix_type}")
    return mat


def normalize(x: np.ndarray, mode: str = "l2", eps: float = 1e-8) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "l2":
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / (n + eps)
    if mode == "zscore":
        mu = x.mean(axis=-1, keepdims=True)
        sig = x.std(axis=-1, keepdims=True)
        return (x - mu) / (sig + eps)
    raise ValueError(f"unknown normalize mode={mode}")


class FixedRandomProjector:
    """视图专属固定随机投影编码器（无需训练）。"""

    def __init__(self, in_dim: int, out_dim: int, seed: int, matrix_type: str, normalize_mode: str = "l2", nonlinearity: str = "none"):
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.seed = int(seed)
        self.matrix_type = str(matrix_type)
        self.normalize_mode = str(normalize_mode)
        self.nonlinearity = str(nonlinearity)
        self.W = make_random_projection_matrix(self.in_dim, self.out_dim, self.seed, self.matrix_type)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: [..., in_dim]
        y = (x @ self.W.T).astype(np.float32)
        if self.nonlinearity == "tanh":
            y = np.tanh(y)
        elif self.nonlinearity == "relu":
            y = np.maximum(y, 0)
        elif self.nonlinearity != "none":
            raise ValueError(f"unknown nonlinearity={self.nonlinearity}")
        y = normalize(y, self.normalize_mode)
        return y
