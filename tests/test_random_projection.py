import numpy as np
from optc_uras.features.random_projection import FixedRandomProjector


def test_determinism():
    x = np.random.RandomState(0).randn(10, 32).astype(np.float32)
    p1 = FixedRandomProjector(32, 16, seed=123, matrix_type="gaussian", normalize_mode="l2")
    p2 = FixedRandomProjector(32, 16, seed=123, matrix_type="gaussian", normalize_mode="l2")
    y1 = p1(x)
    y2 = p2(x)
    assert np.allclose(y1, y2)
