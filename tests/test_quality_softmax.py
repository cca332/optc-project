import numpy as np
from optc_uras.features.quality import QualityWeighter, QualityWeightsConfig


def test_softmax():
    qw = QualityWeighter(QualityWeightsConfig(weights={"coverage": 1.0}, softmax_temperature=1.0))
    w = qw.softmax_weights([0.0, 0.0, 0.0])
    assert np.allclose(w.sum(), 1.0)
