from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float32)

    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # AUC needs both classes present
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["auc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["auprc"] = float("nan")
    return out
