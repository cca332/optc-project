#!/usr/bin/env python3
"""
独立评估脚本：读取 detection_results.csv，与真实攻击标签对齐，计算准确率、召回率、F1、AUC 等指标。
- 模型输出格式：timestamp, host, anomaly_score, adaptive_threshold, is_anomaly, top_attribution
- 标准时间以 UTC 为准；真实标签按 UTC 的 15 分钟槽写死。若模型输出的 CSV 是 EDT，需加 --csv-edt，脚本会将其转为 UTC 再对齐。
"""

import os
import argparse
from typing import Set, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# 真实攻击标签（从 Summary 提取，按 UTC 写死）
# 事件链：Notepad++ 恶意更新 -> meterpreter -> 两台主机 Sysclient0051, Sysclient0351
# 日志时间(EDT) 已换算为 UTC：UTC = EDT + 4；粒度为 15 分钟槽 (date, slot_start)
# ---------------------------------------------------------------------------

# (日期, 槽起始时间, host 标识) 的集合；host 标识为 "0051" 或 "0351"
def _build_ground_truth_utc() -> Set[Tuple[str, str, str]]:
    # 2019-09-25，日志(EDT) -> UTC: 10:29 EDT -> 14:29 UTC 等
    # Sysclient0051 事件(EDT): 10:29,10:31,...,14:24 -> UTC: 14:29,14:31,...,18:24
    # 对应 15 分钟槽(UTC 槽起始): 14:15, 14:30, 14:45, 15:00, 17:30, 18:15
    # Sysclient0351 事件(EDT): 11:23,11:24 -> UTC: 15:23,15:24 -> 槽 15:15
    date = "2019-09-25"
    slots_0051 = [
        "14:15",  # 14:29
        "14:30",  # 14:31, 14:32, 14:33
        "14:45",  # 14:36, 14:37, 14:38, 14:40, 14:44, 14:48, 14:53
        "15:00",  # 15:07
        "17:30",  # 17:42
        "18:15",  # 18:24
    ]
    slots_0351 = [
        "15:15",  # 15:23, 15:24
    ]
    gt = set()
    for slot in slots_0051:
        gt.add((date, slot, "0051"))
    for slot in slots_0351:
        gt.add((date, slot, "0351"))
    return gt


GROUND_TRUTH_UTC = _build_ground_truth_utc()


def normalize_host(host: object) -> Optional[str]:
    """从 host 字符串提取 0051 或 0351，无法识别则返回 None（不参与带 host 的对齐）。"""
    if pd.isna(host):
        return None
    s = str(host).strip().lower()
    if "0051" in s:
        return "0051"
    if "0351" in s:
        return "0351"
    return None


def csv_timestamp_to_utc_slot(ts_str: str, csv_in_edt: bool = False) -> Optional[Tuple[str, str]]:
    """
    将 CSV 的 timestamp 转为 UTC 下的 (date, slot_start) 用于与真实标签对齐。
    标准时间为 UTC；若 CSV 是 EDT，则在此转为 UTC：EDT + 4h = UTC。
    ts_str: 如 '2019-09-25T13:00:00'
    csv_in_edt: 若 True，认为 CSV 为 EDT，会先转为 UTC 再取槽；否则认为 CSV 已是 UTC，不转换。
    返回 (date, slot_start)，解析失败返回 None。
    """
    try:
        dt = pd.to_datetime(ts_str)
        if csv_in_edt:
            # EDT -> UTC
            dt = dt + pd.Timedelta(hours=4)
        # 向下取整到 15 分钟
        minute = (dt.minute // 15) * 15
        slot_dt = dt.replace(minute=minute, second=0, microsecond=0)
        date = slot_dt.strftime("%Y-%m-%d")
        slot_start = slot_dt.strftime("%H:%M")
        return (date, slot_start)
    except Exception:
        return None


def build_y_true_y_pred_y_score(
    df: pd.DataFrame,
    ground_truth: Set[Tuple[str, str, str]],
    csv_in_edt: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按行对齐：每一行 (timestamp, host) 转为 UTC 下的 (date, slot)，得到是否在真实攻击槽内 -> y_true；
    y_pred = is_anomaly, y_score = anomaly_score。
    若 csv_in_edt=True，会先把 CSV 时间从 EDT 转为 UTC 再对齐。
    """
    y_true_list = []
    y_pred_list = []
    y_score_list = []

    for _, row in df.iterrows():
        ts = row.get("timestamp")
        host = row.get("host")
        pred = int(row.get("is_anomaly", 0))
        score = float(row.get("anomaly_score", 0.0))

        host_id = normalize_host(host)
        slot_key = csv_timestamp_to_utc_slot(str(ts), csv_in_edt=csv_in_edt)

        if slot_key is None:
            # 无法解析时间，跳过或视为 benign
            y_true_list.append(0)
            y_pred_list.append(pred)
            y_score_list.append(score)
            continue

        date, slot_start = slot_key
        if host_id is not None and (date, slot_start, host_id) in ground_truth:
            label = 1
        else:
            label = 0

        y_true_list.append(label)
        y_pred_list.append(pred)
        y_score_list.append(score)

    return (
        np.array(y_true_list, dtype=np.int32),
        np.array(y_pred_list, dtype=np.int32),
        np.array(y_score_list, dtype=np.float64),
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["auc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["auprc"] = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["tp"] = int(tp)
    out["tn"] = int(tn)
    out["fp"] = int(fp)
    out["fn"] = int(fn)
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection_results.csv with ground truth.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="detection_results.csv",
        help="Path to detection_results.csv (default: detection_results.csv)",
    )
    parser.add_argument(
        "--csv-edt",
        action="store_true",
        help="CSV 时间为 EDT（东部时间）时加上此参数，脚本会把 CSV 转为 UTC 再与真实标签对齐",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    if not os.path.isfile(csv_path):
        print(f"Error: File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    required = ["timestamp", "host", "anomaly_score", "is_anomaly"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: CSV missing columns: {missing}")
        return

    csv_in_edt = args.csv_edt
    y_true, y_pred, y_score = build_y_true_y_pred_y_score(df, GROUND_TRUTH_UTC, csv_in_edt=csv_in_edt)

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    print(f"[Ground Truth] 正样本(攻击)数: {n_pos}, 负样本数: {n_neg}, 总行数: {len(y_true)}")
    print(f"[时间对齐] 标准时间=UTC；真实标签为 UTC 写死；CSV 视为 {'EDT，已转为 UTC' if csv_in_edt else '已是 UTC，未做转换'}\n")

    metrics = classification_metrics(y_true, y_pred, y_score)
    print("=== 指标 ===")
    print(f"  准确率 (accuracy): {metrics['accuracy']:.4f}")
    print(f"  精确率 (precision): {metrics['precision']:.4f}")
    print(f"  召回率 (recall): {metrics['recall']:.4f}")
    print(f"  F1 (f1): {metrics['f1']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  AUPRC: {metrics['auprc']:.4f}")
    print(f"  混淆矩阵: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")

    # 可选：保存指标到 JSON
    out_json = csv_path.replace(".csv", "_metrics.json") if csv_path.endswith(".csv") else "detection_metrics.json"
    import json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n指标已保存: {out_json}")


if __name__ == "__main__":
    main()
