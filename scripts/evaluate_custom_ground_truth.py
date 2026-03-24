import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection_results.csv with a custom ground truth JSON.")
    parser.add_argument("csv_path", help="Path to detection_results.csv")
    parser.add_argument(
        "--ground-truth",
        default="configs/day325_core_attack_windows.json",
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--csv-edt",
        action="store_true",
        help="Interpret CSV timestamps as EDT and convert to UTC before evaluation",
    )
    args = parser.parse_args()

    ground_truth = evaluate.load_ground_truth(args.ground_truth)
    metrics = evaluate.run_evaluation_with_ground_truth(args.csv_path, ground_truth, csv_in_edt=args.csv_edt)

    print(f"[Ground Truth Source] custom ground truth: {args.ground_truth}")
    print("=== Metrics ===")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall: {metrics['recall']:.4f}")
    print(f"  f1: {metrics['f1']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  AUPRC: {metrics['auprc']:.4f}")
    print(f"  confusion_matrix: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")

    out_json = args.csv_path.replace(".csv", "_core_attack_metrics.json") if args.csv_path.endswith(".csv") else "core_attack_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics: {out_json}")


if __name__ == "__main__":
    main()
