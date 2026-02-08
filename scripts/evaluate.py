from __future__ import annotations

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'src'))


import json
from pathlib import Path

from scripts._common import parse_args, setup
from optc_uras.utils.metrics import classification_metrics


def main() -> None:
    args = parse_args("Evaluate (skeleton)")
    cfg, run_dir = setup(args.config, args.override)

    pred_path = run_dir / "metrics" / "predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"missing {pred_path}; run scripts/infer.py first")

    y_true, y_pred, y_score = [], [], []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            y_true.append(int(r["label"]))
            y_pred.append(int(r["pred"]))
            y_score.append(float(r["score"]))

    mets = classification_metrics(y_true, y_pred, y_score)
    out_path = run_dir / "metrics" / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mets, f, ensure_ascii=False, indent=2)
    print(f"[eval] {mets}")
    print(f"[eval] saved -> {out_path}")


if __name__ == "__main__":
    main()
