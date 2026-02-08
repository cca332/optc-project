from __future__ import annotations

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'src'))


import argparse
from pathlib import Path

from optc_uras.config import load_config
from optc_uras.data.raw_reader import ToyReader, RawReader
from optc_uras.data.processed_io import save_samples_pt, processed_path
from optc_uras.data.splits import make_splits, save_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--toy", action="store_true", help="生成 toy 数据（骨架自检）")
    ap.add_argument("--num_samples", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=[])
    data_root = Path(cfg["paths"]["data_root"])
    processed_dir = Path(cfg["data"]["processed"]["path"])
    splits_dir = Path(cfg["data"]["splits"]["path"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    if args.toy:
        reader: RawReader = ToyReader(num_samples=args.num_samples, seed=int(cfg["project"]["seed"]))
        samples = list(reader.iterate_samples())
    else:
        raise NotImplementedError("请在 src/optc_uras/data/raw_reader.py 实现你们的 RawReader，然后在此处接入。")

    # 保存全量 + splits（示例：随机切分）
    splits = make_splits(samples, seed=int(cfg["project"]["seed"]), val_ratio=0.1, test_ratio=0.2)
    save_splits(splits, str(splits_dir / "splits.json"))

    def subset(split_name: str):
        idxs = splits[split_name]
        return [samples[i] for i in idxs]

    save_samples_pt(subset("train"), processed_path(str(processed_dir), "train"))
    save_samples_pt(subset("val"), processed_path(str(processed_dir), "val"))
    save_samples_pt(subset("test"), processed_path(str(processed_dir), "test"))

    print(f"[preprocess] saved processed samples to: {processed_dir}")
    print(f"[preprocess] saved splits to: {splits_dir / 'splits.json'}")


if __name__ == "__main__":
    main()
