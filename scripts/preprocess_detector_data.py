import argparse
import os
import sys
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocess import process_file


def main():
    parser = argparse.ArgumentParser(description="Preprocess detector-only benign raw data.")
    parser.add_argument("--config", default="configs/final_production.yaml", help="Path to config file")
    parser.add_argument("--path", type=str, default=None, help="Optional override detector raw path")
    parser.add_argument("--prefix", type=str, default=None, help="Optional override detector cache prefix")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {}).get("optc", {})
    if not data_cfg:
        raise ValueError("Config missing data.optc section")

    detector_path = args.path or data_cfg.get("detector_path")
    detector_prefix = args.prefix or data_cfg.get("detector_prefix", "detector")
    cache_dir = data_cfg.get("cache_dir", "data/cache")

    if not detector_path:
        raise ValueError("No detector_path provided. Set data.optc.detector_path or pass --path.")

    os.makedirs(cache_dir, exist_ok=True)
    detector_cfg = dict(data_cfg)

    print(f"[DetectorPreprocess] Path: {detector_path}")
    print(f"[DetectorPreprocess] Prefix: {detector_prefix}")
    print(f"[DetectorPreprocess] Cache dir: {cache_dir}")

    process_file(detector_path, cache_dir, detector_prefix, detector_cfg)


if __name__ == "__main__":
    main()
