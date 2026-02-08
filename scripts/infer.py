from __future__ import annotations

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'src'))


import json
from pathlib import Path
import torch

from scripts._common import parse_args, setup
from optc_uras.data.processed_io import load_samples_pt, processed_path
from optc_uras.pipelines.factory import build_step1, build_teacher_student
from optc_uras.detection.pipeline import DetectorPipeline, Step3Config
from optc_uras.detection.scd import SCDTrainConfig
from optc_uras.detection.mahalanobis import CovConfig
from optc_uras.detection.atc import ATCConfig
from optc_uras.detection.drift import DriftConfig
from optc_uras.detection.explain import ExplainConfig


def _build_s3cfg(cfg):
    scd_cfg = cfg["step3"]["scd"]
    scd_train = SCDTrainConfig(
        epochs=int(scd_cfg["train"]["epochs"]),
        lr=float(scd_cfg["train"]["lr"]),
        lambda_decorr=float(scd_cfg["train"]["loss_weights"]["lambda_decorr"]),
        lambda_recon=float(scd_cfg["train"]["loss_weights"]["lambda_recon"]),
        lambda_var=float(scd_cfg["train"]["loss_weights"]["lambda_var"]),
        variance_floor=float(scd_cfg["train"]["variance_floor"]),
    )
    cov_cfg = CovConfig(mode=cfg["step3"]["detector"]["covariance"]["mode"],
                        shrinkage=float(cfg["step3"]["detector"]["covariance"]["shrinkage"]))
    atc_cfg = ATCConfig(enabled=bool(cfg["step3"]["atc"]["enabled"]),
                        lambda_c=float(cfg["step3"]["atc"]["lambdas"]["lambda_c"]),
                        lambda_r=float(cfg["step3"]["atc"]["lambdas"]["lambda_r"]),
                        lambda_p=float(cfg["step3"]["atc"]["lambdas"]["lambda_p"]))
    drift_cfg = DriftConfig(enabled=bool(cfg["step3"]["drift"]["enabled"]),
                            margin=float(cfg["step3"]["drift"]["margin"]),
                            ema_alpha=float(cfg["step3"]["drift"]["ema_alpha"]),
                            update_covariance=str(cfg["step3"]["drift"]["update_covariance"]))
    explain_cfg = ExplainConfig(enabled=bool(cfg["step3"]["explain"]["enabled"]),
                                top_k=int(cfg["step3"]["explain"]["top_k"]),
                                alpha_reliability=float(cfg["step3"]["explain"]["score"]["alpha_reliability"]))
    return Step3Config(
        use_scd=bool(scd_cfg["enabled"]),
        style_dim=int(scd_cfg["style_dim"]),
        content_dim=int(scd_cfg["content_dim"]),
        scd_train=scd_train,
        cov=cov_cfg,
        q=float(cfg["step3"]["threshold"]["base_quantile_q"]),
        use_squared=bool(cfg["step3"]["detector"]["score"]["use_squared"]),
        atc=atc_cfg,
        drift=drift_cfg,
        explain=explain_cfg,
    )


def main() -> None:
    args = parse_args("Infer (skeleton)")
    cfg, run_dir = setup(args.config, args.override)
    device = cfg["project"]["device"]
    processed_dir = cfg["data"]["processed"]["path"]
    test_samples = load_samples_pt(processed_path(processed_dir, "test"))
    train_samples = load_samples_pt(processed_path(processed_dir, "train"))

    ckpt_path = cfg["paths"]["resume_from"]
    if ckpt_path is None:
        ckpt_path = str(run_dir / "checkpoints" / "student_global.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    step1 = build_step1(ckpt["cfg"]).to(device)
    step1.fit_vocabs_and_init_projectors(train_samples)
    if ckpt["cfg"]["step1"]["quality"]["standardize"] == "train_stats":
        step1.fit_quality_stats(train_samples)
    _, student_heads = build_teacher_student(ckpt["cfg"], step1)
    student_heads = student_heads.to(device)
    step1.load_state_dict(ckpt["step1_state"], strict=False)
    student_heads.load_state_dict(ckpt["student_heads_state"], strict=False)

    # detector artifacts
    det_path = run_dir / "artifacts" / "detector.pt"
    det = torch.load(det_path, map_location="cpu")
    s3cfg = _build_s3cfg(det["cfg"])

    pipe = DetectorPipeline(step1, student_heads, s3cfg, device=device)
    pipe.artifacts = det["artifacts"]

    preds = []
    for s in test_samples:
        out = pipe.infer_one(s)
        preds.append({
            "host": s.get("host"),
            "t0": s.get("t0"),
            "label": int(s.get("label", 0)),
            "score": out.score,
            "threshold": out.threshold,
            "pred": out.pred,
            "evidence": out.evidence,
        })

    out_path = run_dir / "metrics" / "predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[infer] saved predictions -> {out_path}")


if __name__ == "__main__":
    main()
