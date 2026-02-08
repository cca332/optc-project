from __future__ import annotations

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'src'))


import torch
from pathlib import Path

from scripts._common import parse_args, setup
from optc_uras.data.processed_io import load_samples_pt, processed_path
from optc_uras.pipelines.factory import build_step1, build_teacher_student
from optc_uras.pipelines.teacher_train import pretrain_teacher
from optc_uras.federated.simulation import run_federated_pretrain


def main() -> None:
    args = parse_args("Step2 federated pretrain (skeleton)")
    cfg, run_dir = setup(args.config, args.override)
    device = cfg["project"]["device"]

    processed_dir = cfg["data"]["processed"]["path"]
    train_samples = load_samples_pt(processed_path(processed_dir, "train"))

    # build models
    step1 = build_step1(cfg).to(device)
    # fit deterministic stats
    step1.fit_vocabs_and_init_projectors(train_samples)
    if cfg["step1"]["quality"]["standardize"] == "train_stats":
        step1.fit_quality_stats(train_samples)

    teacher, student_heads = build_teacher_student(cfg, step1)
    teacher = teacher.to(device)
    student_heads = student_heads.to(device)

    # teacher offline pretrain on benign-only
    benign_train = [s for s in train_samples if int(s.get("label", 0)) == 0]
    if cfg["step2"]["teacher"]["enabled"]:
        logs = pretrain_teacher(
            teacher,
            benign_train,
            views=cfg["data"]["views"],
            behavior_dim=int(cfg["step2"]["uras"]["behavior_feature_dim"]),
            device=device,
            epochs=int(cfg["step2"]["teacher"]["pretrain_epochs"]),
            batch_size=int(cfg["step2"]["federated"]["batch_size"]),
            lr=float(cfg["step2"]["federated"]["optimizer"]["lr"]),
            temperature=float(cfg["step2"]["teacher"]["temperature"]),
            augmentations=list(cfg["step2"]["teacher"]["augmentations"]),
            seed=int(cfg["project"]["seed"]),
        )
        print(f"[teacher] {logs}")

    # freeze teacher
    for p in teacher.parameters():
        p.requires_grad_(False)

    # federated simulation
    sim_cfg = {
        "seed": cfg["project"]["seed"],
        "views": cfg["data"]["views"],
        "federated": cfg["step2"]["federated"],
        "uras": cfg["step2"]["uras"],
        "losses": cfg["step2"]["losses"],
        "dp": cfg["step2"]["dp"],
    }
    out = run_federated_pretrain(train_samples, step1, student_heads, teacher, sim_cfg, device=device)
    print(f"[federated] {out['history']}")

    # save checkpoint
    ckpt = {
        "step1_state": step1.state_dict(),
        "student_heads_state": student_heads.state_dict(),
        "teacher_state": teacher.state_dict(),
        "cfg": cfg,
    }
    ckpt_path = run_dir / "checkpoints" / "student_global.pt"
    torch.save(ckpt, ckpt_path)
    print(f"[save] checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
