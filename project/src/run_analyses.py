"""后处理编排：等待训练队列完成后，依次执行评价、不确定性、序列、图表、审计与报告。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from common import PROJECT, TABLE_DIR, setup_logger
from common import FEATURE_DIR

PY = sys.executable
SRC = PROJECT / "src"


def run(cmd: list[str], log, name: str, check: bool = False) -> int:
    log.info("RUN %s", " ".join(cmd))
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (PROJECT / "results" / "logs" / f"analysis_{name}.log").write_text(
        (p.stdout or "") + "\n--- STDERR ---\n" + (p.stderr or ""), encoding="utf-8")
    log.info("DONE %s rc=%d %.1f min", name, p.returncode, (time.time() - t0) / 60)
    if check and p.returncode != 0:
        raise RuntimeError(f"{name} failed rc={p.returncode}")
    return p.returncode


def wait_for_training(log, timeout_min: int = 300) -> None:
    required = [f"metrics_swin_t_308_s{i}.json" for i in (1, 2, 3)]
    required += ["metrics_swin_t_224_s1.json", "metrics_swin_t_308_temporal_s1.json",
                 "metrics_rad_dino_224_s1.json"]
    t0 = time.time()
    while True:
        missing = [r for r in required if not (TABLE_DIR / r).exists()]
        if not missing:
            log.info("all training runs present")
            return
        if (time.time() - t0) / 60 > timeout_min:
            log.warning("wait timeout; missing=%s", missing)
            return
        log.info("waiting for training: %s", missing)
        time.sleep(120)


def wait_for_preprocessing(log, expected: int = 35023, timeout_min: int = 300) -> None:
    cache = FEATURE_DIR / "all_320"
    t0 = time.time()
    while True:
        n = len(list(cache.glob("*.npy"))) if cache.exists() else 0
        if n >= expected:
            log.info("all_320 preprocessing complete: %d", n)
            return
        if (time.time() - t0) / 60 > timeout_min:
            log.warning("preprocessing wait timeout: %d/%d", n, expected)
            return
        log.info("waiting for preprocessing: %d/%d", n, expected)
        time.sleep(120)


def main() -> None:
    log = setup_logger("run_analyses")
    wait_for_training(log)
    wait_for_preprocessing(log)

    # 补跑标准化版本的 CNN 基线（与主模型使用相同的目标标准化）
    if not (TABLE_DIR / "metrics_resnet50_224_s1_std.json").exists():
        run([PY, str(SRC / "train.py"), "--backbone", "resnet50", "--size", "224", "--seed", "1",
             "--epochs", "30", "--patience", "6", "--tag", "resnet50_224_s1_std"],
            log, "train_resnet_std", check=False)

    ckpts = [str(PROJECT / "data" / "features" / "checkpoints" / f"swin_t_308_s{i}.pt") for i in (1, 2, 3)]
    ckpts = [c for c in ckpts if Path(c).exists()]
    if len(ckpts) < 3:
        log.warning("only %d swin checkpoints available", len(ckpts))

    # 1) 集成评价
    run([PY, str(SRC / "evaluate.py"), "--tag", "swin_ensemble", "--backbone", "swin_t",
         "--size", "308", "--checkpoints", *ckpts], log, "evaluate_ensemble")
    # 2) 单模型评价（对照）
    for i, ck in enumerate(ckpts, 1):
        run([PY, str(SRC / "evaluate.py"), "--tag", f"swin_s{i}", "--backbone", "swin_t",
             "--size", "308", "--checkpoints", ck], log, f"evaluate_s{i}")
    # 3) 时间外 / 224 / resnet / raddino
    for tag, backbone, size in (("swin_temporal", "swin_t", 308), ("swin_224", "swin_t", 224),
                                ("resnet50", "resnet50", 224), ("rad_dino", "rad_dino", 224)):
        metrics = TABLE_DIR / f"metrics_{tag}.json"
        ck = None
        for cand in (PROJECT / "data" / "features" / "checkpoints").glob(f"{tag}*.pt"):
            ck = cand
        if ck is None and tag == "swin_temporal":
            ck = PROJECT / "data" / "features" / "checkpoints" / "swin_t_308_temporal_s1.pt"
        if ck is None and tag == "swin_224":
            ck = PROJECT / "data" / "features" / "checkpoints" / "swin_t_224_s1.pt"
        if ck is None and tag == "resnet50":
            ck = PROJECT / "data" / "features" / "checkpoints" / "resnet50_224_s1.pt"
        if ck is None and tag == "rad_dino":
            ck = PROJECT / "data" / "features" / "checkpoints" / "rad_dino_224_s1.pt"
        if ck and Path(ck).exists():
            run([PY, str(SRC / "evaluate.py"), "--tag", tag, "--backbone", backbone,
                 "--size", str(size), "--checkpoints", str(ck)], log, f"evaluate_{tag}", check=False)
    # 4) 不确定性 / 剂量
    run([PY, str(SRC / "uncertainty.py"), "--tag", "swin_ensemble"], log, "uncertainty", check=False)
    run([PY, str(SRC / "dose_simulation.py"), "--tag", "swin_ensemble"], log, "dose", check=False)
    # 5) 全集推理 → 构念效度
    run([PY, str(SRC / "predict_all.py"), "--cohort", "cohort_a_all_images.csv",
         "--backbone", "swin_t", "--size", "308", "--out", "preds_all_swin_ensemble.csv",
         "--checkpoints", *ckpts], log, "predict_all", check=False)
    ev = TABLE_DIR / "evaluation_swin_ensemble.json"
    if ev.exists():
        mae = json.loads(ev.read_text(encoding="utf-8"))["overall"]["mae"]
        run([PY, str(SRC / "construct_validity.py"), "--predictions", "preds_all_swin_ensemble.csv",
             "--cross_sectional_mae", str(mae)], log, "construct_validity", check=False)
    # 6) 序列模型
    run([PY, str(SRC / "sequence_features.py"), "--backbone", "swin_t", "--size", "308",
         "--checkpoint", str(ckpts[0]), "--cohort", "cohort_a_all_images.csv",
         "--out", "sequence_features.npy"], log, "sequence_features", check=False)
    run([PY, str(SRC / "sequence_model.py")], log, "sequence_model", check=False)
    # 7) 注意力/遮挡
    run([PY, str(SRC / "attention_occlusion.py"), "--backbone", "swin_t", "--size", "308",
         "--checkpoint", str(ckpts[0]), "--tag", "swin_ensemble"], log, "attention", check=False)
    # 8) 图表 / 审计 / 报告
    run([PY, str(SRC / "figures.py"), "--tag", "swin_ensemble"], log, "figures", check=False)
    run([PY, str(SRC / "final_audit.py")], log, "audit", check=False)
    run([PY, str(SRC / "build_report.py")], log, "report", check=False)
    log.info("analysis pipeline finished")


if __name__ == "__main__":
    main()
