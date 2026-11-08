"""Phase 1 定向 2×2 试点（预注册）：损失函数 × 几何增强。

设计（在运行前写入 phase1_pilot_plan.json）
    loss: huber（对照） vs l1
    augment: default（缩放 0.9–1.1 + 旋转 ±10°） vs geom_off（关闭几何变换）
    seeds: 11, 12
    训练：inner_train（5,182 张）; 选择：inner_val（920 张）; 原验证集确认、测试集封存
    超参固定：lr 1e-4, wd 0.05, batch 32, epochs 40, patience 12, cosine

判定（预先固定）
    对照组 = huber + default；配置胜出需同时满足
      (1) 两 seed 的 inner-val MAE 均值比对照低 ≥ 0.30 kg；
      (2) 两个 seed 的方向一致（各自低于同 seed 的对照）。
    若无配置胜出：停止搜索，结论为“优化已接近该数据的信息上限”。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PY = sys.executable
SRC = Path(__file__).parent
ROOT = SRC.parents[1]
TABLE = ROOT / "project/results/tables"
GRID = [("huber", "default"), ("huber", "geom_off"), ("l1", "default"), ("l1", "geom_off")]
SEEDS = (11, 12)
TAG = "p1_{loss}_{aug}_s{seed}"


def plan() -> dict:
    return {
        "phase": "1 - targeted 2x2 pilot (loss x geometry augmentation)",
        "split_file": "splits_inner.csv", "split_col": "split_inner",
        "train_split": "inner_train", "val_split": "inner_val",
        "confirm_split": "confirm_val", "test": "sealed",
        "grid": [{"loss": l, "augment": a, "seeds": list(SEEDS)} for l, a in GRID],
        "fixed": {"lr": 1e-4, "weight_decay": 0.05, "batch_size": 32, "epochs": 40,
                  "patience": 12, "backbone": "resnet50", "size": 308, "no_test": True},
        "decision_rule": {
            "control": "huber_default",
            "min_mean_gain_kg": 0.30,
            "requires_same_direction_in_both_seeds": True,
            "if_no_winner": "stop searching; optimisation at the information limit of this data",
        },
    }


def run_one(loss: str, aug: str, seed: int, log) -> None:
    tag = TAG.format(loss=loss, aug=aug, seed=seed)
    metrics = TABLE / f"metrics_{tag}.json"
    if metrics.exists():
        log.info("skip existing %s", tag)
        return
    cmd = [PY, str(SRC / "train.py"), "--backbone", "resnet50", "--seed", str(seed),
           "--size", "308", "--epochs", "40", "--lr", "1e-4", "--weight-decay", "0.05",
           "--batch-size", "32", "--patience", "12", "--loss", loss,
           "--augment-profile", aug, "--split-file", "splits_inner.csv",
           "--split-col", "split_inner", "--train-split", "inner_train",
           "--val-split", "inner_val", "--tag", tag, "--no-test"]
    log.info("run %s", " ".join(cmd[2:]))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("FAILED %s\n%s", tag, r.stderr[-2000:])
    else:
        m = json.loads(metrics.read_text(encoding="utf-8"))
        log.info("done %s: best val MAE=%.3f (epoch %d)", tag, m["best"]["val_mae"],
                 m["best"]["epoch"])


def full_precision_eval(log) -> dict:
    """用保存的最优 checkpoint 在全精度下复算 inner-val MAE（与 AMP 训练日志解耦）。"""
    out = {}
    for loss, aug in GRID:
        for seed in SEEDS:
            tag = TAG.format(loss=loss, aug=aug, seed=seed)
            ck = ROOT / "project/data/features/checkpoints" / f"{tag}.pt"
            if not ck.exists():
                continue
            cmd = [PY, str(SRC / "predict_split.py"), "--split", "inner_val",
                   "--split-file", "splits_inner.csv", "--split-col", "split_inner",
                   "--backbone", "resnet50", "--size", "308", "--checkpoints", str(ck),
                   "--out", f"inner_val_{tag}.csv"]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                log.error("eval failed %s: %s", tag, r.stderr[-500:])
                continue
            j = json.loads((TABLE / f"prediction_inner_val_{tag}.json").read_text(encoding="utf-8"))
            out.setdefault(f"{loss}_{aug}", {})[str(seed)] = float(j["metrics"]["mae"])
    return out


def summarise(eval_by_config: dict, log) -> dict:
    means = {k: float(np.mean(list(v.values()))) for k, v in eval_by_config.items() if len(v) == len(SEEDS)}
    control = means.get("huber_default")
    rows = []
    for k, m in means.items():
        per = eval_by_config[k]
        gains = {s: control - per[s] for s in per} if control is not None else {}
        rows.append({"config": k, "mean_val_mae": m,
                     "gain_vs_control_kg": None if control is None else control - m,
                     "per_seed_gain_kg": gains,
                     "same_direction": bool(gains) and all(g > 0 for g in gains.values()),
                     "wins": bool(control is not None and (control - m) >= 0.30
                                  and all(g > 0 for g in gains.values()))})
    winners = [r["config"] for r in rows if r["wins"]]
    summary = {"control_huber_default": control, "configs": rows, "winners": winners,
               "decision": ("proceed to Phase 2 (Optuna refinement)" if winners
                            else "stop: no config reached the 0.30 kg threshold")}
    (TABLE / "phase1_pilot_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        log.info("%-18s mean=%.3f gain=%s same_dir=%s wins=%s", r["config"],
                 r["mean_val_mae"],
                 "NA" if r["gain_vs_control_kg"] is None else f"{r['gain_vs_control_kg']:+.3f}",
                 r["same_direction"], r["wins"])
    log.info("decision: %s", summary["decision"])
    return summary


def main() -> None:
    from common import setup_logger
    log = setup_logger("phase1_pilot")
    p = TABLE / "phase1_pilot_plan.json"
    if not p.exists():
        p.write_text(json.dumps(plan(), ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("plan -> %s", p)
    for loss, aug in GRID:
        for seed in SEEDS:
            run_one(loss, aug, seed, log)
    ev = full_precision_eval(log)
    summarise(ev, log)


if __name__ == "__main__":
    main()
