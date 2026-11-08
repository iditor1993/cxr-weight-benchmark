"""时间外验证：在 temporal 训练集上重训 ResNet-50 与 Swin-T（各 3 seeds），
在 temporal 测试集（n=1,353，内部患者不重叠）上做一次性比较。

背景：temporal 测试患者与随机训练集重叠 70.7%，因此不能直接用随机训练的主模型评估；
必须按时间切分重训。Swin 的 seed 1 已有产物，这里补 seed 2/3 并把 ResNet 跑满 3 seeds。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PY = sys.executable
SRC = Path(__file__).parent
ROOT = SRC.parents[1]
TABLE = ROOT / "project/results/tables"
CKPT = ROOT / "project/data/features/checkpoints"
PROC = ROOT / "project/data/processed"

RUNS = [("resnet50", 1), ("resnet50", 2), ("resnet50", 3),
        ("swin_t", 2), ("swin_t", 3)]  # swin seed 1 已存在
TAGS = {b: [f"{b}_308_temporal_s{i}" for i in (1, 2, 3)] for b in ("resnet50", "swin_t")}


def train(backbone: str, seed: int, log) -> None:
    tag = f"{backbone}_308_temporal_s{seed}"
    if (TABLE / f"metrics_{tag}.json").exists():
        m = json.loads((TABLE / f"metrics_{tag}.json").read_text(encoding="utf-8"))
        if int(m["args"]["seed"]) != seed or m["args"]["tag"] != tag:
            raise RuntimeError(f"stale metrics for {tag}: args={m['args']}")
        log.info("skip existing %s", tag)
        return
    cmd = [PY, str(SRC / "train.py"), "--backbone", backbone, "--split", "temporal",
           "--seed", str(seed),
           "--size", "308", "--epochs", "30", "--lr", "3e-4", "--weight-decay", "0.05",
           "--batch-size", "32", "--patience", "6", "--tag", tag, "--no-test"]
    log.info("train %s", " ".join(cmd[2:]))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("FAILED %s\n%s", tag, r.stderr[-1500:])
    else:
        m = json.loads((TABLE / f"metrics_{tag}.json").read_text(encoding="utf-8"))
        if int(m["args"]["seed"]) != seed or m["args"]["tag"] != tag:
            raise RuntimeError(f"seed not applied for {tag}: args={m['args']}")
        log.info("done %s: val MAE=%.3f (epoch %d)", tag, m["best"]["val_mae"], m["best"]["epoch"])


def evaluate(tag: str, backbone: str, ckpts: list[str], log) -> None:
    cmd = [PY, str(SRC / "evaluate.py"), "--tag", tag, "--checkpoints", *ckpts,
           "--backbone", backbone, "--size", "308", "--split", "temporal"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("eval FAILED %s\n%s", tag, r.stderr[-1000:])
        return
    ev = json.loads((TABLE / f"evaluation_{tag}.json").read_text(encoding="utf-8"))
    log.info("eval %-38s MAE=%.3f hit10=%.1f%% bias=%+.2f", tag, ev["overall"]["mae"],
             100 * ev["overall"]["hit_within_10pct"], ev["overall"]["bias"])


def paired(a_path: Path, b_path: Path, label: str) -> dict:
    a = pd.read_csv(a_path)[["stay_id", "subject_id", "weight_kg", "pred"]]
    b = pd.read_csv(b_path)[["stay_id", "pred"]].rename(columns={"pred": "pred_b"})
    d = a.merge(b, on="stay_id")
    y = d["weight_kg"].values
    diff = np.abs(y - d["pred"].values) - np.abs(y - d["pred_b"].values)
    subj = d["subject_id"].values
    uniq, inv = np.unique(subj, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    cnts = np.bincount(inv)
    rng = np.random.default_rng(20261028)
    idx = rng.integers(0, len(cnts), size=(4000, len(cnts)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    return {"label": label, "n": int(len(d)), "mae_a": float(np.mean(np.abs(y - d["pred"].values))),
            "mae_b": float(np.mean(np.abs(y - d["pred_b"].values))),
            "paired_diff_a_minus_b": float(diff.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}


def main() -> None:
    from common import setup_logger
    log = setup_logger("temporal_validation")
    for backbone, seed in RUNS:
        train(backbone, seed, log)
    # 逐种子评价
    for backbone, tags in TAGS.items():
        for tag in tags:
            ck = CKPT / f"{tag}.pt"
            if ck.exists():
                evaluate(tag, backbone, [str(ck)], log)
    # 3-seed 集成
    for backbone, tags in TAGS.items():
        ckpts = [str(CKPT / f"{t}.pt") for t in tags if (CKPT / f"{t}.pt").exists()]
        if len(ckpts) == 3:
            evaluate(f"{backbone}_308_temporal_ensemble", backbone, ckpts, log)
    # 配对比较
    out = {"split": "temporal test (n=1,353; internally patient-disjoint)",
           "note": "temporal test patients overlap 70.7% with the random training split, "
                   "so models were retrained on the temporal training split"}
    r_ev = TABLE / "evaluation_resnet50_308_temporal_ensemble.json"
    s_ev = TABLE / "evaluation_swin_t_308_temporal_ensemble.json"
    if r_ev.exists() and s_ev.exists():
        out["ensemble_resnet"] = json.loads(r_ev.read_text(encoding="utf-8"))["overall"]
        out["ensemble_swin"] = json.loads(s_ev.read_text(encoding="utf-8"))["overall"]
        out["paired_resnet_vs_swin"] = paired(
            PROC / "test_predictions_resnet50_308_temporal_ensemble.csv",
            PROC / "test_predictions_swin_t_308_temporal_ensemble.csv",
            "ResNet-50 ensemble minus Swin-T ensemble")
    out["per_seed"] = {}
    for backbone, tags in TAGS.items():
        for tag in tags:
            p = TABLE / f"evaluation_{tag}.json"
            if p.exists():
                o = json.loads(p.read_text(encoding="utf-8"))["overall"]
                out["per_seed"][tag] = {"mae": o["mae"], "hit10": o["hit_within_10pct"],
                                        "bias": o["bias"]}
    (TABLE / "temporal_validation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("temporal validation summary -> temporal_validation.json")
    log.info("resnet ensemble: %s", json.dumps(out.get("ensemble_resnet", {}).get("mae")))
    log.info("swin ensemble: %s", json.dumps(out.get("ensemble_swin", {}).get("mae")))
    log.info("paired: %s", json.dumps(out.get("paired_resnet_vs_swin", {})))


if __name__ == "__main__":
    main()
