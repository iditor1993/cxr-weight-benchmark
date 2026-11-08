"""在 confirm_val（原验证集）上独立确认 Phase 1 的 L1 vs Huber 信号。

这两个配置的模型都只用 inner_val 做早停与选择，confirm_val 未参与本轮训练或选择，
因此可作为一次性独立确认。测试集仍然封存。
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
SEEDS = (11, 12)


def run_predict(tag: str) -> Path:
    out = f"confirm_{tag}.csv"
    cmd = [PY, str(SRC / "predict_split.py"), "--split", "confirm_val",
           "--split-file", "splits_inner.csv", "--split-col", "split_inner",
           "--backbone", "resnet50", "--size", "308",
           "--checkpoints", str(CKPT / f"{tag}.pt"), "--out", out]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    return PROC / out


def main() -> None:
    rows = {}
    for loss in ("huber", "l1"):
        for aug in ("default", "geom_off"):
            for seed in SEEDS:
                tag = f"p1_{loss}_{aug}_s{seed}"
                p = run_predict(tag)
                d = pd.read_csv(p)
                rows[(loss, aug, seed)] = d
    res = {}
    for aug in ("default", "geom_off"):
        per = {}
        for loss in ("huber", "l1"):
            per[loss] = {s: float(np.mean(np.abs(rows[(loss, aug, s)]["weight_kg"].values
                                                 - rows[(loss, aug, s)]["pred"].values)))
                         for s in SEEDS}
        # paired (same seed) difference, pooled and clustered by patient
        diffs, subj = [], []
        for s in SEEDS:
            h = rows[("huber", aug, s)]
            l = rows[("l1", aug, s)].set_index("stay_id").loc[h["stay_id"]].reset_index()
            y = h["weight_kg"].values
            diffs.append(np.abs(y - h["pred"].values) - np.abs(y - l["pred"].values))
            subj.append(h["subject_id"].values)
        d = np.concatenate(diffs)
        s = np.concatenate(subj)
        uniq, inv = np.unique(s, return_inverse=True)
        sums = np.bincount(inv, weights=d)
        cnts = np.bincount(inv)
        k = len(cnts)
        rng = np.random.default_rng(20261027)
        idx = rng.integers(0, k, size=(4000, k))
        boot = sums[idx].sum(1) / cnts[idx].sum(1)
        res[aug] = {
            "huber_per_seed": per["huber"], "l1_per_seed": per["l1"],
            "huber_mean": float(np.mean(list(per["huber"].values()))),
            "l1_mean": float(np.mean(list(per["l1"].values()))),
            "paired_gain_kg": float(d.mean()),
            "paired_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "p_gain_above_zero": float((boot > 0).mean()),
        }
    out = {"split": "confirm_val (n=1362), untouched by Phase 1 training/selection",
           "test": "sealed", "results": res}
    (TABLE / "phase1_confirm_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for aug, r in res.items():
        print(f"{aug}: huber={r['huber_mean']:.3f} l1={r['l1_mean']:.3f} "
              f"paired gain={r['paired_gain_kg']:+.3f} CI={r['paired_ci95']} "
              f"P(gain>0)={r['p_gain_above_zero']:.3f}")


if __name__ == "__main__":
    main()
