"""在"入院体重不可靠"亚组内做配对显著性检验（患者级聚类 bootstrap + Wilcoxon）。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import PROCESSED_DIR, TABLE_DIR, save_json, setup_logger
from metrics import bootstrap_ci_arrays


def cluster_boot_diff(label, a, b, subjects, n_boot=1000, seed=20261011):
    """配对差 a-b 的患者级聚类 bootstrap：返回均值与 95% CI。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"s": subjects, "d": np.abs(label - a) - np.abs(label - b)})
    groups = [g["d"].values for _, g in df.groupby("s", sort=False)]
    stats = np.array([np.concatenate([groups[i] for i in
                                      rng.integers(0, len(groups), len(groups))]).mean()
                      for _ in range(n_boot)])
    return float(df["d"].mean()), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def main() -> None:
    log = setup_logger("subgroup_paired_tests")
    df = pd.read_csv(PROCESSED_DIR / "unreliable_subgroup_check_table.csv")
    label = df["weight_kg"].values.astype(float)
    adm = df["admission_weight_kg"].values.astype(float)
    mask = np.isfinite(adm) & (np.abs(adm - label) >= 5)
    log.info("subgroup n=%d", int(mask.sum()))
    comps = {
        "入院体重（原始）": "admission_weight_kg",
        "入院体重（训练集校准）": "calibrated_adm",
        "最优收缩估计": "shrink_opt",
        "人群均值": "population_mean",
        "沿用既往记录": "charted_last_weight_kg",
        "门诊体重(OMR)": "omr_weight_kg",
        "Devine IBW": "Devine IBW",
    }
    out = {}
    for name, col in comps.items():
        v = df[col].values.astype(float)
        ok = mask & np.isfinite(v)
        diff = np.abs(label[ok] - df["dl"].values[ok]) - np.abs(label[ok] - v[ok])
        mean, lo, hi = cluster_boot_diff(label[ok], df["dl"].values[ok], v[ok],
                                         df["subject_id"].values[ok])
        try:
            p = float(wilcoxon(np.abs(label[ok] - df["dl"].values[ok]),
                               np.abs(label[ok] - v[ok])).pvalue)
        except Exception:  # noqa: BLE001
            p = None
        out[name] = {"n": int(ok.sum()), "dl_mae": float(np.mean(np.abs(label[ok] - df['dl'].values[ok]))),
                     "comp_mae": float(np.mean(np.abs(label[ok] - v[ok]))),
                     "paired_diff_dl_minus_comp": mean, "ci95": [lo, hi], "wilcoxon_p": p}
        log.info("%-20s n=%3d DL=%5.2f comp=%5.2f diff=%+5.2f [%+5.2f,%+5.2f] p=%s",
                 name, out[name]["n"], out[name]["dl_mae"], out[name]["comp_mae"],
                 mean, lo, hi, f"{p:.2e}" if p is not None else "NA")
    save_json(TABLE_DIR / "subgroup_paired_tests.json", out)


if __name__ == "__main__":
    main()
