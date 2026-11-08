"""深度学习 vs 各非测量类估算方法：配对差异的 95% CI、Wilcoxon p 值与 FDR 校正。

配对单位为患者级（同一 stay 的 |误差| 之差），CI 用患者级聚类 bootstrap；
p 值用双侧 Wilcoxon 符号秩检验；多重比较用 Benjamini–Hochberg 校正。

产出：project/results/tables/paired_significance.json
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import PROCESSED_DIR, TABLE_DIR, save_json, setup_logger

LABELS = {
    "admission_weight_kg": "Nurse-recorded admission weight",
    "charted_last_weight_kg": "Most recent previous weight",
    "omr_weight_kg": "Outpatient weight",
    "OMR BMI x height²": "Outpatient BMI × height²",
    "BMI=25 x height²": "Fixed BMI 25 × height²",
    "Miller IBW": "Miller IBW", "Hamwi IBW": "Hamwi IBW", "Robinson IBW": "Robinson IBW",
    "Hume IBW / ARDSNet PBW": "Hume IBW / ARDSNet PBW", "Devine IBW": "Devine IBW",
    "sex-specific mean (train)": "Sex-specific mean",
    "population mean (train)": "Population mean",
    "ols_height_sex": "Height-and-sex regression",
}


def stars(q: float) -> str:
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


def boot_ci(diff: np.ndarray, subjects: np.ndarray, n_boot: int = 2000, seed: int = 20261012):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(subjects, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    cnts = np.bincount(inv)
    k = len(uniq)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        out[b] = sums[idx].sum() / cnts[idx].sum()
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def main() -> None:
    log = setup_logger("paired_significance")
    conv = pd.read_csv(PROCESSED_DIR / "conventional_weight_comparison_table.csv")
    conv["ols_height_sex"] = (-69.9350080268858 + 0.9196542642028203 * conv["height_final_cm"]
                              + 0.04498295084068797 * (conv["gender"] == "M").astype(float))
    y = conv["weight_kg"].values
    dl = conv["pred"].values
    subj = conv["subject_id"].values

    rows = []
    for key, label in LABELS.items():
        p = conv[key].values.astype(float)
        m = np.isfinite(p) & np.isfinite(y) & np.isfinite(dl)
        if m.sum() < 20:
            continue
        ae_other, ae_dl = np.abs(p[m] - y[m]), np.abs(dl[m] - y[m])
        diff = ae_dl - ae_other                      # 负 = 深度学习更好
        lo, hi = boot_ci(diff, subj[m])
        try:
            stat, pv = wilcoxon(ae_dl, ae_other)
        except ValueError:
            stat, pv = float("nan"), float("nan")
        rows.append({"method": label, "n": int(m.sum()),
                     "mae_method": float(ae_other.mean()), "mae_dl": float(ae_dl.mean()),
                     "paired_diff": float(diff.mean()), "ci_low": lo, "ci_high": hi,
                     "p_value": float(pv), "wilcoxon_stat": float(stat),
                     "method_better": bool(diff.mean() > 0)})

    # Benjamini–Hochberg 校正
    ps = np.array([r["p_value"] for r in rows])
    order = np.argsort(ps)
    q = np.empty_like(ps)
    m_tests = len(ps)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        rank_pos = m_tests - rank + 1
        val = min(prev, ps[idx] * m_tests / rank_pos)
        q[idx] = val
        prev = val
    for r, qv in zip(rows, q):
        r["q_value"] = float(qv)
        r["significance"] = stars(float(qv))

    rows.sort(key=lambda r: r["paired_diff"])
    save_json(TABLE_DIR / "paired_significance.json", {"n_comparators": len(rows), "rows": rows})
    for r in rows:
        log.info("%-32s n=%4d 配对差 %+6.2f [%+.2f, %+.2f] p=%.2e q=%.2e %s",
                 r["method"], r["n"], r["paired_diff"], r["ci_low"], r["ci_high"],
                 r["p_value"], r["q_value"], r["significance"])


if __name__ == "__main__":
    main()
