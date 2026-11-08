"""检验"记录不可靠亚组中深度学习反超"是否只是回归到均值效应。

在 |入院体重 - 实测| >= 5 kg 的亚组内，比较：
  - 入院体重
  - 胸片深度学习
  - 训练集人群均值 / 性别均值
  - 入院体重与人群均值的 50/50 混合（收缩估计）
  - 训练集校准后的入院体重（OLS: 实测 ~ 入院体重）
  - 沿用既往记录、门诊体重、IBW 公式（如有）
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from common import PROCESSED_DIR, TABLE_DIR, load_config, save_json, setup_logger
from metrics import regression_metrics


def main() -> None:
    log = setup_logger("unreliable_subgroup_check")
    cfg = load_config()
    con = duckdb.connect(cfg["data"]["duckdb"], read_only=True)
    con.execute("PRAGMA threads=8")

    df = pd.read_csv(PROCESSED_DIR / "conventional_weight_comparison_table.csv")
    label = df["weight_kg"].values.astype(float)
    adm = df["admission_weight_kg"].values.astype(float)
    dl = df["pred"].values.astype(float)

    # 训练集：单变量校准 实测 ~ 入院体重（在 train 上拟合）
    train_ids = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")[["stay_id", "subject_id"]]
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    train_ids = train_ids.merge(splits, on="subject_id")
    train_ids = train_ids[train_ids["split_random"] == "train"][["stay_id"]]
    con.register("train_ids", train_ids)
    tr = con.execute("""
        WITH w AS (
            SELECT stay_id,
                   CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END AS kg
            FROM main.chartevents
            WHERE itemid IN (226512, 226531) AND valuenum IS NOT NULL)
        SELECT w.stay_id, avg(w.kg) AS admission_weight_kg
        FROM w JOIN train_ids USING (stay_id)
        WHERE w.kg BETWEEN 20 AND 350 GROUP BY 1
    """).fetchdf()
    train_cohort = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")[["stay_id", "weight_kg"]]
    tr = tr.merge(train_cohort, on="stay_id").dropna()
    beta1, beta0 = np.polyfit(tr["admission_weight_kg"].values, tr["weight_kg"].values, 1)
    mean_w = float(tr["weight_kg"].mean())
    log.info("calibration on train: y = %.3f * adm + %.3f (n=%d); train mean=%.1f",
             beta1, beta0, len(tr), mean_w)

    # 训练集最优混合系数 w：min MSE(w*adm + (1-w)*mean)
    a = tr["admission_weight_kg"].values
    y = tr["weight_kg"].values
    num = float(np.mean((a - mean_w) * (y - mean_w)))
    den = float(np.mean((a - mean_w) ** 2))
    w_opt = num / den if den > 0 else 0.0
    log.info("train-optimal shrinkage weight on admission: %.3f", w_opt)

    df["dl"] = dl
    df["population_mean"] = mean_w
    df["shrink_50_50"] = 0.5 * adm + 0.5 * mean_w
    df["shrink_opt"] = w_opt * adm + (1 - w_opt) * mean_w
    df["calibrated_adm"] = beta1 * adm + beta0

    methods = {
        "入院体重": "admission_weight_kg",
        "胸片深度学习": "dl",
        "人群均值(训练集)": "population_mean",
        "入院体重+均值 50/50 混合": "shrink_50_50",
        "训练集最优混合(shrinkage)": "shrink_opt",
        "训练集校准后的入院体重": "calibrated_adm",
        "沿用既往记录": "charted_last_weight_kg",
        "门诊体重(OMR)": "omr_weight_kg",
        "门诊BMI×身高²": "OMR BMI x height²",
        "Devine IBW": "Devine IBW",
        "Hume/ARDSNet PBW": "Hume IBW / ARDSNet PBW",
    }

    masks = {
        "all_test": np.isfinite(adm),
        "adm_discrepancy_ge5kg": np.isfinite(adm) & (np.abs(adm - label) >= 5),
        "adm_discrepancy_ge10kg": np.isfinite(adm) & (np.abs(adm - label) >= 10),
    }
    out = {"calibration": {"beta1": float(beta1), "beta0": float(beta0),
                           "shrink_opt_w": float(w_opt), "train_mean": mean_w,
                           "n_train": int(len(tr))}, "subgroups": {}}
    for gname, mask in masks.items():
        rows = {}
        for name, col in methods.items():
            v = df[col].values.astype(float)
            ok = mask & np.isfinite(v)
            if ok.sum() < 20:
                continue
            m = regression_metrics(label[ok], v[ok])
            rows[name] = {"n": int(ok.sum()), "mae": m["mae"], "rmse": m["rmse"],
                          "hit_within_10pct": m["hit_within_10pct"],
                          "hit_within_5kg": m["hit_within_5kg"], "bias": m["bias"]}
        out["subgroups"][gname] = rows
        log.info("=== %s (n=%d) ===", gname, int(mask.sum()))
        for name, r in sorted(rows.items(), key=lambda kv: kv[1]["mae"]):
            log.info("  %-28s n=%4d MAE=%6.2f  ±10%%=%.1f%%  ±5kg=%.1f%%  bias=%+.2f",
                     name, r["n"], r["mae"], 100 * r["hit_within_10pct"],
                     100 * r["hit_within_5kg"], r["bias"])
    save_json(TABLE_DIR / "unreliable_subgroup_check.json", out)
    df.to_csv(PROCESSED_DIR / "unreliable_subgroup_check_table.csv", index=False)


if __name__ == "__main__":
    main()
