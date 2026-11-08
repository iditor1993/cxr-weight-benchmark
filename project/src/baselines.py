"""基线：人群均值、身高+性别 OLS、Devine/Hume 公式（含患者级聚类 bootstrap）。"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from common import PROCESSED_DIR, TABLE_DIR, load_config, save_json, setup_logger
from metrics import cluster_bootstrap_ci, metric_fn_factory, regression_metrics


def devine_ibw(height_cm: np.ndarray, male: np.ndarray) -> np.ndarray:
    inches = height_cm / 2.54
    return np.where(male == 1, 50.0 + 2.3 * (inches - 60.0), 45.5 + 2.3 * (inches - 60.0))


def hume_ibw(height_cm: np.ndarray, male: np.ndarray) -> np.ndarray:
    return np.where(male == 1, 50.0 + 0.91 * (height_cm - 152.4), 45.5 + 0.91 * (height_cm - 152.4))


def main() -> None:
    log = setup_logger("baselines")
    cfg = load_config()
    cohort = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    df = cohort.merge(splits, on="subject_id", how="inner")

    # 身高与性别（辅助基线用）
    con = duckdb.connect(cfg["data"]["duckdb"], read_only=True)
    height = con.execute("SELECT stay_id, avg(height) AS height_cm FROM mimiciv_derived.height GROUP BY 1").fetchdf()
    sex = con.execute("SELECT subject_id, max(CASE WHEN gender='M' THEN 1 ELSE 0 END) AS male "
                      "FROM main.patients GROUP BY 1").fetchdf()
    df = df.merge(height, on="stay_id", how="left").merge(sex, on="subject_id", how="left")

    train = df[df["split_random"] == "train"]
    test = df[df["split_random"] == "test"].copy()
    out = {}

    # 1) 人群均值
    mean_train = float(train["weight_kg"].mean())
    test["pred_mean"] = mean_train
    out["population_mean"] = regression_metrics(test["weight_kg"].values, test["pred_mean"].values)
    out["population_mean"]["mean_train_kg"] = mean_train

    # 2) 身高+性别 OLS（仅身高可得）
    tr_h = train.dropna(subset=["height_cm", "male"])
    te_h = test.dropna(subset=["height_cm", "male"]).copy()
    X = np.column_stack([np.ones(len(tr_h)), tr_h["height_cm"].values, tr_h["male"].values])
    y = tr_h["weight_kg"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    Xt = np.column_stack([np.ones(len(te_h)), te_h["height_cm"].values, te_h["male"].values])
    te_h["pred_ols"] = Xt @ beta
    out["ols_height_sex"] = regression_metrics(te_h["weight_kg"].values, te_h["pred_ols"].values)
    out["ols_height_sex"]["beta_intercept_height_sex"] = [float(b) for b in beta]
    out["ols_height_sex"]["n_train"] = int(len(tr_h))

    # 3) Devine / Hume（公式，不做拟合）
    te_h["pred_devine"] = devine_ibw(te_h["height_cm"].values, te_h["male"].values)
    te_h["pred_hume"] = hume_ibw(te_h["height_cm"].values, te_h["male"].values)
    out["devine_ibw"] = regression_metrics(te_h["weight_kg"].values, te_h["pred_devine"].values)
    out["hume_ibw"] = regression_metrics(te_h["weight_kg"].values, te_h["pred_hume"].values)

    # 4) 聚类 bootstrap CI（患者级）
    for name, col in (("population_mean", "pred_mean"), ("ols_height_sex", "pred_ols")):
        subset = test if col == "pred_mean" else te_h
        sub = subset[["subject_id", "weight_kg", col]].rename(columns={col: "pred"})
        pt, lo, hi = cluster_bootstrap_ci(sub, metric_fn_factory("mae"))
        out[name]["mae_ci95"] = [pt, lo, hi]
        pt, lo, hi = cluster_bootstrap_ci(sub, metric_fn_factory("hit10pct"))
        out[name]["hit10_ci95"] = [pt, lo, hi]

    save_json(TABLE_DIR / "baselines.json", out)
    te_h.to_csv(PROCESSED_DIR / "baseline_test_predictions.csv", index=False)
    log.info("baselines:\n%s", json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
