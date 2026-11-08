"""标签粗错敏感性分析。

思路：索引体重标签无法直接验证真值，但可用同一次住院、时间上独立的另一条体重记录
（±6 h，至少相隔 30 min）做一致性核对，并把标签分为：
  corroborated   有独立记录在 2 kg 内吻合，且没有 ≥5 kg 的矛盾记录
  equivocal      只有 2–5 kg 的差异记录
  contradicted   存在 ≥5 kg 的矛盾记录且没有 2 kg 内吻合
  unverifiable   6 h 内没有独立记录

随后在测试集上比较各层的模型误差，并复算主结论（记录可靠性 / BMI / 液体平衡）的敏感性。
"""

from __future__ import annotations

import json

import duckdb
import numpy as np

import pandas as pd

from common import (FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, load_config, main_pred_file,
                    save_json, setup_logger)


def build_flags(con, log) -> pd.DataFrame:
    p = str(PROCESSED_DIR / "cohort_a_index.csv").replace("\\", "/")
    pairs = con.execute(
        """
        WITH lab AS (
            SELECT stay_id, weight_charttime AS t0, weight_kg AS w0
            FROM read_csv_auto(?)
        ), w AS (
            SELECT c.stay_id, c.itemid, c.charttime,
                   CASE WHEN c.itemid = 226531 THEN c.valuenum / 2.2046226218
                        ELSE c.valuenum END AS kg
            FROM main.chartevents c JOIN lab USING (stay_id)
            WHERE c.itemid IN (224639, 226512, 226531) AND c.valuenum IS NOT NULL
        )
        SELECT lab.stay_id, lab.w0, w.itemid, w.kg,
               abs(date_diff('minute', w.charttime, lab.t0)) / 60.0 AS dt_h
        FROM lab JOIN w USING (stay_id)
        WHERE abs(date_diff('minute', w.charttime, lab.t0)) <= 1440
        """, [p]).fetchdf()
    # 独立记录：时间上至少相隔 30 min（排除同一时刻的重复/复制记录）
    pairs = pairs[(pairs["dt_h"] >= 0.5) & (pairs["kg"].between(20, 350))].copy()
    pairs["absdiff"] = (pairs["kg"] - pairs["w0"]).abs()
    log.info("independent comparison records: %d over %d stays", len(pairs),
             pairs["stay_id"].nunique())

    rows = []
    for stay, g in pairs.groupby("stay_id", sort=False):
        g6 = g[g["dt_h"] <= 6]
        rows.append({
            "stay_id": stay,
            "n_other_6h": int(len(g6)),
            "n_other_24h": int(len(g)),
            "min_absdiff_6h": float(g6["absdiff"].min()) if len(g6) else np.nan,
            "max_absdiff_6h": float(g6["absdiff"].max()) if len(g6) else np.nan,
            "any_within_2kg": bool((g6["absdiff"] <= 2).any()),
            "any_beyond_5kg": bool((g6["absdiff"] >= 5).any()),
        })
    flags = pd.DataFrame(rows)
    cohort = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")[["stay_id", "subject_id", "weight_kg"]]
    flags = cohort.merge(flags, on="stay_id", how="left")
    flags[["n_other_6h", "n_other_24h"]] = flags[["n_other_6h", "n_other_24h"]].fillna(0).astype(int)

    def classify(r):
        if r["n_other_6h"] == 0:
            return "unverifiable"
        if r["any_beyond_5kg"] and not r["any_within_2kg"]:
            return "contradicted"
        if r["any_within_2kg"] and not r["any_beyond_5kg"]:
            return "corroborated"
        return "equivocal"

    flags["label_class"] = flags.apply(classify, axis=1)
    return flags


def main() -> None:
    log = setup_logger("label_quality_analysis")
    cfg = load_config()
    con = duckdb.connect(cfg["data"]["duckdb"], read_only=True)
    flags = build_flags(con, log)
    flags.to_csv(PROCESSED_DIR / "label_quality_flags.csv", index=False)
    dist = flags["label_class"].value_counts().to_dict()
    log.info("label classes: %s", dist)

    pred = pd.read_csv(main_pred_file())
    conv = pd.read_csv(PROCESSED_DIR / "conventional_weight_comparison_table.csv")[
        ["stay_id", "admission_weight_kg", "height_final_cm"]]
    test = pred.merge(conv, on="stay_id", how="left")
    test = test.merge(flags[["stay_id", "label_class", "n_other_6h", "min_absdiff_6h"]],
                      on="stay_id", how="left")
    h = test["height_final_cm"].values.astype(float)
    male = (test["gender"].values == "M").astype(int)
    test["ols_height_sex"] = np.where(np.isfinite(h),
                                      -69.9350080268858 + 0.9196542642028203 * h
                                      + 0.04498295084068797 * male, np.nan)
    test["bmi_final"] = np.where(np.isfinite(h), test["weight_kg"].values / (h / 100.0) ** 2, np.nan)
    p = str(PROCESSED_DIR / "test_predictions_resnet50_308_ensemble.csv").replace("\\", "/")
    fb = con.execute(
        """
        WITH base AS (SELECT stay_id, study_datetime FROM read_csv_auto(?)),
        t AS (SELECT b.stay_id, ic.intime AS icu_intime, b.study_datetime
              FROM base b JOIN main.icustays ic ON ic.stay_id = b.stay_id),
        ins AS (SELECT t.stay_id, sum(ie.amount) AS intake FROM t
                JOIN main.inputevents ie ON ie.stay_id = t.stay_id
                  AND ie.starttime BETWEEN t.icu_intime AND t.study_datetime GROUP BY 1),
        outs AS (SELECT t.stay_id, sum(oe.value) AS output FROM t
                 JOIN main.outputevents oe ON oe.stay_id = t.stay_id
                   AND oe.charttime BETWEEN t.icu_intime AND t.study_datetime GROUP BY 1)
        SELECT t.stay_id, (coalesce(ins.intake,0) - coalesce(outs.output,0)) / 1000.0 AS fluid_balance_kg
        FROM t LEFT JOIN ins USING (stay_id) LEFT JOIN outs USING (stay_id)
        """, [p]).fetchdf()
    test = test.merge(fb, on="stay_id", how="left")
    y = test["weight_kg"].values.astype(float)
    dl = test["pred"].values.astype(float)
    adm = test["admission_weight_kg"].values.astype(float)

    def metrics(mask, a=dl, b=None):
        m = mask
        out = {"n": int(m.sum())}
        if m.sum() == 0:
            return out
        out["mae_dl"] = float(np.mean(np.abs(y[m] - a[m])))
        out["bias_dl"] = float(np.mean(a[m] - y[m]))
        out["hit10_dl"] = float(np.mean(np.abs(y[m] - a[m]) <= 0.10 * y[m]))
        if b is not None:
            out["mae_record"] = float(np.mean(np.abs(y[m] - b[m])))
        return out

    classes = ["corroborated", "equivocal", "contradicted", "unverifiable"]
    by_class = {}
    total_abs = float(np.abs(y - dl).sum())
    for c in classes:
        m = (test["label_class"] == c).values
        by_class[c] = metrics(m, dl, adm)
        if m.sum():
            by_class[c]["share_of_total_abs_error"] = float(np.abs(y[m] - dl[m]).sum() / total_abs)
    high = (test["label_class"] == "corroborated").values
    overall = metrics(np.ones(len(test), dtype=bool), dl, adm)
    overall["mae_regression"] = float(np.mean(np.abs(y - test["ols_height_sex"].values)))
    high_metrics = metrics(high, dl, adm)
    high_metrics["mae_regression"] = float(np.mean(
        np.abs(y[high] - test["ols_height_sex"].values[high])))

    # 敏感性：记录可靠性 / BMI / 液体平衡 的主对比在 corroborated 子集内是否保持
    def interaction(mask, group_a, group_b, comparator="admission_weight_kg"):
        comp = test[comparator].values.astype(float)
        d = np.abs(y - dl) - np.abs(y - comp)
        ma = mask & group_a & np.isfinite(comp)
        mb = mask & group_b & np.isfinite(comp)
        if ma.sum() < 20 or mb.sum() < 20:
            return None
        return {"delta_diff": float(d[ma].mean() - d[mb].mean()),
                "n_a": int(ma.sum()), "n_b": int(mb.sum())}

    adm_disc = np.abs(adm - y)
    bmi = test["bmi_final"].values.astype(float)
    fluid = test["fluid_balance_kg"].values.astype(float)
    sens = {
        "record_reliability_ge5_vs_lt2": interaction(high, adm_disc >= 5, adm_disc < 2),
        "bmi_ge40_vs_18_5_25": interaction(high, bmi >= 40, (bmi >= 18.5) & (bmi < 25)),
        "fluid_balance_q4_vs_q1": None,
    }
    if np.isfinite(fluid).sum() > 100:
        q1, q4 = np.nanquantile(fluid, [0.25, 0.75])
        sens["fluid_balance_q4_vs_q1"] = interaction(high, fluid >= q4, fluid <= q1)

    # 排除"矛盾标签"的敏感性（注意：这是选择效应，不是真正的标签清洗）
    excl = {}
    for name, drop in (("drop_contradicted", ["contradicted"]),
                       ("drop_contradicted_and_equivocal", ["contradicted", "equivocal"])):
        keep = ~test["label_class"].isin(drop).values
        m = metrics(keep, dl, adm)
        m["record_reliability_ge5_vs_lt2"] = interaction(
            keep, adm_disc >= 5, adm_disc < 2)
        m["bmi_ge40_vs_18_5_25"] = interaction(
            keep, bmi >= 40, (bmi >= 18.5) & (bmi < 25))
        if np.isfinite(fluid).sum() > 100:
            m["fluid_balance_q4_vs_q1"] = interaction(keep, fluid >= q4, fluid <= q1)
        excl[name] = m

    # copy-forward 证据：被"证实"的标签里有多少是数值完全相同的复制记录
    pairs_all = con.execute(
        """
        WITH lab AS (SELECT stay_id, weight_charttime AS t0, weight_kg AS w0
                     FROM read_csv_auto(?)),
        w AS (SELECT c.stay_id, c.itemid, c.charttime,
                CASE WHEN c.itemid=226531 THEN c.valuenum/2.2046226218 ELSE c.valuenum END AS kg
              FROM main.chartevents c JOIN lab USING (stay_id)
              WHERE c.itemid IN (224639,226512,226531) AND c.valuenum IS NOT NULL)
        SELECT lab.stay_id, lab.w0, w.kg,
               abs(date_diff('minute', w.charttime, lab.t0))/60.0 AS dt_h
        FROM lab JOIN w USING (stay_id)
        WHERE abs(date_diff('minute', w.charttime, lab.t0)) <= 360
        """, [p]).fetchdf()
    pairs_all = pairs_all[(pairs_all.dt_h >= 0.5) & (pairs_all.kg.between(20, 350))]
    pairs_all["absdiff"] = (pairs_all.kg - pairs_all.w0).abs()
    best = (pairs_all.sort_values(["stay_id", "absdiff", "dt_h"])
            .groupby("stay_id").first().reset_index())
    corr = best[best.absdiff <= 2]
    copy = {"n_corroborated": int(len(corr)),
            "share_exact_same_value": float((corr.absdiff == 0).mean()) if len(corr) else None,
            "share_within_0_2kg": float((corr.absdiff <= 0.2).mean()) if len(corr) else None,
            "median_gap_h": float(corr.dt_h.median()) if len(corr) else None,
            "share_same_value_within_2h": float(((corr.absdiff == 0) & (corr.dt_h < 2)).mean())
            if len(corr) else None}

    # 选择偏倚检查：各层的基础特征
    sel = {}
    for c in classes:
        m = (test["label_class"] == c).values
        if m.sum() == 0:
            continue
        sel[c] = {"n": int(m.sum()), "weight_median": float(np.median(y[m])),
                  "age_median": float(np.nanmedian(test["anchor_age"].values[m])),
                  "male_pct": float(np.mean(test["gender"].values[m] == "M")),
                  "ventilated_pct": float(np.nanmean(test["ventilated"].values[m] == 1)),
                  "mae_dl": by_class[c]["mae_dl"]}

    payload = {"n_cohort": int(len(flags)), "n_test": int(len(test)),
               "label_class_distribution_cohort": {k: int(v) for k, v in dist.items()},
               "label_class_distribution_test": {c: int((test["label_class"] == c).sum())
                                                 for c in classes},
               "overall_test": overall, "corroborated_subset": high_metrics,
               "by_class": by_class, "sensitivity_interactions": sens,
               "exclusion_sensitivity": excl, "copy_forward_evidence": copy,
               "selection_check": sel,
               "note": ("corroborated requires an independent record within 2 kg at <=6 h; "
                        "contradicted requires a >=5 kg disagreement and no <=2 kg agreement. "
                        "Most corroborating records are exact copies, so corroboration is not "
                        "independent verification; exclusion analyses are selection effects.")}
    save_json(TABLE_DIR / "label_quality_analysis.json", payload)
    for c in classes:
        if c in by_class:
            log.info("%-13s n=%4d MAE=%.2f (record %.2f) share=%.1f%%",
                     c, by_class[c]["n"], by_class[c].get("mae_dl", float("nan")),
                     by_class[c].get("mae_record", float("nan")),
                     100 * by_class[c].get("share_of_total_abs_error", 0))
    log.info("overall MAE %.2f -> corroborated subset %.2f (n=%d)", overall["mae_dl"],
             high_metrics["mae_dl"], high_metrics["n"])
    log.info("sensitivity: %s", json.dumps(sens, ensure_ascii=False))
    log.info("copy-forward: %s", json.dumps(copy, ensure_ascii=False))
    for k, v in excl.items():
        log.info("%s: n=%d MAE=%.2f (overall %.2f)", k, v["n"], v["mae_dl"], overall["mae_dl"])


if __name__ == "__main__":
    main()
