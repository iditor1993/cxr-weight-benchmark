"""常规体重估算方法 vs 胸片深度学习：同一测试集（n=1,303）配对比较。

常规方法（均可由 MIMIC-IV 计算，不含直接称重）：
  1) 入院体重（护士录入，临床默认）
  2) 既往体重（最近一次历史体重记录，chartevents）
  3) 最近门诊体重（OMR）
  4) 既往或门诊最新记录（"沿用旧记录"最佳实践）
  5-9) IBW 公式：Devine / Hume(=ARDSNet PBW) / Robinson / Miller / Hamwi
  10) BMI 假设法：BMI=25 × height²
  11) 门诊 BMI × height²
  12-13) 性别均值 / 人群均值（训练集估计）
深度学习：Swin-T 3 种子集成（pred）
参考标准：Daily Weight（床秤，kg）
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from common import PROCESSED_DIR, TABLE_DIR, load_config, main_pred_file, save_json, setup_logger
from metrics import bootstrap_ci_arrays, regression_metrics


def ibw_formulas(df: pd.DataFrame) -> dict:
    cm = df["height_cm"].values.astype(float)
    male = (df["gender"].values == "M")
    inch = cm / 2.54
    out = {}
    out["Devine IBW"] = np.where(male, 50 + 2.3 * (inch - 60), 45.5 + 2.3 * (inch - 60))
    out["Hume IBW / ARDSNet PBW"] = np.where(male, 50 + 0.91 * (cm - 152.4),
                                              45.5 + 0.91 * (cm - 152.4))
    out["Robinson IBW"] = np.where(male, 52 + 1.9 * (inch - 60), 49 + 1.7 * (inch - 60))
    out["Miller IBW"] = np.where(male, 56.2 + 1.41 * (inch - 60), 53.1 + 1.41 * (inch - 60))
    out["Hamwi IBW"] = np.where(male, 48 + 2.7 * (inch - 60), 45.5 + 2.2 * (inch - 60))
    out["BMI=25 x height²"] = 25.0 * (cm / 100.0) ** 2
    return out


def main() -> None:
    log = setup_logger("conventional_weight_comparison")
    cfg = load_config()
    con = duckdb.connect(cfg["data"]["duckdb"], read_only=True)
    con.execute("PRAGMA threads=8")

    test = pd.read_csv(main_pred_file())
    test = test[["subject_id", "hadm_id", "stay_id", "study_datetime", "weight_kg",
                 "pred", "gender", "anchor_age", "height_cm"]].copy()
    test.to_csv(PROCESSED_DIR / "conv_test_base.csv", index=False)

    # ---------- 入院体重与既往体重（chartevents） ----------
    chart = con.execute("""
        WITH w AS (
            SELECT subject_id, stay_id, charttime, itemid,
                   CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END AS kg
            FROM main.chartevents
            WHERE itemid IN (224639, 226512, 226531) AND valuenum IS NOT NULL
        )
        SELECT s.stay_id,
               (SELECT avg(w.kg) FROM w
                 WHERE w.stay_id = s.stay_id AND w.itemid IN (226512, 226531)
                   AND w.kg BETWEEN 20 AND 350) AS admission_weight_kg,
               (SELECT w.kg FROM w
                 WHERE w.subject_id = s.subject_id AND w.charttime < s.intime
                   AND (w.stay_id IS NULL OR w.stay_id <> s.stay_id) AND w.kg BETWEEN 20 AND 350
                 ORDER BY w.charttime DESC LIMIT 1) AS prior_weight_kg,
               (SELECT w.charttime FROM w
                 WHERE w.subject_id = s.subject_id AND w.charttime < s.intime
                   AND (w.stay_id IS NULL OR w.stay_id <> s.stay_id) AND w.kg BETWEEN 20 AND 350
                 ORDER BY w.charttime DESC LIMIT 1) AS prior_weight_time,
               s.intime
        FROM main.icustays s
        WHERE s.stay_id IN (SELECT stay_id FROM read_csv_auto(?))
    """, [str(PROCESSED_DIR / "conv_test_base.csv").replace("\\", "/")]).fetchdf()
    test = test.merge(chart, on="stay_id", how="left")

    # ---------- OMR（门诊记录） ----------
    omr = con.execute("""
        SELECT subject_id, chartdate, result_name, try_cast(result_value AS DOUBLE) AS value
        FROM main.omr
        WHERE result_name IN ('Weight (Lbs)', 'Weight', 'BMI (kg/m2)', 'BMI',
                              'Height (Inches)', 'Height')
    """).fetchdf()
    omr["chartdate"] = pd.to_datetime(omr["chartdate"])
    omr = omr.dropna(subset=["value"])
    omr["value_std"] = np.where(omr["result_name"].str.contains("Lbs", na=False), omr["value"] * 0.45359237,
                        np.where(omr["result_name"].str.startswith("BMI"), omr["value"],
                        np.where(omr["result_name"].str.contains("Inches", na=False), omr["value"] * 2.54,
                                 omr["value"])))

    test_ids = set(test["subject_id"].unique())
    omr_sub = omr[omr["subject_id"].isin(test_ids)]
    omr_by = {sid: g.sort_values("chartdate") for sid, g in omr_sub.groupby("subject_id")}

    def latest_omr(row, kinds, lo, hi):
        g = omr_by.get(row["subject_id"])
        if g is None:
            return np.nan, pd.NaT
        sub = g[g["result_name"].isin(kinds) &
                (g["chartdate"] <= row["study_datetime"]) &
                (g["value_std"].between(lo, hi))]
        if len(sub) == 0:
            return np.nan, pd.NaT
        last = sub.iloc[-1]
        return float(last["value_std"]), last["chartdate"]

    rows_log = []
    for _, r in test.iterrows():
        w, wt = latest_omr(r, ["Weight (Lbs)", "Weight"], 20, 350)
        b, bt = latest_omr(r, ["BMI (kg/m2)", "BMI"], 10, 80)
        h, ht = latest_omr(r, ["Height (Inches)", "Height"], 100, 220)
        rows_log.append({"omr_weight_kg": w, "omr_weight_date": wt,
                         "omr_bmi": b, "omr_bmi_date": bt, "omr_height_cm": h,
                         "omr_height_date": ht})
    test = pd.concat([test.reset_index(drop=True), pd.DataFrame(rows_log)], axis=1)

    # 身高补齐：ICU height -> OMR height
    test["height_final_cm"] = test["height_cm"].fillna(test["omr_height_cm"])
    # 沿用旧记录：取"既往住院体重"与"门诊体重"中时间更晚者
    test["prior_weight_time"] = pd.to_datetime(test["prior_weight_time"])
    use_omr = (test["omr_weight_kg"].notna() &
               (test["prior_weight_kg"].isna() |
                (test["omr_weight_date"] > test["prior_weight_time"])))
    test["charted_last_weight_kg"] = np.where(use_omr, test["omr_weight_kg"], test["prior_weight_kg"])

    # ---------- 公式与基线 ----------
    test_for_formula = test.copy()
    test_for_formula["height_cm"] = test_for_formula["height_final_cm"]
    formulas = ibw_formulas(test_for_formula)
    for k, v in formulas.items():
        test[k] = np.where(test["height_final_cm"].notna(), v, np.nan)
    test["OMR BMI x height²"] = np.where(
        test["omr_bmi"].notna() & test["height_final_cm"].notna(),
        test["omr_bmi"] * (test["height_final_cm"] / 100.0) ** 2, np.nan)
    # 训练集均值（避免测试集泄漏）：用 splits_random 的 train 部分
    cohort = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")[["subject_id", "weight_kg"]]
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    train_w = cohort.merge(splits, on="subject_id")
    train_w = train_w[train_w["split_random"] == "train"]["weight_kg"]
    test["population mean (train)"] = float(train_w.mean())
    # 性别均值：直接用 patients 性别与训练集
    sex = con.execute("SELECT subject_id, gender FROM main.patients").fetchdf()
    train_sex = cohort.merge(splits, on="subject_id").merge(sex, on="subject_id")
    train_sex = train_sex[train_sex["split_random"] == "train"]
    m_mean = float(train_sex.loc[train_sex["gender"] == "M", "weight_kg"].mean())
    f_mean = float(train_sex.loc[train_sex["gender"] == "F", "weight_kg"].mean())
    test["sex-specific mean (train)"] = np.where(test["gender"] == "M", m_mean, f_mean)

    methods = ["admission_weight_kg", "charted_last_weight_kg", "omr_weight_kg",
               "Devine IBW", "Hume IBW / ARDSNet PBW", "Robinson IBW", "Miller IBW",
               "Hamwi IBW", "BMI=25 x height²", "OMR BMI x height²",
               "sex-specific mean (train)", "population mean (train)"]
    label = test["weight_kg"].values.astype(float)
    dl = test["pred"].values.astype(float)

    results = {}
    for m in methods:
        v = test[m].values.astype(float)
        ok = np.isfinite(v) & np.isfinite(label)
        if ok.sum() < 30:
            continue
        metrics = regression_metrics(label[ok], v[ok])
        rec = {"n": int(ok.sum()), "mae": metrics["mae"], "rmse": metrics["rmse"],
               "hit_within_10pct": metrics["hit_within_10pct"],
               "hit_within_5kg": metrics["hit_within_5kg"], "bias": metrics["bias"]}
        # 与 DL 配对（同一子集）
        pt, lo, hi = bootstrap_ci_arrays(label[ok], v[ok], test["subject_id"].values[ok], "mae")
        rec["mae_ci95"] = [pt, lo, hi]
        dl_ok = dl[ok]
        mae_dl_sub = float(np.mean(np.abs(label[ok] - dl_ok)))
        diff = np.abs(label[ok] - dl_ok) - np.abs(label[ok] - v[ok])
        rec["dl_mae_same_subset"] = mae_dl_sub
        rec["paired_dl_minus_method"] = float(diff.mean())
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        rec["paired_ci95"] = [float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)]
        results[m] = rec
        log.info("%-28s n=%4d MAE=%6.2f [%5.2f,%5.2f] | DL same-subset MAE=%6.2f | "
                 "paired DL-method=%+6.2f [%+6.2f,%+6.2f]",
                 m, rec["n"], rec["mae"], rec["mae_ci95"][1], rec["mae_ci95"][2],
                 mae_dl_sub, rec["paired_dl_minus_method"], rec["paired_ci95"][0],
                 rec["paired_ci95"][1])

    full = regression_metrics(label, dl)
    results["__DL ensemble (full test set)__"] = {"n": int(len(label)), **full}
    log.info("DL ensemble full test set: n=%d MAE=%.2f hit10=%.1f%%", len(label),
             full["mae"], 100 * full["hit_within_10pct"])

    # 入院体重可靠性分层：|入院-实测| <2 / 2-5 / >=5 kg 时 DL 的表现
    adm = test["admission_weight_kg"].values.astype(float)
    ok = np.isfinite(adm)
    strata = {"adm_discrepancy_lt2kg": np.abs(adm - label) < 2,
              "adm_discrepancy_2to5kg": (np.abs(adm - label) >= 2) & (np.abs(adm - label) < 5),
              "adm_discrepancy_ge5kg": np.abs(adm - label) >= 5}
    strat = {}
    for k, mask in strata.items():
        m = ok & mask
        if m.sum() >= 20:
            strat[k] = {"n": int(m.sum()),
                        "admission_mae": float(np.mean(np.abs(label[m] - adm[m]))),
                        "dl_mae": float(np.mean(np.abs(label[m] - dl[m])))}
    results["admission_weight_strata"] = strat
    log.info("admission strata: %s", json.dumps(strat, ensure_ascii=False))

    save_json(TABLE_DIR / "conventional_weight_comparison.json", results)
    test.to_csv(PROCESSED_DIR / "conventional_weight_comparison_table.csv", index=False)


if __name__ == "__main__":
    main()
