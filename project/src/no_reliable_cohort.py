"""构建"无可信体重记录"分析队列。

定义
    目标人群：有床旁 AP 胸片、但拍片 ±24 h 内**没有床秤实测体重**（itemid 224639）的 ICU 住院。
    - 只有护理录入的入院体重（226512/226531）:  n = 4,656（cohort_b \ cohort_a）
    - 连任何体重记录都没有:                     n = 537
    评估子集：在上述人群中，能在拍片 ±96 h 内找到一次床秤实测作为**替代参考**的住院。

产出
    project/data/processed/cohort_nrr_index.csv    分析表（含替代参考、护理记录、身高、既往/门诊记录）
    project/data/processed/image_manifest_nrr.csv  待预处理影像清单（供 preprocess_images.py --set nrr）
    project/data/processed/cohort_nrr_flow.json    人群流程数字
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from common import PROCESSED_DIR, load_config, save_json, setup_logger

REF_MAX_GAP_H = 96.0


def main() -> None:
    log = setup_logger("no_reliable_cohort")
    cfg = load_config()
    con = duckdb.connect(cfg["data"]["duckdb"], read_only=True)
    con.execute("PRAGMA threads=8")

    a = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")
    b = pd.read_csv(PROCESSED_DIR / "cohort_b_index.csv")
    nrr = b[~b["stay_id"].isin(set(a["stay_id"]))].copy()   # cohort_b \ cohort_a
    nrr["nurse_adm_kg"] = nrr["weight_kg"]
    nrr["nurse_itemid"] = nrr["weight_itemid"]
    log.info("B\\A = %d 次住院（仅护理录入体重）", len(nrr))

    p = str((PROCESSED_DIR / "cohort_b_index.csv")).replace("\\", "/")
    nrr["study_dt"] = pd.to_datetime(nrr["study_datetime"])
    con.register("nrr", nrr[["subject_id", "stay_id", "study_datetime"]])

    # 替代参考：拍片前后各 96 h 内最近一次床秤实测（Daily Weight 224639）
    ref = con.execute("""
        WITH w AS (
            SELECT subject_id, stay_id, charttime, valuenum
            FROM main.chartevents
            WHERE itemid = 224639 AND valuenum BETWEEN 20 AND 350),
        j AS (
            SELECT n.stay_id, w.charttime, w.valuenum,
                   date_diff('minute', w.charttime, CAST(n.study_datetime AS TIMESTAMP)) / 60.0 AS gap_h,
                   row_number() OVER (PARTITION BY n.stay_id
                       ORDER BY abs(date_diff('minute', w.charttime,
                                              CAST(n.study_datetime AS TIMESTAMP)))) AS rn,
                   count(1) OVER (PARTITION BY n.stay_id) AS n_daily
            FROM nrr n JOIN w ON w.subject_id = n.subject_id AND w.stay_id = n.stay_id)
        SELECT stay_id, valuenum AS ref_kg, charttime AS ref_charttime, gap_h AS ref_gap_hours,
               n_daily AS n_daily_in_stay
        FROM j WHERE rn = 1
    """).fetchdf()
    nrr = nrr.merge(ref, on="stay_id", how="left")
    # 主分析窗口：替代参考须在拍片 ±96 h 内；更远的实测只作为"更长窗口"敏感性
    too_far = nrr["ref_gap_hours"].abs() > REF_MAX_GAP_H
    n_too_far = int(too_far.sum())
    nrr.loc[too_far, ["ref_kg", "ref_charttime", "ref_gap_hours"]] = np.nan
    log.info("替代参考超出 ±%d h 被排除：%d 例", int(REF_MAX_GAP_H), n_too_far)

    # 护理录入体重、既往体重、住院前置天数
    ctx = con.execute("""
        WITH w AS (
            SELECT subject_id, stay_id, charttime, itemid,
                   CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END AS kg
            FROM main.chartevents
            WHERE itemid IN (224639, 226512, 226531) AND valuenum IS NOT NULL)
        SELECT s.stay_id,
               (SELECT avg(w.kg) FROM w WHERE w.stay_id = s.stay_id
                  AND w.itemid IN (226512, 226531) AND w.kg BETWEEN 20 AND 350) AS adm_chart_kg,
               (SELECT w.kg FROM w WHERE w.subject_id = s.subject_id AND w.charttime < s.intime
                  AND (w.stay_id IS NULL OR w.stay_id <> s.stay_id) AND w.kg BETWEEN 20 AND 350
                  ORDER BY w.charttime DESC LIMIT 1) AS prior_weight_kg,
               (SELECT w.charttime FROM w WHERE w.subject_id = s.subject_id AND w.charttime < s.intime
                  AND (w.stay_id IS NULL OR w.stay_id <> s.stay_id) AND w.kg BETWEEN 20 AND 350
                  ORDER BY w.charttime DESC LIMIT 1) AS prior_weight_time,
               date_diff('hour', a.admittime, s.intime) / 24.0 AS hosp_days_before_icu
        FROM main.icustays s JOIN main.admissions a ON a.hadm_id = s.hadm_id
        WHERE s.stay_id IN (SELECT stay_id FROM read_csv_auto(?))
    """, [p]).fetchdf()
    nrr = nrr.merge(ctx, on="stay_id", how="left")

    height = con.execute(
        "SELECT stay_id, avg(height) AS height_icu FROM mimiciv_derived.height GROUP BY 1").fetchdf()
    sex = con.execute("SELECT subject_id, gender, anchor_age FROM main.patients").fetchdf()
    nrr = nrr.merge(height, on="stay_id", how="left").merge(sex, on="subject_id", how="left")

    # OMR：门诊身高/体重/BMI（用于身高补齐与敏感性分析）
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
    omr_by = {sid: g.sort_values("chartdate") for sid, g in omr.groupby("subject_id")}

    def latest(row, kinds, lo, hi):
        g = omr_by.get(row["subject_id"])
        if g is None:
            return np.nan, pd.NaT
        sub = g[g["result_name"].isin(kinds) & (g["chartdate"] <= row["study_dt"])
                & g["value_std"].between(lo, hi)]
        if len(sub) == 0:
            return np.nan, pd.NaT
        last = sub.iloc[-1]
        return float(last["value_std"]), last["chartdate"]

    rows = []
    for _, r in nrr.iterrows():
        w, wt = latest(r, ["Weight (Lbs)", "Weight"], 20, 350)
        bmi, _ = latest(r, ["BMI (kg/m2)", "BMI"], 10, 80)
        h, _ = latest(r, ["Height (Inches)", "Height"], 100, 220)
        rows.append({"omr_weight_kg": w, "omr_weight_date": wt, "omr_bmi": bmi, "omr_height_cm": h})
    nrr = pd.concat([nrr.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    nrr["height_final_cm"] = nrr["height_icu"].fillna(nrr["omr_height_cm"])

    # flow 数字
    total_stays = 13960
    has_ref = nrr["ref_kg"].notna()
    flow = {
        "n_stays_portable_ap": total_stays,
        "n_cohort_a_measured_24h": int(len(a)),
        "n_no_reliable_measure_record_only": int(len(nrr)),
        "n_no_any_weight_record": 537,
        "n_nrr_with_delayed_reference_96h": int(has_ref.sum()),
        "n_nrr_with_delayed_reference_48h": int((nrr["ref_gap_hours"].abs() <= 48).sum()),
        "n_excluded_reference_beyond_96h": n_too_far,
        "ref_max_gap_hours": REF_MAX_GAP_H,
        "median_ref_gap_hours": float(nrr.loc[has_ref, "ref_gap_hours"].abs().median()),
        "interquartile_ref_gap_hours": [float(nrr.loc[has_ref, "ref_gap_hours"].abs().quantile(q))
                                        for q in (0.25, 0.75)],
        "frac_reference_after_film": float((nrr.loc[has_ref, "ref_gap_hours"] > 0).mean()),
        "n_with_height": int(nrr.loc[has_ref, "height_final_cm"].notna().sum()),
        "n_with_omr_weight": int(nrr.loc[has_ref, "omr_weight_kg"].notna().sum()),
        "n_with_prior_weight": int(nrr.loc[has_ref, "prior_weight_kg"].notna().sum()),
    }
    save_json(PROCESSED_DIR / "cohort_nrr_flow.json", flow)
    log.info("flow: %s", json.dumps(flow, ensure_ascii=False))

    keep = ["subject_id", "hadm_id", "stay_id", "study_id", "dicom_id", "study_datetime",
            "first_careunit", "nurse_adm_kg", "nurse_itemid", "adm_chart_kg", "ref_kg",
            "ref_charttime", "ref_gap_hours", "n_daily_in_stay", "prior_weight_kg",
            "prior_weight_time", "omr_weight_kg", "omr_weight_date", "omr_bmi",
            "omr_height_cm", "height_icu", "height_final_cm", "gender", "anchor_age",
            "hosp_days_before_icu"]
    out = nrr[keep].copy()
    out.to_csv(PROCESSED_DIR / "cohort_nrr_index.csv", index=False)
    log.info("分析表 -> %s（%d 行，其中有替代参考 %d 行）",
             PROCESSED_DIR / "cohort_nrr_index.csv", len(out), int(has_ref.sum()))

    # 影像清单（用于预处理）
    man = pd.read_csv(PROCESSED_DIR / "image_manifest_cohort_b.csv")
    man = man[man["stay_id"].isin(set(nrr.loc[has_ref, "stay_id"]))].copy()
    man.to_csv(PROCESSED_DIR / "image_manifest_nrr.csv", index=False)
    log.info("影像清单 -> %s（%d 张）", PROCESSED_DIR / "image_manifest_nrr.csv", len(man))
    con.close()


if __name__ == "__main__":
    main()
