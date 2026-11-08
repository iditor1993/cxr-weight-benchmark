"""Step 6: clinical framing checks - is weight really missing, and what is the formula baseline?"""

import json

import duckdb
import numpy as np
import pandas as pd

DB = r"E:\clinical_research\MIMIC_IV_3.1\mimic_iv_3_1.duckdb"
con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA threads=8")


def q(sql):
    df = con.execute(sql).fetchdf()
    with pd.option_context("display.max_columns", 60, "display.width", 240):
        print(df.to_string())
    return df


out = {}

q("""
CREATE OR REPLACE TEMP TABLE weight_events AS
SELECT stay_id, subject_id, charttime, itemid,
       CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END AS weight_kg
FROM main.chartevents
WHERE itemid IN (224639, 226512, 226531, 226846) AND valuenum IS NOT NULL
  AND (CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END) BETWEEN 20 AND 350
""")

q("""
CREATE OR REPLACE TEMP TABLE core AS
SELECT c.subject_id, c.stay_id, c.study_id, c.dicom_id, c.intime, c.study_datetime,
       c.view_position, c.procedure_desc, w.charttime, w.weight_kg, w.itemid,
       abs(epoch(c.study_datetime - w.charttime))/3600.0 AS gap_h
FROM (SELECT i.subject_id, i.stay_id, i.intime, m.study_id, m.dicom_id, m.study_datetime,
             m.view_position, m.procedure_desc
      FROM main.icustays i
      JOIN main.cxr_metadata m
        ON m.subject_id = i.subject_id AND m.study_datetime BETWEEN i.intime AND i.outtime) c
JOIN weight_events w
  ON w.stay_id = c.stay_id
 AND w.charttime BETWEEN c.study_datetime - INTERVAL 24 HOUR AND c.study_datetime + INTERVAL 24 HOUR
WHERE c.view_position='AP' AND c.procedure_desc ILIKE '%PORTABLE%'
QUALIFY row_number() OVER (PARTITION BY c.dicom_id ORDER BY abs(epoch(c.study_datetime - w.charttime))) = 1
""")

print("== timing: first portable AP CXR vs first charted weight, per stay ==")
out["timing"] = q("""
WITH firsts AS (
  SELECT c.stay_id, c.intime,
         min(c.study_datetime) AS first_cxr,
         min(w.charttime) AS first_weight
  FROM core c
  JOIN weight_events w USING (stay_id)
  GROUP BY 1,2)
SELECT count(*) AS n_stays,
       sum(CASE WHEN first_cxr < first_weight THEN 1 ELSE 0 END) AS cxr_before_first_weight,
       round(100.0*sum(CASE WHEN first_cxr < first_weight THEN 1 ELSE 0 END)/count(*),1) AS pct,
       median(CASE WHEN first_cxr < first_weight
                   THEN epoch(first_weight - first_cxr)/3600.0 END) AS med_hours_earlier,
       sum(CASE WHEN first_cxr < first_weight AND epoch(first_weight-first_cxr)/3600.0 >= 6 THEN 1 ELSE 0 END) AS cxr_ge6h_earlier,
       sum(CASE WHEN first_cxr < first_weight AND epoch(first_weight-first_cxr)/3600.0 >= 12 THEN 1 ELSE 0 END) AS cxr_ge12h_earlier
FROM firsts
""").to_dict("records")

print("\n== which item provides the first charted weight? ==")
out["label_source"] = q("""
WITH fw AS (SELECT stay_id, min(charttime) AS t0 FROM weight_events GROUP BY 1),
first_item AS (
  SELECT w.stay_id, w.itemid
  FROM weight_events w JOIN fw USING (stay_id)
  WHERE w.charttime = fw.t0
  QUALIFY row_number() OVER (PARTITION BY w.stay_id ORDER BY w.itemid) = 1)
SELECT itemid, count(*) AS n_stays, round(100.0*count(*)/sum(count(*)) OVER (),1) AS pct
FROM first_item GROUP BY 1 ORDER BY n_stays DESC
""").to_dict("records")

print("\n== weight volatility within stay ==")
out["volatility"] = q("""
WITH s AS (
  SELECT stay_id, max(weight_kg)-min(weight_kg) AS rng
  FROM weight_events GROUP BY 1)
SELECT count(*) AS n_stays, median(rng) AS med_range,
       quantile_cont(rng,0.75) AS p75, quantile_cont(rng,0.95) AS p95,
       sum(CASE WHEN rng>=10 THEN 1 ELSE 0 END) AS ge10kg
FROM s
""").to_dict("records")

print("\n== formula baseline (Devine IBW) vs actual weight, core cohort with height ==")
df = q("""
WITH core_stay AS (
  SELECT stay_id, subject_id, median(weight_kg) AS actual_kg FROM core GROUP BY 1,2),
sex AS (
  SELECT subject_id, max(CASE WHEN gender='M' THEN 1 ELSE 0 END) AS male
  FROM main.patients GROUP BY 1),
h AS (SELECT stay_id, avg(height) AS height_cm FROM mimiciv_derived.height GROUP BY 1)
SELECT c.subject_id, c.actual_kg, s.male, h.height_cm
FROM core_stay c JOIN sex s USING (subject_id) JOIN h USING (stay_id)
WHERE h.height_cm BETWEEN 120 AND 220
""")
df["height_in"] = df["height_cm"] / 2.54
df["ibw"] = np.where(df["male"] == 1,
                     50 + 2.3 * (df["height_in"] - 60),
                     45.5 + 2.3 * (df["height_in"] - 60))
valid = df[(df["ibw"] > 20) & (df["ibw"] < 200)]
mae = float(np.mean(np.abs(valid["actual_kg"] - valid["ibw"])))
rmse = float(np.sqrt(np.mean((valid["actual_kg"] - valid["ibw"]) ** 2)))
print(f"n={len(valid)}, IBW MAE={mae:.2f} kg, RMSE={rmse:.2f} kg")
out["ibw_baseline"] = {"n": int(len(valid)), "mae_kg": round(mae, 2), "rmse_kg": round(rmse, 2),
                       "median_actual": float(valid["actual_kg"].median())}

with open("lit_search/mimic_probe6.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("\nsaved lit_search/mimic_probe6.json")
