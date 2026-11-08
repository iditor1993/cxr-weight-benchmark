"""Freeze-ready cohort numbers for the two label definitions and window sensitivity."""

import json

import duckdb
import pandas as pd

DB = r"E:\clinical_research\MIMIC_IV_3.1\mimic_iv_3_1.duckdb"
con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA threads=8")


def q(sql):
    df = con.execute(sql).fetchdf()
    with pd.option_context("display.max_columns", 40, "display.width", 220):
        print(df.to_string())
    return df


out = {}

con.execute("""
CREATE OR REPLACE TEMP TABLE w AS
SELECT stay_id, subject_id, charttime, itemid,
       CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END AS kg
FROM main.chartevents
WHERE itemid IN (224639, 226512, 226531, 226846) AND valuenum IS NOT NULL
  AND (CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END) BETWEEN 20 AND 350
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE pairs AS
SELECT c.subject_id, c.stay_id, c.study_id, c.dicom_id, c.study_datetime,
       w.charttime, w.kg, w.itemid,
       abs(epoch(c.study_datetime - w.charttime))/3600.0 AS gap_h
FROM (SELECT i.subject_id, i.stay_id, m.study_id, m.dicom_id, m.study_datetime
      FROM main.icustays i
      JOIN main.cxr_metadata m
        ON m.subject_id = i.subject_id AND m.study_datetime BETWEEN i.intime AND i.outtime
      WHERE m.view_position='AP' AND m.procedure_desc ILIKE '%PORTABLE%') c
JOIN w ON w.stay_id = c.stay_id
 AND w.charttime BETWEEN c.study_datetime - INTERVAL 48 HOUR AND c.study_datetime + INTERVAL 48 HOUR
QUALIFY row_number() OVER (PARTITION BY c.dicom_id ORDER BY abs(epoch(c.study_datetime - w.charttime))) = 1
""")

print("== window sensitivity, all weight sources ==")
out["all_windows"] = q("""
SELECT w AS window_h,
       count(*) AS n_images, count(DISTINCT stay_id) AS n_stays, count(DISTINCT subject_id) AS n_patients
FROM (SELECT *, CASE WHEN gap_h<=6 THEN 6 WHEN gap_h<=12 THEN 12 WHEN gap_h<=24 THEN 24 ELSE 48 END AS w FROM pairs)
GROUP BY 1 ORDER BY 1
""").to_dict("records")

print("\n== cohort A: daily weight (224639) anchored, by window ==")
out["daily_windows"] = q("""
SELECT CASE WHEN gap_h<=6 THEN 6 WHEN gap_h<=12 THEN 12 WHEN gap_h<=24 THEN 24 ELSE 48 END AS window_h,
       count(*) AS n_images, count(DISTINCT stay_id) AS n_stays, count(DISTINCT subject_id) AS n_patients
FROM pairs WHERE itemid = 224639
GROUP BY 1 ORDER BY 1
""").to_dict("records")

print("\n== cohort B: admission weight (226512/226531) anchored, by window ==")
out["admission_windows"] = q("""
SELECT CASE WHEN gap_h<=6 THEN 6 WHEN gap_h<=12 THEN 12 WHEN gap_h<=24 THEN 24 ELSE 48 END AS window_h,
       count(*) AS n_images, count(DISTINCT stay_id) AS n_stays, count(DISTINCT subject_id) AS n_patients
FROM pairs WHERE itemid IN (226512, 226531)
GROUP BY 1 ORDER BY 1
""").to_dict("records")

print("\n== cohort A index-image design: one image per stay (earliest) ==")
out["cohort_a_index"] = q("""
WITH idx AS (
  SELECT *, row_number() OVER (PARTITION BY stay_id ORDER BY study_datetime, dicom_id) AS rn
  FROM pairs WHERE itemid = 224639 AND gap_h <= 24)
SELECT count(*) AS n_index_images, count(DISTINCT stay_id) AS n_stays,
       count(DISTINCT subject_id) AS n_patients,
       median(kg) AS med_kg, quantile_cont(kg,0.05) AS p05, quantile_cont(kg,0.95) AS p95,
       sum(CASE WHEN kg < 50 THEN 1 ELSE 0 END) AS n_lt50,
       sum(CASE WHEN kg >= 100 THEN 1 ELSE 0 END) AS n_ge100
FROM idx WHERE rn = 1
""").to_dict("records")

print("\n== cohort A: patient-level split feasibility (70/15/15) ==")
out["cohort_a_split"] = q("""
WITH idx AS (
  SELECT DISTINCT subject_id FROM pairs WHERE itemid = 224639 AND gap_h <= 24),
t AS (SELECT subject_id, row_number() OVER (ORDER BY hash(subject_id)) AS rn, count(*) OVER () AS n FROM idx)
SELECT sum(CASE WHEN rn <= 0.7*n THEN 1 ELSE 0 END) AS train_pt,
       sum(CASE WHEN rn > 0.7*n AND rn <= 0.85*n THEN 1 ELSE 0 END) AS val_pt,
       sum(CASE WHEN rn > 0.85*n THEN 1 ELSE 0 END) AS test_pt,
       max(n) AS total_pt
FROM t
""").to_dict("records")

print("\n== cohort A: longitudinal (>=2 daily-anchored images per stay) ==")
out["cohort_a_longitudinal"] = q("""
WITH s AS (
  SELECT stay_id, count(DISTINCT dicom_id) AS n_img, max(kg)-min(kg) AS rng
  FROM pairs WHERE itemid = 224639 AND gap_h <= 24 GROUP BY 1)
SELECT count(*) AS n_stays_multi,
       sum(CASE WHEN rng >= 1 THEN 1 ELSE 0 END) AS rng_ge1,
       sum(CASE WHEN rng >= 3 THEN 1 ELSE 0 END) AS rng_ge3,
       sum(CASE WHEN rng >= 5 THEN 1 ELSE 0 END) AS rng_ge5,
       median(rng) AS med_rng
FROM s WHERE n_img >= 2
""").to_dict("records")

print("\n== cohort B index-image design (admission weight anchor, <=24h) ==")
out["cohort_b_index"] = q("""
WITH idx AS (
  SELECT *, row_number() OVER (PARTITION BY stay_id ORDER BY study_datetime, dicom_id) AS rn
  FROM pairs WHERE itemid IN (226512,226531) AND gap_h <= 24)
SELECT count(*) AS n_index_images, count(DISTINCT stay_id) AS n_stays,
       count(DISTINCT subject_id) AS n_patients,
       median(kg) AS med_kg, quantile_cont(kg,0.05) AS p05, quantile_cont(kg,0.95) AS p95
FROM idx WHERE rn = 1
""").to_dict("records")

with open("lit_search/cohort_definition.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("\nsaved lit_search/cohort_definition.json")
