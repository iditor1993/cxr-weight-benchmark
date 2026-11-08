"""Step 3b: cleaned weight events + CXR pairing (portable/AP focus)."""

import json
import time

import duckdb
import pandas as pd

DB = r"E:\clinical_research\MIMIC_IV_3.1\mimic_iv_3_1.duckdb"
con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA threads=8")


def q(sql):
    t0 = time.time()
    df = con.execute(sql).fetchdf()
    print(f"    ({time.time()-t0:.1f}s, {len(df)} rows)")
    with pd.option_context("display.max_columns", 60, "display.width", 240):
        print(df.to_string())
    return df


out = {}

# --- cleaned weight events (kg) ---------------------------------------------
con.execute("""
CREATE OR REPLACE TEMP TABLE weight_events AS
SELECT stay_id, subject_id, charttime, itemid,
       CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END AS weight_kg
FROM main.chartevents
WHERE itemid IN (224639, 226512, 226531, 226846)
  AND valuenum IS NOT NULL
  AND (CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END) BETWEEN 20 AND 350
""")

print("== cleaned weight events ==")
out["cleaned_weight_events"] = q("""
SELECT count(*) AS n_events, count(DISTINCT stay_id) AS n_stays,
       count(DISTINCT subject_id) AS n_subjects,
       min(weight_kg) AS min_kg, median(weight_kg) AS med_kg, max(weight_kg) AS max_kg
FROM weight_events
""").to_dict("records")

# --- ICU-linked CXR table ----------------------------------------------------
con.execute("DROP TABLE IF EXISTS cxr_icu")
con.execute("""
CREATE TEMP TABLE cxr_icu AS
SELECT i.subject_id, i.stay_id, i.intime, i.outtime,
       m.study_id, m.dicom_id, m.study_datetime, m.view_position,
       m.patient_orientation_code_meaning AS orientation,
       m.procedure_desc,
       (m.study_datetime BETWEEN i.intime AND i.outtime) AS inside_stay
FROM main.icustays i
JOIN main.cxr_metadata m
  ON m.subject_id = i.subject_id
 AND m.study_datetime BETWEEN i.intime - INTERVAL 24 HOUR AND i.outtime + INTERVAL 24 HOUR
""")

print("\n== procedure descriptions, ICU-linked images ==")
out["procedure_desc"] = q("""
SELECT procedure_desc, count(*) AS n_images, count(DISTINCT dicom_id) AS n_unique,
       count(DISTINCT stay_id) AS n_stays
FROM cxr_icu GROUP BY 1 ORDER BY n_images DESC LIMIT 20
""").to_dict("records")

print("\n== inside-stay image counts ==")
out["inside_stay"] = q("""
SELECT count(*) AS n_images, count(DISTINCT dicom_id) AS n_unique_images,
       count(DISTINCT stay_id) AS n_stays, count(DISTINCT subject_id) AS n_patients,
       count(DISTINCT CASE WHEN view_position='AP' THEN dicom_id END) AS n_ap,
       count(DISTINCT CASE WHEN procedure_desc ILIKE '%PORTABLE%' THEN dicom_id END) AS n_portable,
       count(DISTINCT CASE WHEN view_position='AP' AND procedure_desc ILIKE '%PORTABLE%' THEN dicom_id END) AS n_ap_portable
FROM cxr_icu WHERE inside_stay
""").to_dict("records")

# --- pairing -----------------------------------------------------------------
con.execute("DROP TABLE IF EXISTS cxr_weight")
con.execute("""
CREATE TEMP TABLE cxr_weight AS
SELECT c.subject_id, c.stay_id, c.study_id, c.dicom_id, c.study_datetime,
       c.view_position, c.orientation, c.procedure_desc, c.inside_stay,
       w.charttime, w.weight_kg, w.itemid,
       abs(epoch(c.study_datetime - w.charttime)) / 3600.0 AS gap_h
FROM cxr_icu c
JOIN weight_events w
  ON w.stay_id = c.stay_id
 AND w.charttime BETWEEN c.study_datetime - INTERVAL 48 HOUR AND c.study_datetime + INTERVAL 48 HOUR
QUALIFY row_number() OVER (PARTITION BY c.dicom_id ORDER BY abs(epoch(c.study_datetime - w.charttime))) = 1
""")

print("\n== pairing summary (all ICU-linked images, nearest weight within 48h) ==")
out["pairing_all"] = q("""
SELECT count(*) AS n_images, count(DISTINCT subject_id) AS n_patients,
       count(DISTINCT stay_id) AS n_stays,
       sum(CASE WHEN gap_h<=6 THEN 1 ELSE 0 END) AS le6h,
       sum(CASE WHEN gap_h<=12 THEN 1 ELSE 0 END) AS le12h,
       sum(CASE WHEN gap_h<=24 THEN 1 ELSE 0 END) AS le24h,
       median(gap_h) AS med_gap_h
FROM cxr_weight
""").to_dict("records")

print("\n== pairing restricted to images inside the ICU stay ==")
out["pairing_inside"] = q("""
SELECT
  count(*) AS n_images,
  count(DISTINCT subject_id) AS n_patients,
  count(DISTINCT stay_id) AS n_stays,
  sum(CASE WHEN gap_h<=6 THEN 1 ELSE 0 END) AS le6h,
  sum(CASE WHEN gap_h<=12 THEN 1 ELSE 0 END) AS le12h,
  sum(CASE WHEN gap_h<=24 THEN 1 ELSE 0 END) AS le24h,
  sum(CASE WHEN gap_h<=24 AND view_position='AP' THEN 1 ELSE 0 END) AS ap_le24h,
  sum(CASE WHEN gap_h<=24 AND procedure_desc ILIKE '%PORTABLE%' THEN 1 ELSE 0 END) AS portable_le24h,
  sum(CASE WHEN gap_h<=24 AND view_position='AP' AND procedure_desc ILIKE '%PORTABLE%' THEN 1 ELSE 0 END) AS ap_portable_le24h
FROM cxr_weight WHERE inside_stay
""").to_dict("records")

print("\n== core cohort: AP portable, inside stay, |gap|<=24h ==")
out["core_cohort"] = q("""
SELECT count(*) AS n_images, count(DISTINCT subject_id) AS n_patients,
       count(DISTINCT stay_id) AS n_stays,
       median(weight_kg) AS med_weight, min(weight_kg) AS min_w, max(weight_kg) AS max_w,
       median(gap_h) AS med_gap_h,
       count(DISTINCT CASE WHEN weight_kg < 50 THEN dicom_id END) AS n_lt50,
       count(DISTINCT CASE WHEN weight_kg >= 100 THEN dicom_id END) AS n_ge100
FROM cxr_weight
WHERE inside_stay AND view_position='AP' AND procedure_desc ILIKE '%PORTABLE%' AND gap_h<=24
""").to_dict("records")

print("\n== weight itemid composition in core cohort ==")
out["core_itemids"] = q("""
SELECT itemid, count(*) AS n_images, count(DISTINCT stay_id) AS n_stays,
       median(weight_kg) AS med_kg
FROM cxr_weight
WHERE inside_stay AND view_position='AP' AND procedure_desc ILIKE '%PORTABLE%' AND gap_h<=24
GROUP BY 1 ORDER BY n_images DESC
""").to_dict("records")

print("\n== per-patient image counts, core cohort ==")
out["per_patient"] = q("""
WITH t AS (
  SELECT subject_id, count(DISTINCT dicom_id) AS n
  FROM cxr_weight
  WHERE inside_stay AND view_position='AP' AND procedure_desc ILIKE '%PORTABLE%' AND gap_h<=24
  GROUP BY 1)
SELECT CASE WHEN n=1 THEN '1' WHEN n<=3 THEN '2-3' WHEN n<=10 THEN '4-10' ELSE '>10' END AS bucket,
       count(*) AS n_patients
FROM t GROUP BY 1 ORDER BY 1
""").to_dict("records")

with open("lit_search/mimic_probe3b.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("\nsaved lit_search/mimic_probe3b.json")
