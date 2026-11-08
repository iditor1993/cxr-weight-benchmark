"""Temporal split and clustering structure for the primary (daily-weight) cohort."""

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
WHERE itemid = 224639 AND valuenum IS NOT NULL AND valuenum BETWEEN 20 AND 350
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE a AS
SELECT * FROM (
  SELECT c.subject_id, c.stay_id, c.study_id, c.dicom_id, c.study_datetime,
         w.charttime, w.kg, abs(epoch(c.study_datetime - w.charttime))/3600.0 AS gap_h,
         row_number() OVER (PARTITION BY c.stay_id ORDER BY c.study_datetime, c.dicom_id) AS img_rank
  FROM (SELECT i.subject_id, i.stay_id, m.study_id, m.dicom_id, m.study_datetime
        FROM main.icustays i
        JOIN main.cxr_metadata m
          ON m.subject_id = i.subject_id AND m.study_datetime BETWEEN i.intime AND i.outtime
        WHERE m.view_position='AP' AND m.procedure_desc ILIKE '%PORTABLE%') c
  JOIN w ON w.stay_id = c.stay_id
   AND w.charttime BETWEEN c.study_datetime - INTERVAL 24 HOUR AND c.study_datetime + INTERVAL 24 HOUR
  QUALIFY row_number() OVER (PARTITION BY c.dicom_id ORDER BY abs(epoch(c.study_datetime - w.charttime))) = 1
) WHERE img_rank = 1
""")

print("== primary cohort: one index image per stay ==")
out["primary"] = q("""
SELECT count(*) AS n_images, count(DISTINCT stay_id) AS n_stays,
       count(DISTINCT subject_id) AS n_patients,
       median(kg) AS med_kg, min(kg) AS min_kg, max(kg) AS max_kg
FROM a
""").to_dict("records")

print("\n== stays per patient (clustering) ==")
out["stays_per_patient"] = q("""
WITH t AS (SELECT subject_id, count(DISTINCT stay_id) AS n FROM a GROUP BY 1)
SELECT CASE WHEN n=1 THEN '1' WHEN n=2 THEN '2' WHEN n=3 THEN '3' ELSE '>=4' END AS bucket,
       count(*) AS n_patients FROM t GROUP BY 1 ORDER BY 1
""").to_dict("records")

print("\n== temporal split: earliest 70% / next 15% / last 15% of index images by time ==")
out["temporal_split"] = q("""
WITH t AS (SELECT *, row_number() OVER (ORDER BY study_datetime, dicom_id) AS rn, count(*) OVER () AS n FROM a)
SELECT sum(CASE WHEN rn <= 0.7*n THEN 1 ELSE 0 END) AS train_img,
       sum(CASE WHEN rn > 0.7*n AND rn <= 0.85*n THEN 1 ELSE 0 END) AS val_img,
       sum(CASE WHEN rn > 0.85*n THEN 1 ELSE 0 END) AS test_img,
       min(study_datetime) AS first_dt, max(study_datetime) AS last_dt
FROM t
""").to_dict("records")

print("\n== temporal split: patient overlap check ==")
out["temporal_patient_overlap"] = q("""
WITH t AS (SELECT *, row_number() OVER (ORDER BY study_datetime, dicom_id) AS rn, count(*) OVER () AS n FROM a),
lab AS (SELECT subject_id, CASE WHEN rn <= 0.7*n THEN 'train' WHEN rn <= 0.85*n THEN 'val' ELSE 'test' END AS split FROM t)
SELECT split, count(DISTINCT subject_id) AS n_patients FROM lab GROUP BY 1 ORDER BY 1
""").to_dict("records")

with open("lit_search/cohort_definition2.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("\nsaved")
