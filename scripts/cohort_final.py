"""Single clean definition of the primary analysis sets (no ambiguity)."""

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

# weights: daily only, cleaned
con.execute("""
CREATE OR REPLACE TEMP TABLE w_daily AS
SELECT stay_id, subject_id, charttime, valuenum AS kg
FROM main.chartevents
WHERE itemid = 224639 AND valuenum IS NOT NULL AND valuenum BETWEEN 20 AND 350
""")

# candidate portable AP images during ICU stay
con.execute("""
CREATE OR REPLACE TEMP TABLE cxr AS
SELECT i.stay_id, i.subject_id, m.study_id, m.dicom_id, m.study_datetime
FROM main.icustays i
JOIN main.cxr_metadata m
  ON m.subject_id = i.subject_id AND m.study_datetime BETWEEN i.intime AND i.outtime
WHERE m.view_position = 'AP' AND m.procedure_desc ILIKE '%PORTABLE%'
""")

# one nearest daily weight per image, within 24h
con.execute("""
CREATE OR REPLACE TEMP TABLE pairs AS
SELECT * FROM (
  SELECT c.stay_id, c.subject_id, c.study_id, c.dicom_id, c.study_datetime,
         w.charttime, w.kg, abs(epoch(c.study_datetime - w.charttime))/3600.0 AS gap_h,
         row_number() OVER (PARTITION BY c.dicom_id
                            ORDER BY abs(epoch(c.study_datetime - w.charttime))) AS w_rank
  FROM cxr c
  JOIN w_daily w ON w.stay_id = c.stay_id
   AND w.charttime BETWEEN c.study_datetime - INTERVAL 24 HOUR AND c.study_datetime + INTERVAL 24 HOUR
) WHERE w_rank = 1
""")

# index image: earliest matched image per stay
con.execute("""
CREATE OR REPLACE TEMP TABLE index_img AS
SELECT * FROM (
  SELECT *, row_number() OVER (PARTITION BY stay_id ORDER BY study_datetime, dicom_id) AS img_rank
  FROM pairs
) WHERE img_rank = 1
""")

print("== primary label set (daily weight, <=24h), all matched images ==")
out["all_matched"] = q("""
SELECT count(*) AS n_images, count(DISTINCT study_id) AS n_studies,
       count(DISTINCT stay_id) AS n_stays, count(DISTINCT subject_id) AS n_patients,
       median(gap_h) AS med_gap_h, sum(CASE WHEN gap_h<=6 THEN 1 ELSE 0 END) AS le6h,
       sum(CASE WHEN gap_h<=12 THEN 1 ELSE 0 END) AS le12h
FROM pairs
""").to_dict("records")

print("\n== index-image analysis set (one per stay) ==")
out["index_set"] = q("""
SELECT count(*) AS n_index_images, count(DISTINCT stay_id) AS n_stays,
       count(DISTINCT subject_id) AS n_patients,
       median(kg) AS med_kg, quantile_cont(kg,0.05) AS p05, quantile_cont(kg,0.95) AS p95,
       sum(CASE WHEN kg<50 THEN 1 ELSE 0 END) AS lt50,
       sum(CASE WHEN kg>=100 THEN 1 ELSE 0 END) AS ge100
FROM index_img
""").to_dict("records")

print("\n== index set: stays per patient ==")
out["stays_per_patient"] = q("""
WITH t AS (SELECT subject_id, count(*) AS n FROM index_img GROUP BY 1)
SELECT CASE WHEN n=1 THEN '1' WHEN n=2 THEN '2' ELSE '>=3' END AS bucket, count(*) AS n_patients
FROM t GROUP BY 1 ORDER BY 1
""").to_dict("records")

print("\n== index set: patient-level 70/15/15 random split ==")
out["random_split"] = q("""
WITH p AS (SELECT DISTINCT subject_id FROM index_img),
t AS (SELECT subject_id, row_number() OVER (ORDER BY hash(subject_id)) AS rn, count(*) OVER () AS n FROM p)
SELECT sum(CASE WHEN rn<=0.7*n THEN 1 ELSE 0 END) AS train_pt,
       sum(CASE WHEN rn>0.7*n AND rn<=0.85*n THEN 1 ELSE 0 END) AS val_pt,
       sum(CASE WHEN rn>0.85*n THEN 1 ELSE 0 END) AS test_pt
FROM t
""").to_dict("records")

print("\n== index set: temporal 70/15/15 by time ==")
out["temporal_split"] = q("""
WITH t AS (SELECT *, row_number() OVER (ORDER BY study_datetime, dicom_id) AS rn, count(*) OVER () AS n FROM index_img)
SELECT sum(CASE WHEN rn<=0.7*n THEN 1 ELSE 0 END) AS train_n,
       sum(CASE WHEN rn>0.7*n AND rn<=0.85*n THEN 1 ELSE 0 END) AS val_n,
       sum(CASE WHEN rn>0.85*n THEN 1 ELSE 0 END) AS test_n,
       min(study_datetime) AS first_dt, max(study_datetime) AS last_dt
FROM t
""").to_dict("records")

print("\n== index set: patient overlap under temporal split ==")
out["temporal_overlap"] = q("""
WITH t AS (SELECT *, row_number() OVER (ORDER BY study_datetime, dicom_id) AS rn, count(*) OVER () AS n FROM index_img),
lab AS (SELECT subject_id, CASE WHEN rn<=0.7*n THEN 'train' WHEN rn<=0.85*n THEN 'val' ELSE 'test' END AS split FROM t)
SELECT count(*) AS n_patients_total,
       count(DISTINCT CASE WHEN split='test' THEN subject_id END) AS n_in_test,
       count(*) - count(DISTINCT subject_id) AS n_patients_in_multiple_splits
FROM lab
""").to_dict("records")

print("\n== secondary analysis: images per stay (all matched) ==")
out["images_per_stay"] = q("""
WITH t AS (SELECT stay_id, count(*) AS n FROM pairs GROUP BY 1)
SELECT CASE WHEN n=1 THEN '1' WHEN n<=3 THEN '2-3' WHEN n<=10 THEN '4-10' ELSE '>10' END AS bucket,
       count(*) AS n_stays FROM t GROUP BY 1 ORDER BY 1
""").to_dict("records")

with open("lit_search/cohort_final.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("\nsaved lit_search/cohort_final.json")
