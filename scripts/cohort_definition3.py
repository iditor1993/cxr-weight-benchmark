"""Final window sensitivity for the daily-weight-only pairing (primary label definition)."""

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
CREATE OR REPLACE TEMP TABLE dw AS
SELECT stay_id, subject_id, charttime, valuenum AS kg
FROM main.chartevents
WHERE itemid = 224639 AND valuenum IS NOT NULL AND valuenum BETWEEN 20 AND 350
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE p AS
SELECT c.subject_id, c.stay_id, c.study_id, c.dicom_id, c.study_datetime,
       w.charttime, w.kg, abs(epoch(c.study_datetime - w.charttime))/3600.0 AS gap_h
FROM (SELECT i.subject_id, i.stay_id, m.study_id, m.dicom_id, m.study_datetime
      FROM main.icustays i
      JOIN main.cxr_metadata m
        ON m.subject_id = i.subject_id AND m.study_datetime BETWEEN i.intime AND i.outtime
      WHERE m.view_position='AP' AND m.procedure_desc ILIKE '%PORTABLE%') c
JOIN dw w ON w.stay_id = c.stay_id
 AND w.charttime BETWEEN c.study_datetime - INTERVAL 24 HOUR AND c.study_datetime + INTERVAL 24 HOUR
QUALIFY row_number() OVER (PARTITION BY c.dicom_id ORDER BY abs(epoch(c.study_datetime - w.charttime))) = 1
""")

print("== daily-weight-only pairing, within 24h ==")
out["daily_total"] = q("""
SELECT count(*) AS n_images, count(DISTINCT study_id) AS n_studies,
       count(DISTINCT stay_id) AS n_stays, count(DISTINCT subject_id) AS n_patients,
       median(gap_h) AS med_gap,
       sum(CASE WHEN gap_h<=6 THEN 1 ELSE 0 END) AS le6h,
       sum(CASE WHEN gap_h<=12 THEN 1 ELSE 0 END) AS le12h
FROM p
""").to_dict("records")

print("\n== daily-weight-only, images per stay distribution ==")
out["daily_per_stay"] = q("""
WITH t AS (SELECT stay_id, count(DISTINCT dicom_id) AS n FROM p GROUP BY 1)
SELECT CASE WHEN n=1 THEN '1' WHEN n<=3 THEN '2-3' WHEN n<=10 THEN '4-10' ELSE '>10' END AS bucket,
       count(*) AS n_stays FROM t GROUP BY 1 ORDER BY 1
""").to_dict("records")

print("\n== daily-weight-only, longitudinal (>=2 images, weight range) ==")
out["daily_longitudinal"] = q("""
WITH s AS (SELECT stay_id, count(DISTINCT dicom_id) AS n, max(kg)-min(kg) AS rng FROM p GROUP BY 1)
SELECT count(*) AS n_multi, sum(CASE WHEN rng>=1 THEN 1 ELSE 0 END) AS ge1,
       sum(CASE WHEN rng>=3 THEN 1 ELSE 0 END) AS ge3, sum(CASE WHEN rng>=5 THEN 1 ELSE 0 END) AS ge5,
       median(rng) AS med_rng
FROM s WHERE n>=2
""").to_dict("records")

with open("lit_search/cohort_definition3.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("\nsaved")
