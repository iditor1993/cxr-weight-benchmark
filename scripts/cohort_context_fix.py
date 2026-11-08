"""Correct stay-level counts (avoid join fan-out)."""

import json

import duckdb

DB = r"E:\clinical_research\MIMIC_IV_3.1\mimic_iv_3_1.duckdb"
con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA threads=8")

out = {}
cols = con.execute("""SELECT column_name FROM information_schema.columns
                      WHERE table_name='icustays' ORDER BY ordinal_position""").fetchdf()
print(cols["column_name"].tolist())

con.execute("""
CREATE OR REPLACE TEMP TABLE w_daily AS
SELECT stay_id, charttime, valuenum AS kg FROM main.chartevents
WHERE itemid = 224639 AND valuenum IS NOT NULL AND valuenum BETWEEN 20 AND 350
""")
con.execute("""
CREATE OR REPLACE TEMP TABLE matched AS
SELECT * FROM (
  SELECT c.stay_id, c.subject_id, c.study_id, c.dicom_id, c.study_datetime,
         row_number() OVER (PARTITION BY c.dicom_id
           ORDER BY abs(epoch(c.study_datetime - w.charttime))) AS w_rank
  FROM (SELECT i.stay_id, i.subject_id, m.study_id, m.dicom_id, m.study_datetime
        FROM main.icustays i JOIN main.cxr_metadata m
          ON m.subject_id=i.subject_id AND m.study_datetime BETWEEN i.intime AND i.outtime
        WHERE m.view_position='AP' AND m.procedure_desc ILIKE '%PORTABLE%') c
  JOIN w_daily w ON w.stay_id=c.stay_id
   AND w.charttime BETWEEN c.study_datetime - INTERVAL 24 HOUR AND c.study_datetime + INTERVAL 24 HOUR
) WHERE w_rank=1
""")
con.execute("""
CREATE OR REPLACE TEMP TABLE index_img AS
SELECT stay_id, subject_id FROM (
  SELECT stay_id, subject_id,
         row_number() OVER (PARTITION BY stay_id ORDER BY study_datetime) AS rn
  FROM matched) WHERE rn=1
""")

n = con.execute("SELECT count(*) FROM index_img").fetchone()[0]
print("index stays:", n)

h = con.execute("""
SELECT count(DISTINCT x.stay_id)
FROM index_img x
WHERE EXISTS (SELECT 1 FROM mimiciv_derived.height hh WHERE hh.stay_id = x.stay_id)
""").fetchone()[0]
print("with height:", h, round(100*h/n, 1))

v = con.execute("""
SELECT count(DISTINCT x.stay_id)
FROM index_img x
WHERE EXISTS (SELECT 1 FROM mimiciv_derived.ventilation vv
              WHERE vv.stay_id = x.stay_id AND vv.ventilation_status='InvasiveVent')
""").fetchone()[0]
print("invasive ventilation:", v, round(100*v/n, 1))

rrt = con.execute("""
SELECT count(DISTINCT x.stay_id)
FROM index_img x
WHERE EXISTS (SELECT 1 FROM mimiciv_derived.rrt rr WHERE rr.stay_id = x.stay_id)
""").fetchone()[0]
print("rrt:", rrt, round(100*rrt/n, 1))

out = {"n_index_stays": int(n), "n_with_height": int(h), "pct_height": round(100*h/n, 1),
       "n_invasive_vent": int(v), "pct_invasive_vent": round(100*v/n, 1),
       "n_rrt": int(rrt), "pct_rrt": round(100*rrt/n, 1)}
with open("lit_search/cohort_context_fix.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False))
