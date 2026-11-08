"""ICU context variables for the primary cohort: care unit mix, ventilation, height availability."""

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


print("== icustays columns ==")
cols = q("""SELECT column_name FROM information_schema.columns
            WHERE table_name='icustays' ORDER BY ordinal_position""")
print(cols["column_name"].tolist())

out = {}

con.execute("""
CREATE OR REPLACE TEMP TABLE w_daily AS
SELECT stay_id, subject_id, charttime, valuenum AS kg
FROM main.chartevents
WHERE itemid = 224639 AND valuenum IS NOT NULL AND valuenum BETWEEN 20 AND 350
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE matched AS
SELECT * FROM (
  SELECT c.stay_id, c.subject_id, c.study_id, c.dicom_id, c.study_datetime, w.kg, w.charttime,
         row_number() OVER (PARTITION BY c.dicom_id
                            ORDER BY abs(epoch(c.study_datetime - w.charttime))) AS w_rank
  FROM (SELECT i.stay_id, i.subject_id, m.study_id, m.dicom_id, m.study_datetime
        FROM main.icustays i
        JOIN main.cxr_metadata m
          ON m.subject_id = i.subject_id AND m.study_datetime BETWEEN i.intime AND i.outtime
        WHERE m.view_position='AP' AND m.procedure_desc ILIKE '%PORTABLE%') c
  JOIN w_daily w ON w.stay_id = c.stay_id
   AND w.charttime BETWEEN c.study_datetime - INTERVAL 24 HOUR AND c.study_datetime + INTERVAL 24 HOUR
) WHERE w_rank = 1
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE index_img AS
SELECT * FROM (
  SELECT *, row_number() OVER (PARTITION BY stay_id ORDER BY study_datetime, dicom_id) AS img_rank
  FROM matched
) WHERE img_rank = 1
""")

print("\n== primary cohort: first ICU care unit ==")
out["care_unit"] = q("""
SELECT i.first_careunit AS unit, count(*) AS n
FROM index_img x JOIN main.icustays i USING (stay_id)
GROUP BY 1 ORDER BY n DESC
""").to_dict("records")

print("\n== primary cohort: height availability ==")
out["height"] = q("""
SELECT count(*) AS n_stays, count(DISTINCT h.stay_id) AS n_with_height,
       round(100.0*count(DISTINCT h.stay_id)/count(*),1) AS pct
FROM index_img x LEFT JOIN mimiciv_derived.height h USING (stay_id)
""").to_dict("records")

print("\n== primary cohort: invasive ventilation ==")
out["ventilation"] = q("""
SELECT count(*) AS n_stays, count(DISTINCT v.stay_id) AS n_ventilated,
       round(100.0*count(DISTINCT v.stay_id)/count(*),1) AS pct
FROM index_img x LEFT JOIN mimiciv_derived.ventilation v
  ON v.stay_id = x.stay_id AND v.ventilation_status = 'InvasiveVent'
""").to_dict("records")

print("\n== primary cohort: ICU length of stay ==")
out["los"] = q("""
SELECT median(date_diff('hour', i.intime, i.outtime)/24.0) AS med_los_days,
       quantile_cont(date_diff('hour', i.intime, i.outtime)/24.0, 0.25) AS p25,
       quantile_cont(date_diff('hour', i.intime, i.outtime)/24.0, 0.75) AS p75
FROM index_img x JOIN main.icustays i USING (stay_id)
""").to_dict("records")

with open("lit_search/cohort_context.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("\nsaved lit_search/cohort_context.json")
