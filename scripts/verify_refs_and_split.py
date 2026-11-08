"""Verify cited DOIs and compute the patient-level temporal split for the primary cohort."""

import duckdb
import requests

S = requests.Session()
S.headers.update({"User-Agent": "paper-lookup/1.0 (mailto:research@example.org)"})

print("== DOI existence checks ==")
for doi in [
    "10.1136/bmj-2023-078378",      # TRIPOD+AI
    "10.1038/s41597-019-0322-0",    # MIMIC-CXR
    "10.1038/s41597-022-01899-x",   # MIMIC-IV
    "10.1111/nicc.13271",
    "10.1007/s12194-023-00697-3",
    "10.1007/s12194-026-01028-y",
]:
    try:
        r = S.get(f"https://api.crossref.org/works/{doi}", timeout=45)
        if r.status_code == 200:
            m = r.json()["message"]
            print(f"OK  {doi} | {(m.get('title') or [''])[0][:95]} | "
                  f"{(m.get('container-title') or [''])[0][:45]} | "
                  f"{m.get('issued', {}).get('date-parts')}")
        else:
            print(f"MISS {doi} -> HTTP {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERR {doi} -> {type(exc).__name__}")

print("\n== patient-level temporal split (last 15% of patients by first index image) ==")
DB = r"E:\clinical_research\MIMIC_IV_3.1\mimic_iv_3_1.duckdb"
con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA threads=8")
con.execute("""
CREATE OR REPLACE TEMP TABLE w AS
SELECT stay_id, charttime, valuenum AS kg FROM main.chartevents
WHERE itemid = 224639 AND valuenum IS NOT NULL AND valuenum BETWEEN 20 AND 350
""")
con.execute("""
CREATE OR REPLACE TEMP TABLE m AS
SELECT * FROM (
  SELECT c.stay_id, c.subject_id, c.dicom_id, c.study_datetime,
         row_number() OVER (PARTITION BY c.dicom_id
           ORDER BY abs(epoch(c.study_datetime - w.charttime))) AS r
  FROM (SELECT i.stay_id, i.subject_id, m2.dicom_id, m2.study_datetime
        FROM main.icustays i JOIN main.cxr_metadata m2
          ON m2.subject_id=i.subject_id AND m2.study_datetime BETWEEN i.intime AND i.outtime
        WHERE m2.view_position='AP' AND m2.procedure_desc ILIKE '%PORTABLE%') c
  JOIN w ON w.stay_id=c.stay_id
   AND w.charttime BETWEEN c.study_datetime - INTERVAL 24 HOUR AND c.study_datetime + INTERVAL 24 HOUR
) WHERE r=1
""")
con.execute("""
CREATE OR REPLACE TEMP TABLE idx AS
SELECT * FROM (
  SELECT *, row_number() OVER (PARTITION BY stay_id ORDER BY study_datetime, dicom_id) AS k
  FROM m) WHERE k=1
""")
df = con.execute("""
WITH firsts AS (
  SELECT subject_id, min(study_datetime) AS first_dt FROM idx GROUP BY 1),
ranked AS (
  SELECT subject_id, first_dt,
         row_number() OVER (ORDER BY first_dt, subject_id) AS rn,
         count(*) OVER () AS n
  FROM firsts),
lab AS (
  SELECT subject_id, CASE WHEN rn <= 0.7*n THEN 'train'
                          WHEN rn <= 0.85*n THEN 'val' ELSE 'test' END AS split
  FROM ranked)
SELECT l.split, count(DISTINCT i.subject_id) AS n_patients,
       count(*) AS n_stays, min(i.study_datetime) AS first_img, max(i.study_datetime) AS last_img
FROM idx i JOIN lab l USING (subject_id)
GROUP BY 1 ORDER BY 1
""").fetchdf()
print(df.to_string())
