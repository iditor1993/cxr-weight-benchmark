"""Step 4: label anchors, longitudinal structure, split feasibility, DICOM access."""

import json
import os
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

con.execute("""
CREATE OR REPLACE TEMP TABLE weight_events AS
SELECT stay_id, subject_id, charttime, itemid,
       CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END AS weight_kg
FROM main.chartevents
WHERE itemid IN (224639, 226512, 226531, 226846)
  AND valuenum IS NOT NULL
  AND (CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END) BETWEEN 20 AND 350
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE core AS
SELECT c.subject_id, c.stay_id, c.study_id, c.dicom_id, c.study_datetime,
       c.view_position, c.procedure_desc, w.charttime, w.weight_kg, w.itemid,
       abs(epoch(c.study_datetime - w.charttime)) / 3600.0 AS gap_h
FROM (SELECT i.subject_id, i.stay_id, i.intime, i.outtime,
             m.study_id, m.dicom_id, m.study_datetime, m.view_position, m.procedure_desc
      FROM main.icustays i
      JOIN main.cxr_metadata m
        ON m.subject_id = i.subject_id
       AND m.study_datetime BETWEEN i.intime AND i.outtime) c
JOIN weight_events w
  ON w.stay_id = c.stay_id
 AND w.charttime BETWEEN c.study_datetime - INTERVAL 24 HOUR AND c.study_datetime + INTERVAL 24 HOUR
WHERE c.view_position = 'AP' AND c.procedure_desc ILIKE '%PORTABLE%'
QUALIFY row_number() OVER (PARTITION BY c.dicom_id ORDER BY abs(epoch(c.study_datetime - w.charttime))) = 1
""")

print("== core cohort ==")
out["core"] = q("""
SELECT count(*) AS n_images, count(DISTINCT subject_id) AS n_patients,
       count(DISTINCT stay_id) AS n_stays, median(weight_kg) AS med_w,
       quantile_cont(weight_kg,0.05) AS p05, quantile_cont(weight_kg,0.95) AS p95
FROM core
""").to_dict("records")

print("\n== label anchor: admission-weight availability ==")
out["admission_anchor"] = q("""
WITH per_stay AS (
  SELECT stay_id,
         max(CASE WHEN itemid=226512 THEN 1 ELSE 0 END) AS has_adm_kg,
         max(CASE WHEN itemid=226531 THEN 1 ELSE 0 END) AS has_adm_lbs,
         max(CASE WHEN itemid=224639 THEN 1 ELSE 0 END) AS has_daily
  FROM weight_events GROUP BY 1)
SELECT count(DISTINCT c.stay_id) AS n_stays_core,
       count(DISTINCT CASE WHEN p.has_adm_kg=1 OR p.has_adm_lbs=1 THEN c.stay_id END) AS stays_with_admission_wt,
       count(DISTINCT CASE WHEN p.has_daily=1 THEN c.stay_id END) AS stays_with_daily_wt,
       count(DISTINCT CASE WHEN c.itemid IN (226512,226531) THEN c.dicom_id END) AS images_anchored_admission,
       count(DISTINCT CASE WHEN c.itemid=224639 THEN c.dicom_id END) AS images_anchored_daily
FROM core c JOIN per_stay p USING (stay_id)
""").to_dict("records")

print("\n== per-stay image counts ==")
out["per_stay"] = q("""
WITH t AS (SELECT stay_id, count(DISTINCT dicom_id) AS n FROM core GROUP BY 1)
SELECT CASE WHEN n=1 THEN '1' WHEN n<=3 THEN '2-3' WHEN n<=10 THEN '4-10' ELSE '>10' END AS bucket,
       count(*) AS n_stays FROM t GROUP BY 1 ORDER BY 1
""").to_dict("records")

print("\n== longitudinal: stays with >=2 images at different charted weights ==")
out["longitudinal"] = q("""
WITH per_stay AS (
  SELECT stay_id, count(DISTINCT dicom_id) AS n_img,
         max(weight_kg) - min(weight_kg) AS wt_range
  FROM core GROUP BY 1)
SELECT count(*) AS n_stays_multi,
       sum(CASE WHEN wt_range >= 1 THEN 1 ELSE 0 END) AS stays_range_ge1kg,
       sum(CASE WHEN wt_range >= 3 THEN 1 ELSE 0 END) AS stays_range_ge3kg,
       sum(CASE WHEN wt_range >= 5 THEN 1 ELSE 0 END) AS stays_range_ge5kg,
       median(wt_range) AS med_range
FROM per_stay WHERE n_img >= 2
""").to_dict("records")

print("\n== temporal distribution (MIMIC shifted years) ==")
out["year_dist"] = q("""
SELECT year(study_datetime) AS yr, count(*) AS n_images, count(DISTINCT subject_id) AS n_patients
FROM core GROUP BY 1 ORDER BY 1
""").to_dict("records")

print("\n== patient-level split feasibility (70/15/15 by subject) ==")
out["split"] = q("""
WITH pts AS (SELECT DISTINCT subject_id FROM core),
t AS (SELECT subject_id, row_number() OVER (ORDER BY hash(subject_id)) AS rn, count(*) OVER () AS n FROM pts)
SELECT sum(CASE WHEN rn <= 0.7*n THEN 1 ELSE 0 END) AS n_train_pt,
       sum(CASE WHEN rn > 0.7*n AND rn <= 0.85*n THEN 1 ELSE 0 END) AS n_val_pt,
       sum(CASE WHEN rn > 0.85*n THEN 1 ELSE 0 END) AS n_test_pt
FROM t
""").to_dict("records")

print("\n== image counts by split ==")
out["split_images"] = q("""
WITH pts AS (SELECT DISTINCT subject_id FROM core),
t AS (SELECT subject_id, row_number() OVER (ORDER BY hash(subject_id)) AS rn, count(*) OVER () AS n FROM pts),
lab AS (SELECT subject_id, CASE WHEN rn <= 0.7*n THEN 'train' WHEN rn <= 0.85*n THEN 'val' ELSE 'test' END AS split FROM t)
SELECT l.split, count(*) AS n_images, count(DISTINCT c.subject_id) AS n_patients
FROM core c JOIN lab l USING (subject_id) GROUP BY 1 ORDER BY 1
""").to_dict("records")

# --- DICOM access check
print("\n== DICOM file access check ==")
sample = q("""
SELECT f.dicom_id, f.zip_path, f.inner_path, f.size_bytes
FROM main.cxr_files f JOIN core c USING (dicom_id) LIMIT 3
""")
out["sample_files"] = sample.to_dict("records")
paths = []
for _, row in sample.iterrows():
    zp = str(row["zip_path"]).replace("\\", "/")
    if zp.startswith("Z:") or zp.startswith("z:"):
        zp = "Z:" + zp[2:]
    paths.append((zp, row["inner_path"]))

import zipfile
for zp, inner in paths:
    try:
        ok = os.path.exists(zp)
        size = os.path.getsize(zp) / 1e9 if ok else None
        with zipfile.ZipFile(zp) as zf:
            names = set(zf.namelist())
        print(f"{zp} exists={ok} size={size} inner_in_zip={inner in names} inner={inner}")
        out.setdefault("dicom_check", []).append(
            {"zip": zp, "exists": ok, "gb": size, "inner_in_zip": inner in names})
    except Exception as exc:  # noqa: BLE001
        print("ERR", zp, type(exc).__name__, str(exc)[:150])
        out.setdefault("dicom_check", []).append({"zip": zp, "error": str(exc)[:200]})

with open("lit_search/mimic_probe4.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("\nsaved lit_search/mimic_probe4.json")
