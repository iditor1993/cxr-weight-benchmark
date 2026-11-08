"""Step 3: the decisive feasibility numbers - ICU stays x CXR x weight pairing."""

import json
import time

import duckdb

DB = r"E:\clinical_research\MIMIC_IV_3.1\mimic_iv_3_1.duckdb"
con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA threads=8")


def q(sql):
    t0 = time.time()
    df = con.execute(sql).fetchdf()
    print(f"    ({time.time()-t0:.1f}s, {len(df)} rows)")
    with __import__("pandas").option_context("display.max_columns", 60, "display.width", 230):
        print(df.to_string())
    return df


out = {}

print("== weight chartevents inventory ==")
out["weight_items"] = q("""
SELECT c.itemid, d.label, d.unitname, count(*) AS n_rows,
       count(DISTINCT c.stay_id) AS n_stays, count(DISTINCT c.subject_id) AS n_subjects,
       min(c.valuenum) AS min_v, median(c.valuenum) AS med_v, max(c.valuenum) AS max_v
FROM main.chartevents c
JOIN main.d_items d USING (itemid)
WHERE c.itemid IN (224639, 226512, 226531, 226846)
GROUP BY 1,2,3
ORDER BY n_rows DESC
""").to_dict("records")

print("\n== ICU stays with any weight ==")
out["icu_weight_coverage"] = q("""
SELECT count(DISTINCT i.stay_id) AS n_stays_total,
       count(DISTINCT CASE WHEN c.stay_id IS NOT NULL THEN i.stay_id END) AS n_stays_with_weight,
       count(DISTINCT CASE WHEN c.stay_id IS NOT NULL THEN i.subject_id END) AS n_patients_with_weight
FROM main.icustays i
LEFT JOIN (SELECT DISTINCT stay_id FROM main.chartevents WHERE itemid IN (224639,226512,226531,226846)) c
  ON c.stay_id = i.stay_id
""").to_dict("records")


print("\n== CXR (any view) linked to an ICU stay window (+/-24h) ==")
con.execute("""
CREATE OR REPLACE TEMP TABLE cxr_icu AS
SELECT i.subject_id, i.stay_id, i.intime, i.outtime,
       m.study_id, m.dicom_id, m.study_datetime, m.view_position,
       m.patient_orientation_code_meaning AS orientation
FROM main.icustays i
JOIN main.cxr_metadata m
  ON m.subject_id = i.subject_id
 AND m.study_datetime BETWEEN i.intime - INTERVAL 24 HOUR AND i.outtime + INTERVAL 24 HOUR
""")
out["cxr_icu_overall"] = q("""
SELECT count(*) AS n_image_rows,
       count(DISTINCT stay_id) AS n_stays,
       count(DISTINCT subject_id) AS n_patients,
       count(DISTINCT study_id) AS n_studies,
       count(DISTINCT dicom_id) AS n_images
FROM cxr_icu
""").to_dict("records")

print("\n== view position / orientation breakdown for ICU-linked images ==")
out["cxr_icu_views"] = q("""
SELECT view_position, orientation, count(*) AS n_images,
       count(DISTINCT stay_id) AS n_stays, count(DISTINCT subject_id) AS n_subjects
FROM cxr_icu
GROUP BY 1,2 ORDER BY n_images DESC
""").to_dict("records")

print("\n== images inside the ICU stay itself (not +/-24h) ==")
out["cxr_inside_stay"] = q("""
SELECT count(*) AS n_images, count(DISTINCT stay_id) AS n_stays,
       count(DISTINCT subject_id) AS n_patients,
       count(DISTINCT CASE WHEN view_position='AP' THEN dicom_id END) AS n_ap_images,
       count(DISTINCT CASE WHEN patient_orientation_code_meaning='Recumbent' THEN dicom_id END) AS n_recumbent_images
FROM cxr_icu
WHERE study_datetime BETWEEN intime AND outtime
""").to_dict("records")

print("\n== nearest weight measurement to each ICU-linked CXR image (within 72h) ==")
con.execute("""
CREATE OR REPLACE TEMP TABLE cxr_weight AS
SELECT c.subject_id, c.stay_id, c.study_id, c.dicom_id, c.study_datetime, c.view_position, c.orientation,
       w.charttime, w.valuenum AS weight, w.itemid,
       abs(epoch(c.study_datetime - w.charttime)) / 3600.0 AS gap_h
FROM cxr_icu c
JOIN (SELECT stay_id, subject_id, charttime, valuenum, itemid
      FROM main.chartevents
      WHERE itemid IN (224639, 226512, 226846)
        AND valuenum BETWEEN 20 AND 350) w
  ON w.stay_id = c.stay_id
 AND w.charttime BETWEEN c.study_datetime - INTERVAL 72 HOUR AND c.study_datetime + INTERVAL 72 HOUR
QUALIFY row_number() OVER (PARTITION BY c.dicom_id ORDER BY abs(epoch(c.study_datetime - w.charttime))) = 1
""")
out["cxr_weight_gap"] = q("""
SELECT
  count(*) AS n_pairs_72h,
  count(DISTINCT subject_id) AS n_patients,
  count(DISTINCT stay_id) AS n_stays,
  count(DISTINCT dicom_id) AS n_images,
  sum(CASE WHEN gap_h <= 6 THEN 1 ELSE 0 END) AS n_gap_le6h,
  sum(CASE WHEN gap_h <= 12 THEN 1 ELSE 0 END) AS n_gap_le12h,
  sum(CASE WHEN gap_h <= 24 THEN 1 ELSE 0 END) AS n_gap_le24h,
  quantile_cont(gap_h, 0.25) AS p25_gap, median(gap_h) AS med_gap, quantile_cont(gap_h, 0.75) AS p75_gap
FROM cxr_weight
""").to_dict("records")

print("\n== same, restricted to images inside an ICU stay ==")
out["cxr_weight_inside"] = q("""
SELECT
  count(*) AS n_pairs,
  count(DISTINCT c.subject_id) AS n_patients,
  count(DISTINCT c.stay_id) AS n_stays,
  count(DISTINCT c.dicom_id) AS n_images,
  sum(CASE WHEN c.gap_h <= 6 THEN 1 ELSE 0 END) AS n_gap_le6h,
  sum(CASE WHEN c.gap_h <= 12 THEN 1 ELSE 0 END) AS n_gap_le12h,
  sum(CASE WHEN c.gap_h <= 24 THEN 1 ELSE 0 END) AS n_gap_le24h
FROM cxr_weight c
JOIN main.icustays i ON i.stay_id = c.stay_id
WHERE c.study_datetime BETWEEN i.intime AND i.outtime
""").to_dict("records")

print("\n== by view / orientation, gap <= 24h, image inside stay ==")
out["cxr_weight_views"] = q("""
SELECT c.view_position, c.orientation,
       count(*) AS n_pairs, count(DISTINCT c.dicom_id) AS n_images,
       count(DISTINCT c.stay_id) AS n_stays,
       count(DISTINCT c.subject_id) AS n_patients,
       median(c.gap_h) AS med_gap_h
FROM cxr_weight c
JOIN main.icustays i ON i.stay_id = c.stay_id
WHERE c.study_datetime BETWEEN i.intime AND i.outtime AND c.gap_h <= 24
GROUP BY 1,2 ORDER BY n_pairs DESC
""").to_dict("records")

print("\n== number of paired images per patient (gap<=24h, inside stay) ==")
out["per_patient_distribution"] = q("""
WITH t AS (
  SELECT c.subject_id, count(DISTINCT c.dicom_id) AS n
  FROM cxr_weight c
  JOIN main.icustays i ON i.stay_id = c.stay_id
  WHERE c.study_datetime BETWEEN i.intime AND i.outtime AND c.gap_h <= 24
  GROUP BY 1)
SELECT n_images_per_patient, count(*) AS n_patients
FROM (SELECT n AS n_images_per_patient FROM t)
GROUP BY 1 ORDER BY 1
""").to_dict("records")

with open("lit_search/mimic_probe3.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("\nsaved lit_search/mimic_probe3.json")
