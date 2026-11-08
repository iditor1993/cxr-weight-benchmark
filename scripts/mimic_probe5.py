"""Step 5: domain shift checks (pathology/devices/orientation) + DICOM decode smoke test."""

import io
import json
import zipfile

import duckdb
import numpy as np
import pandas as pd

DB = r"E:\clinical_research\MIMIC_IV_3.1\mimic_iv_3_1.duckdb"
con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA threads=8")


def q(sql):
    df = con.execute(sql).fetchdf()
    with pd.option_context("display.max_columns", 60, "display.width", 240):
        print(df.to_string())
    return df


out = {}

q("""
CREATE OR REPLACE TEMP TABLE weight_events AS
SELECT stay_id, subject_id, charttime, itemid,
       CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END AS weight_kg
FROM main.chartevents
WHERE itemid IN (224639, 226512, 226531, 226846) AND valuenum IS NOT NULL
  AND (CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END) BETWEEN 20 AND 350
""")

q("""
CREATE OR REPLACE TEMP TABLE core AS
SELECT c.subject_id, c.stay_id, c.study_id, c.dicom_id, c.study_datetime,
       c.view_position, c.procedure_desc, c.orientation AS patient_orientation,
       w.charttime, w.weight_kg, w.itemid,
       abs(epoch(c.study_datetime - w.charttime)) / 3600.0 AS gap_h
FROM (SELECT i.subject_id, i.stay_id, m.study_id, m.dicom_id, m.study_datetime,
             m.view_position, m.procedure_desc, m.patient_orientation_code_meaning AS orientation
      FROM main.icustays i
      JOIN main.cxr_metadata m
        ON m.subject_id = i.subject_id
       AND m.study_datetime BETWEEN i.intime AND i.outtime) c
JOIN weight_events w
  ON w.stay_id = c.stay_id
 AND w.charttime BETWEEN c.study_datetime - INTERVAL 24 HOUR AND c.study_datetime + INTERVAL 24 HOUR
WHERE c.view_position='AP' AND c.procedure_desc ILIKE '%PORTABLE%'
QUALIFY row_number() OVER (PARTITION BY c.dicom_id ORDER BY abs(epoch(c.study_datetime - w.charttime))) = 1
""")

print("== core cohort: study-level counts ==")
out["study_level"] = q("""
SELECT count(*) AS n_images, count(DISTINCT study_id) AS n_studies,
       count(DISTINCT stay_id) AS n_stays, count(DISTINCT subject_id) AS n_patients,
       sum(CASE WHEN patient_orientation='Recumbent' THEN 1 ELSE 0 END) AS n_recumbent,
       sum(CASE WHEN patient_orientation='Erect' THEN 1 ELSE 0 END) AS n_erect,
       sum(CASE WHEN patient_orientation IS NULL THEN 1 ELSE 0 END) AS n_no_orientation
FROM core
""").to_dict("records")

print("\n== CheXpert pathology prevalence in core cohort vs all MIMIC-CXR ==")
out["chexpert"] = q("""
WITH allcxr AS (
  SELECT m.dicom_id, ch.*
  FROM main.cxr_metadata m JOIN main.cxr_chexpert ch USING (subject_id, study_id))
SELECT 'core' AS cohort, count(*) AS n,
       avg(CASE WHEN "No Finding"=1 THEN 1 ELSE 0 END) AS no_finding,
       avg(CASE WHEN "Support Devices"=1 THEN 1 ELSE 0 END) AS support_devices,
       avg(CASE WHEN Edema=1 THEN 1 ELSE 0 END) AS edema,
       avg(CASE WHEN "Pleural Effusion"=1 THEN 1 ELSE 0 END) AS pleural_effusion,
       avg(CASE WHEN Atelectasis=1 THEN 1 ELSE 0 END) AS atelectasis,
       avg(CASE WHEN Cardiomegaly=1 THEN 1 ELSE 0 END) AS cardiomegaly,
       avg(CASE WHEN Consolidation=1 THEN 1 ELSE 0 END) AS consolidation
FROM allcxr WHERE dicom_id IN (SELECT dicom_id FROM core)
UNION ALL
SELECT 'all_mimic_cxr', count(*),
       avg(CASE WHEN "No Finding"=1 THEN 1 ELSE 0 END),
       avg(CASE WHEN "Support Devices"=1 THEN 1 ELSE 0 END),
       avg(CASE WHEN Edema=1 THEN 1 ELSE 0 END),
       avg(CASE WHEN "Pleural Effusion"=1 THEN 1 ELSE 0 END),
       avg(CASE WHEN Atelectasis=1 THEN 1 ELSE 0 END),
       avg(CASE WHEN Cardiomegaly=1 THEN 1 ELSE 0 END),
       avg(CASE WHEN Consolidation=1 THEN 1 ELSE 0 END)
FROM allcxr
""").to_dict("records")

print("\n== DICOM decode smoke test ==")
sample = q("""
SELECT f.dicom_id, f.zip_path, f.inner_path FROM main.cxr_files f
WHERE f.dicom_id IN (SELECT dicom_id FROM core) LIMIT 2
""")

import pydicom

decode_report = []
for _, row in sample.iterrows():
    zp = str(row["zip_path"]).replace("\\", "/")
    inner = str(row["inner_path"])
    try:
        with zipfile.ZipFile(zp) as zf:
            data = zf.read(inner)
        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array
        info = {"dicom_id": row["dicom_id"], "shape": list(arr.shape), "dtype": str(arr.dtype),
                "min": int(arr.min()), "max": int(arr.max()),
                "photometric": ds.get("PhotometricInterpretation"),
                "transfer_syntax": str(ds.file_meta.TransferSyntaxUID),
                "view": ds.get("ViewPosition"), "rows": int(ds.Rows), "cols": int(ds.Columns)}
        print(info)
        decode_report.append(info)
    except Exception as exc:  # noqa: BLE001
        print("DECODE ERROR", type(exc).__name__, str(exc)[:300])
        decode_report.append({"dicom_id": row["dicom_id"], "error": f"{type(exc).__name__}: {str(exc)[:300]}"})

out["decode"] = decode_report

print("\n== total DICOM payload size for core cohort ==")
out["payload"] = q("""
SELECT count(*) AS n_files, sum(f.size_bytes)/1e9 AS total_gb,
       avg(f.size_bytes)/1e6 AS avg_mb, max(f.size_bytes)/1e6 AS max_mb
FROM main.cxr_files f WHERE f.dicom_id IN (SELECT dicom_id FROM core)
""").to_dict("records")

with open("lit_search/mimic_probe5.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("\nsaved lit_search/mimic_probe5.json")
