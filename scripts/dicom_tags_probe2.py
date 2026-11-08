"""Larger sample: distribution of surviving technical tags in the ICU portable-AP cohort."""

import io
import json
import zipfile
from collections import Counter

import duckdb
import pydicom

DB = r"E:\clinical_research\MIMIC_IV_3.1\mimic_iv_3_1.duckdb"
con = duckdb.connect(DB, read_only=True)

rows = con.execute("""
SELECT f.dicom_id, f.zip_path, f.inner_path
FROM main.cxr_files f
WHERE f.view_position='AP' AND f.procedure_desc ILIKE '%PORTABLE%'
USING SAMPLE 200 ROWS (reservoir, 42)
""").fetchdf()

counters = {k: Counter() for k in ["ImagerPixelSpacing", "KVP", "ExposureIndex", "DetectorType",
                                   "PhotometricInterpretation", "BitsStored", "Rows", "Columns"]}
missing = Counter()
ok = 0
for _, row in rows.iterrows():
    try:
        with zipfile.ZipFile(row["zip_path"]) as zf:
            ds = pydicom.dcmread(io.BytesIO(zf.read(row["inner_path"])), stop_before_pixels=True)
        ok += 1
        for k in counters:
            v = ds.get(k)
            if v is None:
                missing[k] += 1
            else:
                counters[k][str(v)] += 1
    except Exception as exc:  # noqa: BLE001
        print("ERR", type(exc).__name__, str(exc)[:100])

print(f"read OK: {ok}/{len(rows)}")
for k, c in counters.items():
    print(f"{k}: {c.most_common(8)} (missing={missing[k]})")

with open("lit_search/dicom_tags_probe2.json", "w", encoding="utf-8") as fh:
    json.dump({k: dict(c.most_common(30)) for k, c in counters.items()} | {"missing": dict(missing)},
              fh, ensure_ascii=False, indent=1)
print("saved")
