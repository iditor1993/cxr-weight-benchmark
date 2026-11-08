"""Check which technical DICOM tags survive de-identification (for device-stratified validation)."""

import io
import json
import zipfile

import duckdb
import pydicom

DB = r"E:\clinical_research\MIMIC_IV_3.1\mimic_iv_3_1.duckdb"
con = duckdb.connect(DB, read_only=True)

rows = con.execute("""
SELECT f.dicom_id, f.zip_path, f.inner_path
FROM main.cxr_files f
WHERE f.view_position='AP' AND f.procedure_desc ILIKE '%PORTABLE%'
LIMIT 40
""").fetchdf()

TAGS = ["Manufacturer", "ManufacturerModelName", "StationName", "SoftwareVersions",
        "BodyPartExamined", "ViewPosition", "PatientPosition", "DetectorType",
        "PhotometricInterpretation", "ImagerPixelSpacing", "PixelSpacing",
        "DistanceSourceToDetector", "DistanceSourceToPatient", "ExposureIndex",
        "KVP", "ProtocolName", "SeriesDescription", "StudyDescription"]

found = {t: {} for t in TAGS}
ok = 0
for _, row in rows.iterrows():
    try:
        with zipfile.ZipFile(row["zip_path"]) as zf:
            ds = pydicom.dcmread(io.BytesIO(zf.read(row["inner_path"])))
        ok += 1
        for t in TAGS:
            v = ds.get(t)
            if v is not None and str(v).strip() != "":
                found[t][str(v)] = found[t].get(str(v), 0) + 1
    except Exception as exc:  # noqa: BLE001
        print("ERR", type(exc).__name__, str(exc)[:120])

print(f"read OK: {ok}/{len(rows)}")
for t in TAGS:
    if found[t]:
        top = sorted(found[t].items(), key=lambda kv: -kv[1])[:6]
        print(f"{t}: {top}")

with open("lit_search/dicom_tags_probe.json", "w", encoding="utf-8") as fh:
    json.dump({t: dict(sorted(v.items(), key=lambda kv: -kv[1])[:20]) for t, v in found.items()},
              fh, ensure_ascii=False, indent=1)
print("saved lit_search/dicom_tags_probe.json")
