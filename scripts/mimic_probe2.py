"""Step 2: inventory of CXR tables, ICU stays and weight tables (main schema)."""

import json

import duckdb

DB = r"E:\clinical_research\MIMIC_IV_3.1\mimic_iv_3_1.duckdb"
con = duckdb.connect(DB, read_only=True)


def q(sql):
    return con.execute(sql).fetchdf()


def show(title, sql, save=None, out=None):
    print("\n==", title, "==")
    df = q(sql)
    with __import__("pandas").option_context("display.max_columns", 50, "display.width", 200):
        print(df.to_string())
    if save is not None and out is not None:
        out[save] = df.to_dict("records")
    return df


out = {}

show("icustays", """
SELECT count(*) AS n_stays, count(DISTINCT subject_id) AS n_patients,
       count(DISTINCT hadm_id) AS n_admissions,
       min(intime) AS first_intime, max(outtime) AS last_outtime
FROM main.icustays
""", "icustays", out)

show("cxr_metadata columns", "SELECT * FROM main.cxr_metadata LIMIT 3")
cols = q("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='cxr_metadata' ORDER BY ordinal_position")
print(cols.to_string())
out["cxr_metadata_cols"] = cols.to_dict("records")

show("cxr_metadata counts", """
SELECT count(*) AS n_rows, count(DISTINCT subject_id) AS n_subjects,
       count(DISTINCT study_id) AS n_studies, count(DISTINCT dicom_id) AS n_images
FROM main.cxr_metadata
""", "cxr_metadata_counts", out)

for tbl in ("cxr_records", "cxr_studies", "cxr_frontal_studies", "cxr_files", "cxr_zip_index",
            "cxr_chexpert", "cxr_negbio", "cxr_split"):
    try:
        c = q(f"SELECT count(*) AS n FROM main.{tbl}")
        cols = q(f"SELECT column_name FROM information_schema.columns WHERE table_name='{tbl}' ORDER BY ordinal_position")
        print(f"{tbl}: rows={int(c['n'][0])} cols={cols['column_name'].tolist()}")
        out.setdefault("table_info", {})[tbl] = {
            "rows": int(c["n"][0]), "cols": cols["column_name"].tolist()}
    except Exception as exc:  # noqa: BLE001
        print(tbl, "ERR", exc)

show("frontal studies sample", "SELECT * FROM main.cxr_frontal_studies LIMIT 3")

show("icustays + derived weight tables", """
SELECT 'first_day_weight' AS t, count(*) AS n_rows, count(DISTINCT stay_id) AS n_stays
FROM mimiciv_derived.first_day_weight
UNION ALL
SELECT 'weight_durations', count(*), count(DISTINCT stay_id) FROM mimiciv_derived.weight_durations
UNION ALL
SELECT 'height', count(*), count(DISTINCT stay_id) FROM mimiciv_derived.height
""", "derived_weight_tables", out)

show("weight_durations sample", "SELECT * FROM mimiciv_derived.weight_durations LIMIT 5")

show("d_items weight labels", """
SELECT itemid, label, abbreviation, linksto, category, unitname
FROM main.d_items
WHERE lower(label) LIKE '%weight%' OR lower(label) LIKE '%wt%'
ORDER BY itemid
""", "weight_items", out)

with open("lit_search/mimic_probe2.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("\nsaved lit_search/mimic_probe2.json")
