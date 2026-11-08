"""Step 1: schema + weight-label inventory in the local MIMIC-IV 3.1 DuckDB."""

import json

import duckdb

DB = r"E:\clinical_research\MIMIC_IV_3.1\mimic_iv_3_1.duckdb"
con = duckdb.connect(DB, read_only=True)


def q(sql):
    return con.execute(sql).fetchdf()


out = {}

print("== schemas/tables ==")
tables = q("""
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema NOT IN ('information_schema','pg_catalog')
ORDER BY 1,2
""")
print(tables.to_string())
out["tables"] = tables.to_dict("records")

print("\n== icustays ==")
c = q("""
SELECT count(*) AS n_stays,
       count(DISTINCT subject_id) AS n_patients,
       count(DISTINCT hadm_id) AS n_admissions
FROM icu.icustays
""")
print(c.to_string())
out["icustays"] = c.to_dict("records")[0]

print("\n== weight-like items in d_items ==")
d = q("""
SELECT itemid, label, abbreviation, linksto, category, unitname, param_type
FROM icu.d_items
WHERE lower(label) LIKE '%weight%' OR lower(label) LIKE '%wt%'
ORDER BY itemid
""")
print(d.to_string())
out["weight_items"] = d.to_dict("records")

ids = [int(x) for x in d["itemid"].tolist()]
if ids:
    idlist = ",".join(str(i) for i in ids)
    print("\n== chartevents rows per weight item ==")
    e = q(f"""
    SELECT itemid, count(*) AS n_rows,
           count(DISTINCT stay_id) AS n_stays,
           count(DISTINCT subject_id) AS n_subjects
    FROM icu.chartevents
    WHERE itemid IN ({idlist})
    GROUP BY itemid
    ORDER BY n_rows DESC
    """)
    print(e.to_string())
    out["weight_event_counts"] = e.to_dict("records")

    print("\n== value ranges (kg items only) ==")
    v = q(f"""
    SELECT itemid,
           count(*) AS n,
           sum(CASE WHEN valuenum IS NOT NULL THEN 1 ELSE 0 END) AS n_numeric,
           min(valuenum) AS min_v, quantile_cont(valuenum, 0.01) AS p01,
           median(valuenum) AS med, quantile_cont(valuenum, 0.99) AS p99,
           max(valuenum) AS max_v
    FROM icu.chartevents
    WHERE itemid IN ({idlist})
    GROUP BY itemid
    """)
    print(v.to_string())
    out["weight_value_ranges"] = v.to_dict("records")

with open("lit_search/mimic_probe1.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("\nsaved lit_search/mimic_probe1.json")
