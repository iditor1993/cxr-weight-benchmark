"""Step 7: admission-vs-daily weight discrepancy + fair baseline comparison."""

import json

import duckdb
import numpy as np
import pandas as pd

DB = r"E:\clinical_research\MIMIC_IV_3.1\mimic_iv_3_1.duckdb"
con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA threads=8")


def q(sql):
    return con.execute(sql).fetchdf()


out = {}

q("""
CREATE OR REPLACE TEMP TABLE w AS
SELECT stay_id, subject_id, charttime, itemid,
       CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END AS kg
FROM main.chartevents
WHERE itemid IN (224639, 226512, 226531, 226846) AND valuenum IS NOT NULL
  AND (CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END) BETWEEN 20 AND 350
""")

# admission weight (nurse-documented, often estimated) vs first daily weight (bed scale)
df = q("""
WITH adm AS (
  SELECT stay_id, subject_id, charttime, kg,
         row_number() OVER (PARTITION BY stay_id ORDER BY charttime) AS rn
  FROM w WHERE itemid IN (226512, 226531)),
day AS (
  SELECT stay_id, min(charttime) AS t_daily, avg(kg) AS kg_daily
  FROM w WHERE itemid = 224639 GROUP BY 1)
SELECT a.stay_id, a.subject_id, a.kg AS adm_kg, d.kg_daily,
       abs(a.kg - d.kg_daily) AS abs_diff,
       abs(a.kg - d.kg_daily)/d.kg_daily AS rel_diff
FROM adm a JOIN day d USING (stay_id)
WHERE a.rn = 1
""")

print(f"== admission weight vs first daily weight: n={len(df)} stays ==")
stats = {
    "n_stays": int(len(df)),
    "median_abs_diff_kg": round(float(df["abs_diff"].median()), 2),
    "p75_abs_diff_kg": round(float(df["abs_diff"].quantile(0.75)), 2),
    "p90_abs_diff_kg": round(float(df["abs_diff"].quantile(0.90)), 2),
    "pct_ge5kg": round(float((df["abs_diff"] >= 5).mean() * 100), 1),
    "pct_ge10kg": round(float((df["abs_diff"] >= 10).mean() * 100), 1),
    "pct_rel_ge10pct": round(float((df["rel_diff"] >= 0.10).mean() * 100), 1),
    "pct_rel_ge20pct": round(float((df["rel_diff"] >= 0.20).mean() * 100), 1),
}
for k, v in stats.items():
    print(f"  {k}: {v}")
out["adm_vs_daily"] = stats

# --- baseline comparison on core cohort stays with height ---------------------
df2 = q("""
WITH core_stay AS (
  SELECT c.stay_id, c.subject_id, median(c.weight_kg) AS actual_kg
  FROM (
    SELECT i.subject_id, i.stay_id, m.dicom_id, m.study_datetime,
           (SELECT CASE WHEN w2.itemid=226531 THEN w2.valuenum/2.2046226218 ELSE w2.valuenum END
            FROM main.chartevents w2
            WHERE w2.stay_id=i.stay_id AND w2.itemid IN (224639,226512,226531,226846)
              AND w2.valuenum IS NOT NULL
              AND (CASE WHEN w2.itemid=226531 THEN w2.valuenum/2.2046226218 ELSE w2.valuenum END) BETWEEN 20 AND 350
              AND w2.charttime BETWEEN m.study_datetime - INTERVAL 24 HOUR AND m.study_datetime + INTERVAL 24 HOUR
            ORDER BY abs(epoch(w2.charttime - m.study_datetime)) LIMIT 1) AS weight_kg
    FROM main.icustays i
    JOIN main.cxr_metadata m ON m.subject_id=i.subject_id
     AND m.study_datetime BETWEEN i.intime AND i.outtime
    WHERE m.view_position='AP' AND m.procedure_desc ILIKE '%PORTABLE%') c
  WHERE c.weight_kg IS NOT NULL
  GROUP BY 1,2),
sex AS (SELECT subject_id, max(CASE WHEN gender='M' THEN 1 ELSE 0 END) AS male
        FROM main.patients GROUP BY 1),
h AS (SELECT stay_id, avg(height) AS height_cm FROM mimiciv_derived.height GROUP BY 1)
SELECT c.stay_id, c.actual_kg, s.male, h.height_cm
FROM core_stay c JOIN sex s USING (subject_id) JOIN h USING (stay_id)
WHERE h.height_cm BETWEEN 120 AND 220
""")

df2["height_in"] = df2["height_cm"] / 2.54
df2["ibw"] = np.where(df2["male"] == 1,
                      50 + 2.3 * (df2["height_in"] - 60),
                      45.5 + 2.3 * (df2["height_in"] - 60))
valid = df2[(df2["ibw"] > 20) & (df2["ibw"] < 200)].copy()

# OLS: weight ~ height + sex
X = np.column_stack([np.ones(len(valid)), valid["height_cm"].values, valid["male"].values])
y = valid["actual_kg"].values
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
pred_ols = X @ beta

mean_pred = np.full(len(valid), y.mean())
mae = lambda a, b: float(np.mean(np.abs(a - b)))
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

res = {
    "n_stays": int(len(valid)),
    "median_actual_kg": round(float(np.median(y)), 1),
    "mean_actual_kg": round(float(y.mean()), 1),
    "baseline_population_mean": {"mae": round(mae(y, mean_pred), 2), "rmse": round(rmse(y, mean_pred), 2)},
    "baseline_devine_ibw": {"mae": round(mae(y, valid["ibw"].values), 2),
                            "rmse": round(rmse(y, valid["ibw"].values), 2)},
    "baseline_ols_height_sex": {"mae": round(mae(y, pred_ols), 2), "rmse": round(rmse(y, pred_ols), 2),
                                "beta": [round(float(b), 3) for b in beta]},
}

print("\n== baseline comparison (ICU stays with portable AP CXR + height) ==")
for k, v in res.items():
    print(f"  {k}: {v}")
out["baselines"] = res

with open("lit_search/mimic_probe7.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("\nsaved lit_search/mimic_probe7.json")
