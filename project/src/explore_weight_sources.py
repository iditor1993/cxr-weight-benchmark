"""探查 MIMIC-IV 中可用于"常规体重估算"的字段：OMR、入院体重、既往体重、身高。"""

from __future__ import annotations

import duckdb
import pandas as pd

from common import PROCESSED_DIR, load_config

con = duckdb.connect(load_config()["data"]["duckdb"], read_only=True)
con.execute("PRAGMA threads=8")

print("=== omr result_name 分布 ===")
omr = con.execute("""
    SELECT result_name, count(*) AS n, count(DISTINCT subject_id) AS n_subjects
    FROM main.omr GROUP BY 1 ORDER BY n DESC LIMIT 20
""").fetchdf()
print(omr.to_string())

test = pd.read_csv(PROCESSED_DIR / "test_predictions_swin_ensemble.csv")
test_ids = test["subject_id"].unique().tolist()
con.register("test_ids", pd.DataFrame({"subject_id": test_ids}))

print("\n=== 测试集（1,303 次住院）可用来源 ===")
q = con.execute("""
WITH stays AS (
    SELECT t.subject_id, t.stay_id, t.hadm_id, t.study_datetime, t.weight_kg, t.height_cm
    FROM read_csv_auto(?) t
)
SELECT
  count(*) AS n_stays,
  sum(CASE WHEN height_cm IS NOT NULL THEN 1 ELSE 0 END) AS with_height,
  sum(CASE WHEN EXISTS (SELECT 1 FROM main.chartevents c
                        WHERE c.stay_id = s.stay_id AND c.itemid IN (226512,226531)
                          AND (CASE WHEN c.itemid=226531 THEN c.valuenum/2.2046226218 ELSE c.valuenum END) BETWEEN 20 AND 350)
           THEN 1 ELSE 0 END) AS with_admission_weight,
  sum(CASE WHEN EXISTS (SELECT 1 FROM main.chartevents c
                        WHERE c.subject_id = s.subject_id AND c.charttime < s.study_datetime
                          AND c.itemid IN (224639,226512,226531)
                          AND (CASE WHEN c.itemid=226531 THEN c.valuenum/2.2046226218 ELSE c.valuenum END) BETWEEN 20 AND 350)
           THEN 1 ELSE 0 END) AS with_prior_weight,
  sum(CASE WHEN EXISTS (SELECT 1 FROM main.omr o
                        WHERE o.subject_id = s.subject_id AND lower(o.result_name) LIKE '%weight%'
                          AND try_cast(o.result_value AS DOUBLE) BETWEEN 20 AND 350)
           THEN 1 ELSE 0 END) AS with_omr_weight,
  sum(CASE WHEN EXISTS (SELECT 1 FROM main.omr o
                        WHERE o.subject_id = s.subject_id AND lower(o.result_name) LIKE '%bmi%'
                          AND try_cast(o.result_value AS DOUBLE) BETWEEN 10 AND 80)
           THEN 1 ELSE 0 END) AS with_omr_bmi
FROM stays s
""", [str(PROCESSED_DIR / "test_predictions_swin_ensemble.csv").replace("\\", "/")]).fetchdf()
print(q.to_string())
