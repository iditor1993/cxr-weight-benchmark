"""Stage 1：队列冻结、标签清洗、索引影像、切分、纳排计数与数据字典输入。

对应 研究方案_v4_定稿.md 第 3–6 节。
输出（project/data/processed/）：
  cohort_a_all_images.csv  队列 A 全部配对影像（35k 级）
  cohort_a_index.csv       队列 A 索引影像（每次住院一张）
  cohort_b_index.csv       队列 B（任意体重来源）索引影像（敏感性）
  splits_random.csv        患者级随机 70/15/15
  splits_temporal.csv      患者级时间外切分
  cohort_flow.json         纳排计数（流程图数据）
  qc_summary.json          标签清洗与质控汇总
  cohort_context.json      ICU 构成、通气、CRRT、LOS、身高可得性
"""

from __future__ import annotations

import json
from collections import OrderedDict

import duckdb
import pandas as pd

from common import (
    PROCESSED_DIR,
    PROJECT,
    load_config,
    read_sql,
    save_json,
    setup_logger,
    stable_fraction,
)


def main() -> None:
    log = setup_logger("extract_cohort")
    cfg = load_config()
    db_path = cfg["data"]["duckdb"]
    con = duckdb.connect(db_path, read_only=True)
    con.execute("PRAGMA threads=8")

    log.info("loading SQL definitions")
    for sql_name in ("01_weights.sql", "02_cxr_candidates.sql", "03_pairs_daily.sql",
                     "04_pairs_any.sql", "05_index_image.sql"):
        con.execute(read_sql(sql_name))
        log.info("executed %s", sql_name)

    # ---------- 纳排计数（流程图） ----------
    flow = OrderedDict()
    flow["n_icustays_total"] = int(con.execute("SELECT count(*) FROM main.icustays").fetchone()[0])
    flow["n_stays_with_portable_ap"] = int(
        con.execute("SELECT count(DISTINCT stay_id) FROM cxr").fetchone()[0])
    flow["n_stays_without_portable_ap"] = flow["n_icustays_total"] - flow["n_stays_with_portable_ap"]
    flow["n_portable_ap_images"] = int(con.execute("SELECT count(*) FROM cxr").fetchone()[0])
    flow["n_portable_ap_studies"] = int(
        con.execute("SELECT count(DISTINCT study_id) FROM cxr").fetchone()[0])
    flow["n_images_with_daily_weight_24h"] = int(
        con.execute("SELECT count(*) FROM pairs_daily").fetchone()[0])
    flow["n_images_without_any_weight_24h"] = flow["n_portable_ap_images"] - int(
        con.execute("SELECT count(*) FROM pairs_any").fetchone()[0])
    flow["n_stays_cohort_a"] = int(con.execute("SELECT count(DISTINCT stay_id) FROM pairs_daily").fetchone()[0])
    flow["n_patients_cohort_a"] = int(con.execute("SELECT count(DISTINCT subject_id) FROM pairs_daily").fetchone()[0])
    flow["n_stays_cohort_b"] = int(con.execute("SELECT count(DISTINCT stay_id) FROM pairs_any").fetchone()[0])
    flow["n_patients_cohort_b"] = int(con.execute("SELECT count(DISTINCT subject_id) FROM pairs_any").fetchone()[0])
    flow["n_index_images_cohort_a"] = int(con.execute("SELECT count(*) FROM index_daily").fetchone()[0])
    flow["n_index_images_cohort_b"] = int(con.execute("SELECT count(*) FROM index_any").fetchone()[0])
    log.info("flow: %s", json.dumps(flow, ensure_ascii=False))

    # ---------- 标签清洗 QC ----------
    qc = OrderedDict()
    qc["weight_rows_raw"] = int(con.execute("SELECT count(*) FROM w_raw").fetchone()[0])
    qc["weight_rows_after_null_filter"] = int(con.execute("SELECT count(*) FROM w_clean").fetchone()[0])
    qc["weight_rows_out_of_range"] = int(
        con.execute("SELECT count(*) FROM w_clean WHERE NOT in_range").fetchone()[0])
    qc["weight_rows_in_range"] = int(
        con.execute("SELECT count(*) FROM w_clean WHERE in_range").fetchone()[0])
    qc["lbs_converted_rows"] = int(
        con.execute("SELECT count(*) FROM w_clean WHERE itemid = 226531").fetchone()[0])
    qc["by_itemid"] = con.execute("""
        SELECT itemid,
               count(*) AS n_rows,
               sum(CASE WHEN in_range THEN 1 ELSE 0 END) AS n_in_range,
               min(CASE WHEN in_range THEN weight_kg END) AS min_kg,
               median(CASE WHEN in_range THEN weight_kg END) AS median_kg,
               max(CASE WHEN in_range THEN weight_kg END) AS max_kg
        FROM w_clean GROUP BY 1 ORDER BY 1
    """).fetchdf().to_dict("records")
    qc["duplicate_stay_time_item"] = int(con.execute("""
        SELECT count(*) FROM (
            SELECT stay_id, charttime, itemid, count(*) AS n
            FROM w_clean WHERE in_range GROUP BY 1,2,3 HAVING count(*) > 1)
    """).fetchone()[0])
    qc["weight_24h_change_gt10pct_stays"] = int(con.execute("""
        WITH t AS (
            SELECT stay_id, charttime, weight_kg,
                   lag(weight_kg) OVER (PARTITION BY stay_id ORDER BY charttime) AS prev
            FROM w_clean WHERE in_range AND itemid = 224639)
        SELECT count(DISTINCT stay_id) FROM t
        WHERE prev IS NOT NULL AND abs(weight_kg - prev) / prev > 0.10
    """).fetchone()[0])
    log.info("qc: %s", json.dumps(qc, ensure_ascii=False))

    # ---------- 导出配对表 ----------
    cohort_a_all = con.execute("SELECT * FROM pairs_daily ORDER BY subject_id, study_datetime").fetchdf()
    cohort_a_index = con.execute("SELECT * FROM index_daily ORDER BY subject_id, study_datetime").fetchdf()
    cohort_b_index = con.execute("SELECT * FROM index_any ORDER BY subject_id, study_datetime").fetchdf()
    for df in (cohort_a_all, cohort_a_index, cohort_b_index):
        df.drop(columns=["w_rank", "img_rank"], inplace=True, errors="ignore")
    cohort_a_all.to_csv(PROCESSED_DIR / "cohort_a_all_images.csv", index=False)
    cohort_a_index.to_csv(PROCESSED_DIR / "cohort_a_index.csv", index=False)
    cohort_b_index.to_csv(PROCESSED_DIR / "cohort_b_index.csv", index=False)
    log.info("exported cohort tables: A_all=%d, A_index=%d, B_index=%d",
             len(cohort_a_all), len(cohort_a_index), len(cohort_b_index))

    # ---------- 患者级切分 ----------
    seed = cfg["split"]["seed"]

    def assign_random(subject_ids: pd.Series) -> pd.Series:
        frac = subject_ids.map(lambda s: stable_fraction(str(s), seed))
        return pd.cut(frac, bins=[-0.001, 0.70, 0.85, 1.0], labels=["train", "val", "test"]).astype(str)

    subj_index = cohort_a_index[["subject_id"]].drop_duplicates().copy()
    subj_index["split_random"] = assign_random(subj_index["subject_id"])
    subj_index.to_csv(PROCESSED_DIR / "splits_random.csv", index=False)

    # temporal：按患者首次索引影像时间排序，前 70% train / 次 15% val / 末 15% test
    first_dt = (cohort_a_index.groupby("subject_id", as_index=False)["study_datetime"]
                .min().sort_values(["study_datetime", "subject_id"]).reset_index(drop=True))
    n = len(first_dt)
    first_dt["split_temporal"] = "train"
    first_dt.loc[int(0.70 * n):int(0.85 * n) - 1, "split_temporal"] = "val"
    first_dt.loc[int(0.85 * n):, "split_temporal"] = "test"
    first_dt[["subject_id", "split_temporal"]].to_csv(PROCESSED_DIR / "splits_temporal.csv", index=False)
    log.info("splits: random=%s, temporal=%s",
             subj_index["split_random"].value_counts().to_dict(),
             first_dt["split_temporal"].value_counts().to_dict())

    # ---------- 队列背景信息 ----------
    context = {}
    context["care_unit"] = con.execute("""
        SELECT first_careunit AS unit, count(*) AS n
        FROM index_daily GROUP BY 1 ORDER BY n DESC
    """).fetchdf().to_dict("records")
    context["height_available"] = con.execute("""
        SELECT count(*) AS n_stays,
               sum(CASE WHEN EXISTS (SELECT 1 FROM mimiciv_derived.height h
                                     WHERE h.stay_id = x.stay_id) THEN 1 ELSE 0 END) AS n_with_height
        FROM index_daily x
    """).fetchdf().to_dict("records")[0]
    context["invasive_ventilation"] = con.execute("""
        SELECT count(*) AS n_stays,
               sum(CASE WHEN EXISTS (SELECT 1 FROM mimiciv_derived.ventilation v
                                     WHERE v.stay_id = x.stay_id
                                       AND v.ventilation_status = 'InvasiveVent')
                        THEN 1 ELSE 0 END) AS n_ventilated
        FROM index_daily x
    """).fetchdf().to_dict("records")[0]
    context["rrt"] = con.execute("""
        SELECT count(*) AS n_stays,
               sum(CASE WHEN EXISTS (SELECT 1 FROM mimiciv_derived.rrt r
                                     WHERE r.stay_id = x.stay_id) THEN 1 ELSE 0 END) AS n_rrt
        FROM index_daily x
    """).fetchdf().to_dict("records")[0]
    context["los_days"] = con.execute("""
        SELECT median(date_diff('hour', i.intime, i.outtime)/24.0) AS median,
               quantile_cont(date_diff('hour', i.intime, i.outtime)/24.0, 0.25) AS p25,
               quantile_cont(date_diff('hour', i.intime, i.outtime)/24.0, 0.75) AS p75
        FROM index_daily x JOIN main.icustays i USING (stay_id)
    """).fetchdf().to_dict("records")[0]
    context["weight_distribution_index"] = con.execute("""
        SELECT count(*) AS n, median(weight_kg) AS median,
               quantile_cont(weight_kg, 0.05) AS p05, quantile_cont(weight_kg, 0.95) AS p95,
               sum(CASE WHEN weight_kg < 50 THEN 1 ELSE 0 END) AS lt50,
               sum(CASE WHEN weight_kg >= 100 THEN 1 ELSE 0 END) AS ge100
        FROM index_daily
    """).fetchdf().to_dict("records")[0]
    context["images_per_stay"] = con.execute("""
        WITH t AS (SELECT stay_id, count(*) AS n FROM pairs_daily GROUP BY 1)
        SELECT CASE WHEN n=1 THEN '1' WHEN n<=3 THEN '2-3' WHEN n<=10 THEN '4-10' ELSE '>10' END AS bucket,
               count(*) AS n_stays
        FROM t GROUP BY 1 ORDER BY 1
    """).fetchdf().to_dict("records")
    context["longitudinal"] = con.execute("""
        WITH s AS (
            SELECT stay_id, count(*) AS n, max(weight_kg)-min(weight_kg) AS rng
            FROM pairs_daily GROUP BY 1)
        SELECT count(*) AS n_multi,
               sum(CASE WHEN rng >= 1 THEN 1 ELSE 0 END) AS ge1,
               sum(CASE WHEN rng >= 3 THEN 1 ELSE 0 END) AS ge3,
               sum(CASE WHEN rng >= 5 THEN 1 ELSE 0 END) AS ge5,
               median(rng) AS median_rng
        FROM s WHERE n >= 2
    """).fetchdf().to_dict("records")[0]
    context["stays_per_patient"] = con.execute("""
        WITH t AS (SELECT subject_id, count(*) AS n FROM index_daily GROUP BY 1)
        SELECT CASE WHEN n=1 THEN '1' WHEN n=2 THEN '2' ELSE '>=3' END AS bucket,
               count(*) AS n_patients FROM t GROUP BY 1 ORDER BY 1
    """).fetchdf().to_dict("records")
    context["support_devices"] = con.execute("""
        SELECT avg(CASE WHEN ch."Support Devices" = 1 THEN 1.0 ELSE 0.0 END) AS pct_support_devices,
               avg(CASE WHEN ch."No Finding" = 1 THEN 1.0 ELSE 0.0 END) AS pct_no_finding
        FROM index_daily x
        LEFT JOIN main.cxr_chexpert ch ON ch.subject_id = x.subject_id AND ch.study_id = x.study_id
    """).fetchdf().to_dict("records")[0]

    save_json(PROCESSED_DIR / "cohort_flow.json", dict(flow))
    save_json(PROCESSED_DIR / "qc_summary.json", dict(qc))
    save_json(PROCESSED_DIR / "cohort_context.json", context)
    log.info("context: %s", json.dumps(context, ensure_ascii=False, default=str)[:1500])

    # ---------- 版本信息 ----------
    meta = {
        "plan_version": cfg["plan_version"],
        "duckdb": db_path,
        "n_icustays_total": flow["n_icustays_total"],
        "cohort_a_images": len(cohort_a_all),
        "cohort_a_index": len(cohort_a_index),
        "cohort_b_index": len(cohort_b_index),
        "note": "Stage 1 输出；质控回退规则将在影像预处理阶段更新索引影像",
    }
    save_json(PROJECT / "data" / "processed" / "stage1_meta.json", meta)
    log.info("Stage 1 完成")


if __name__ == "__main__":
    main()
