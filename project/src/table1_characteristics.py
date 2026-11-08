"""生成 Table 1（队列特征）所需的汇总数字，供稿件直接引用。"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from common import PROCESSED_DIR, TABLE_DIR, load_config, save_json, setup_logger


def main() -> None:
    log = setup_logger("table1_characteristics")
    coh = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    coh = coh.merge(splits[["subject_id", "split_random"]], on="subject_id", how="left")
    con = duckdb.connect(load_config()["data"]["duckdb"], read_only=True)
    con.execute("PRAGMA threads=8")
    p = str(PROCESSED_DIR / "cohort_a_index.csv").replace("\\", "/")
    sex = con.execute("SELECT subject_id, gender, anchor_age FROM main.patients").fetchdf()
    los = con.execute("""
        SELECT stay_id, date_diff('minute', intime, outtime)/1440.0 AS los_days
        FROM main.icustays WHERE stay_id IN (SELECT stay_id FROM read_csv_auto(?))""",
        [p]).fetchdf()
    vent = con.execute("""
        SELECT stay_id, count(1) AS n FROM main.chartevents
        WHERE itemid = 223848 AND stay_id IN (SELECT stay_id FROM read_csv_auto(?))
        GROUP BY 1""", [p]).fetchdf()
    con.close()
    df = coh.merge(sex, on="subject_id", how="left").merge(los, on="stay_id", how="left")
    df["ventilated"] = df["stay_id"].isin(set(vent["stay_id"]))
    out = {
        "n_stays": int(len(df)),
        "n_patients": int(df["subject_id"].nunique()),
        "age_median": float(df["anchor_age"].median()),
        "age_iqr": [float(df["anchor_age"].quantile(0.25)), float(df["anchor_age"].quantile(0.75))],
        "male_n": int((df["gender"] == "M").sum()),
        "male_pct": float((df["gender"] == "M").mean()),
        "weight_median": float(df["weight_kg"].median()),
        "weight_iqr": [float(df["weight_kg"].quantile(0.25)), float(df["weight_kg"].quantile(0.75))],
        "weight_p5": float(df["weight_kg"].quantile(0.05)),
        "weight_p95": float(df["weight_kg"].quantile(0.95)),
        "n_lt50": int((df["weight_kg"] < 50).sum()),
        "n_ge100": int((df["weight_kg"] >= 100).sum()),
        "los_median_days": float(df["los_days"].median()),
        "los_iqr_days": [float(df["los_days"].quantile(0.25)), float(df["los_days"].quantile(0.75))],
        "ventilated_pct": float(df["ventilated"].mean()),
        "top_units": df["first_careunit"].value_counts().head(6).to_dict(),
        "gap_hours_median": float(df["gap_hours"].abs().median()),
        "gap_within_6h_pct": float((df["gap_hours"].abs() <= 6).mean()),
        "splits": df["split_random"].value_counts().to_dict(),
        "test_sex": {"male": int((df.loc[df.split_random == "test", "gender"] == "M").sum()),
                     "female": int((df.loc[df.split_random == "test", "gender"] == "F").sum())},
    }
    save_json(TABLE_DIR / "table1_characteristics.json", out)
    log.info(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
