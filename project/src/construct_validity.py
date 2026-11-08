"""构念效度分析（研究方案第 10 节）：Δ预测体重 vs Δ真实体重。"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import PROCESSED_DIR, TABLE_DIR, save_json, setup_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="predict_all 输出的逐影像预测 CSV")
    ap.add_argument("--cross_sectional_mae", type=float, required=True)
    args = ap.parse_args()
    log = setup_logger("construct_validity")
    df = pd.read_csv(PROCESSED_DIR / args.predictions)
    df = df.dropna(subset=["pred", "weight_kg"]).sort_values(["stay_id", "study_datetime"])
    rows = []
    for stay_id, g in df.groupby("stay_id"):
        if len(g) < 2:
            continue
        g = g.drop_duplicates(subset=["weight_kg", "pred"], keep="first")
        if len(g) < 2:
            continue
        first, last = g.iloc[0], g.iloc[-1]
        rows.append({
            "stay_id": stay_id,
            "d_weight_pred": last["pred"] - first["pred"],
            "d_weight_true": last["weight_kg"] - first["weight_kg"],
            "n_images": len(g),
        })
    d = pd.DataFrame(rows)
    if len(d) < 20:
        log.warning("too few longitudinal pairs: %d", len(d))
        return
    r = float(np.corrcoef(d["d_weight_true"], d["d_weight_pred"])[0, 1])
    mae_delta = float(np.mean(np.abs(d["d_weight_true"] - d["d_weight_pred"])))
    slope = float(np.polyfit(d["d_weight_true"], d["d_weight_pred"], 1)[0])
    rule = {
        "r": r, "mae_delta_kg": mae_delta, "slope": slope,
        "cross_sectional_mae_kg": args.cross_sectional_mae,
        "r_below_0.5": r < 0.5,
        "mae_delta_above_1.5x": mae_delta > 1.5 * args.cross_sectional_mae,
    }
    rule["construct_validity_insufficient"] = bool(rule["r_below_0.5"] or rule["mae_delta_above_1.5x"])
    save_json(TABLE_DIR / "construct_validity.json", rule)
    d.to_csv(PROCESSED_DIR / "construct_validity_pairs.csv", index=False)
    log.info(json.dumps(rule, ensure_ascii=False))


if __name__ == "__main__":
    main()
