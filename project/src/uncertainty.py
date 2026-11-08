"""不确定性评估（研究方案 8.4）：种子集成的区间覆盖率、宽度分布与校准。"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import PROCESSED_DIR, TABLE_DIR, save_json, setup_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--predictions", default=None)
    args = ap.parse_args()
    log = setup_logger(f"uncertainty_{args.tag}")
    path = (PROCESSED_DIR / args.predictions) if args.predictions else (
        PROCESSED_DIR / f"test_predictions_{args.tag}.csv")
    df = pd.read_csv(path)
    if "pred_sigma" in df.columns and df["pred_sigma"].notna().any():
        df["pred_std"] = df["pred_sigma"]
        log.info("using heteroscedastic sigma as predictive std")
    if "pred_std" not in df.columns or df["pred_std"].max() == 0:
        log.warning("no ensemble std available; skipping")
        return
    z90 = 1.6448536269514722
    err = np.abs(df["weight_kg"] - df["pred"])
    covered = err <= z90 * df["pred_std"]
    width = 2 * z90 * df["pred_std"]
    out = {
        "n": int(len(df)),
        "coverage_90_nominal": float(covered.mean()),
        "mean_interval_width_kg": float(width.mean()),
        "median_interval_width_kg": float(np.median(width)),
        "p90_interval_width_kg": float(np.quantile(width, 0.90)),
        "calibration_slope_abs_err_on_std": float(np.polyfit(df["pred_std"], err, 1)[0]),
        "pct_width_gt_10kg": float(np.mean(width > 10)),
        "pct_width_gt_15kg": float(np.mean(width > 15)),
        "pct_width_gt_20kg": float(np.mean(width > 20)),
        "coverage_by_weight_stratum": {},
    }
    for name, g in df.groupby(pd.cut(df["weight_kg"], [0, 50, 80, 100, 1000],
                                     labels=["<50", "50-80", "80-100", ">=100"]) , observed=True):
        if len(g) >= 20:
            e = np.abs(g["weight_kg"] - g["pred"])
            out["coverage_by_weight_stratum"][str(name)] = float((e <= z90 * g["pred_std"]).mean())
    save_json(TABLE_DIR / f"uncertainty_{args.tag}.json", out)
    log.info(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
