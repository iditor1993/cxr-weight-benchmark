"""假设性剂量影响分析（研究方案 9.4）。

模型输出 ABW；对需要 AdjBW/IBW 的药物，仅在有身高亚组内计算。
MCID：剂量偏差 >10% 视为有临床意义。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import PROCESSED_DIR, TABLE_DIR, save_json, setup_logger

# 药物 -> (公式名, 每 kg 剂量, 依据的体重标量)
DRUGS = {
    "vancomycin_load_25mgkg": ("25 mg/kg", 25.0, "ABW"),
    "enoxaparin_treatment_1mgkg": ("1 mg/kg", 1.0, "ABW"),
    "propofol_induction_2mgkg": ("2 mg/kg", 2.0, "ABW"),
    "rocuronium_0.6mgkg": ("0.6 mg/kg", 0.6, "ABW"),
    "gentamicin_5mgkg_adjbw": ("5 mg/kg (AdjBW)", 5.0, "AdjBW"),
}


def devine_ibw(height_cm: np.ndarray, male: np.ndarray) -> np.ndarray:
    inches = height_cm / 2.54
    return np.where(male == 1, 50.0 + 2.3 * (inches - 60.0), 45.5 + 2.3 * (inches - 60.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    log = setup_logger(f"dose_simulation_{args.tag}")
    df = pd.read_csv(PROCESSED_DIR / f"test_predictions_{args.tag}.csv")
    df = df.dropna(subset=["weight_kg", "pred"])
    male = (df["gender"] == "M").astype(int).values

    rows = []
    for name, (label, mg_per_kg, scalar) in DRUGS.items():
        if scalar == "ABW":
            dose_true = mg_per_kg * df["weight_kg"].values
            dose_pred = mg_per_kg * df["pred"].values
            n = len(df)
        else:
            sub = df.dropna(subset=["height_cm"]).copy()
            if len(sub) < 20:
                continue
            ibw = devine_ibw(sub["height_cm"].values, (sub["gender"] == "M").astype(int).values)
            adj_true = ibw + 0.4 * (sub["weight_kg"].values - ibw)
            adj_pred = ibw + 0.4 * (sub["pred"].values - ibw)
            dose_true = mg_per_kg * adj_true
            dose_pred = mg_per_kg * adj_pred
            n = len(sub)
        rel = np.abs(dose_pred - dose_true) / np.maximum(dose_true, 1e-6)
        rows.append({
            "drug": name, "regimen": label, "weight_scalar": scalar, "n": int(n),
            "median_abs_pct_diff": float(np.median(rel) * 100),
            "p90_abs_pct_diff": float(np.quantile(rel, 0.90) * 100),
            "pct_dose_diff_gt10": float(np.mean(rel > 0.10) * 100),
            "pct_dose_diff_gt20": float(np.mean(rel > 0.20) * 100),
        })
    save_json(TABLE_DIR / f"dose_simulation_{args.tag}.json", {"rows": rows, "mcid": 0.10})
    for r in rows:
        log.info("%s: n=%d >10%%=%.1f%% median|Δ%%|=%.1f", r["drug"], r["n"],
                 r["pct_dose_diff_gt10"], r["median_abs_pct_diff"])


if __name__ == "__main__":
    main()
