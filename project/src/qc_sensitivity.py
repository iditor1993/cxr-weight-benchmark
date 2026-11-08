"""自动质控敏感性分析：在严格 QC 子集上重算集成模型性能。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import PROCESSED_DIR, TABLE_DIR, main_pred_file, save_json, setup_logger
from metrics import bootstrap_ci_arrays, regression_metrics


def main() -> None:
    log = setup_logger("qc_sensitivity")
    preds = pd.read_csv(main_pred_file())
    qc = pd.read_csv(PROCESSED_DIR / "qc_images_index_320.csv")
    df = preds.merge(qc, on="dicom_id", how="left")
    # 严格规则：宽高比 0.5–2.0、均值强度 1%–99% 分位、饱和像素 <1%、KVP 缺失视为不满足
    aspect = df["rows"] / df["cols"]
    mi = df["mean_intensity"]
    strict = ((aspect.between(0.5, 2.0))
              & (mi.between(mi.quantile(0.01), mi.quantile(0.99)))
              & (df["frac_saturated"] < 0.01)
              & (df["kvp"].notna()))
    out = {"n_all": int(len(df)), "n_strict": int(strict.sum()),
           "strict_rule": "aspect 0.5-2.0 & mean intensity 1-99pct & saturated<1% & KVP present"}
    if strict.sum() >= 50:
        d = df[strict]
        m = regression_metrics(d["weight_kg"].values, d["pred"].values)
        pt, lo, hi = bootstrap_ci_arrays(d["weight_kg"].values, d["pred"].values,
                                         d["subject_id"].values, "mae")
        out["strict_metrics"] = m
        out["strict_mae_ci95"] = [pt, lo, hi]
    m_all = regression_metrics(df["weight_kg"].values, df["pred"].values)
    out["all_metrics"] = m_all
    save_json(TABLE_DIR / "qc_sensitivity.json", out)
    log.info(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
