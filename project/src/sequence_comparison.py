"""序列模型 vs 同批影像的单张集成预测（同集对照）。"""

from __future__ import annotations

import json

import pandas as pd

from common import MAIN_TAG, PROCESSED_DIR, TABLE_DIR, save_json, setup_logger
from metrics import regression_metrics


def main() -> None:
    log = setup_logger("sequence_comparison")
    seq = pd.read_csv(PROCESSED_DIR / "sequence_test_predictions.csv")
    single = pd.read_csv(PROCESSED_DIR / f"preds_all_{MAIN_TAG}.csv")
    single = single[["dicom_id", "pred"]].rename(columns={"pred": "pred_single"})
    merged = seq.merge(single, left_on="last_dicom_id", right_on="dicom_id", how="left")
    if merged["pred_single"].isna().any():
        log.warning("missing single-image predictions: %d", int(merged["pred_single"].isna().sum()))
    m_seq = regression_metrics(merged["weight_kg"].values, merged["pred"].values)
    m_single = regression_metrics(merged["weight_kg"].values, merged["pred_single"].values)
    out = {"n": int(len(merged)), "sequence_model": m_seq, "single_image_same_set": m_single,
           "mae_difference_kg": float(m_single["mae"] - m_seq["mae"])}
    save_json(TABLE_DIR / "sequence_comparison.json", out)
    log.info(json.dumps(out, ensure_ascii=False, indent=1)[:1200])


if __name__ == "__main__":
    main()
