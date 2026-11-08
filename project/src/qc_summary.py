"""汇总自动质控指标，并生成 100 张人工复核样表（供双人一致性评估）。"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from common import FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, save_json, setup_logger


def main() -> None:
    log = setup_logger("qc_summary")
    qc = pd.read_csv(PROCESSED_DIR / "qc_images_index_320.csv")
    summary = {
        "n_images": int(len(qc)),
        "n_ok": int(qc["ok"].sum()),
        "ok_rate": float(qc["ok"].mean()),
        "rows_min": int(qc["rows"].min()), "rows_max": int(qc["rows"].max()),
        "cols_min": int(qc["cols"].min()), "cols_max": int(qc["cols"].max()),
        "kvp_missing": int(qc["kvp"].isna().sum()),
        "kvp_median": float(qc["kvp"].median()),
        "exposure_index_missing": int(qc["exposure_index"].isna().sum()),
        "imager_pixel_spacing_unique": sorted(set(qc["imager_pixel_spacing"].dropna().round(4))),
        "mean_intensity_p05": float(qc["mean_intensity"].quantile(0.05)),
        "mean_intensity_p95": float(qc["mean_intensity"].quantile(0.95)),
        "frac_saturated_p95": float(qc["frac_saturated"].quantile(0.95)),
    }
    save_json(TABLE_DIR / "qc_image_summary.json", summary)
    log.info(json.dumps(summary, ensure_ascii=False))

    # 100 张人工复核样表（固定随机种子）
    sample = qc.sample(n=min(100, len(qc)), random_state=20261004)
    sample[["dicom_id", "rows", "cols", "kvp", "exposure_index", "mean_intensity",
            "frac_saturated", "imager_pixel_spacing"]].to_csv(
        PROCESSED_DIR / "qc_manual_review_sample.csv", index=False)
    ncol, nrow = 10, 10
    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 22))
    for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):
        img = np.load(PROCESSED_DIR.parent / "features" / "index_320" / f"{row['dicom_id']}.npy")
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(str(row["dicom_id"])[:8], fontsize=6)
        ax.axis("off")
    fig.suptitle("QC 人工复核样表（100 张，随机种子 20261004）", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_qc_review_sheet.png", dpi=120)
    plt.close(fig)
    log.info("manual review sheet -> %s", FIGURE_DIR / "fig_qc_review_sheet.png")


if __name__ == "__main__":
    main()
