"""生成两项人工复核工具包：
  1) QC 复核：100 张编号缩略图 + 填写模板（rater1/rater2）
  2) 器械 CAM 复核：100 张 CAM 叠加图 + 填写模板
产物在 project/docs/human_review/ 下，可直接交给两名评估者。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import DOCS_DIR, FEATURE_DIR, PROCESSED_DIR, setup_logger
from dataset import WeightCXR
from models import build_model, normalization_for

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REVIEW_DIR = DOCS_DIR / "human_review"


def write_template(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_qc_package(cache: Path) -> None:
    qc = pd.read_csv(PROCESSED_DIR / "qc_images_index_320.csv")
    sample = qc.sample(n=min(100, len(qc)), random_state=20261004).reset_index(drop=True)
    sample.insert(0, "idx", np.arange(1, len(sample) + 1))
    # 编号缩略图（10×10）
    fig, axes = plt.subplots(10, 10, figsize=(22, 24))
    for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):
        img = np.load(cache / f"{row['dicom_id']}.npy")
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"#{row['idx']}", fontsize=7)
        ax.axis("off")
    fig.suptitle("QC 人工复核（100 张，编号对应 qc_review_template.csv）", fontsize=15)
    fig.tight_layout()
    fig.savefig(REVIEW_DIR / "qc_review_sheet_numbered.png", dpi=110)
    plt.close(fig)
    rows = [{"idx": int(r["idx"]), "dicom_id": r["dicom_id"],
             "rater1_ok (1=合格/0=不合格)": "", "rater2_ok (1=合格/0=不合格)": "",
             "rater1_notes": "", "rater2_notes": ""} for _, r in sample.iterrows()]
    write_template(REVIEW_DIR / "qc_review_template.csv", rows,
                   ["idx", "dicom_id", "rater1_ok (1=合格/0=不合格)",
                    "rater2_ok (1=合格/0=不合格)", "rater1_notes", "rater2_notes"])


def make_cam_package(cache: Path, checkpoint: Path, backbone: str = "swin_t", size: int = 308) -> None:
    from attention_occlusion import gradcam

    ds = WeightCXR("test", split_kind="random", size=size, train=False,
                   mean=normalization_for(backbone)[0], std=normalization_for(backbone)[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(backbone, img_size=size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rng = np.random.default_rng(20261004)
    idxs = rng.choice(len(ds), size=min(100, len(ds)), replace=False)
    fig, axes = plt.subplots(10, 10, figsize=(22, 24))
    rows = []
    for n, (ax, i) in enumerate(zip(axes.ravel(), idxs), start=1):
        sample = ds[int(i)]
        x = sample["image"].unsqueeze(0).to(device)
        cam = gradcam(model, x, backbone, stage=-2)[0]
        ax.imshow(sample["image"][0].numpy(), cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(f"#{n}", fontsize=7)
        ax.axis("off")
        rows.append({"idx": n, "dicom_id": sample["dicom_id"],
                     "rater1_mask_ok (1/0)": "", "rater2_mask_ok (1/0)": "",
                     "rater1_notes": "", "rater2_notes": ""})
    fig.suptitle("器械/注意力 CAM 掩膜复核（100 张，编号对应 cam_review_template.csv）", fontsize=15)
    fig.tight_layout()
    fig.savefig(REVIEW_DIR / "cam_review_sheet_numbered.png", dpi=110)
    plt.close(fig)
    write_template(REVIEW_DIR / "cam_review_template.csv", rows,
                   ["idx", "dicom_id", "rater1_mask_ok (1/0)", "rater2_mask_ok (1/0)",
                    "rater1_notes", "rater2_notes"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(FEATURE_DIR / "checkpoints" / "swin_t_308_s1.pt"))
    args = ap.parse_args()
    log = setup_logger("human_review_packages")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    cache = FEATURE_DIR / "index_320"
    make_qc_package(cache)
    log.info("QC package -> %s", REVIEW_DIR)
    if Path(args.checkpoint).exists():
        make_cam_package(cache, Path(args.checkpoint))
        log.info("CAM package -> %s", REVIEW_DIR)
    else:
        log.warning("checkpoint not found: %s", args.checkpoint)
    (REVIEW_DIR / "README.md").write_text(
        "# 人工复核说明\n\n"
        "本目录包含两项待人工完成的复核（研究方案 D9/D10）：\n\n"
        "## 1. 影像质控一致性（QC）\n"
        "- 打开 `qc_review_sheet_numbered.png`，对照 `qc_review_template.csv` 中的编号；\n"
        "- 两名评估者独立在 rater1_ok / rater2_ok 列填写 1（合格）或 0（不合格）；\n"
        "- 填写完成后运行：\n"
        "  `python project/src/review_agreement.py --template project/docs/human_review/qc_review_template.csv "
        "--col1 \"rater1_ok (1=合格/0=不合格)\" --col2 \"rater2_ok (1=合格/0=不合格)\" "
        "--out project/results/tables/qc_agreement.json`\n\n"
        "## 2. 器械/注意力 CAM 掩膜复核\n"
        "- 打开 `cam_review_sheet_numbered.png`，对照 `cam_review_template.csv`；\n"
        "- 两名评估者独立判断 CAM 高亮区域是否与器械/导管区域一致（1=一致/0=不一致）；\n"
        "- 填写完成后运行：\n"
        "  `python project/src/review_agreement.py --template project/docs/human_review/cam_review_template.csv "
        "--col1 \"rater1_mask_ok (1/0)\" --col2 \"rater2_mask_ok (1/0)\" "
        "--out project/results/tables/cam_agreement.json`\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
