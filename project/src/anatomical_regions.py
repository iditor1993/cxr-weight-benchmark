"""对测试集影像运行胸片解剖分割，导出肺/纵隔/膈下区域掩膜并做自动 QC。"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

import torchxrayvision.baseline_models.chestx_anatomy as ca
from common import FEATURE_DIR, FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, load_config, save_json, setup_logger
from dataset import WeightCXR
from models import normalization_for

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLASSES = {"right_lung": "right lung", "left_lung": "left lung",
           "cardiomediastinum": "cardiomediastinum",
           "left_hemidiaphragm": "left hemidiaphragm",
           "right_hemidiaphragm": "right hemidiaphragm",
           "liver": "liver", "stomach": "stomach", "small_bowel": "small bowel",
           "duodenum": "duodenum", "pancreas": "pancreas",
           "kidney_left": "kidney left", "kidney_right": "kidney right"}


def largest_component_fraction(mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0.0
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return float(sizes.max() / mask.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--split", default="random")
    ap.add_argument("--tag", default="anatomy_regions_test60")
    args = ap.parse_args()
    log = setup_logger(args.tag)
    device = torch.device("cuda")
    model = ca.UNetResNet50(weights=True).to(device).eval()
    mean, std = normalization_for("swin_t")
    ds = WeightCXR("test", split_kind=args.split, size=308, train=False, mean=mean, std=std)
    rng = np.random.default_rng(20261007)      # 与 pilot 相同的 60 张
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)

    out_masks, qc_rows, examples = {}, [], []
    for k, i in enumerate(idxs, 1):
        s = ds[int(i)]
        img = s["image"][0].numpy()  # 取单通道；已归一化 [-1,1]
        dicom_id = s["dicom_id"]
        x = torch.from_numpy(img)[None, None] * std + mean    # 还原到 0-1
        x = x * 2048 - 1024                                    # xrv 归一化
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False).to(device)
        with torch.no_grad():
            logits = model(x)
        prob = torch.sigmoid(logits[0])
        masks512 = (prob > 0.5).float().cpu().numpy()
        # 下采样回 308 与模型输入对齐
        m = torch.from_numpy(masks512).unsqueeze(0)
        m308 = (F.interpolate(m, size=(308, 308), mode="bilinear", align_corners=False)[0] > 0.5)
        m308 = m308.numpy()
        region = {k2: m308[ca.TARGETS.index(v)].astype(np.uint8) for k2, v in CLASSES.items()}
        sub_dia = np.clip(region["liver"] + region["stomach"] + region["small_bowel"] +
                          region["duodenum"] + region["pancreas"] +
                          region["kidney_left"] + region["kidney_right"], 0, 1).astype(np.uint8)
        region["subdiaphragmatic"] = sub_dia
        lung = np.clip(region["right_lung"] + region["left_lung"], 0, 1)
        region["lungs"] = lung.astype(np.uint8)
        area = float(lung.mean())
        r_area, l_area = float(region["right_lung"].mean()), float(region["left_lung"].mean())
        ratio = (l_area + 1e-6) / (r_area + 1e-6)
        cm_area = float(region["cardiomediastinum"].mean())
        frag = max(1.0 - largest_component_fraction(region["right_lung"] > 0),
                   1.0 - largest_component_fraction(region["left_lung"] > 0))
        accepted = bool(0.05 <= area <= 0.35 and 0.5 <= ratio <= 2.0
                        and 0.03 <= cm_area <= 0.30 and frag <= 0.30
                        and r_area >= 0.02 and l_area >= 0.02)
        qc_rows.append({"idx": int(i), "dicom_id": dicom_id, "lung_area": area,
                        "left_lung_area": l_area, "right_lung_area": r_area,
                        "lr_ratio": ratio, "cardiomediastinum_area": cm_area,
                        "fragmentation": frag, "accepted": accepted})
        out_masks[dicom_id] = {k2: v for k2, v in region.items()}
        if len(examples) < 6:
            examples.append((img, region, accepted, dicom_id))
        if k % 10 == 0:
            log.info("segmented %d/%d", k, len(idxs))

    dicom_ids = list(out_masks.keys())
    region_names = ["right_lung", "left_lung", "cardiomediastinum", "subdiaphragmatic", "lungs",
                    "left_hemidiaphragm", "right_hemidiaphragm", "liver", "stomach"]
    arrays = {name: np.stack([out_masks[d][name] for d in dicom_ids]) for name in region_names}
    np.savez_compressed(PROCESSED_DIR / f"{args.tag}.npz",
                        dicom_ids=np.array(dicom_ids), **arrays)
    save_json(TABLE_DIR / f"{args.tag}_qc.json", {"rows": qc_rows,
              "n": len(qc_rows), "n_accepted": int(sum(r["accepted"] for r in qc_rows))})
    log.info("accepted %d/%d", sum(r["accepted"] for r in qc_rows), len(qc_rows))

    n_ex = len(examples)
    fig, axes = plt.subplots(n_ex, 3, figsize=(10, 3.3 * n_ex))
    for r, (img, region, acc, did) in enumerate(examples):
        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_title(("接受 " if acc else "拒绝 ") + did[:8] if r == 0 else "")
        ov = np.zeros((*img.shape, 3), dtype=np.float32)
        ov[..., 0] = region["right_lung"] + 0.6 * region["left_lung"]
        ov[..., 1] = region["cardiomediastinum"]
        ov[..., 2] = region["subdiaphragmatic"]
        axes[r, 1].imshow(img, cmap="gray"); axes[r, 1].imshow(np.clip(ov, 0, 1), alpha=0.45)
        axes[r, 1].set_title("肺(红)/纵隔(绿)/膈下(蓝)" if r == 0 else "")
        axes[r, 2].imshow(img, cmap="gray")
        axes[r, 2].imshow(region["lungs"], cmap="Reds", alpha=0.45)
        axes[r, 2].set_title("肺掩膜" if r == 0 else "")
        for c in range(3):
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.tag}_examples.png", dpi=150)
    plt.close(fig)
    log.info("saved masks -> %s", PROCESSED_DIR / f"{args.tag}.npz")


if __name__ == "__main__":
    main()
