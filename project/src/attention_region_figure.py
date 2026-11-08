"""可视化：主模型（ResNet-50 @308 三种子集成）的 Grad-CAM 空间分布。

产出 figS4：代表性病例热图、120 例平均热图、几何分区密度，
并并列展示解剖遮挡（有因果含义）与线性探针结果（信息分布）。

注意：CAM 只描述模型对输入的敏感度，不等于因果机制；本项目的删除/插入实验显示
CAM 排序并不优于随机，因此图中同时给出遮挡与探针证据。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

from attention_occlusion import gradcam
from common import FEATURE_DIR, FIGURE_DIR, TABLE_DIR, main_pred_file, save_json, setup_logger
from models import build_model, normalization_for

SIZE = 308
SEED = 20261106
PER_STRATUM = 30
CKPTS = [FEATURE_DIR / "checkpoints" / f"resnet50_308_s{i}.pt" for i in (1, 2, 3)]
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verma": "#D55E00",
      "purple": "#CC79A7", "grey": "#999999", "black": "#000000"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600})
MM = 1 / 25.4


def regions(size: int = SIZE) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size]
    fy, fx = yy / size, xx / size
    upper = fy < 0.70
    r = {
        "Left lung": upper & (fx >= 0.08) & (fx < 0.40),
        "Right lung": upper & (fx >= 0.60) & (fx < 0.92),
        "Mediastinum": upper & (fx >= 0.42) & (fx < 0.58),
        "Subdiaphragmatic": (fy >= 0.70) & (fy < 0.98) & (fx >= 0.08) & (fx < 0.92),
    }
    used = np.zeros((size, size), dtype=bool)
    for m in r.values():
        used |= m
    r["Periphery / other"] = ~used
    return r


def load_image(dicom_id: str) -> np.ndarray:
    return np.load(FEATURE_DIR / "index_320" / f"{dicom_id}.npy")


def to_tensor(img: np.ndarray, mean: float, std: float, device) -> torch.Tensor:
    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    x = torch.nn.functional.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    return x.repeat(1, 3, 1, 1).to(device)


def main() -> None:
    log = setup_logger("attention_region_figure")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = normalization_for("resnet50")
    preds = pd.read_csv(main_pred_file())
    preds = preds[preds["dicom_id"].map(
        lambda d: (FEATURE_DIR / "index_320" / f"{d}.npy").exists())].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    picks = []
    for lo, hi in ((0, 50), (50, 80), (80, 100), (100, 400)):
        g = preds[(preds.weight_kg >= lo) & (preds.weight_kg < hi)]
        n = min(PER_STRATUM, len(g))
        picks.append(g.iloc[rng.choice(len(g), size=n, replace=False)])
    sample = pd.concat(picks).reset_index(drop=True)
    log.info("sample n=%d", len(sample))

    models = []
    for ck in CKPTS:
        m = build_model("resnet50", img_size=SIZE).to(device)
        m.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])
        m.eval()
        models.append(m)

    masks = regions()
    cams, region_vals = [], []
    for i, row in sample.iterrows():
        img = load_image(row["dicom_id"])
        x = to_tensor(img, mean, std, device)
        per_seed = []
        for m in models:
            with torch.enable_grad():
                per_seed.append(gradcam(m, x, "resnet50")[0])
        cam = np.mean(per_seed, axis=0)
        hi = np.quantile(cam, 0.99)
        cam = np.clip(cam / max(hi, 1e-8), 0, 1)
        cams.append(cam)
        overall = cam.mean()
        region_vals.append({k: float(cam[m].mean() / max(overall, 1e-8)) for k, m in masks.items()})
        if (i + 1) % 20 == 0:
            log.info("processed %d/%d", i + 1, len(sample))
    cams = np.stack(cams)
    mean_cam = cams.mean(axis=0)
    mean_cam = mean_cam / max(mean_cam.max(), 1e-8)
    reg = pd.DataFrame(region_vals).mean().to_dict()
    reg_sd = pd.DataFrame(region_vals).std().to_dict()

    # 代表病例：接近 55 / 80 / 110 / 140 kg，且误差 <10 kg
    target_w = [55, 80, 110, 140]
    example_rows = []
    for tw in target_w:
        cand = sample.assign(dw=(sample.weight_kg - tw).abs(),
                             ae=(sample.pred - sample.weight_kg).abs())
        cand = cand[cand.ae < 12].sort_values(["dw", "ae"])
        if len(cand):
            example_rows.append(cand.iloc[0])

    # 因果证据
    occ = json.loads((TABLE_DIR / "anatomical_occlusion_matched.json").read_text(encoding="utf-8"))["regions"]
    causal = [("Lungs", occ["lungs"]["paired_diff"], occ["lungs"]["paired_ci95"]),
              ("Mediastinum", occ["cardiomediastinum"]["paired_diff"],
               occ["cardiomediastinum"]["paired_ci95"]),
              ("Subdiaphragm", occ["subdiaphragmatic"]["paired_diff"],
               occ["subdiaphragmatic"]["paired_ci95"]),
              ("Liver", occ["liver"]["paired_diff"], occ["liver"]["paired_ci95"])]

    fig = plt.figure(figsize=(170 * MM, 104 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[0.78, 1.22], hspace=0.06, wspace=0.22)
    for j, r in enumerate(example_rows):
        ax = fig.add_subplot(gs[0, j])
        base = load_image(r["dicom_id"])
        cam = cams[int(r.name)]
        ax.imshow(base, cmap="gray", vmin=0, vmax=255)
        ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{r['weight_kg']:.0f} kg \u2192 {r['pred']:.0f} kg", fontsize=7.2, pad=2)
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(mean_cam, cmap="jet", vmin=0, vmax=1)
    for name, m in masks.items():
        if name == "Periphery / other":
            continue
        ys, xs = np.where(m)
        if len(ys):
            ax.add_patch(Rectangle((xs.min(), ys.min()), xs.max() - xs.min(), ys.max() - ys.min(),
                                   fill=False, edgecolor="white", linewidth=0.7, alpha=0.85))
    ax.text(0.02, 0.98, "Mean CAM (n=%d)" % len(sample), transform=ax.transAxes, va="top",
            ha="left", fontsize=7, color="white",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.2))
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 1])
    names = list(reg.keys())
    vals = [reg[k] for k in names]
    err = [reg_sd[k] for k in names]
    colors = [OI["blue"], OI["blue"], OI["verma"], OI["purple"], OI["grey"]]
    ax.barh(np.arange(len(names)), vals, xerr=err, color=colors, height=0.6, capsize=2)
    ax.axvline(1.0, color=OI["black"], lw=0.8, ls=":")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(["L lung", "R lung", "Mediast.", "Subdiaph.", "Periphery"], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("CAM density (\u00d7 image mean)", fontsize=7)
    ax.set_title("Where CAM mass sits", loc="left", fontsize=7.5)
    ax = fig.add_subplot(gs[1, 2])
    lab = [c[0] for c in causal]
    val = [c[1] for c in causal]
    lo = [c[1] - c[2][0] for c in causal]
    hi = [c[2][1] - c[1] for c in causal]
    cols = [OI["green"] if v > 0 else OI["grey"] for v in val]
    ax.barh(np.arange(len(lab)), val, xerr=[lo, hi], color=cols, height=0.6, capsize=2)
    ax.axvline(0, color=OI["black"], lw=0.8)
    ax.set_yticks(np.arange(len(lab))); ax.set_yticklabels(lab, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("\u0394|error| vs control (kg)", fontsize=7)
    ax.set_title("Occlusion (causal)", loc="left", fontsize=7.5)
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    ax.text(0, 1.0, "How to read", fontsize=7.5, fontweight="bold", va="top")
    ax.text(0, 0.88,
            "\u2022 CAM = sensitivity, not mechanism.\n"
            "\u2022 Deletion/insertion: CAM and\n  RISE were no better than random.\n"
            "\u2022 Occlusion: only lungs beat a\n  matched control (+1.6 kg).\n"
            "\u2022 Linear probe: weight is decodable\n  from every patch (R\u00b2 0.56\u20130.66).",
            fontsize=5.8, va="top", linespacing=1.35)
    for name in ("figS4_attention_regions",):
        fig.savefig(FIGURE_DIR / f"{name}.png", facecolor="white")
        fig.savefig(FIGURE_DIR / f"{name}.svg", facecolor="white")
        ioi = Path("Insights_into_Imaging/figures")
        if ioi.exists():
            fig.savefig(ioi / f"{name}.png", facecolor="white")
            fig.savefig(ioi / f"{name}.svg", facecolor="white")
    plt.close(fig)

    out = {"n": int(len(sample)), "region_density_mean": reg, "region_density_sd": reg_sd,
           "examples": [{"dicom_id": r["dicom_id"], "weight_kg": float(r["weight_kg"]),
                         "pred_kg": float(r["pred"])} for r in example_rows],
           "causal_occlusion": {c[0]: {"delta": c[1], "ci95": c[2]} for c in causal},
           "note": "CAM is descriptive; occlusion and linear probing provide causal/representational evidence"}
    save_json(TABLE_DIR / "attention_region_figure.json", out)
    log.info("region density: %s", json.dumps({k: round(v, 2) for k, v in reg.items()}))


if __name__ == "__main__":
    main()
