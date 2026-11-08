"""实验 A2：内容匹配对照的解剖分区遮挡。

对照定义（比"裸随机平移"更严格）：
  - 掩膜 ≥95% 落在身体轮廓内（Otsu 身体掩膜 + 最大连通域 + 填洞）
  - 与原解剖区的 IoU ≤ 0.2（避免对照落到同一区域）
  - 每区域取 2 个满足条件的随机平移取平均
附加：左肺 vs 右肺（同组织、对侧）配对对照。
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.stats import wilcoxon
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import binary_closing, disk

from common import FEATURE_DIR, FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, save_json, setup_logger
from dataset import WeightCXR
from models import build_model, normalization_for

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
REGIONS = ["lungs", "cardiomediastinum", "subdiaphragmatic", "liver"]
LABELS = {"lungs": "双侧肺野", "cardiomediastinum": "纵隔/心影",
          "subdiaphragmatic": "膈下", "liver": "肝脏"}


def body_mask_from_img(img01: np.ndarray) -> np.ndarray:
    thr = threshold_otsu(img01)
    b = img01 > max(thr * 0.8, 0.02)
    b = binary_closing(b, disk(3))
    lab = label(b)
    if lab.max() == 0:
        return np.zeros_like(b)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    b = lab == sizes.argmax()
    return binary_fill_holes(b)


def match_shift(mask: np.ndarray, body: np.ndarray, rng, tries: int = 300):
    area = mask.sum()
    for _ in range(tries):
        sy, sx = int(rng.integers(0, mask.shape[0])), int(rng.integers(0, mask.shape[1]))
        m = np.roll(np.roll(mask, sy, 0), sx, 1)
        inside = float((m & body).sum() / max(area, 1))
        iou = float((m & mask).sum() / max((m | mask).sum(), 1))
        if inside >= 0.95 and iou <= 0.20:
            return m, inside, iou
    return None, None, None


def predict_kg(model, x, tm, ts):
    with torch.no_grad():
        p = model(x).squeeze(-1)
        if p.ndim > 0:
            p = p[..., 0] if p.ndim > 1 else p
        return float(p.item()) * ts + tm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="anatomy_regions_test1303")
    ap.add_argument("--shifts", type=int, default=2)
    ap.add_argument("--blur-sigma", type=float, default=20.0)
    args = ap.parse_args()
    log = setup_logger("anatomical_occlusion_matched")
    device = torch.device("cuda")
    ckpt = torch.load(FEATURE_DIR / "checkpoints" / "swin_t_308_s1.pt",
                      map_location=device, weights_only=False)
    model = build_model("swin_t", img_size=308).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tm, ts = float(ckpt.get("target_mean", 0.0)), float(ckpt.get("target_std", 1.0))
    mean, std = normalization_for("swin_t")
    ds = WeightCXR("test", split_kind="random", size=308, train=False, mean=mean, std=std)
    idx_by_dicom = {d: i for i, d in enumerate(ds.df["dicom_id"])}
    qc = json.loads((TABLE_DIR / f"{args.tag}_qc.json").read_text(encoding="utf-8"))
    accepted = [r for r in qc["rows"] if r["accepted"]]
    npz = np.load(PROCESSED_DIR / f"{args.tag}.npz", allow_pickle=True)
    dicom_ids = list(npz["dicom_ids"])
    rng = np.random.default_rng(20261010)

    res = {r: {"delta": [], "ctrl": [], "inside": [], "iou": []} for r in REGIONS}
    lr = {"left": [], "right": []}
    examples = []
    n_used, n_skipped = 0, 0
    for row in accepted:
        did = row["dicom_id"]
        if did not in idx_by_dicom or did not in dicom_ids:
            continue
        j = dicom_ids.index(did)
        s = ds[idx_by_dicom[did]]
        x = s["image"].unsqueeze(0).to(device)
        y = float(s["weight"])
        x_blur = TF.gaussian_blur(x, kernel_size=61, sigma=args.blur_sigma)
        img01 = np.clip(s["image"][0].numpy() * 0.5 + 0.5, 0, 1)
        body = body_mask_from_img(img01)
        if body.mean() < 0.05:
            n_skipped += 1
            continue
        base = predict_kg(model, x, tm, ts)
        base_err = abs(base - y)
        ok = False
        for region in REGIONS:
            mask = npz[region][j].astype(bool)
            if mask.sum() < 50:
                continue
            xo = x.clone()
            mt = torch.from_numpy(mask).to(device)
            xo[0][:, mt] = x_blur[0][:, mt]
            d = abs(predict_kg(model, xo, tm, ts) - y) - base_err
            ctrls, insides, ious = [], [], []
            for _ in range(args.shifts):
                m2, inside, iou = match_shift(mask, body, rng)
                if m2 is None:
                    continue
                xs = x.clone()
                mt2 = torch.from_numpy(m2).to(device)
                xs[0][:, mt2] = x_blur[0][:, mt2]
                ctrls.append(abs(predict_kg(model, xs, tm, ts) - y) - base_err)
                insides.append(inside); ious.append(iou)
            if not ctrls:
                continue
            res[region]["delta"].append(d)
            res[region]["ctrl"].append(float(np.mean(ctrls)))
            res[region]["inside"].append(float(np.mean(insides)))
            res[region]["iou"].append(float(np.mean(ious)))
            ok = True
            if region == "lungs" and len(examples) < 4:
                examples.append((img01, mask, match_shift(mask, body, rng)[0], body))
        # 左/右肺同组织对照
        for side in ("left", "right"):
            m = npz[f"{side}_lung"][j].astype(bool)
            if m.sum() < 50:
                continue
            xo = x.clone()
            mt = torch.from_numpy(m).to(device)
            xo[0][:, mt] = x_blur[0][:, mt]
            lr[side].append(abs(predict_kg(model, xo, tm, ts) - y) - base_err)
        if ok:
            n_used += 1
        if n_used % 100 == 0:
            log.info("processed %d", n_used)

    out = {"n_images": n_used, "n_skipped_body": n_skipped, "regions": {}}
    for region, v in res.items():
        if not v["delta"]:
            continue
        a, b = np.array(v["delta"]), np.array(v["ctrl"])
        diff = a - b
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        out["regions"][region] = {
            "n": len(diff),
            "delta_mean": float(a.mean()), "delta_std": float(a.std()),
            "ctrl_mean": float(b.mean()), "ctrl_std": float(b.std()),
            "paired_diff": float(diff.mean()),
            "paired_ci95": [float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)],
            "wilcoxon_p": float(wilcoxon(a, b).pvalue),
            "ctrl_inside_body_mean": float(np.mean(v["inside"])),
            "ctrl_iou_with_region_mean": float(np.mean(v["iou"])),
            "per_image_delta": [float(x) for x in a],
            "per_image_ctrl": [float(x) for x in b],
        }
    if lr["left"] and lr["right"]:
        a, b = np.array(lr["left"]), np.array(lr["right"])
        n = min(len(a), len(b))
        diff = a[:n] - b[:n]
        se = diff.std(ddof=1) / np.sqrt(n)
        out["left_vs_right_lung"] = {
            "n": int(n), "left_mean": float(a[:n].mean()), "right_mean": float(b[:n].mean()),
            "paired_diff": float(diff.mean()),
            "paired_ci95": [float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)],
            "wilcoxon_p": float(wilcoxon(a[:n], b[:n]).pvalue)}
    save_json(TABLE_DIR / "anatomical_occlusion_matched.json", out)
    for k, v in out["regions"].items():
        log.info("%s (n=%d): region=%+.2f ctrl=%+.2f paired=%+.2f CI95=[%+.2f,%+.2f] p=%.2e "
                 "| ctrl inside body=%.3f iou=%.3f",
                 k, v["n"], v["delta_mean"], v["ctrl_mean"], v["paired_diff"],
                 v["paired_ci95"][0], v["paired_ci95"][1], v["wilcoxon_p"],
                 v["ctrl_inside_body_mean"], v["ctrl_iou_with_region_mean"])
    if "left_vs_right_lung" in out:
        v = out["left_vs_right_lung"]
        log.info("left vs right lung: left=%+.2f right=%+.2f diff=%+.2f p=%.2e",
                 v["left_mean"], v["right_mean"], v["paired_diff"], v["wilcoxon_p"])

    if examples:
        fig, axes = plt.subplots(len(examples), 4, figsize=(12, 3.2 * len(examples)))
        for r, (img, mask, ctrl, body) in enumerate(examples):
            axes[r, 0].imshow(img, cmap="gray"); axes[r, 0].set_title("原图" if r == 0 else "")
            axes[r, 1].imshow(img, cmap="gray"); axes[r, 1].imshow(body, alpha=0.35, cmap="Greens")
            axes[r, 1].set_title("身体掩膜" if r == 0 else "")
            axes[r, 2].imshow(img, cmap="gray"); axes[r, 2].imshow(mask, alpha=0.45, cmap="Reds")
            axes[r, 2].set_title("肺野掩膜" if r == 0 else "")
            axes[r, 3].imshow(img, cmap="gray")
            if ctrl is not None:
                axes[r, 3].imshow(ctrl, alpha=0.45, cmap="Blues")
            axes[r, 3].set_title("内容匹配对照" if r == 0 else "")
            for c in range(4):
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "anatomical_occlusion_matched_examples.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    keys = list(out["regions"].keys())
    xpos = np.arange(len(keys))
    ax.bar(xpos - 0.2, [out["regions"][k]["delta_mean"] for k in keys], width=0.4,
           color="#2C6FA6", label="遮挡解剖区")
    ax.bar(xpos + 0.2, [out["regions"][k]["ctrl_mean"] for k in keys], width=0.4,
           color="#C98A7F", label="内容匹配对照（体内+低重叠）")
    ax.axhline(0, color="gray", ls=":")
    ax.set_xticks(xpos); ax.set_xticklabels([LABELS.get(k, k) for k in keys])
    ax.set_ylabel("误差增加 Δ|error| (kg)")
    ax.set_title(f"解剖分区遮挡 vs 内容匹配对照（n={out['n_images']}）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "anatomical_occlusion_matched.png", dpi=200)
    plt.close(fig)
    log.info("figures saved")


if __name__ == "__main__":
    main()
