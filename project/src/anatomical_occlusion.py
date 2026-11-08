"""实验 A：解剖分区（肺野/纵隔/膈下）区域遮挡。

- 分割：torchxrayvision chestx_anatomy（512→308 对齐）
- 填充：高斯模糊（σ=20）
- 对照：同形状掩膜随机平移 4 次取均值
- 指标：Δ|error|（误差增加）、Δpred，配对 Wilcoxon
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

from common import FEATURE_DIR, FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, load_config, save_json, setup_logger
from dataset import WeightCXR
from models import build_model, normalization_for

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REGIONS = ["lungs", "cardiomediastinum", "subdiaphragmatic", "liver"]
LABELS = {"lungs": "双侧肺野", "cardiomediastinum": "纵隔/心影",
          "subdiaphragmatic": "膈下（肝/胃/肠）", "liver": "肝脏"}


def predict_kg(model, x, tm, ts):
    with torch.no_grad():
        p = model(x).squeeze(-1)
        if p.ndim > 0:
            p = p[..., 0] if p.ndim > 1 else p
        return float(p.item()) * ts + tm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="anatomy_regions_test60")
    ap.add_argument("--shifts", type=int, default=4)
    ap.add_argument("--blur-sigma", type=float, default=20.0)
    args = ap.parse_args()
    log = setup_logger("anatomical_occlusion")
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
    rng = np.random.default_rng(20261008)

    results = {r: {"delta_err": [], "delta_pred": [], "delta_err_shift": [], "area": []}
               for r in REGIONS}
    results["all_regions"] = {"delta_err": [], "delta_pred": [], "area": []}
    n_used = 0
    for row in accepted:
        did = row["dicom_id"]
        if did not in idx_by_dicom or did not in dicom_ids:
            continue
        j = dicom_ids.index(did)
        s = ds[idx_by_dicom[did]]
        x = s["image"].unsqueeze(0).to(device)
        y = float(s["weight"])
        x_blur = TF.gaussian_blur(x, kernel_size=61, sigma=args.blur_sigma)
        base = predict_kg(model, x, tm, ts)
        base_err = abs(base - y)
        n_used += 1

        for region in REGIONS:
            mask = npz[region][j].astype(bool)
            if mask.sum() < 50:
                continue
            xo = x.clone()
            mt = torch.from_numpy(mask).to(device)
            xo[0][:, mt] = x_blur[0][:, mt]
            p = predict_kg(model, xo, tm, ts)
            results[region]["delta_err"].append(abs(p - y) - base_err)
            results[region]["delta_pred"].append(p - base)
            results[region]["area"].append(float(mask.mean()))
            shifts = []
            for _ in range(args.shifts):
                m2 = np.roll(np.roll(mask, int(rng.integers(0, 308)), 0),
                             int(rng.integers(0, 308)), 1)
                xs = x.clone()
                mt2 = torch.from_numpy(m2).to(device)
                xs[0][:, mt2] = x_blur[0][:, mt2]
                shifts.append(abs(predict_kg(model, xs, tm, ts) - y) - base_err)
            results[region]["delta_err_shift"].append(float(np.mean(shifts)))

        union = np.zeros((308, 308), dtype=bool)
        for region in ("lungs", "cardiomediastinum", "subdiaphragmatic"):
            union |= npz[region][j].astype(bool)
        xo = x.clone()
        mt = torch.from_numpy(union).to(device)
        xo[0][:, mt] = x_blur[0][:, mt]
        p = predict_kg(model, xo, tm, ts)
        results["all_regions"]["delta_err"].append(abs(p - y) - base_err)
        results["all_regions"]["delta_pred"].append(p - base)
        results["all_regions"]["area"].append(float(union.mean()))

    out = {"n_images": n_used, "regions": {}}
    for region, v in results.items():
        if not v["delta_err"]:
            continue
        rec = {"n": len(v["delta_err"]),
               "area_mean": float(np.mean(v["area"])),
               "delta_err_mean": float(np.mean(v["delta_err"])),
               "delta_err_std": float(np.std(v["delta_err"])),
               "delta_pred_mean": float(np.mean(v["delta_pred"])) if v["delta_pred"] else None,
               "per_image_delta_err": [float(x) for x in v["delta_err"]]}
        if v.get("delta_err_shift"):
            rec["delta_err_shift_mean"] = float(np.mean(v["delta_err_shift"]))
            rec["delta_err_shift_std"] = float(np.std(v["delta_err_shift"]))
            rec["per_image_delta_err_shift"] = [float(x) for x in v["delta_err_shift"]]
            diff = np.array(v["delta_err"]) - np.array(v["delta_err_shift"])
            rec["paired_diff_mean"] = float(diff.mean())
            rec["paired_diff_ci95"] = [float(diff.mean() - 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))),
                                       float(diff.mean() + 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)))]
            try:
                rec["wilcoxon_region_vs_shift_p"] = float(
                    wilcoxon(v["delta_err"], v["delta_err_shift"]).pvalue)
            except Exception:  # noqa: BLE001
                rec["wilcoxon_region_vs_shift_p"] = None
            rec["frac_region_worse"] = float(np.mean(np.array(v["delta_err"]) >
                                                     np.array(v["delta_err_shift"])))
        out["regions"][region] = rec
    save_json(TABLE_DIR / "anatomical_occlusion.json", out)
    for k, v in out["regions"].items():
        log.info("%s (n=%d, area=%.3f): Δerr=%+.2f±%.2f | shift=%+.2f | p=%s | worse=%.2f",
                 k, v["n"], v["area_mean"], v["delta_err_mean"], v["delta_err_std"],
                 v.get("delta_err_shift_mean", float("nan")), v.get("wilcoxon_region_vs_shift_p"),
                 v.get("frac_region_worse", float("nan")))

    fig, ax = plt.subplots(figsize=(9, 5))
    keys = [k for k in out["regions"] if k != "all_regions"]
    xpos = np.arange(len(keys) + 1)
    vals = [out["regions"][k]["delta_err_mean"] for k in keys] + \
           [out["regions"]["all_regions"]["delta_err_mean"]]
    errs = [out["regions"][k]["delta_err_std"] / np.sqrt(out["regions"][k]["n"]) for k in keys] + \
           [out["regions"]["all_regions"]["delta_err_std"] /
            np.sqrt(out["regions"]["all_regions"]["n"])]
    shift = [out["regions"][k].get("delta_err_shift_mean", 0) for k in keys] + [0]
    labels = [LABELS.get(k, k) for k in keys] + ["三区合并"]
    ax.bar(xpos - 0.2, vals, width=0.4, yerr=errs, capsize=4, color="#2C6FA6", label="遮挡解剖区")
    ax.bar(xpos + 0.2, shift, width=0.4, color="#C98A7F", label="同形状随机平移")
    ax.axhline(0, color="gray", ls=":")
    ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_ylabel("误差增加 Δ|error| (kg)")
    ax.set_title(f"解剖分区遮挡（n={n_used}，blur 填充）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "anatomical_occlusion.png", dpi=200)
    plt.close(fig)
    log.info("figure -> %s", FIGURE_DIR / "anatomical_occlusion.png")


if __name__ == "__main__":
    main()
