"""归因预实验 v3：4,000 掩膜 × 16×16 单元的完整 RISE，与 CAM/梯度显著图同台对比。

- 图像：与 pilot2_n60 完全相同的 60 张测试图像（同 seed）
- RISE：4,000 masks、16×16 cells、blur 填充、batch 256（本机实测最优）
- 评价：删除/插入曲线（误差导向，8 档比例）、随机排序 4 次重复、配对 Wilcoxon
- 输出：JSON + 曲线图 + 定性对比图
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.stats import spearmanr, wilcoxon

from attention_occlusion import gradcam
from attribution_pilot2 import curves_for_order, grad_saliency
from common import FEATURE_DIR, FIGURE_DIR, TABLE_DIR, save_json, setup_logger
from dataset import WeightCXR
from models import build_model, normalization_for

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def predict_kg(model, xb, tm, ts):
    with torch.no_grad():
        p = model(xb).squeeze(-1)
        if p.ndim > 1:
            p = p[..., 0]
        return p.float().cpu().numpy() * ts + tm


def rise_map_streaming(model, x, x_blur, tm, ts, n_masks, cell, side, rng, chunk=256):
    """流式 RISE：先存低分辨率掩膜与预测，再按全局权重累积重要性图（避免显存峰值）。"""
    low_masks = np.zeros((n_masks, cell, cell), dtype=bool)
    preds = np.zeros(n_masks, dtype=np.float32)
    t0 = time.perf_counter()
    for s in range(0, n_masks, chunk):
        bs = min(chunk, n_masks - s)
        low = rng.random((bs, cell, cell)) > 0.5
        low_masks[s:s + bs] = low
        m = torch.from_numpy(low.astype(np.float32)).unsqueeze(1)
        m = torch.nn.functional.interpolate(m, size=(side, side), mode="bilinear",
                                            align_corners=False)[:, 0]
        m = (m > 0.5).float().to(x.device)
        xb = x * m.unsqueeze(1) + x_blur * (1 - m.unsqueeze(1))
        preds[s:s + bs] = predict_kg(model, xb, tm, ts)
    w = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
    imp = np.zeros((side, side), dtype=np.float64)
    for s in range(0, n_masks, chunk):
        bs = min(chunk, n_masks - s)
        m = torch.from_numpy(low_masks[s:s + bs].astype(np.float32)).unsqueeze(1)
        m = torch.nn.functional.interpolate(m, size=(side, side), mode="bilinear",
                                            align_corners=False)[:, 0]
        m = (m > 0.5).numpy().astype(np.float64)
        imp += (m * w[s:s + bs, None, None]).sum(axis=0)
    imp /= (w.sum() + 1e-8)
    return imp, preds, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rise-m", type=int, default=4000)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--fractions", type=int, default=8)
    ap.add_argument("--random-repeats", type=int, default=4)
    ap.add_argument("--blur-sigma", type=float, default=20.0)
    ap.add_argument("--stage", type=int, default=-2)
    ap.add_argument("--tag", default="attribution_pilot3_rise")
    args = ap.parse_args()

    log = setup_logger(args.tag)
    device = torch.device("cuda")
    ckpt = torch.load(FEATURE_DIR / "checkpoints" / "swin_t_308_s1.pt",
                      map_location=device, weights_only=False)
    model = build_model("swin_t", img_size=308).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tm, ts = float(ckpt.get("target_mean", 0.0)), float(ckpt.get("target_std", 1.0))
    mean, std = normalization_for("swin_t")
    ds = WeightCXR("test", split_kind="random", size=308, train=False, mean=mean, std=std)
    rng = np.random.default_rng(20261007)          # 与 pilot2_n60 相同种子
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    side, flat = 308, 308 * 308
    fracs = np.linspace(1.0 / args.fractions, 1.0, args.fractions)
    border = np.zeros((side, side), dtype=bool)
    bw = int(0.15 * side)
    border[:bw, :] = border[-bw:, :] = border[:, :bw] = border[:, -bw:] = True

    methods = ["cam", "grad", "rise", "random"]
    del_curves = {m: [] for m in methods}
    ins_curves = {m: [] for m in methods}
    del_auc = {m: [] for m in methods}
    ins_auc = {m: [] for m in methods}
    corr_rows, border_frac = [], {m: [] for m in ("cam", "grad", "rise")}
    base_errs, blur_errs = [], []
    rise_times = []
    qual = []

    t_start = time.perf_counter()
    for k, i in enumerate(idxs, 1):
        s = ds[int(i)]
        x = s["image"].unsqueeze(0).to(device)
        y = float(s["weight"])
        x_blur = TF.gaussian_blur(x, kernel_size=61, sigma=args.blur_sigma)
        base = float(predict_kg(model, x, tm, ts)[0])
        blurp = float(predict_kg(model, x_blur, tm, ts)[0])
        base_err = abs(base - y)
        base_errs.append(base_err)
        blur_errs.append(abs(blurp - y))

        cam = gradcam(model, x, "swin_t", stage=args.stage)[0]
        grad = grad_saliency(model, x)
        rise, rise_preds, rise_dt = rise_map_streaming(model, x, x_blur, tm, ts,
                                                       args.rise_m, args.cell, side, rng,
                                                       chunk=args.chunk)
        rise_times.append(rise_dt)
        maps = {"cam": cam, "grad": grad, "rise": rise}
        for m, mp in maps.items():
            border_frac[m].append(float(border[mp >= np.quantile(mp, 0.80)].mean()))
        corr_rows.append({"cam_grad": float(spearmanr(cam.ravel(), grad.ravel()).statistic),
                          "cam_rise": float(spearmanr(cam.ravel(), rise.ravel()).statistic),
                          "grad_rise": float(spearmanr(grad.ravel(), rise.ravel()).statistic)})
        if len(qual) < 4:
            qual.append({"image": s["image"][0].numpy().copy(), "cam": cam.copy(),
                         "grad": grad.copy(), "rise": rise.copy()})

        for m, mp in maps.items():
            d, ins = curves_for_order(model, x, x_blur, np.argsort(-mp.ravel()),
                                      fracs, tm, ts, side, flat)
            err_d = np.abs(d - y) - base_err
            err_i = np.abs(ins - y)
            del_curves[m].append(err_d.tolist())
            ins_curves[m].append(err_i.tolist())
            del_auc[m].append(float(np.trapezoid(err_d, fracs)))
            ins_auc[m].append(float(np.trapezoid(err_i, fracs)))
        d_r_acc, i_r_acc = [], []
        for _ in range(args.random_repeats):
            d_r, i_r = curves_for_order(model, x, x_blur, rng.permutation(flat),
                                        fracs, tm, ts, side, flat)
            d_r_acc.append(np.abs(d_r - y) - base_err)
            i_r_acc.append(np.abs(i_r - y))
        del_curves["random"].append(np.mean(d_r_acc, axis=0).tolist())
        ins_curves["random"].append(np.mean(i_r_acc, axis=0).tolist())
        del_auc["random"].append(float(np.trapezoid(np.mean(d_r_acc, axis=0), fracs)))
        ins_auc["random"].append(float(np.trapezoid(np.mean(i_r_acc, axis=0), fracs)))
        if k % 5 == 0 or k == 1:
            log.info("progress %d/%d | RISE %.1f s/img | elapsed %.1f min",
                     k, len(idxs), rise_dt, (time.perf_counter() - t_start) / 60)

    out = {
        "n": len(idxs), "rise_masks": args.rise_m, "rise_cell": args.cell,
        "rise_batch_chunk": args.chunk, "fracs": fracs.tolist(),
        "rise_seconds_per_image_mean": float(np.mean(rise_times)),
        "total_minutes": (time.perf_counter() - t_start) / 60,
        "sanity": {"base_mae_mean": float(np.mean(base_errs)),
                   "blur_mae_mean": float(np.mean(blur_errs))},
        "border_fraction_top20pct": {m: float(np.mean(v)) for m, v in border_frac.items()},
        "map_correlations": {k: float(np.mean([r[k] for r in corr_rows]))
                             for k in ("cam_grad", "cam_rise", "grad_rise")},
    }
    for phase, curves, aucs, better in (("deletion", del_curves, del_auc, "higher"),
                                        ("insertion", ins_curves, ins_auc, "lower")):
        out[phase] = {"better": better}
        for m in methods:
            out[phase][m] = {"curve": np.mean(curves[m], axis=0).tolist(),
                             "auc_mean": float(np.mean(aucs[m])),
                             "auc_std": float(np.std(aucs[m]))}
        out[phase]["wilcoxon_vs_random_p"] = {}
        for m in ("cam", "grad", "rise"):
            try:
                out[phase]["wilcoxon_vs_random_p"][m] = float(wilcoxon(aucs[m], aucs["random"]).pvalue)
            except Exception:  # noqa: BLE001
                out[phase]["wilcoxon_vs_random_p"][m] = None
    save_json(TABLE_DIR / f"{args.tag}.json", out)
    log.info("RISE %.2f s/img; total %.1f min", out["rise_seconds_per_image_mean"], out["total_minutes"])
    log.info("map corr: %s", out["map_correlations"])
    for phase in ("deletion", "insertion"):
        log.info("%s (%s better): %s | p=%s", phase, out[phase]["better"],
                 {m: round(out[phase][m]["auc_mean"], 3) for m in methods},
                 out[phase]["wilcoxon_vs_random_p"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in methods:
        axes[0].plot(fracs, out["deletion"][m]["curve"], "-o", label=m)
        axes[1].plot(fracs, out["insertion"][m]["curve"], "-o", label=m)
    axes[0].axhline(0, color="gray", ls=":")
    axes[0].set_xlabel("删除比例"); axes[0].set_ylabel("误差增加 Δ|error| (kg)")
    axes[0].set_title(f"删除曲线（越高越好；RISE {args.rise_m} 掩膜）"); axes[0].legend()
    axes[1].axhline(out["sanity"]["blur_mae_mean"], color="gray", ls=":", label="全模糊")
    axes[1].axhline(out["sanity"]["base_mae_mean"], color="black", ls=":", label="原图")
    axes[1].set_xlabel("插入比例"); axes[1].set_ylabel("绝对误差 (kg)")
    axes[1].set_title("插入曲线（越低越好）"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.tag}_curves.png", dpi=200)
    plt.close(fig)

    if qual:
        fig, axes = plt.subplots(len(qual), 4, figsize=(14, 3.6 * len(qual)))
        for r, p in enumerate(qual):
            axes[r, 0].imshow(p["image"], cmap="gray"); axes[r, 0].set_title("原图" if r == 0 else "")
            for c, key, title in ((1, "cam", "CAM"), (2, "grad", "输入梯度显著图"),
                                  (3, "rise", f"RISE（{args.rise_m} 掩膜）")):
                mm = p[key]
                mm = (mm - mm.min()) / (mm.max() - mm.min() + 1e-8)
                axes[r, c].imshow(p["image"], cmap="gray")
                axes[r, c].imshow(mm, cmap="jet", alpha=0.45)
                axes[r, c].set_title(title if r == 0 else "")
            for c in range(4):
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{args.tag}_qualitative.png", dpi=170)
        plt.close(fig)
    log.info("figures saved")


if __name__ == "__main__":
    main()
