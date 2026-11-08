"""更严格归因预实验 v2（回归任务的误差导向指标）。

指标定义：
- 删除曲线：err_del(f) = |pred_del(f) - y| - |pred_base - y|，即删除前 f 比例
  "最重要"像素后误差的增加量；AUC 越大表示该归因排序越能抓住关键像素。
- 插入曲线：err_ins(f) = |pred_ins(f) - y|，即只保留前 f 比例"最重要"像素
  （其余用模糊图填充）时的绝对误差；AUC 越小表示恢复越快、归因排序越好。
- 对照：随机排序（多次重复取平均）、RISE、输入梯度显著图。
- 另做误差导向的模糊遮挡（高显著区 vs 同形状随机平移 vs 边缘环 vs 中心），配对 Wilcoxon。

用法：python project/src/attribution_pilot2.py --n 30 --fractions 8 --random-repeats 4 --shifts 4 --rise-m 96
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
from scipy.stats import spearmanr, wilcoxon

from attention_occlusion import gradcam
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


def grad_saliency(model, x):
    xg = x.clone().detach().requires_grad_(True)
    out = model(xg).squeeze(-1)
    if out.ndim > 1:
        out = out[..., 0]
    out.sum().backward()
    return xg.grad.detach().abs().mean(dim=1)[0].cpu().numpy()


def rise_map(model, x, x_blur, tm, ts, m_masks, cell, side, rng, chunk=32):
    low = (rng.random((m_masks, cell, cell)) > 0.5).astype(np.float32)
    masks = torch.from_numpy(low).unsqueeze(1)
    masks = torch.nn.functional.interpolate(masks, size=(side, side), mode="bilinear",
                                            align_corners=False)[:, 0]
    masks = (masks > 0.5).float()
    preds = []
    for s in range(0, m_masks, chunk):
        mb = masks[s:s + chunk].to(x.device)
        xb = x.repeat(mb.shape[0], 1, 1, 1) * mb.unsqueeze(1) + \
             x_blur.repeat(mb.shape[0], 1, 1, 1) * (1 - mb.unsqueeze(1))
        preds.append(predict_kg(model, xb, tm, ts))
    preds = np.concatenate(preds)
    w = np.clip(preds - preds.min(), 0, None)
    return (masks.numpy() * w[:, None, None]).sum(axis=0) / (w.sum() + 1e-8)


def masks_from_order(flat, order, frac, side):
    mm = np.zeros(flat, dtype=bool)
    mm[order[: int(frac * flat)]] = True
    return mm.reshape(side, side)


def curves_for_order(model, x, x_blur, order, fracs, tm, ts, side, flat):
    xd = x.repeat(len(fracs), 1, 1, 1).clone()
    xi = x_blur.repeat(len(fracs), 1, 1, 1).clone()
    for j, f in enumerate(fracs):
        m = masks_from_order(flat, order, f, side)
        mt = torch.from_numpy(m).to(x.device)
        xd[j][:, mt] = x_blur[0][:, mt]
        xi[j][:, mt] = x[0][:, mt]
    return predict_kg(model, xd, tm, ts), predict_kg(model, xi, tm, ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--fractions", type=int, default=8)
    ap.add_argument("--random-repeats", type=int, default=4)
    ap.add_argument("--shifts", type=int, default=4)
    ap.add_argument("--rise-m", type=int, default=96)
    ap.add_argument("--blur-sigma", type=float, default=20.0)
    ap.add_argument("--stage", type=int, default=-2)
    ap.add_argument("--backbone", default="swin_t")
    ap.add_argument("--size", type=int, default=308)
    ap.add_argument("--tag", default="attribution_pilot2")
    args = ap.parse_args()

    log = setup_logger(args.tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(FEATURE_DIR / "checkpoints" / "swin_t_308_s1.pt",
                      map_location=device, weights_only=False)
    model = build_model(args.backbone, img_size=args.size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tm, ts = float(ckpt.get("target_mean", 0.0)), float(ckpt.get("target_std", 1.0))
    mean, std = normalization_for(args.backbone)
    ds = WeightCXR("test", split_kind="random", size=args.size, train=False, mean=mean, std=std)
    rng = np.random.default_rng(20261007)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    side = args.size
    flat = side * side
    fracs = np.linspace(1.0 / args.fractions, 1.0, args.fractions)
    border = np.zeros((side, side), dtype=bool)
    bw = int(0.15 * side)
    border[:bw, :] = border[-bw:, :] = border[:, :bw] = border[:, -bw:] = True

    methods = ["cam", "grad", "rise", "random"]
    del_err_curves = {m: [] for m in methods}
    ins_err_curves = {m: [] for m in methods}
    del_auc = {m: [] for m in methods}
    ins_auc = {m: [] for m in methods}
    corr_rows, border_frac = [], {m: [] for m in ("cam", "grad", "rise")}
    err_occl = {k: [] for k in ("salient", "shifted", "border", "center")}
    sanity = {"base_mae": [], "blur_mae": [], "del_full_equals_blur": []}
    qual_panels = []

    for i in idxs:
        s = ds[int(i)]
        x = s["image"].unsqueeze(0).to(device)
        y = float(s["weight"])
        x_blur = TF.gaussian_blur(x, kernel_size=61, sigma=args.blur_sigma)
        base = float(predict_kg(model, x, tm, ts)[0])
        blurp = float(predict_kg(model, x_blur, tm, ts)[0])
        base_err = abs(base - y)
        sanity["base_mae"].append(base_err)
        sanity["blur_mae"].append(abs(blurp - y))

        cam = gradcam(model, x, args.backbone, stage=args.stage)[0]
        grad = grad_saliency(model, x)
        rise = rise_map(model, x, x_blur, tm, ts, args.rise_m, 7, side, rng)
        maps = {"cam": cam, "grad": grad, "rise": rise}
        if len(qual_panels) < 4:
            qual_panels.append({"image": s["image"][0].numpy().copy(),
                                "cam": cam.copy(), "grad": grad.copy(), "rise": rise.copy()})
        for m, mp in maps.items():
            q = np.quantile(mp, 0.80)
            border_frac[m].append(float(border[mp >= q].mean()))
        corr_rows.append({"cam_grad": float(spearmanr(cam.ravel(), grad.ravel()).statistic),
                          "cam_rise": float(spearmanr(cam.ravel(), rise.ravel()).statistic),
                          "grad_rise": float(spearmanr(grad.ravel(), rise.ravel()).statistic)})

        for m, mp in maps.items():
            d, ins = curves_for_order(model, x, x_blur, np.argsort(-mp.ravel()),
                                      fracs, tm, ts, side, flat)
            err_d = np.abs(d - y) - base_err
            err_i = np.abs(ins - y)
            del_err_curves[m].append(err_d.tolist())
            ins_err_curves[m].append(err_i.tolist())
            del_auc[m].append(float(np.trapezoid(err_d, fracs)))
            ins_auc[m].append(float(np.trapezoid(err_i, fracs)))
            if m == "cam":
                # 验证：删除 100% 时应等于全模糊预测
                sanity["del_full_equals_blur"].append(float(abs(d[-1] - blurp)))

        d_r_acc, i_r_acc = [], []
        for _ in range(args.random_repeats):
            d_r, i_r = curves_for_order(model, x, x_blur, rng.permutation(flat),
                                        fracs, tm, ts, side, flat)
            d_r_acc.append(np.abs(d_r - y) - base_err)
            i_r_acc.append(np.abs(i_r - y))
        del_err_curves["random"].append(np.mean(d_r_acc, axis=0).tolist())
        ins_err_curves["random"].append(np.mean(i_r_acc, axis=0).tolist())
        del_auc["random"].append(float(np.trapezoid(np.mean(d_r_acc, axis=0), fracs)))
        ins_auc["random"].append(float(np.trapezoid(np.mean(i_r_acc, axis=0), fracs)))

        def occ_err(mask):
            xo = x.clone()
            mt = torch.from_numpy(mask).to(device)
            xo[0][:, mt] = x_blur[0][:, mt]
            return abs(float(predict_kg(model, xo, tm, ts)[0]) - y) - base_err

        cam_mask = cam >= np.quantile(cam, 0.80)
        err_occl["salient"].append(occ_err(cam_mask))
        err_occl["shifted"].append(float(np.mean([
            occ_err(np.roll(np.roll(cam_mask, int(rng.integers(0, side)), 0),
                            int(rng.integers(0, side)), 1)) for _ in range(args.shifts)])))
        area = int(cam_mask.sum())
        bflat = np.flatnonzero(border.ravel())[:area]
        bmask2d = np.zeros((side, side), dtype=bool)
        bmask2d.ravel()[bflat] = True
        cside = int(np.sqrt(area))
        cy = cx = side // 2
        yy, xx = np.mgrid[0:side, 0:side]
        cmask2d = (np.abs(yy - cy) <= cside // 2) & (np.abs(xx - cx) <= cside // 2)
        err_occl["border"].append(occ_err(bmask2d))
        err_occl["center"].append(occ_err(cmask2d))

    out = {"n": len(idxs), "fracs": fracs.tolist(),
           "metric": "deletion: Δ|error| increase (higher=better); insertion: |error| (lower=better)"}
    out["sanity"] = {k: float(np.mean(v)) for k, v in sanity.items()}
    for phase, curves, aucs, better in (("deletion", del_err_curves, del_auc, "higher"),
                                        ("insertion", ins_err_curves, ins_auc, "lower")):
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
    out["border_fraction_top20pct"] = {m: float(np.mean(v)) for m, v in border_frac.items()}
    out["map_correlations"] = {k: float(np.mean([r[k] for r in corr_rows]))
                               for k in ("cam_grad", "cam_rise", "grad_rise")}
    out["error_oriented_occlusion"] = {k: {"mean_delta_abs_error": float(np.mean(v)),
                                           "std": float(np.std(v))} for k, v in err_occl.items()}
    try:
        out["error_oriented_occlusion"]["wilcoxon_salient_vs_shifted_p"] = float(
            wilcoxon(err_occl["salient"], err_occl["shifted"]).pvalue)
        out["error_oriented_occlusion"]["wilcoxon_border_vs_center_p"] = float(
            wilcoxon(err_occl["border"], err_occl["center"]).pvalue)
    except Exception:  # noqa: BLE001
        pass
    save_json(TABLE_DIR / f"{args.tag}.json", out)

    log.info("sanity: base MAE=%.2f, blur MAE=%.2f, |del(100%%)-blur|=%.4f",
             out["sanity"]["base_mae"], out["sanity"]["blur_mae"],
             out["sanity"]["del_full_equals_blur"])
    log.info("border fraction top20%%: %s", out["border_fraction_top20pct"])
    log.info("map corr: %s", out["map_correlations"])
    log.info("error occlusion: %s", json.dumps(out["error_oriented_occlusion"], ensure_ascii=False))
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
    axes[0].set_title("删除曲线（越高越好）"); axes[0].legend()
    axes[1].axhline(out["sanity"]["blur_mae"], color="gray", ls=":", label="全模糊")
    axes[1].axhline(out["sanity"]["base_mae"], color="black", ls=":", label="原图")
    axes[1].set_xlabel("插入比例"); axes[1].set_ylabel("绝对误差 (kg)")
    axes[1].set_title("插入曲线（越低越好）"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.tag}_curves.png", dpi=200)
    plt.close(fig)

    if qual_panels:
        fig, axes = plt.subplots(len(qual_panels), 4, figsize=(14, 3.6 * len(qual_panels)))
        for r, p in enumerate(qual_panels):
            axes[r, 0].imshow(p["image"], cmap="gray"); axes[r, 0].set_title("原图" if r == 0 else "")
            for c, key, title in ((1, "cam", "CAM"), (2, "grad", "输入梯度显著图"),
                                  (3, "rise", "RISE")):
                m = p[key]
                m = (m - m.min()) / (m.max() - m.min() + 1e-8)
                axes[r, c].imshow(p["image"], cmap="gray")
                axes[r, c].imshow(m, cmap="jet", alpha=0.45)
                axes[r, c].set_title(title if r == 0 else "")
            for c in range(4):
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{args.tag}_qualitative.png", dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
