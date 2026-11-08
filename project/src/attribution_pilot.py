"""更严格的归因预实验（pilot）：

A. 删除/插入曲线（blur 填充，CAM 排序 vs 随机排序，多次随机重复 + Wilcoxon）
B. 模糊遮挡位置对照（高显著区 vs 多个随机平移，配对检验）
C. 小规模 RISE 重要性图与 CAM 的秩相关

用法示例：
  python project/src/attribution_pilot.py --n 20 --fractions 10 --random-repeats 5 --shifts 5 --rise-m 128 --rise-n 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def predict_kg(model, xb, target_mean, target_std):
    with torch.no_grad():
        p = model(xb).squeeze(-1)
        if p.ndim > 1:
            p = p[..., 0]
        return (p.float().cpu().numpy() * target_std + target_mean)


def masks_from_order(flat_size: int, order: np.ndarray, frac: float, side: int) -> np.ndarray:
    m = np.zeros(flat_size, dtype=bool)
    m[order[: int(frac * flat_size)]] = True
    return m.reshape(side, side)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="参与删除/插入曲线的图像数")
    ap.add_argument("--fractions", type=int, default=10)
    ap.add_argument("--random-repeats", type=int, default=5)
    ap.add_argument("--shifts", type=int, default=5, help="每张图的随机平移次数（模糊填充）")
    ap.add_argument("--rise-m", type=int, default=128)
    ap.add_argument("--rise-n", type=int, default=12)
    ap.add_argument("--blur-sigma", type=float, default=20.0)
    ap.add_argument("--stage", type=int, default=-2)
    ap.add_argument("--backbone", default="swin_t")
    ap.add_argument("--size", type=int, default=308)
    ap.add_argument("--tag", default="attribution_pilot")
    args = ap.parse_args()

    log = setup_logger(args.tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = FEATURE_DIR / "checkpoints" / "swin_t_308_s1.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(args.backbone, img_size=args.size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    target_mean = float(ckpt.get("target_mean", 0.0))
    target_std = float(ckpt.get("target_std", 1.0))
    mean, std = normalization_for(args.backbone)
    ds = WeightCXR("test", split_kind="random", size=args.size, train=False, mean=mean, std=std)
    rng = np.random.default_rng(20261006)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)

    side = args.size
    flat = side * side
    fracs = np.linspace(1.0 / args.fractions, 1.0, args.fractions)
    results = {"n": len(idxs), "fractions": fracs.tolist(),
               "blur_sigma": args.blur_sigma, "cam_stage": args.stage}

    del_cam_curves, del_rand_curves = [], []
    ins_cam_curves, ins_rand_curves = [], []
    del_cam_auc, del_rand_auc, ins_cam_auc, ins_rand_auc = [], [], [], []
    occ_salient, occ_shifted, occ_pairs = [], [], []

    for i in idxs:
        sample = ds[int(i)]
        x = sample["image"].unsqueeze(0).to(device)
        x_blur = TF.gaussian_blur(x, kernel_size=61, sigma=args.blur_sigma)
        base = float(predict_kg(model, x, target_mean, target_std)[0])
        blurp = float(predict_kg(model, x_blur, target_mean, target_std)[0])
        cam = gradcam(model, x, args.backbone, stage=args.stage)[0]
        order = np.argsort(-cam.ravel())

        # ---- A. 删除 / 插入（CAM 排序） ----
        def run_curves(order_vec):
            xd = x.repeat(len(fracs), 1, 1, 1).clone()
            xi = x_blur.repeat(len(fracs), 1, 1, 1).clone()
            for j, f in enumerate(fracs):
                m = masks_from_order(flat, order_vec, f, side)
                mt = torch.from_numpy(m).to(device)
                xd[j][:, mt] = x_blur[0][:, mt]      # 删除：用模糊替换
                xi[j][:, mt] = x[0][:, mt]           # 插入：只保留前 f 的原始像素
            return predict_kg(model, xd, target_mean, target_std), predict_kg(model, xi, target_mean, target_std)

        d_cam, i_cam = run_curves(order)
        d_rand_acc, i_rand_acc = [], []
        for _ in range(args.random_repeats):
            perm = rng.permutation(flat)
            d_r, i_r = run_curves(perm)
            d_rand_acc.append(d_r)
            i_rand_acc.append(i_r)
        d_rand = np.mean(d_rand_acc, axis=0)
        i_rand = np.mean(i_rand_acc, axis=0)

        del_cam_curves.append(d_cam - base)
        del_rand_curves.append(d_rand - base)
        denom = (base - blurp) if abs(base - blurp) > 1e-6 else np.nan
        ins_cam_curves.append((i_cam - blurp) / denom)
        ins_rand_curves.append((i_rand - blurp) / denom)
        del_cam_auc.append(float(np.trapezoid(d_cam - base, fracs)))
        del_rand_auc.append(float(np.trapezoid(d_rand - base, fracs)))
        ins_cam_auc.append(float(np.trapezoid((i_cam - blurp) / denom, fracs)))
        ins_rand_auc.append(float(np.trapezoid((i_rand - blurp) / denom, fracs)))

        # ---- B. 模糊填充的位置对照 ----
        thr = np.quantile(cam, 0.80)
        mask = cam >= thr
        x_sal = x.clone()
        mt = torch.from_numpy(mask).to(device)
        x_sal[0][:, mt] = x_blur[0][:, mt]
        p_sal = float(predict_kg(model, x_sal, target_mean, target_std)[0])
        occ_salient.append(p_sal - base)
        shifts = []
        for _ in range(args.shifts):
            sy, sx = int(rng.integers(0, side)), int(rng.integers(0, side))
            m2 = np.roll(np.roll(mask, sy, axis=0), sx, axis=1)
            x_sh = x.clone()
            mt2 = torch.from_numpy(m2).to(device)
            x_sh[0][:, mt2] = x_blur[0][:, mt2]
            shifts.append(float(predict_kg(model, x_sh, target_mean, target_std)[0]) - base)
        occ_shifted.append(float(np.mean(shifts)))
        occ_pairs.append((p_sal - base, float(np.mean(shifts))))

    def arr_summary(v):
        v = np.asarray(v, dtype=float)
        return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                "median": float(np.nanmedian(v))}

    results["deletion"] = {
        "curve_cam": np.mean(del_cam_curves, axis=0).tolist(),
        "curve_random": np.mean(del_rand_curves, axis=0).tolist(),
        "auc_cam": arr_summary(del_cam_auc), "auc_random": arr_summary(del_rand_auc),
    }
    results["insertion"] = {
        "curve_cam": np.nanmean(ins_cam_curves, axis=0).tolist(),
        "curve_random": np.nanmean(ins_rand_curves, axis=0).tolist(),
        "auc_cam": arr_summary(ins_cam_auc), "auc_random": arr_summary(ins_rand_auc),
    }
    try:
        w_del = wilcoxon(del_cam_auc, del_rand_auc)
        results["deletion"]["wilcoxon_p"] = float(w_del.pvalue)
    except Exception as exc:  # noqa: BLE001
        results["deletion"]["wilcoxon_p"] = None
        results["deletion"]["wilcoxon_error"] = str(exc)
    try:
        w_ins = wilcoxon(ins_cam_auc, ins_rand_auc)
        results["insertion"]["wilcoxon_p"] = float(w_ins.pvalue)
    except Exception as exc:  # noqa: BLE001
        results["insertion"]["wilcoxon_p"] = None
        results["insertion"]["wilcoxon_error"] = str(exc)

    sal = np.asarray(occ_salient)
    shi = np.asarray(occ_shifted)
    results["blur_occlusion"] = {
        "n": len(sal), "shifts_per_image": args.shifts,
        "salient_delta_mean": float(sal.mean()), "salient_delta_std": float(sal.std()),
        "shifted_delta_mean": float(shi.mean()), "shifted_delta_std": float(shi.std()),
        "paired_salient_minus_shifted_mean": float((sal - shi).mean()),
        "frac_salient_worse": float((sal > shi).mean()),
        "wilcoxon_p": float(wilcoxon(sal, shi).pvalue) if len(sal) > 5 else None,
    }
    save_json(TABLE_DIR / f"{args.tag}.json", results)
    log.info("deletion AUC cam=%s random=%s p=%s", results["deletion"]["auc_cam"],
             results["deletion"]["auc_random"], results["deletion"].get("wilcoxon_p"))
    log.info("insertion AUC cam=%s random=%s p=%s", results["insertion"]["auc_cam"],
             results["insertion"]["auc_random"], results["insertion"].get("wilcoxon_p"))
    log.info("blur occlusion: %s", json.dumps(results["blur_occlusion"], ensure_ascii=False))

    # ---- 曲线图 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(fracs, results["deletion"]["curve_cam"], "-o", label="CAM 排序删除")
    axes[0].plot(fracs, results["deletion"]["curve_random"], "-s", label="随机排序删除")
    axes[0].set_xlabel("删除比例")
    axes[0].set_ylabel("预测变化 Δweight (kg)")
    axes[0].set_title("删除曲线（模糊填充）")
    axes[0].legend()
    axes[1].plot(fracs, results["insertion"]["curve_cam"], "-o", label="CAM 排序插入")
    axes[1].plot(fracs, results["insertion"]["curve_random"], "-s", label="随机排序插入")
    axes[1].set_xlabel("插入比例")
    axes[1].set_ylabel("恢复比例（0=全模糊，1=原图）")
    axes[1].set_title("插入曲线（模糊填充）")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.tag}_curves.png", dpi=200)
    plt.close(fig)

    # ---- C. RISE（小规模） ----
    rise_rows = []
    for i in idxs[: args.rise_n]:
        sample = ds[int(i)]
        x = sample["image"].unsqueeze(0).to(device)
        x_blur = TF.gaussian_blur(x, kernel_size=61, sigma=args.blur_sigma)
        cam = gradcam(model, x, args.backbone, stage=args.stage)[0]
        cell = 7
        low = (rng.random((args.rise_m, cell, cell)) > 0.5).astype(np.float32)
        masks = torch.from_numpy(low).unsqueeze(1)
        masks = torch.nn.functional.interpolate(masks, size=(side, side), mode="bilinear",
                                                align_corners=False)[:, 0].to(device)
        masks = (masks > 0.5).float()
        xb = x.repeat(args.rise_m, 1, 1, 1)
        xb = xb * masks.unsqueeze(1) + x_blur.repeat(args.rise_m, 1, 1, 1) * (1 - masks.unsqueeze(1))
        preds = predict_kg(model, xb, target_mean, target_std)
        w = np.clip(preds - preds.min(), 0, None)
        imp = (masks.cpu().numpy() * w[:, None, None]).sum(axis=0) / (w.sum() + 1e-8)
        rho = float(spearmanr(imp.ravel(), cam.ravel()).statistic)
        rise_rows.append({"idx": int(i), "spearman_cam_rise": rho})
    results["rise"] = {"n": len(rise_rows), "m_masks": args.rise_m,
                       "spearman_mean": float(np.mean([r["spearman_cam_rise"] for r in rise_rows])),
                       "rows": rise_rows}
    save_json(TABLE_DIR / f"{args.tag}.json", results)
    log.info("RISE vs CAM spearman mean=%.3f", results["rise"]["spearman_mean"])


if __name__ == "__main__":
    main()
