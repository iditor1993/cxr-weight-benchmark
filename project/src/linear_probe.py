"""实验 B：冻结 Swin-T 特征的线性探针（全局 + 空间 patch），定位体重相关信息。"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from common import FEATURE_DIR, FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, save_json, setup_logger
from metrics import regression_metrics

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def fit_ridge(Xtr, ytr, Xte, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    yc = ytr - ytr.mean()
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(d, dtype=np.float32)
    b = Xtr.T @ yc
    coef = np.linalg.solve(A, b)
    return Xte @ coef + ytr.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="swin_features")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--tag", default="linear_probe")
    args = ap.parse_args()
    log = setup_logger(args.tag)

    index = pd.read_csv(PROCESSED_DIR / f"{args.features}_index.csv")
    cohort = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")[["dicom_id", "subject_id"]]
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    df = index.merge(cohort, on="dicom_id", how="left").merge(splits, on="subject_id", how="left")
    df["split_random"] = df["split_random"].fillna("train")

    gl = np.load(FEATURE_DIR / f"{args.features}_global.npy").astype(np.float32)
    sp = np.load(FEATURE_DIR / f"{args.features}_spatial.npy", mmap_mode="r")
    n, c, p = sp.shape
    log.info("features: global %s spatial %s", gl.shape, sp.shape)
    tr = (df["split_random"] == "train").values
    va = (df["split_random"] == "val").values
    te = (df["split_random"] == "test").values
    y = df["weight_kg"].values.astype(np.float32)
    log.info("split sizes: train=%d val=%d test=%d", tr.sum(), va.sum(), te.sum())

    # ---- B1 全局线性探针 ----
    alphas = [0.1, 1, 10, 100, 1000, 10000]
    best = None
    for a in alphas:
        pv = fit_ridge(gl[tr], y[tr], gl[va], a)
        mae = float(np.mean(np.abs(pv - y[va])))
        if best is None or mae < best[1]:
            best = (a, mae)
    alpha_global = best[0]
    pred_te = fit_ridge(gl[tr], y[tr], gl[te], alpha_global)
    global_metrics = {"alpha": alpha_global, "val_mae": best[1],
                      "test": regression_metrics(y[te], pred_te)}
    log.info("global probe: alpha=%.1f val MAE=%.2f test MAE=%.2f R2=%.3f",
             alpha_global, best[1], global_metrics["test"]["mae"], global_metrics["test"]["r2"])

    # ---- B2 空间 patch 线性探针 ----
    r2_map = np.zeros((20, 20), dtype=np.float32)
    mae_map = np.zeros((20, 20), dtype=np.float32)
    p_map = np.ones((20, 20), dtype=np.float32)
    perm_cache = []
    ytr_mean, ytr_std = float(y[tr].mean()), float(y[tr].std())
    rng = np.random.default_rng(20261009)
    ytr = y[tr]
    yte = y[te]
    for j in range(20):
        for i in range(20):
            idx = j * 20 + i
            Xtr = np.asarray(sp[tr, :, idx], dtype=np.float32)
            Xte = np.asarray(sp[te, :, idx], dtype=np.float32)
            pred = fit_ridge(Xtr, ytr, Xte, alpha_global)
            m = regression_metrics(yte, pred)
            r2_map[j, i] = m["r2"]
            mae_map[j, i] = m["mae"]
            # 置换：打乱训练标签重拟合（同一 Gram 矩阵，成本低）
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Xs = (Xtr - mu) / sd
            Xt = (Xte - mu) / sd
            A = Xs.T @ Xs + alpha_global * np.eye(Xs.shape[1], dtype=np.float32)
            Ainv = np.linalg.inv(A)
            null_r2 = []
            for _ in range(args.n_perm):
                yp = rng.permutation(ytr)
                ypc = yp - yp.mean()
                coef = Ainv @ (Xs.T @ ypc)
                pp = Xt @ coef + yp.mean()
                ss_res = float(np.sum((yte - pp) ** 2))
                ss_tot = float(np.sum((yte - yte.mean()) ** 2))
                null_r2.append(1 - ss_res / ss_tot)
            p_map[j, i] = float((1 + np.sum(np.array(null_r2) >= m["r2"])) / (1 + args.n_perm))
        if j % 5 == 0:
            log.info("patch row %d/20 done", j)

    # BH-FDR
    flat_p = p_map.ravel()
    order = np.argsort(flat_p)
    ranked = flat_p[order]
    q = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q_map = np.ones_like(flat_p)
    q_map[order] = np.clip(q, 0, 1)
    q_map = q_map.reshape(20, 20)
    n_sig = int((q_map < 0.05).sum())

    # 与解剖区域比较（来自 60 张测试图的分割均值）
    region_cmp = {}
    npz_path = PROCESSED_DIR / "anatomy_pooled_1303.npz"
    if not npz_path.exists():
        npz_path = PROCESSED_DIR / "anatomy_regions_test60.npz"
    if npz_path.exists():
        npz = np.load(npz_path, allow_pickle=True)
        if "pooled" in npz_path.name:
            pooled = {k: npz[k] for k in ("lungs", "cardiomediastinum", "subdiaphragmatic")}
        else:
            def pool(m):
                out = np.zeros((20, 20), dtype=np.float32)
                for j in range(20):
                    for i in range(20):
                        out[j, i] = m[j * 15:(j + 1) * 15, i * 15:(i + 1) * 15].mean()
                return out
            pooled = {k: pool(npz[k]) for k in ("lungs", "cardiomediastinum", "subdiaphragmatic")}
        for k, v in pooled.items():
            inside = r2_map[v > 0.05]
            outside = r2_map[v <= 0.05]
            if len(inside) > 3 and len(outside) > 3:
                region_cmp[k] = {"n_inside": int(len(inside)),
                                 "r2_inside_mean": float(inside.mean()),
                                 "r2_outside_mean": float(outside.mean()),
                                 "mannwhitney_p": float(mannwhitneyu(inside, outside).pvalue)}
    out = {"n": int(n), "split_sizes": {"train": int(tr.sum()), "val": int(va.sum()),
                                        "test": int(te.sum())},
           "global_probe": global_metrics,
           "spatial_probe": {"alpha": alpha_global,
                             "r2_map": r2_map.tolist(), "mae_map": mae_map.tolist(),
                             "q_map": q_map.tolist(),
                             "n_patches_fdr05": n_sig,
                             "r2_mean": float(r2_map.mean()), "r2_max": float(r2_map.max()),
                             "mae_min": float(mae_map.min())},
           "region_comparison": region_cmp}
    save_json(TABLE_DIR / f"{args.tag}.json", out)
    log.info("spatial probe: mean R2=%.3f max=%.3f | FDR<0.05 patches=%d/400",
             r2_map.mean(), r2_map.max(), n_sig)
    log.info("region comparison: %s", json.dumps(region_cmp, ensure_ascii=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].imshow(r2_map, cmap="magma")
    axes[0].set_title("每 patch 的测试集 R²（20×20）")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(mae_map, cmap="viridis_r")
    axes[1].set_title("每 patch 的测试集 MAE (kg)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(np.where(q_map < 0.05, r2_map, np.nan), cmap="magma")
    axes[2].set_title(f"FDR<0.05 的 patch（{n_sig}/400）")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.tag}_r2_map.png", dpi=200)
    plt.close(fig)
    log.info("figure -> %s", FIGURE_DIR / f"{args.tag}_r2_map.png")


if __name__ == "__main__":
    main()
