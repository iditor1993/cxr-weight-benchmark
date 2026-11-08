"""公式法（IBW/人体测量学）估计 ICU 体重的分层失真分析。

目的：回答"什么时候不能用公式估体重"——给出可直接写进论文讨论的分层证据。

分层维度
    体重层：<50 / 50–80 / 80–100 / ≥100 kg
    BMI 层：<18.5 / 18.5–25 / 25–30 / 30–40 / ≥40（需身高，n=986）
    性别、肥胖（BMI ≥30）

指标
    MAE（患者级聚类 bootstrap 95% CI）、bias（有符号误差）、RMSE、
    ±10% / ±20% 命中率、>20% 严重偏差比例、失真斜率（error ~ 真实体重 的回归斜率）

可接受性判读（事前定义）
    可接受：MAE ≤5 kg 且 ±10% 命中率 ≥80%
    边缘：  MAE ≤10 kg 且 ±10% 命中率 ≥60%
    不可接受：其余；若 >20% 严重偏差比例 ≥20% 另标"危险"

产出
    project/results/tables/formula_stratified_analysis.json
    project/results/figures/formula_stratified_analysis.png
    project/docs/formula_stratified_analysis.md（另由报告脚本撰写）
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, save_json, setup_logger

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
SEED = 20261012

FORMULA_METHODS = ["Devine IBW", "Hume IBW / ARDSNet PBW", "Robinson IBW", "Miller IBW",
                   "Hamwi IBW", "BMI=25 x height²", "OMR BMI x height²",
                   "population mean (train)", "sex-specific mean (train)", "OLS 身高+性别"]
CONTEXT_METHODS = ["admission_weight_kg", "charted_last_weight_kg", "pred"]
LABELS = {
    "Devine IBW": "Devine IBW", "Hume IBW / ARDSNet PBW": "Hume IBW/PBW",
    "Robinson IBW": "Robinson IBW", "Miller IBW": "Miller IBW", "Hamwi IBW": "Hamwi IBW",
    "BMI=25 x height²": "BMI=25×身高²", "OMR BMI x height²": "门诊BMI×身高²",
    "population mean (train)": "人群均值", "sex-specific mean (train)": "性别均值",
    "OLS 身高+性别": "身高+性别回归", "admission_weight_kg": "护理录入入院体重",
    "charted_last_weight_kg": "沿用既往/门诊记录", "pred": "胸片深度学习",
}


def cell_metrics(y, p, subjects, n_boot=1000, seed=SEED):
    m = np.isfinite(y) & np.isfinite(p)
    y, p, subjects = y[m], p[m], subjects[m]
    if len(y) < 5:
        return {"n": int(len(y))}
    err = p - y
    ae = np.abs(err)
    out = {"n": int(len(y)), "mae": float(ae.mean()), "median_ae": float(np.median(ae)),
           "rmse": float(np.sqrt((err ** 2).mean())), "bias": float(err.mean()),
           "median_ratio": float(np.median(p / y)),
           "hit_within_10pct": float((np.abs(err) / y <= 0.10).mean()),
           "hit_within_20pct": float((np.abs(err) / y <= 0.20).mean()),
           "severe_gt20pct": float((np.abs(err) / y > 0.20).mean())}
    uniq, inv = np.unique(subjects, return_inverse=True)
    csum = np.bincount(inv, weights=ae)
    cn = np.bincount(inv)
    rng = np.random.default_rng(seed)
    k = len(uniq)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        vals[b] = csum[idx].sum() / cn[idx].sum()
    out["mae_ci95"] = [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
    return out


def distortion(y, p, subjects, n_boot=1000, seed=SEED):
    """error = a + b * 真实体重；b 越负说明越"压扁"（越重越低估）。"""
    m = np.isfinite(y) & np.isfinite(p)
    y, p, subjects = y[m], p[m], subjects[m]
    if len(y) < 20:
        return {}
    err = p - y
    X = np.column_stack([np.ones(len(y)), y])
    beta = np.linalg.lstsq(X, err, rcond=None)[0]
    uniq, inv = np.unique(subjects, return_inverse=True)
    rng = np.random.default_rng(seed)
    k = len(uniq)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, k, k)
        rows = np.isin(inv, idx)
        if rows.sum() < 20:
            continue
        slopes.append(np.linalg.lstsq(np.column_stack([np.ones(rows.sum()), y[rows]]),
                                      err[rows], rcond=None)[0][1])
    return {"slope": float(beta[1]),
            "slope_ci95": [float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))]
            if slopes else None,
            "intercept": float(beta[0])}


def verdict(met):
    if "mae" not in met:
        return "样本不足"
    ok = met["mae"] <= 5 and met["hit_within_10pct"] >= 0.80
    marginal = met["mae"] <= 10 and met["hit_within_10pct"] >= 0.60
    tag = "可接受" if ok else ("边缘" if marginal else "不可接受")
    if met["severe_gt20pct"] >= 0.20:
        tag += "（危险）"
    return tag


def main() -> None:
    log = setup_logger("formula_stratified_analysis")
    d = pd.read_csv(PROCESSED_DIR / "conventional_weight_comparison_table.csv")
    d["male"] = (d["gender"] == "M").astype(float)
    # 身高+性别 OLS（在队列 A 训练集上拟合，见 baselines.json）
    d["OLS 身高+性别"] = -69.9350080268858 + 0.9196542642028203 * d["height_final_cm"] \
        + 0.04498295084068797 * d["male"]
    d["bmi_true"] = d["weight_kg"] / (d["height_final_cm"] / 100.0) ** 2

    y = d["weight_kg"].values
    subj = d["subject_id"].values
    methods = FORMULA_METHODS + CONTEXT_METHODS
    res = {"n_test": int(len(d)), "definitions": {
        "weight_strata_kg": [0, 50, 80, 100, 400],
        "bmi_strata": [0, 18.5, 25, 30, 40, 100],
        "acceptability": "可接受 = MAE≤5 kg 且 ±10%≥80%；边缘 = MAE≤10 kg 且 ±10%≥60%；危险 = >20% 偏差占比≥20%"},
        "overall": {}, "by_weight": {}, "by_bmi": {}, "by_sex": {}, "by_obesity": {},
        "distortion": {}}

    for m in methods:
        res["overall"][LABELS.get(m, m)] = cell_metrics(y, d[m].values, subj)
    for m in methods:
        met = res["overall"][LABELS.get(m, m)]
        met["verdict"] = verdict(met)
        res["distortion"][LABELS.get(m, m)] = distortion(y, d[m].values, subj)
    log.info("总体 MAE（公式类）：%s", json.dumps(
        {LABELS.get(m, m): round(res["overall"][LABELS.get(m, m)]["mae"], 1)
         for m in FORMULA_METHODS}, ensure_ascii=False))

    ws = pd.cut(d["weight_kg"], [0, 50, 80, 100, 400], labels=["<50", "50-80", "80-100", "≥100"])
    bs = pd.cut(d["bmi_true"], [0, 18.5, 25, 30, 40, 100],
                labels=["<18.5", "18.5-25", "25-30", "30-40", "≥40"])
    for tag, cut in (("by_weight", ws), ("by_bmi", bs)):
        for stratum in cut.cat.categories:
            mask = (cut == stratum).values
            res[tag][str(stratum)] = {
                "n": int(mask.sum()),
                "methods": {LABELS.get(m, m): cell_metrics(y[mask], d.loc[mask, m].values, subj[mask],
                                                           n_boot=400)
                            for m in methods}}
            for k, v in res[tag][str(stratum)]["methods"].items():
                v["verdict"] = verdict(v)
            log.info("%s %-8s n=%4d | Devine MAE=%.1f bias=%+.1f | 均值 MAE=%.1f | 深度学习 MAE=%.1f | 记录 MAE=%.1f",
                     tag, stratum, mask.sum(),
                     res[tag][str(stratum)]["methods"]["Devine IBW"]["mae"],
                     res[tag][str(stratum)]["methods"]["Devine IBW"]["bias"],
                     res[tag][str(stratum)]["methods"]["人群均值"]["mae"],
                     res[tag][str(stratum)]["methods"]["胸片深度学习"]["mae"],
                     res[tag][str(stratum)]["methods"]["护理录入入院体重"].get("mae", float("nan")))
    for name, mask in (("男性", (d["male"] == 1).values), ("女性", (d["male"] == 0).values),
                       ("肥胖 BMI≥30", (d["bmi_true"] >= 30).values),
                       ("非肥胖 BMI<30", (d["bmi_true"] < 30).values)):
        res["by_sex" if "性" in name else "by_obesity"][name] = {
            "n": int(mask.sum()),
            "methods": {LABELS.get(m, m): cell_metrics(y[mask], d.loc[mask, m].values, subj[mask],
                                                       n_boot=400)
                        for m in methods}}
    # Devine 在肥胖/非肥胖中的偏差（写讨论用）
    for nm in ("肥胖 BMI≥30", "非肥胖 BMI<30"):
        dv = res["by_obesity"][nm]["methods"]["Devine IBW"]
        log.info("%s（n=%d）：Devine MAE=%.1f，bias=%+.1f，±10%%=%.1f%%",
                 nm, res["by_obesity"][nm]["n"], dv["mae"], dv["bias"], 100 * dv["hit_within_10pct"])

    save_json(TABLE_DIR / "formula_stratified_analysis.json", res)

    # ---------- 图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    hs_methods = ["Devine IBW", "Hume IBW/PBW", "Robinson IBW", "Miller IBW",
                  "Hamwi IBW", "BMI=25×身高²", "门诊BMI×身高²", "人群均值", "性别均值", "身高+性别回归",
                  "胸片深度学习", "护理录入入院体重"]
    cols = ["<50", "50-80", "80-100", "≥100"]
    M = np.array([[res["by_weight"][c]["methods"][m].get("mae", np.nan) for c in cols]
                  for m in hs_methods])
    im = axes[0, 0].imshow(M, cmap="YlOrRd", aspect="auto", vmin=0, vmax=55)
    axes[0, 0].set_xticks(range(len(cols))), axes[0, 0].set_xticklabels(cols)
    axes[0, 0].set_yticks(range(len(hs_methods))), axes[0, 0].set_yticklabels(hs_methods, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                axes[0, 0].text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8)
    axes[0, 0].set_title("MAE (kg)：按体重分层"), axes[0, 0].set_xlabel("真实体重 (kg)")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)

    for m in ["Devine IBW", "Hume IBW/PBW", "门诊BMI×身高²", "人群均值", "身高+性别回归", "胸片深度学习",
              "护理录入入院体重"]:
        xs, ys = [], []
        for i, c in enumerate(cols):
            met = res["by_weight"][c]["methods"][m]
            if "bias" in met:
                xs.append(i), ys.append(met["bias"])
        axes[0, 1].plot(xs, ys, "o-", label=m)
    axes[0, 1].axhline(0, color="black", lw=1)
    axes[0, 1].set_xticks(range(len(cols))), axes[0, 1].set_xticklabels(cols)
    axes[0, 1].set_ylabel("bias（估计 − 真实，kg）"), axes[0, 1].set_xlabel("真实体重 (kg)")
    axes[0, 1].set_title("系统性偏差：负值 = 低估"); axes[0, 1].legend(fontsize=8)

    sev = ["Devine IBW", "Hume IBW/PBW", "BMI=25×身高²", "人群均值", "身高+性别回归",
           "胸片深度学习", "护理录入入院体重"]
    x = np.arange(len(cols))
    w = 0.8 / len(sev)
    for i, m in enumerate(sev):
        vals = [100 * res["by_weight"][c]["methods"][m].get("severe_gt20pct", np.nan) for c in cols]
        axes[1, 0].bar(x + i * w - 0.4 + w / 2, vals, w, label=m)
    axes[1, 0].set_xticks(x), axes[1, 0].set_xticklabels(cols)
    axes[1, 0].set_ylabel("偏差 >20% 的患者比例 (%)"), axes[1, 0].set_xlabel("真实体重 (kg)")
    axes[1, 0].set_title("严重偏差（>20%）比例"); axes[1, 0].legend(fontsize=7)

    bcols = ["<18.5", "18.5-25", "25-30", "30-40", "≥40"]
    Mb = np.array([[res["by_bmi"][c]["methods"][m].get("mae", np.nan) for c in bcols]
                   for m in hs_methods])
    imb = axes[1, 1].imshow(Mb, cmap="YlOrRd", aspect="auto", vmin=0, vmax=70)
    axes[1, 1].set_xticks(range(len(bcols))), axes[1, 1].set_xticklabels(bcols)
    axes[1, 1].set_yticks(range(len(hs_methods))), axes[1, 1].set_yticklabels(hs_methods, fontsize=8)
    for i in range(Mb.shape[0]):
        for j in range(Mb.shape[1]):
            if np.isfinite(Mb[i, j]):
                axes[1, 1].text(j, i, f"{Mb[i, j]:.1f}", ha="center", va="center", fontsize=8)
    axes[1, 1].set_title("MAE (kg)：按 BMI 分层"), axes[1, 1].set_xlabel("BMI (kg/m²)")
    fig.colorbar(imb, ax=axes[1, 1], fraction=0.046)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "formula_stratified_analysis.png", dpi=200)
    plt.close(fig)
    log.info("图 -> %s", FIGURE_DIR / "formula_stratified_analysis.png")
    log.info("总体可接受性：%s", json.dumps(
        {LABELS.get(m, m): res["overall"][LABELS.get(m, m)].get("verdict") for m in methods},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
