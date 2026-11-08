"""生成 Insights into Imaging 投稿用图（6）与表（5）。

遵循科研可视化规范：
    - 不篡改数据；条形图含零基线；明确标注误差条含义与 n；
    - 颜色使用 Okabe-Ito 色盲友好调色板，并辅以直接标注/纹理；
    - 物理尺寸按期刊要求（半页宽 85 mm，整页宽 170 mm），导出 600 dpi PNG；
    - 同时输出 SVG（矢量）供期刊优先使用。

产出
    Insights_into_Imaging/figures/*.png|.svg
    Insights_into_Imaging/tables/*.csv|.md
    Insights_into_Imaging/figures/legends.md
"""

from __future__ import annotations

import json
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(".")
T = ROOT / "project/results/tables"
P = ROOT / "project/data/processed"
FIG = ROOT / "Insights_into_Imaging/figures"
TAB = ROOT / "Insights_into_Imaging/tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# 2026-10-08: main model switched to the ResNet-50 @308 3-seed ensemble
# (see project/results/tables/attention_ablation_decision.json).
MAIN_TAG = os.environ.get("CR_MAIN_TAG", "resnet50_308_ensemble")

# Okabe-Ito 色盲友好调色板
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verma": "#D55E00", "purple": "#CC79A7", "grey": "#999999",
      "black": "#000000"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600,
})
MM = 1 / 25.4


def f1(x: float) -> str:
    """一位小数，四舍五入（避免 Python 二进制浮点造成的 67.35→67.3）。"""
    return str(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


ev = load(T / f"evaluation_{MAIN_TAG}.json")
conv = load(T / "conventional_weight_comparison.json")
psig = load(T / "paired_significance.json")
base = load(T / "baselines.json")
strat = load(T / "formula_stratified_analysis.json")
nrr = load(T / "no_reliable_record_analysis.json")
unc = load(T / f"uncertainty_{MAIN_TAG}.json")
dose = load(T / f"dose_simulation_{MAIN_TAG}.json")
t1 = load(T / "table1_characteristics.json")
flow = load(P / "cohort_flow.json")
flow_nrr = load(P / "cohort_nrr_flow.json")
pred = pd.read_csv(P / f"test_predictions_{MAIN_TAG}.csv")

# 配对显著性：方法名 → {配对差, CI, p, q, 星号}
SIG = {}
for r in psig["rows"]:
    key = r["method"]
    SIG[key] = r
    if key == "Height-and-sex regression":
        SIG["OLS 身高+性别"] = r
    if key == "Nurse-recorded admission weight":
        SIG["admission_weight_kg"] = r
    if key == "Most recent previous weight":
        SIG["charted_last_weight_kg"] = r
    if key == "Outpatient weight":
        SIG["omr_weight_kg"] = r
    if key == "Outpatient BMI × height²":
        SIG["OMR BMI x height²"] = r
    if key == "Fixed BMI 25 × height²":
        SIG["BMI=25 x height²"] = r
    if key == "Population mean":
        SIG["population mean (train)"] = r
    if key == "Sex-specific mean":
        SIG["sex-specific mean (train)"] = r
    for ibw in ("Devine IBW", "Hume IBW / ARDSNet PBW", "Robinson IBW", "Miller IBW", "Hamwi IBW"):
        if key == ibw:
            SIG[ibw] = r

LAB = {"pred": "Portable CXR deep learning", "admission_weight_kg": "Nurse-recorded admission weight",
       "charted_last_weight_kg": "Most recent previous weight", "omr_weight_kg": "Outpatient weight",
       "OMR BMI x height²": "Outpatient BMI × height²", "BMI=25 x height²": "Fixed BMI 25 × height²",
       "Miller IBW": "Miller IBW", "Hamwi IBW": "Hamwi IBW", "Robinson IBW": "Robinson IBW",
       "Hume IBW / ARDSNet PBW": "Hume IBW / ARDSNet PBW", "Devine IBW": "Devine IBW",
       "sex-specific mean (train)": "Sex-specific mean", "population mean (train)": "Population mean",
       "ols_height_sex": "Height-and-sex regression"}


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches=None, facecolor="white")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- Fig 1 流程图
def fig1() -> None:
    fig, ax = plt.subplots(figsize=(170 * MM, 95 * MM), layout="constrained")
    ax.axis("off")
    boxes = [
        (0.5, 0.90, f"ICU stays screened\nn = {flow['n_icustays_total']:,}", OI["sky"]),
        (0.5, 0.72, f"Portable anteroposterior chest radiograph\nn = {flow['n_stays_with_portable_ap']:,} stays",
         OI["sky"]),
        (0.5, 0.52, f"Bed-scale weight within 24 h of the radiograph\n(reference standard)\n"
                    f"n = {flow['n_stays_cohort_a']:,} stays, {flow['n_patients_cohort_a']:,} patients",
         OI["green"]),
        (0.5, 0.30, "Patient-level split", OI["blue"]),
    ]
    for x, y, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x - 0.32, y - 0.055), 0.64, 0.11,
                                    boxstyle="round,pad=0.012", linewidth=0.9,
                                    edgecolor=col, facecolor="white", transform=ax.transAxes))
        ax.text(x, y, txt, ha="center", va="center", transform=ax.transAxes, fontsize=7.5)
    for y0, y1 in ((0.845, 0.78), (0.665, 0.58), (0.465, 0.36)):
        ax.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1), arrowstyle="-|>", mutation_scale=9,
                                     linewidth=0.9, color=OI["grey"], transform=ax.transAxes))
    for x, lab, val in ((0.20, "Training", 6102), (0.50, "Validation", 1362), (0.80, "Test", 1303)):
        ax.add_patch(FancyBboxPatch((x - 0.13, 0.14), 0.26, 0.09, boxstyle="round,pad=0.01",
                                    linewidth=0.9, edgecolor=OI["blue"], facecolor="white",
                                    transform=ax.transAxes))
        ax.text(x, 0.185, f"{lab}\nn = {val:,}", ha="center", va="center", fontsize=7.5,
                transform=ax.transAxes)
    ax.text(0.5, 0.045, f"Post hoc: {flow_nrr['n_no_reliable_measure_record_only']:,} stays without a "
                        f"contemporaneous measured weight\n({flow_nrr['n_nrr_with_delayed_reference_96h']} "
                        "with a delayed reference within 96 h)",
            ha="center", va="center", fontsize=7, style="italic", transform=ax.transAxes)
    save(fig, "fig1_cohort_flow")


# ---------------------------------------------------- Fig 2 基准与配对差异
def fig2() -> None:
    rows = [("Portable CXR deep learning", ev["overall"]["mae"], ev["overall"]["mae_ci95"][1],
             ev["overall"]["mae_ci95"][2], None)]
    for key in ("admission_weight_kg", "charted_last_weight_kg", "omr_weight_kg", "OMR BMI x height²",
                "BMI=25 x height²", "Miller IBW", "Hamwi IBW", "Robinson IBW",
                "Hume IBW / ARDSNet PBW", "Devine IBW", "sex-specific mean (train)",
                "population mean (train)"):
        v = conv[key]
        rows.append((LAB[key], v["mae"], v["mae_ci95"][1], v["mae_ci95"][2], v.get("paired_dl_minus_method")))
    reg = strat["overall"]["身高+性别回归"]
    rows.append((LAB["ols_height_sex"], reg["mae"], reg["mae_ci95"][0], reg["mae_ci95"][1], None))
    paired_ci = {}
    for key in ("admission_weight_kg", "charted_last_weight_kg", "omr_weight_kg", "OMR BMI x height²",
                "BMI=25 x height²", "Miller IBW", "Hamwi IBW", "Robinson IBW",
                "Hume IBW / ARDSNet PBW", "Devine IBW", "sex-specific mean (train)",
                "population mean (train)"):
        paired_ci[LAB[key]] = conv[key]["paired_ci95"]
    df = pd.DataFrame(rows, columns=["method", "mae", "lo", "hi", "paired"])
    # 两栏统一行序：配对差降序 = 记录（比模型更准）→ 深度学习基准 → 公式/均值/回归（更差）
    df["value"] = [SIG[m]["paired_diff"] if m in SIG else 0.0 for m in df["method"]]
    df["v_lo"] = [SIG[m]["ci_low"] if m in SIG else 0.0 for m in df["method"]]
    df["v_hi"] = [SIG[m]["ci_high"] if m in SIG else 0.0 for m in df["method"]]
    df["stars"] = [SIG[m]["significance"] if m in SIG else "" for m in df["method"]]
    df = df.sort_values("value", ascending=False).reset_index(drop=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(170 * MM, 115 * MM), layout="constrained",
                                   width_ratios=[0.88, 1.12])
    y = np.arange(len(df))
    short = {"Nurse-recorded admission weight": "Admission weight (recorded)",
             "Most recent previous weight": "Previous weight", "Outpatient weight": "Outpatient weight",
             "Outpatient BMI × height²": "Outpatient BMI × h²", "Fixed BMI 25 × height²": "Fixed BMI 25",
             "Height-and-sex regression": "Height+sex regression",
             "Portable CXR deep learning": "CXR deep learning",
             "Hume IBW / ARDSNet PBW": "Hume IBW/PBW", "Population mean": "Population mean",
             "Sex-specific mean": "Sex-specific mean"}
    record_labels = {"Nurse-recorded admission weight", "Outpatient weight",
                     "Most recent previous weight", "Outpatient BMI × height²"}
    colors = [OI["verma"] if m == "Portable CXR deep learning"
              else (OI["blue"] if m in record_labels else OI["grey"]) for m in df["method"]]

    # ---- 左栏：MAE 条形 + 95% CI（保留原设计）----
    axL.barh(y, df["mae"], color=colors, height=0.62)
    axL.errorbar(df["mae"], y, xerr=[df["mae"] - df["lo"], df["hi"] - df["mae"]], fmt="none",
                 ecolor=OI["black"], elinewidth=0.7, capsize=1.6)
    axL.set_yticks(y)
    axL.set_yticklabels([short.get(m, m) for m in df["method"]], fontsize=7)
    axL.invert_yaxis()                       # 第一行在最上，与右栏一一对应
    axL.set_xlabel("MAE (kg), 95% CI")
    axL.set_xlim(0, 30)
    axL.set_title("Accuracy on the test set", loc="left")
    for m, hi, yy in zip(df["mae"], df["hi"], y):
        axL.text(hi + 0.5, yy, f"{m:.1f}", va="center", fontsize=6.6)

    # ---- 右栏：以深度学习为基线的发散柱状图（更准向上、更差向下）----
    axR.axhline(0, color=OI["black"], linewidth=0.9)
    axR.bar(y, df["value"], color=[OI["black"] if m == "Portable CXR deep learning" else c
                                   for m, c in zip(df["method"], colors)], width=0.62)
    axR.errorbar(y, df["value"], yerr=[df["value"] - df["v_lo"], df["v_hi"] - df["value"]],
                 fmt="none", ecolor=OI["grey"], elinewidth=0.8, capsize=2.0, zorder=3)
    dl_idx = list(df["method"]).index("Portable CXR deep learning")
    axR.plot([y[dl_idx] - 0.31, y[dl_idx] + 0.31], [0, 0], color=OI["black"], linewidth=2.6,
             solid_capstyle="butt", zorder=5)
    axR.set_xticks(y)
    axR.set_xticklabels([short.get(m, m) for m in df["method"]], fontsize=6.6, rotation=40, ha="right")
    axR.set_ylabel("MAE(method) − MAE(deep learning) (kg)")
    axR.set_title("Accuracy relative to deep learning", loc="left")
    v_span = float(df["v_hi"].max() - df["v_lo"].min())
    y_lo = float(df["v_lo"].min()) - 0.30 * v_span
    y_hi = float(df["v_hi"].max()) + 0.30 * v_span
    axR.set_ylim(y_lo, y_hi)
    span2 = y_hi - y_lo
    off_val = 0.022 * span2
    # 正负两组各用一条统一基线放星号：避免相邻柱子的标注互相碰撞
    star_y_pos = float(df.loc[df["value"] > 0, "v_hi"].max()) + 0.045 * span2
    star_y_neg = float(df.loc[df["value"] < 0, "v_lo"].min()) - 0.045 * span2
    # 数值标签交错排布（相邻柱用两种偏移），避免细柱之间文字相撞
    pos_rank = neg_rank = 0
    for yy, v, st, lo, hi in zip(y, df["value"], df["stars"], df["v_lo"], df["v_hi"]):
        if abs(v) < 0.5:
            continue
        if v > 0:
            extra = 0.0 if pos_rank % 2 == 0 else 0.052 * span2
            pos_rank += 1
            axR.text(yy, hi + off_val + extra, f"{v:+.1f}", ha="center", va="bottom",
                     fontsize=6.3, color="#222222", zorder=6)
        else:
            extra = 0.0 if neg_rank % 2 == 0 else 0.052 * span2
            neg_rank += 1
            axR.text(yy, lo - off_val - extra, f"{v:+.1f}", ha="center", va="top",
                     fontsize=6.3, color="#222222", zorder=6)
        if st:
            axR.text(yy, star_y_pos if v > 0 else star_y_neg, st, ha="center",
                     va="bottom" if v > 0 else "top", fontsize=7.0, zorder=6)
    # 说明文字放在坐标区内部右上角（右对齐、分两行），避免压到 y 轴与刻度标签
    axR.text(0.985, 0.975, "↑ up = more accurate than deep learning\n↓ down = less accurate",
             transform=axR.transAxes, ha="right", va="top", fontsize=6.4, color="#444444",
             linespacing=1.35)
    axR.text(0.5, 0.015, "Error bars: 95% CI; stars: BH-adjusted q (*** q<0.001)",
             transform=axR.transAxes, ha="center", va="bottom", fontsize=6.4, style="italic",
             color="#444444")
    save(fig, "fig2_benchmark")


# ------------------------------------------------ Fig 3 公式失真（三层）
def fig3() -> None:
    methods = ["Devine IBW", "Hume IBW/PBW", "Robinson IBW", "Miller IBW", "Hamwi IBW",
               "BMI=25×身高²", "人群均值", "身高+性别回归", "胸片深度学习", "护理录入入院体重"]
    display = {"BMI=25×身高²": "Fixed BMI 25 × height²", "人群均值": "Population mean",
               "身高+性别回归": "Height-and-sex regression",
               "胸片深度学习": "Portable CXR deep learning",
               "护理录入入院体重": "Nurse-recorded admission weight"}
    wcols = ["<50", "50-80", "80-100", "≥100"]
    bcols = ["<18.5", "18.5-25", "25-30", "30-40", "≥40"]
    M = np.array([[strat["by_weight"][c]["methods"][m].get("mae", np.nan) for c in wcols]
                  for m in methods])
    Mb = np.array([[strat["by_bmi"][c]["methods"][m].get("mae", np.nan) for c in bcols]
                   for m in methods])
    fig = plt.figure(figsize=(170 * MM, 135 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.05, 1.0])
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])
    for ax, Mx, cols, title, vmax in ((ax1, M, wcols, "Error by measured weight", 55),
                                      (ax2, Mb, bcols, "Error by body mass index", 70)):
        im = ax.imshow(Mx, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{c}" for c in cols], rotation=30, ha="right")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([display.get(m, m) for m in methods] if ax is ax1 else ["" for _ in methods],
                           fontsize=6.6)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Weight (kg)" if ax is ax1 else "BMI (kg/m²)")
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                if np.isfinite(Mx[i, j]):
                    ax.text(j, i, f"{Mx[i, j]:.1f}", ha="center", va="center", fontsize=6.2)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="MAE (kg)")
    strata_x = [37, 65, 90, 125]
    for m, col, lab in (("Devine IBW", OI["verma"], "Devine IBW"),
                        ("人群均值", OI["purple"], "Population mean"),
                        ("身高+性别回归", OI["sky"], "Height-and-sex regression"),
                        ("胸片深度学习", OI["green"], "Portable CXR deep learning"),
                        ("护理录入入院体重", OI["blue"], "Nurse-recorded admission weight")):
        ys = [strat["by_weight"][c]["methods"][m]["bias"] for c in wcols]
        ax3.plot(strata_x, ys, marker="o", markersize=3.2, linewidth=1.0, color=col, label=lab)
    ax3.axhline(0, color=OI["black"], linewidth=0.8)
    ax3.set_xlabel("Measured body weight (kg, stratum midpoint)")
    ax3.set_ylabel("Bias (estimate − measured, kg)")
    ax3.set_title("Systematic bias across body size (negative = underestimation)", loc="left")
    ax3.legend(frameon=False, ncol=3, loc="lower left", fontsize=6.4)
    ax3.set_xticks(strata_x)
    ax3.set_xticklabels(["<50", "50-80", "80-100", "≥100"])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    save(fig, "fig3_formula_failure")


# ------------------------------------------- Fig 4 无同期实测人群
def fig4() -> None:
    m = nrr["methods"]
    pairs = [("胸片深度学习（本次）", "Portable CXR deep learning"),
             ("护理录入的入院体重", "Nurse-recorded admission weight"),
             ("身高+性别回归（同队列训练）", "Height-and-sex regression"),
             ("Devine IBW", "Devine IBW"),
             ("人群均值", "Population mean")]
    df = pd.DataFrame([{"method": lab, "mae": m[k]["mae"], "lo": m[k]["mae_ci95"][0],
                        "hi": m[k]["mae_ci95"][1]} for k, lab in pairs]).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 70 * MM), layout="constrained",
                             width_ratios=[1.15, 1.0])
    y = np.arange(len(df))
    colors = [OI["verma"] if "deep learning" in s else (OI["blue"] if "admission" in s else OI["grey"])
              for s in df["method"]]
    axes[0].barh(y, df["mae"], color=colors, height=0.6)
    axes[0].errorbar(df["mae"], y, xerr=[df["mae"] - df["lo"], df["hi"] - df["mae"]], fmt="none",
                     ecolor=OI["black"], elinewidth=0.7, capsize=1.6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["method"])
    axes[0].set_xlabel("Mean absolute error (kg)")
    axes[0].set_xlim(0, 33)
    axes[0].set_title(f"Stays without a contemporaneous measurement (n={nrr['n_evaluable']})", loc="left")
    for v, hi, yi in zip(df["mae"], df["hi"], y):
        axes[0].text(hi + 0.5, yi, f"{v:.1f}", va="center", fontsize=6.8)
    pv = nrr.get("paired_vs_dl", {})
    names, diffs, los, his = [], [], [], []
    for k, lab in pairs[1:]:
        key = {"护理录入的入院体重": "护理录入的入院体重",
               "身高+性别回归（同队列训练）": "身高+性别回归（同队列训练）",
               "Devine IBW": "Devine IBW", "人群均值": "人群均值"}[k]
        if key in pv:
            names.append(lab)
            diffs.append(pv[key]["diff"])
            los.append(pv[key]["ci95"][0])
            his.append(pv[key]["ci95"][1])
    yy = np.arange(len(names))
    axes[1].axvline(0, color=OI["black"], linewidth=0.8)
    axes[1].errorbar(diffs, yy, xerr=[np.array(diffs) - np.array(los), np.array(his) - np.array(diffs)],
                     fmt="o", color=OI["verma"], markersize=3.4, ecolor=OI["grey"], capsize=1.6,
                     elinewidth=0.7)
    axes[1].set_yticks(yy)
    axes[1].set_yticklabels(names, fontsize=6.8)
    axes[1].set_xlabel("Paired difference (kg)")
    axes[1].set_title("Negative favours deep learning", loc="left", fontsize=8)
    save(fig, "fig4_no_measurement")


# ------------------------------------- Fig 5 安全边界与不确定性
def clustered_ci(values: np.ndarray, subjects: np.ndarray, n_boot: int = 1000, seed: int = 7):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(subjects, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    cnts = np.bincount(inv)
    k = len(uniq)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        out[b] = sums[idx].sum() / cnts[idx].sum()
    return np.quantile(out, 0.025), np.quantile(out, 0.975)


def fig5() -> None:
    err = pred["pred"].values - pred["weight_kg"].values
    strata = [(pred["weight_kg"] < 50), (pred["weight_kg"].between(50, 80, inclusive="left")),
              (pred["weight_kg"].between(80, 100, inclusive="left")), (pred["weight_kg"] >= 100)]
    labels = ["<50", "50-80", "80-100", "≥100"]
    bias = [err[m].mean() for m in strata]
    ci = [clustered_ci(err[m], pred["subject_id"].values[m]) for m in strata]
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 62 * MM), layout="constrained")
    x = np.arange(4)
    axes[0].axhline(0, color=OI["black"], linewidth=0.8)
    axes[0].errorbar(x, bias, yerr=[np.array(bias) - np.array([c[0] for c in ci]),
                                    np.array([c[1] for c in ci]) - np.array(bias)],
                     fmt="o", color=OI["verma"], markersize=4, capsize=2, elinewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Measured weight stratum (kg)")
    axes[0].set_ylabel("Bias (kg)")
    axes[0].set_title("Model bias by weight stratum (95% CI)", loc="left")
    cov = unc["coverage_by_weight_stratum"]
    covv = [100 * cov[k]["coverage"] if isinstance(cov[k], dict) and "coverage" in cov[k]
            else 100 * cov[k] for k in ("<50", "50-80", "80-100", ">=100")]
    axes[1].bar(x, covv, color=OI["sky"], width=0.6)
    axes[1].axhline(85, color=OI["verma"], linestyle="--", linewidth=0.9)
    axes[1].text(3.35, 86.5, "prespecified 85%", color=OI["verma"], fontsize=6.5, ha="right")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("Measured weight stratum (kg)")
    axes[1].set_ylabel("Empirical 90% interval coverage (%)")
    axes[1].set_title("Prediction-interval coverage by stratum", loc="left")
    for xi, v in zip(x, covv):
        axes[1].text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=6.8)
    save(fig, "fig5_safety_uncertainty")


# ------------------------------------------- Fig 6 决策摘要（示意 + 数据）
def fig6() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 70 * MM), layout="constrained",
                             width_ratios=[1.0, 1.05])
    ibw_keys = ("Devine IBW", "Hume IBW / ARDSNet PBW", "Robinson IBW", "Miller IBW", "Hamwi IBW")
    ibw_mean = float(np.mean([conv[k]["mae"] for k in ibw_keys]))
    sources = [("Admission\nweight", conv["admission_weight_kg"]["mae"], OI["blue"]),
               ("CXR deep\nlearning", ev["overall"]["mae"], OI["verma"]),
               ("Height–sex\nregression", strat["overall"]["身高+性别回归"]["mae"], OI["sky"]),
               ("Population\nmean", conv["population mean (train)"]["mae"], OI["grey"]),
               ("IBW\nformulas", ibw_mean, OI["purple"])]
    x = np.arange(len(sources))
    axes[0].bar(x, [s[1] for s in sources], color=[s[2] for s in sources], width=0.62)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s[0] for s in sources], fontsize=6.6)
    axes[0].set_ylabel("MAE (kg)")
    axes[0].set_title("Weight sources when weighing is not possible", loc="left", fontsize=8.5)
    for xi, s in zip(x, sources):
        axes[0].text(xi, s[1] + 0.5, f"{s[1]:.1f}", ha="center", fontsize=6.8)
    axes[1].axis("off")
    rows = [("BMI <18.5", "Formulas overestimate", "Use history or imaging"),
            ("BMI 18.5-25", "Formulas acceptable (MAE ≈5 kg)", "Measurement still preferred"),
            ("BMI 25-30", "Formulas fail (MAE 15 kg)", "Do not substitute"),
            ("BMI 30-40", "Formulas fail (MAE 35 kg, 0% within 10%)", "Do not substitute"),
            ("BMI ≥40", "Formulas fail (MAE 69 kg)", "Do not substitute")]
    axes[1].set_title("When formula-based substitutes fail", loc="left", fontsize=8.5)
    for i, (a, b, c) in enumerate(rows):
        ypos = 0.86 - i * 0.18
        col = OI["green"] if "acceptable" in b else OI["verma"]
        axes[1].add_patch(FancyBboxPatch((0.02, ypos - 0.055), 0.96, 0.13,
                                         boxstyle="round,pad=0.012", linewidth=0.8,
                                         edgecolor=col, facecolor="white",
                                         transform=axes[1].transAxes))
        axes[1].text(0.05, ypos + 0.02, a, fontsize=7.2, fontweight="bold",
                     transform=axes[1].transAxes)
        axes[1].text(0.34, ypos + 0.02, b, fontsize=7.0, transform=axes[1].transAxes)
        axes[1].text(0.34, ypos - 0.028, c, fontsize=6.6, style="italic", color="#444444",
                     transform=axes[1].transAxes)
    save(fig, "fig6_decision_summary")


# ------------------------------------------------------------------ 表格
def tables() -> None:
    # Table 1
    t1_rows = [
        ("ICU stays, n", f"{t1['n_stays']:,}"),
        ("Patients, n", f"{t1['n_patients']:,}"),
        ("Age, median (IQR), years", f"{t1['age_median']:.0f} ({t1['age_iqr'][0]:.0f}-{t1['age_iqr'][1]:.0f})"),
        ("Male, n (%)", f"{t1['male_n']:,} ({100*t1['male_pct']:.1f})"),
        ("Measured weight, median (IQR), kg",
         f"{f1(t1['weight_median'])} ({f1(t1['weight_iqr'][0])}-{f1(t1['weight_iqr'][1])})"),
        ("Weight <50 kg, n", f"{t1['n_lt50']}"),
        ("Weight ≥100 kg, n", f"{t1['n_ge100']:,}"),
        ("ICU length of stay, median (IQR), days",
         f"{t1['los_median_days']:.1f} ({t1['los_iqr_days'][0]:.1f}-{t1['los_iqr_days'][1]:.1f})"),
        ("Invasive mechanical ventilation, %", f"{100*t1['ventilated_pct']:.1f}"),
        ("Interval from radiograph to weight, median, h", f"{t1['gap_hours_median']:.1f}"),
        ("Test-set sex (male/female), n", f"{t1['test_sex']['male']}/{t1['test_sex']['female']}"),
    ]
    pd.DataFrame(t1_rows, columns=["Characteristic", "Value"]).to_csv(
        TAB / "table1_cohort_characteristics.csv", index=False)
    (TAB / "table1_cohort_characteristics.md").write_text(
        "| Characteristic | Value |\n|---|---|\n" +
        "\n".join(f"| {a} | {b} |" for a, b in t1_rows) + "\n", encoding="utf-8")

    # Table 2 基准
    rows2 = [("Portable CXR deep learning", ev["overall"]["n"], ev["overall"]["mae"],
              ev["overall"]["mae_ci95"][1], ev["overall"]["mae_ci95"][2], ev["overall"]["bias"],
              100 * ev["overall"]["hit_within_10pct"], None)]
    for key in ("admission_weight_kg", "charted_last_weight_kg", "omr_weight_kg", "OMR BMI x height²",
                "BMI=25 x height²", "Miller IBW", "Hamwi IBW", "Robinson IBW",
                "Hume IBW / ARDSNet PBW", "Devine IBW", "sex-specific mean (train)",
                "population mean (train)"):
        v = conv[key]
        rows2.append((LAB[key], v["n"], v["mae"], v["mae_ci95"][1], v["mae_ci95"][2], v["bias"],
                      100 * v["hit_within_10pct"], v.get("paired_dl_minus_method")))
    # 身高+性别回归：与配对检验同子集（strat 中的 986 例），避免"MAE 子集 ≠ 配对子集"的错配
    reg = strat["overall"]["身高+性别回归"]
    rows2.append((LAB["ols_height_sex"], reg["n"], reg["mae"],
                  reg["mae_ci95"][0], reg["mae_ci95"][1],
                  reg["bias"], 100 * reg["hit_within_10pct"], None))
    # 附加：配对差的 95% CI、Wilcoxon p、BH 校正 q 与显著性星号
    name2sig = {"Portable CXR deep learning": None}
    for r in psig["rows"]:
        name2sig[r["method"]] = r
    t2_rows = []
    for r in rows2:
        sig = name2sig.get(r[0])
        t2_rows.append({
            "Method": r[0], "n": r[1], "MAE (kg)": r[2], "MAE 95% CI lower": r[3],
            "MAE 95% CI upper": r[4], "Bias (kg)": r[5], "Within ±10% (%)": r[6],
            "Paired difference vs model (kg)": r[7],
            "Paired 95% CI lower": None if sig is None else sig["ci_low"],
            "Paired 95% CI upper": None if sig is None else sig["ci_high"],
            "Wilcoxon p": None if sig is None else sig["p_value"],
            "BH-adjusted q": None if sig is None else sig["q_value"],
            "Significance": "reference" if sig is None else sig["significance"]})
    t2 = pd.DataFrame(t2_rows)
    t2.to_csv(TAB / "table2_benchmark.csv", index=False)
    md2 = ["| Method | n | MAE (kg) | MAE 95% CI | Bias (kg) | Within ±10% | Paired diff. vs model (kg) | Paired 95% CI | p (Wilcoxon) | q (BH) | Sig. |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t2_rows:
        md2.append(
            f"| {r['Method']} | {r['n']} | {r['MAE (kg)']:.2f} | {r['MAE 95% CI lower']:.2f} to "
            f"{r['MAE 95% CI upper']:.2f} | {r['Bias (kg)']:+.2f} | {r['Within ±10% (%)']:.1f}% | "
            + ("—" if r["Paired difference vs model (kg)"] is None
               else f"{r['Paired difference vs model (kg)']:+.2f}")
            + " | " + ("—" if r["Paired 95% CI lower"] is None
                       else f"{r['Paired 95% CI lower']:+.2f} to {r['Paired 95% CI upper']:+.2f}")
            + " | " + ("—" if r["Wilcoxon p"] is None else f"{r['Wilcoxon p']:.1e}")
            + " | " + ("—" if r["BH-adjusted q"] is None else f"{r['BH-adjusted q']:.1e}")
            + " | " + str(r["Significance"]) + " |")
    md2.append("")
    md2.append("Significance: *** q<0.001, ** q<0.01, * q<0.05 (Benjamini–Hochberg across the 13 comparisons "
               "against the model); negative paired differences favour the deep learning model.")
    (TAB / "table2_benchmark.md").write_text("\n".join(md2) + "\n", encoding="utf-8")

    # Table 3 分层失真
    methods = ["Devine IBW", "Hume IBW/PBW", "Robinson IBW", "Miller IBW", "Hamwi IBW",
               "BMI=25×身高²", "门诊BMI×身高²", "人群均值", "身高+性别回归", "胸片深度学习", "护理录入入院体重"]
    md3 = ["| Method | metric | " + " | ".join(["<50 kg", "50-80 kg", "80-100 kg", "≥100 kg"]) + " |",
           "|---|---|---|---|---|---|"]
    for m in methods:
        for metric, key in (("MAE (kg)", "mae"), ("Bias (kg)", "bias"), ("Within ±10% (%)", "hit_within_10pct")):
            vals = []
            for c in ("<50", "50-80", "80-100", "≥100"):
                v = strat["by_weight"][c]["methods"][m].get(key)
                if v is None:
                    vals.append("—")
                elif key == "hit_within_10pct":
                    vals.append(f"{100*v:.1f}")
                else:
                    vals.append(f"{v:.1f}")
            md3.append(f"| {m} | {metric} | " + " | ".join(vals) + " |")
    md3 += ["", "| Method | metric | " + " | ".join(["<18.5", "18.5-25", "25-30", "30-40", "≥40"]) + " |",
            "|---|---|---|---|---|---|---|"]
    for m in methods:
        for metric, key in (("MAE (kg)", "mae"), ("Bias (kg)", "bias"), ("Within ±10% (%)", "hit_within_10pct")):
            vals = []
            for c in ("<18.5", "18.5-25", "25-30", "30-40", "≥40"):
                v = strat["by_bmi"][c]["methods"][m].get(key)
                if v is None:
                    vals.append("—")
                elif key == "hit_within_10pct":
                    vals.append(f"{100*v:.1f}")
                else:
                    vals.append(f"{v:.1f}")
            md3.append(f"| {m} | {metric} | " + " | ".join(vals) + " |")
    (TAB / "table3_stratified_errors.md").write_text("\n".join(md3) + "\n", encoding="utf-8")

    # Table 4 无同期实测
    md4 = ["| Method | n | MAE (kg) | 95% CI | Bias (kg) | Within ±10% | Paired diff. vs model |",
           "|---|---|---|---|---|---|---|"]
    for k, lab in (("胸片深度学习（本次）", "Portable CXR deep learning"),
                   ("护理录入的入院体重", "Nurse-recorded admission weight"),
                   ("身高+性别回归（同队列训练）", "Height-and-sex regression"),
                   ("Devine IBW", "Devine IBW"), ("人群均值", "Population mean"),
                   ("性别均值", "Sex-specific mean")):
        v = nrr["methods"][k]
        p = nrr.get("paired_vs_dl", {}).get(k, {}).get("diff")
        md4.append(f"| {lab} | {v['n']} | {v['mae']:.2f} | {v['mae_ci95'][0]:.2f} to {v['mae_ci95'][1]:.2f} | "
                   f"{v['bias']:+.2f} | {100*v['hit_within_10pct']:.1f}% | "
                   + ("—" if p is None else f"{p:+.2f}") + " |")
    (TAB / "table4_no_measurement.md").write_text("\n".join(md4) + "\n", encoding="utf-8")

    # Table 5 不确定性与剂量
    abw = next(r for r in dose["rows"] if r["weight_scalar"] == "ABW")
    adj = next(r for r in dose["rows"] if r["weight_scalar"] == "AdjBW")
    rows5 = [
        ("Nominal 90% interval coverage, %", f"{100*unc['coverage_90_nominal']:.1f}"),
        ("Mean interval width, kg", f"{unc['mean_interval_width_kg']:.1f}"),
        ("Dose deviation >10%, actual body weight dosing, %", f"{abw['pct_dose_diff_gt10']:.1f}"),
        ("Dose deviation >20%, actual body weight dosing, %", f"{abw['pct_dose_diff_gt20']:.1f}"),
        ("Median absolute dose deviation, actual body weight dosing, %",
         f"{abw['median_abs_pct_diff']:.1f}"),
        ("Dose deviation >10%, adjusted body weight dosing, %", f"{adj['pct_dose_diff_gt10']:.1f}"),
        ("Median absolute dose deviation, adjusted body weight dosing, %",
         f"{adj['median_abs_pct_diff']:.1f}"),
    ]
    (TAB / "table5_uncertainty_dosing.md").write_text(
        "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows5) + "\n",
        encoding="utf-8")

    # 图注
    legends = """# Figure titles and legends (Insights into Imaging)

**Fig. 1 Cohort flow diagram.**
Of 94,458 ICU stays screened, 13,960 had a portable anteroposterior chest radiograph and 8,767 had a
bed-scale measured weight within 24 hours, forming the analysis cohort of 7,166 patients. Stays were
split at the patient level into training (6,102), validation (1,362) and test (1,303) sets. Post hoc,
4,656 stays without a contemporaneous measured weight were identified, of which 482 had a delayed
reference measurement within 96 hours.

**Fig. 2 Benchmark of the imaging model against routinely available weight estimates.**
Left, mean absolute error with 95% confidence intervals (patient-level cluster bootstrap) on the test
set. Right, the same comparisons expressed relative to the deep learning model, which appears as the
reference category at zero: bars above zero denote methods more accurate than the model, bars below
zero denote methods less accurate than the model, and whiskers are 95% confidence intervals of the
paired difference. Stars denote Benjamini-Hochberg adjusted significance (*** q<0.001), plotted on a
common level for bars above and below zero. Both panels share the same row order and bars include a
zero baseline. n = 1,303 stays for the model and the population or sex means, 1,265 for the
nurse-recorded admission weight, 859 and 692 for previous and outpatient weights, 618 for outpatient
BMI × height², and 986 for formula-based estimates and the height-and-sex regression.

**Fig. 3 Distortion of formula-based estimates across body size.**
Top left, mean absolute error by measured weight stratum. Top right, mean absolute error by body mass
index stratum. Bottom, signed bias across weight strata, where negative values indicate underestimation.
Devine error reaches 69.4 kg at a body mass index of 40 or above, where no estimate fell within 10%.

**Fig. 4 Benchmark in stays without a contemporaneous measured weight.**
Left, mean absolute error with 95% confidence intervals in the 482 evaluable stays, using a delayed
measured weight (median 38.4 hours) as reference. Right, paired differences against the deep learning
model. Nurse-recorded admission weight remained the most accurate value in this population.

**Fig. 5 Safety boundaries and uncertainty of the imaging model.**
Left, model bias by measured weight stratum with 95% confidence intervals from patient-level cluster
bootstrap. Right, empirical coverage of nominal 90% prediction intervals by stratum, with the
prespecified 85% target shown. Intervals were derived from the dispersion of three training seeds.

**Fig. 6 Weight sources and the limits of formula-based substitution.**
Left, mean absolute error of each weight source available when weighing is not possible. Right,
stratum-specific interpretation of formula accuracy, showing that substitution is defensible only
within the normal body mass index range. Values are taken from the test-set benchmark.
"""
    (FIG / "legends.md").write_text(legends, encoding="utf-8")
    print("tables + legends written")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    tables()
