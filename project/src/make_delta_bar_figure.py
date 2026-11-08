"""以胸片深度学习为基线的发散柱状图：更准向上、更差向下，并标注显著性。

定义：value = 配对差 = |误差_深度学习| − |误差_该方法|
    value > 0 → 该方法比深度学习更准（向上）
    value < 0 → 该方法比深度学习更差（向下）
误差条为配对差的 95% CI（患者级聚类 bootstrap）；星号为 Benjamini–Hochberg 校正后显著性。

产出：Insights_into_Imaging/figures/fig2b_delta_vs_dl.png|svg
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TABLE = Path("project/results/tables/paired_significance.json")
FIG = Path("Insights_into_Imaging/figures")
FIG.mkdir(parents=True, exist_ok=True)

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verma": "#D55E00", "purple": "#CC79A7", "grey": "#999999",
      "black": "#000000"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600})
MM = 1 / 25.4

SHORT = {
    "Nurse-recorded admission weight": "Admission weight (recorded)",
    "Most recent previous weight": "Previous weight",
    "Outpatient weight": "Outpatient weight",
    "Outpatient BMI × height²": "Outpatient BMI × h²",
    "Height-and-sex regression": "Height+sex regression",
    "Sex-specific mean": "Sex-specific mean",
    "Population mean": "Population mean",
    "Fixed BMI 25 × height²": "Fixed BMI 25",
    "Miller IBW": "Miller IBW", "Hamwi IBW": "Hamwi IBW", "Robinson IBW": "Robinson IBW",
    "Hume IBW / ARDSNet PBW": "Hume IBW/PBW", "Devine IBW": "Devine IBW",
}


def main() -> None:
    rows = json.loads(TABLE.read_text(encoding="utf-8"))["rows"]
    # 配对差定义为 |err_DL| − |err_method|：正 = 该方法更准（向上），负 = 该方法更差（向下）
    for r in rows:
        r["value"] = r["paired_diff"]
        r["lo"] = r["ci_low"]
        r["hi"] = r["ci_high"]
    records = sorted([r for r in rows if r["value"] > 0], key=lambda r: -r["value"])
    others = sorted([r for r in rows if r["value"] < 0], key=lambda r: -r["value"])
    # 基线插到"更准"与"更差"之间
    labels = [SHORT[r["method"]] for r in records] + ["CXR deep learning\n(reference)"] + \
             [SHORT[r["method"]] for r in others]
    values = [r["value"] for r in records] + [0.0] + [r["value"] for r in others]
    los = [r["lo"] for r in records] + [0.0] + [r["lo"] for r in others]
    his = [r["hi"] for r in records] + [0.0] + [r["hi"] for r in others]
    stars = [r["significance"] for r in records] + [""] + [r["significance"] for r in others]
    colors = [OI["blue"]] * len(records) + [OI["black"]] + [OI["verma"]] * len(others)

    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(170 * MM, 100 * MM), layout="constrained")
    ax.axhline(0, color=OI["black"], linewidth=0.9)
    ax.bar(x, values, color=colors, width=0.62)
    ax.errorbar(x, values, yerr=[np.array(values) - np.array(los), np.array(his) - np.array(values)],
                fmt="none", ecolor=OI["grey"], elinewidth=0.8, capsize=2.2, zorder=3)
    # 基线组：在 0 处画一条短粗线表示参照
    ax.plot([x[len(records)] - 0.31, x[len(records)] + 0.31], [0, 0], color=OI["black"],
            linewidth=2.6, solid_capstyle="butt", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=32, ha="right")
    ax.set_ylabel("MAE(method) − MAE(deep learning) (kg)")
    ax.set_title("Accuracy relative to portable chest radiograph deep learning", loc="left")
    # 先确定坐标范围，再据此安排标注位置：数值移到柱外（正向上、负向下），星号再外移一层
    data_lo, data_hi = min(los), max(his)
    span = data_hi - data_lo
    ymin, ymax = data_lo - 0.22 * span, data_hi + 0.22 * span
    ax.set_ylim(ymin, ymax)
    span2 = ymax - ymin
    off_val = 0.022 * span2      # 数值与置信区间端帽的距离
    off_star = 0.062 * span2     # 星号与数值的距离
    for xi, v, st, lo, hi in zip(x, values, stars, los, his):
        if abs(v) > 0.5:                         # 蓝色柱：数值在端帽上方；橙色柱：数值在端帽下方
            y_val = hi + off_val if v > 0 else lo - off_val
            ax.text(xi, y_val, f"{v:+.1f}", ha="center", va="bottom" if v > 0 else "top",
                    fontsize=6.9, color="#222222", zorder=6)
            if st:                               # 星号再向外一层，避免与数值重叠
                y_star = hi + off_star if v > 0 else lo - off_star
                ax.text(xi, y_star, st, ha="center", va="bottom" if v > 0 else "top",
                        fontsize=7.4, zorder=6)
    ax.text(0.02, 0.99, "↑ up = more accurate than deep learning", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.2, color="#444444")
    ax.text(0.98, 0.99, "↓ down = less accurate than deep learning", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.2, color="#444444")
    ax.text(0.5, 0.015, "Error bars: 95% CI (patient-level cluster bootstrap). "
                        "Stars: Benjamini–Hochberg adjusted q (*** q<0.001).",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.6, style="italic",
            color="#444444")
    fig.savefig(FIG / "fig2b_delta_vs_dl.png", facecolor="white")
    fig.savefig(FIG / "fig2b_delta_vs_dl.svg", facecolor="white")
    plt.close(fig)
    print("saved fig2b_delta_vs_dl (png+svg)")


if __name__ == "__main__":
    main()
