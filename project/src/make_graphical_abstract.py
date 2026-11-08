"""生成 Insights into Imaging 投稿所需的 Graphical abstract（期刊模板要求的单幅图）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path("Insights_into_Imaging/figures")
FIG.mkdir(parents=True, exist_ok=True)
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2",
      "verma": "#D55E00", "purple": "#CC79A7", "grey": "#999999", "black": "#000000"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "savefig.dpi": 600})

fig, ax = plt.subplots(figsize=(170 / 25.4, 85 / 25.4), layout="constrained")
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

# 左：公式替代值随 BMI 崩塌
ax.text(5.0, 4.85, "In intensive care, substitute weights mislead, imaging is the fallback, "
                   "measurement stays first", ha="center", fontsize=8.6, fontweight="bold", va="top")
ax.text(0.15, 4.35, "Formula-based substitutes fail as body size increases",
        fontsize=8.2, fontweight="bold", va="top")
bmi = ["BMI\n18.5-25", "BMI\n30-40", "BMI\n≥40"]
mae = [5.0, 35.3, 69.4]
x0, base, scale = 0.20, 1.05, 20.0
for i, (b, m) in enumerate(zip(bmi, mae)):
    ax.add_patch(plt.Rectangle((x0 + i * 0.95, base), 0.72, m / scale, color=OI["verma"],
                               alpha=0.85))
    ax.text(x0 + i * 0.95 + 0.36, base + m / scale + 0.10, f"{m:.0f} kg", ha="center", fontsize=7.5)
    ax.text(x0 + i * 0.95 + 0.36, base - 0.05, b, ha="center", va="top", fontsize=7)
ax.text(1.47, 0.30, "Devine ideal body weight vs 1,303 measured weights", ha="center",
        fontsize=6.8, style="italic")

# 中部：床旁胸片深度学习
ax.add_patch(FancyBboxPatch((3.55, 1.5), 2.9, 2.4, boxstyle="round,pad=0.08",
                            linewidth=1.1, edgecolor=OI["green"], facecolor="white"))
ax.text(5.0, 3.5, "Portable chest radiograph\ndeep learning", ha="center", fontsize=8.5,
        fontweight="bold")
ax.text(5.0, 2.85, "MAE 7.91 kg", ha="center", fontsize=9, color=OI["green"], fontweight="bold")
ax.text(5.0, 2.35, "9.3-17.1 kg better than every\nformula or mean-based estimate",
        ha="center", fontsize=7)
ax.text(5.0, 1.75, "available for every patient", ha="center", fontsize=7, style="italic")

# 右：真实记录仍最优
ax.add_patch(FancyBboxPatch((7.0, 1.5), 2.85, 2.4, boxstyle="round,pad=0.08",
                            linewidth=1.1, edgecolor=OI["blue"], facecolor="white"))
ax.text(8.42, 3.5, "Charted weight\n(any real measurement)", ha="center", fontsize=8.5,
        fontweight="bold")
ax.text(8.42, 2.9, "MAE 4.58 kg", ha="center", fontsize=9, color=OI["blue"], fontweight="bold")
ax.text(8.42, 2.35, "remains the best estimate,\neven when not contemporaneous",
        ha="center", fontsize=7)

ax.add_patch(FancyArrowPatch((3.35, 2.7), (3.55, 2.7), arrowstyle="-|>", mutation_scale=11,
                             linewidth=1.0, color=OI["grey"]))
ax.add_patch(FancyArrowPatch((6.5, 2.7), (6.95, 2.7), arrowstyle="-|>", mutation_scale=11,
                             linewidth=1.0, color=OI["grey"]))

ax.text(6.15, 0.30, "Directional error: underweight patients are overestimated, "
                    "obese patients underestimated", ha="center", fontsize=6.8, style="italic")

for ext in ("png", "svg"):
    fig.savefig(FIG / f"graphical_abstract.{ext}", facecolor="white")
plt.close(fig)
print("graphical abstract written")
