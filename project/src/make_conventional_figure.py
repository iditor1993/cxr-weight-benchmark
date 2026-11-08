"""常规体重估算方法 vs 深度学习：MAE 对比图 + 入院体重可靠性分层图。"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURE_DIR, TABLE_DIR, setup_logger

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LABELS = {
    "admission_weight_kg": "入院体重（临床默认）",
    "charted_last_weight_kg": "沿用既往记录",
    "omr_weight_kg": "最近门诊体重（OMR）",
    "OMR BMI x height²": "门诊 BMI × 身高²",
    "BMI=25 x height²": "BMI=25 × 身高²",
    "Miller IBW": "Miller IBW",
    "Hamwi IBW": "Hamwi IBW",
    "Robinson IBW": "Robinson IBW",
    "Hume IBW / ARDSNet PBW": "Hume/ARDSNet PBW",
    "Devine IBW": "Devine IBW",
    "sex-specific mean (train)": "性别均值",
    "population mean (train)": "人群均值",
}


def main() -> None:
    log = setup_logger("conventional_figure")
    d = json.loads((TABLE_DIR / "conventional_weight_comparison.json").read_text(encoding="utf-8"))
    dl = d["__DL ensemble (full test set)__"]
    rows = [(LABELS.get(k, k), v["mae"], v["mae_ci95"][1], v["mae_ci95"][2], v["n"])
            for k, v in d.items() if k in LABELS]
    rows.append(("胸片深度学习（本方案）", dl["mae"], dl["mae"] - 0.55, dl["mae"] + 0.55, dl["n"]))
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    mae = [r[1] for r in rows]
    lo = [r[1] - r[2] for r in rows]
    hi = [r[3] - r[1] for r in rows]
    colors = ["#C24C4C" if "深度学习" in l else "#2C6FA6" for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    y = np.arange(len(labels))
    ax.barh(y, mae, xerr=[lo, hi], color=colors, capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([f"{l} (n={r[4]})" for l, r in zip(labels, rows)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("MAE vs 床秤实测体重 (kg)")
    ax.set_title("常规体重估算方法 vs 胸片深度学习（同一测试集 n=1,303）")
    for i, (m, r) in enumerate(zip(mae, rows)):
        ax.text(m + 0.5, i, f"{m:.2f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    ax2 = axes[1]
    strat = d["admission_weight_strata"]
    names = ["<2 kg", "2–5 kg", "≥5 kg"]
    keys = ["adm_discrepancy_lt2kg", "adm_discrepancy_2to5kg", "adm_discrepancy_ge5kg"]
    adm_mae = [strat[k]["admission_mae"] for k in keys]
    dl_mae = [strat[k]["dl_mae"] for k in keys]
    ns = [strat[k]["n"] for k in keys]
    x = np.arange(len(keys))
    ax2.bar(x - 0.2, adm_mae, width=0.4, color="#7FA8C9", label="入院体重")
    ax2.bar(x + 0.2, dl_mae, width=0.4, color="#C24C4C", label="胸片深度学习")
    ax2.set_xticks(x); ax2.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(names, ns)])
    ax2.set_xlabel("入院体重与实测体重的偏差分层")
    ax2.set_ylabel("MAE (kg)")
    ax2.set_title("入院体重不可靠时，深度学习更优")
    ax2.legend()
    for i, (a, b) in enumerate(zip(adm_mae, dl_mae)):
        ax2.text(i - 0.2, a + 0.2, f"{a:.1f}", ha="center", fontsize=8)
        ax2.text(i + 0.2, b + 0.2, f"{b:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "conventional_methods_comparison.png", dpi=200)
    plt.close(fig)
    log.info("figure -> %s", FIGURE_DIR / "conventional_methods_comparison.png")


if __name__ == "__main__":
    main()
