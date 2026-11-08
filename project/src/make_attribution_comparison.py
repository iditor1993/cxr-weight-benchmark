"""汇总 RISE-96 与 RISE-4000 的删除/插入 AUC，生成对比图与表格。"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURE_DIR, TABLE_DIR, setup_logger

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    log = setup_logger("attribution_comparison")
    p2 = json.loads((TABLE_DIR / "attribution_pilot2_n60.json").read_text(encoding="utf-8"))
    p3 = json.loads((TABLE_DIR / "attribution_pilot3_rise.json").read_text(encoding="utf-8"))
    labels = ["CAM", "梯度显著图", "RISE-96", "RISE-4000", "随机"]
    keys = ["cam", "grad", "rise", "rise", "random"]
    del_auc = [p2["deletion"][k]["auc_mean"] if not (i == 3) else p3["deletion"]["rise"]["auc_mean"]
               for i, k in enumerate(keys)]
    ins_auc = [p2["insertion"][k]["auc_mean"] if not (i == 3) else p3["insertion"]["rise"]["auc_mean"]
               for i, k in enumerate(keys)]
    del_sd = [p2["deletion"][k]["auc_std"] for k in keys[:3]] + [p3["deletion"]["rise"]["auc_std"], p2["deletion"]["random"]["auc_std"]]
    ins_sd = [p2["insertion"][k]["auc_std"] for k in keys[:3]] + [p3["insertion"]["rise"]["auc_std"], p2["insertion"]["random"]["auc_std"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(labels))
    axes[0].bar(x, del_auc, yerr=del_sd, capsize=4, color="#2C6FA6")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_ylabel("删除 AUC（越大越好）")
    axes[0].set_title(f"删除曲线对比（n={p2['n']}）")
    axes[0].axhline(p2["deletion"]["random"]["auc_mean"], color="crimson", ls="--", label="随机基准")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, ins_auc, yerr=ins_sd, capsize=4, color="#9BC49B")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_ylabel("插入 AUC（越小越好）")
    axes[1].set_title("插入曲线对比")
    axes[1].axhline(p2["insertion"]["random"]["auc_mean"], color="crimson", ls="--", label="随机基准")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "attribution_method_comparison.png", dpi=200)
    plt.close(fig)

    rows = []
    for lab, d, i, ds, isd in zip(labels, del_auc, ins_auc, del_sd, ins_sd):
        rows.append({"method": lab, "deletion_auc": round(d, 2), "deletion_sd": round(ds, 2),
                     "insertion_auc": round(i, 2), "insertion_sd": round(isd, 2)})
    log.info("comparison table:")
    for r in rows:
        log.info("%s | deletion %.2f±%.2f | insertion %.2f±%.2f",
                 r["method"], r["deletion_auc"], r["deletion_sd"],
                 r["insertion_auc"], r["insertion_sd"])


if __name__ == "__main__":
    main()
