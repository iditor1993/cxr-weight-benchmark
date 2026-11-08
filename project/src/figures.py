"""结果图表：流程图、体重分布、基线对比、Bland–Altman、亚组命中率。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURE_DIR, MAIN_SEED_TAG, MAIN_TAG, PROCESSED_DIR, TABLE_DIR, setup_logger

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def fig_flow() -> None:
    flow = json.loads((PROCESSED_DIR / "cohort_flow.json").read_text(encoding="utf-8"))
    steps = [
        ("ICU 住院总数", flow["n_icustays_total"]),
        ("有床旁便携 AP 片", flow["n_stays_with_portable_ap"]),
        ("±24h 内有每日体重（队列 A）", flow["n_stays_cohort_a"]),
        ("索引影像分析集", flow["n_index_images_cohort_a"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    y = 0.9
    for i, (label, value) in enumerate(steps):
        ax.add_patch(plt.Rectangle((0.15, y - 0.1), 0.7, 0.14, fill=True,
                                   facecolor="#E8F1FA", edgecolor="#2C6FA6"))
        ax.text(0.5, y - 0.03, f"{label}: {value:,}", ha="center", va="center", fontsize=12)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.13), xytext=(0.5, y - 0.10),
                        arrowprops=dict(arrowstyle="->", color="#2C6FA6"))
        y -= 0.22
    ax.set_title("CR_Weight 队列纳排流程（队列 A）", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig1_cohort_flow.png", dpi=200)
    plt.close(fig)


def fig_weight_distribution() -> None:
    df = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["weight_kg"], bins=60, color="#2C6FA6", alpha=0.85)
    ax.axvline(df["weight_kg"].median(), color="crimson", ls="--",
               label=f"中位 {df['weight_kg'].median():.1f} kg")
    ax.set_xlabel("体重（kg）")
    ax.set_ylabel("住院次数")
    ax.set_title("队列 A 索引影像体重分布（n=%d）" % len(df))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig2_weight_distribution.png", dpi=200)
    plt.close(fig)


def fig_baseline_comparison() -> None:
    path = TABLE_DIR / "baselines.json"
    if not path.exists():
        return
    base = json.loads(path.read_text(encoding="utf-8"))
    names = [k for k in base if "mae" in base[k]]
    mae = [base[k]["mae"] for k in names]
    hit = [base[k]["hit_within_10pct"] * 100 for k in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(names, mae, color="#7FA8C9")
    axes[0].set_ylabel("MAE (kg)")
    axes[0].set_title("基线模型 MAE")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, hit, color="#9BC49B")
    axes[1].axhline(80, color="crimson", ls="--", label="S1 目标 80%")
    axes[1].set_ylabel("±10% 命中率 (%)")
    axes[1].set_title("基线模型命中率")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig3_baselines.png", dpi=200)
    plt.close(fig)


def fig_bland_altman(tag: str) -> None:
    path = PROCESSED_DIR / f"test_predictions_{tag}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    diff = df["pred"] - df["weight_kg"]
    mean = (df["pred"] + df["weight_kg"]) / 2
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean, diff, s=6, alpha=0.4, color="#2C6FA6")
    ax.axhline(diff.mean(), color="black", ls="-", label=f"偏差 {diff.mean():.2f} kg")
    ax.axhline(diff.mean() + 1.96 * diff.std(), color="crimson", ls="--",
               label=f"±1.96 SD ({diff.mean() + 1.96 * diff.std():.1f})")
    ax.axhline(diff.mean() - 1.96 * diff.std(), color="crimson", ls="--")
    ax.set_xlabel("(预测 + 实测)/2 (kg)")
    ax.set_ylabel("预测 − 实测 (kg)")
    ax.set_title(f"Bland–Altman（{tag}）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"fig4_bland_altman_{tag}.png", dpi=200)
    plt.close(fig)


def fig_subgroup(tag: str) -> None:
    path = TABLE_DIR / f"evaluation_{tag}.json"
    if not path.exists():
        return
    ev = json.loads(path.read_text(encoding="utf-8"))
    rows = ev["subgroups"].get("weight_stratum", [])
    if not rows:
        return
    labels = [r["group"] for r in rows]
    mae = [r["mae"] for r in rows]
    hit = [r["hit_within_10pct"] * 100 for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, mae, width=0.4, color="#7FA8C9", label="MAE (kg)")
    ax1.set_ylabel("MAE (kg)")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, hit, width=0.4, color="#9BC49B", label="±10% 命中率 (%)")
    ax2.axhline(65, color="crimson", ls=":", label="S3 亚组门槛 65%")
    ax2.set_ylabel("命中率 (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("体重分层 (kg)")
    ax1.set_title(f"体重分层性能（{tag}）")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"fig5_subgroup_weight_{tag}.png", dpi=200)
    plt.close(fig)


def fig_model_comparison() -> None:
    tags = [("resnet50_224_s1", "ResNet-50 224"),
            (f"{MAIN_SEED_TAG}_s1", "主模型 ResNet-50 308 s1"),
            (f"{MAIN_SEED_TAG}_s2", "主模型 ResNet-50 308 s2"),
            (f"{MAIN_SEED_TAG}_s3", "主模型 ResNet-50 308 s3"),
            ("swin_t_308_s1", "Swin-T 308 s1"),
            ("swin_t_308_s2", "Swin-T 308 s2"), ("swin_t_308_s3", "Swin-T 308 s3"),
            ("swin_t_224_s1", "Swin-T 224"), ("swin_t_308_temporal_s1", "Swin-T 时间外"),
            ("rad_dino_224_s1", "RadDINO 224")]
    labels, maes, hits = [], [], []
    for tag, label in tags:
        p = TABLE_DIR / f"metrics_{tag}.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            labels.append(label)
            maes.append(m["test"]["mae"])
            hits.append(m["test"]["hit_within_10pct"] * 100)
    p = TABLE_DIR / f"evaluation_{MAIN_TAG}.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))["overall"]
        labels.append("主模型集成")
        maes.append(m["mae"])
        hits.append(m["hit_within_10pct"] * 100)
    if not labels:
        return
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, maes, width=0.4, color="#2C6FA6", label="MAE (kg)")
    ax1.axhline(5, color="crimson", ls="--", label="S2 目标 MAE 5 kg")
    ax1.set_ylabel("MAE (kg)")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, hits, width=0.4, color="#9BC49B", label="±10% 命中率 (%)")
    ax2.axhline(80, color="orange", ls=":", label="S1 目标 80%")
    ax2.set_ylabel("±10% 命中率 (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_title("模型对比：MAE 与 ±10% 命中率")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig6_model_comparison.png", dpi=200)
    plt.close(fig)


def fig_uncertainty_comparison() -> None:
    ens = TABLE_DIR / f"uncertainty_{MAIN_TAG}.json"
    het = TABLE_DIR / "uncertainty_swin_hetero.json"
    if not ens.exists():
        return
    e = json.loads(ens.read_text(encoding="utf-8"))
    rows = [("3 种子集成", e["coverage_90_nominal"] * 100, e["mean_interval_width_kg"])]
    if het.exists():
        h = json.loads(het.read_text(encoding="utf-8"))
        rows.append(("异方差回归", h["coverage_90_nominal"] * 100, h["mean_interval_width_kg"]))
    labels = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, [r[1] for r in rows], color="#2C6FA6")
    axes[0].axhline(85, color="crimson", ls="--", label="可接受下限 85%")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("90% 区间实际覆盖率 (%)")
    axes[0].set_title("校准（覆盖率）")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, [r[2] for r in rows], color="#C98A7F")
    axes[1].set_ylabel("平均区间宽度 (kg)")
    axes[1].set_title("锐度（区间宽度）")
    for ax in axes:
        ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig7_uncertainty_comparison.png", dpi=200)
    plt.close(fig)


def fig_dose_distribution(tag: str = "swin_ensemble") -> None:
    path = PROCESSED_DIR / f"test_predictions_{tag}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    rel_abw = np.abs(df["pred"] - df["weight_kg"]) / df["weight_kg"] * 100
    sub = df.dropna(subset=["height_cm"]).copy()
    if len(sub) > 20:
        male = (sub["gender"] == "M").astype(int).values
        inches = sub["height_cm"].values / 2.54
        ibw = np.where(male == 1, 50.0 + 2.3 * (inches - 60.0), 45.5 + 2.3 * (inches - 60.0))
        adj_true = ibw + 0.4 * (sub["weight_kg"].values - ibw)
        adj_pred = ibw + 0.4 * (sub["pred"].values - ibw)
        rel_adj = np.abs(adj_pred - adj_true) / np.maximum(adj_true, 1e-6) * 100
    else:
        rel_adj = None
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 40, 41)
    ax.hist(rel_abw, bins=bins, alpha=0.7, color="#2C6FA6", label="ABW 线性药物（n=%d）" % len(rel_abw))
    if rel_adj is not None:
        ax.hist(rel_adj, bins=bins, alpha=0.7, color="#9BC49B", label="AdjBW 药物（n=%d）" % len(rel_adj))
    ax.axvline(10, color="crimson", ls="--", label="MCID 10%")
    ax.set_xlabel("模拟剂量相对偏差 (%)")
    ax.set_ylabel("患者数")
    ax.set_title("假设性剂量偏差分布")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig8_dose_distribution.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    log = setup_logger("figures")
    fig_flow()
    fig_weight_distribution()
    fig_baseline_comparison()
    if args.tag:
        fig_bland_altman(args.tag)
        fig_subgroup(args.tag)
    fig_model_comparison()
    fig_uncertainty_comparison()
    fig_dose_distribution(args.tag or MAIN_TAG)
    log.info("figures written to %s", FIGURE_DIR)


if __name__ == "__main__":
    main()
