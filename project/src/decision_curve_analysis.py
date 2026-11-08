"""决策曲线与策略比较：把影像模型作为"是否复称"的分诊工具，并比较各体重来源的剂量误差。

目标事件：护理录入的入院体重不可靠（|入院体重 − 实测| ≥5 kg；次要 ≥10 kg）。
分诊策略：固定规则 |影像 − 入院| ≥5 kg（或 ≥10 kg）即复称；对照为全部复称 / 不复称；
已拟合标记模型与单变量校准曲线复用 flagging_model_v2.json 的净收益结果。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, main_pred_file, save_json, setup_logger

SEED = 20261028
PT = np.round(np.arange(0.05, 0.81, 0.05), 2)
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verma": "#D55E00",
      "purple": "#CC79A7", "grey": "#999999", "black": "#000000"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600})
MM = 1 / 25.4


def net_benefit(y: np.ndarray, flagged: np.ndarray, t: float) -> float:
    n = len(y)
    tp = float(((y == 1) & flagged).sum())
    fp = float(((y == 0) & flagged).sum())
    return tp / n - fp / n * t / (1 - t)


def boot_nb(y, flagged, subjects, t, n_boot=2000, seed=SEED):
    uniq, inv = np.unique(subjects, return_inverse=True)
    tp = np.bincount(inv, weights=((y == 1) & flagged).astype(float))
    fp = np.bincount(inv, weights=((y == 0) & flagged).astype(float))
    nn = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    tp_s = tp[idx].sum(1); fp_s = fp[idx].sum(1); n_s = nn[idx].sum(1)
    return tp_s / n_s - fp_s / n_s * t / (1 - t)


def curve_value(mapping: dict, t: float) -> float:
    for key in (f"{t:.2f}", f"{t:g}", str(float(t))):
        if key in mapping:
            return float(mapping[key])
    raise KeyError(f"{t} not in {list(mapping)[:5]}")


def main() -> None:
    log = setup_logger("decision_curve")
    pred = pd.read_csv(main_pred_file())
    conv = pd.read_csv(PROCESSED_DIR / "conventional_weight_comparison_table.csv")
    need = ["stay_id", "admission_weight_kg", "Devine IBW", "height_final_cm"]
    df = pred.merge(conv[need], on="stay_id", how="left")
    y = df["weight_kg"].values.astype(float)
    dl = df["pred"].values.astype(float)
    adm = df["admission_weight_kg"].values.astype(float)
    subj = df["subject_id"].values
    ok = np.isfinite(adm) & np.isfinite(dl) & np.isfinite(y)
    df, y, dl, adm, subj = df[ok].reset_index(drop=True), y[ok], dl[ok], adm[ok], subj[ok]
    gap = np.abs(dl - adm)
    unreliable5 = (np.abs(adm - y) >= 5).astype(int)
    unreliable10 = (np.abs(adm - y) >= 10).astype(int)

    # 1) 固定规则的净收益
    rules = {"gap_ge5": gap >= 5, "gap_ge10": gap >= 10}
    nb = {k: {float(t): net_benefit(unreliable5, f, t) for t in PT} for k, f in rules.items()}
    ci = {}
    for k, f in rules.items():
        ci[k] = {}
        for t in (0.1, 0.2, 0.3):
            b = boot_nb(unreliable5, f, subj, t, seed=SEED + int(t * 100))
            ci[k][f"{t:.1f}"] = {"nb": float(net_benefit(unreliable5, f, t)),
                                 "ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]}
    prev5 = float(unreliable5.mean()); prev10 = float(unreliable10.mean())
    nb_all = {float(t): prev5 - (1 - prev5) * t / (1 - t) for t in PT}

    # 2) 复用既有校准净收益（标记模型 / 单变量）
    flag = json.loads((TABLE_DIR / "flagging_model_v2.json").read_text(encoding="utf-8"))
    nb_existing = flag["net_benefit"]

    # 3) 剂量误差策略比较（ABW 与 AdjBW）
    ibw = df["Devine IBW"].values.astype(float)
    strategies = {
        "admission record": adm,
        "imaging model": dl,
        "rule: imaging if gap>=5 kg": np.where(gap >= 5, dl, adm),
        "rule: imaging if gap>=10 kg": np.where(gap >= 10, dl, adm),
        "rule: imaging if gap>=5 and low uncertainty":
            np.where((gap >= 5) & (df["pred_std"].values <= np.median(df["pred_std"].values)), dl, adm),
        "Devine IBW": ibw,
    }
    dose_rows = []
    for name, est in strategies.items():
        m = np.isfinite(est) & np.isfinite(y)
        rel = np.abs(est[m] - y[m]) / y[m]
        row = {"strategy": name, "n": int(m.sum()),
               "pct_gt10": float(100 * (rel > 0.10).mean()),
               "pct_gt20": float(100 * (rel > 0.20).mean()),
               "median_abs_pct": float(100 * np.median(rel))}
        adj_ok = m & np.isfinite(ibw)
        if adj_ok.sum() > 50:
            adj_true = ibw[adj_ok] + 0.4 * (y[adj_ok] - ibw[adj_ok])
            adj_est = ibw[adj_ok] + 0.4 * (est[adj_ok] - ibw[adj_ok])
            rel_adj = np.abs(adj_est - adj_true) / np.maximum(adj_true, 1e-6)
            row["adjbw_pct_gt10"] = float(100 * (rel_adj > 0.10).mean())
            row["adjbw_n"] = int(adj_ok.sum())
        dose_rows.append(row)

    # 4) 图：净收益 + 剂量误差
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 70 * MM), layout="constrained",
                             width_ratios=[1.05, 1.0])
    ax = axes[0]
    ax.plot(PT, [nav for nav in nb_all.values()], color=OI["grey"], lw=1.1, label="re-weigh all")
    ax.plot(PT, [0] * len(PT), color=OI["black"], lw=0.8, ls=":", label="re-weigh none")
    ax.plot(PT, [curve_value(nb_existing["single_var"], t) for t in PT],
            color=OI["blue"], lw=1.2, label="|imaging − record| (calibrated)")
    ax.plot(PT, [curve_value(nb_existing["model"], t) for t in PT],
            color=OI["verma"], lw=1.2, label="fitted reliability model")
    ax.plot(PT, [nb["gap_ge5"][float(t)] for t in PT], color=OI["green"], lw=1.0, ls="--",
            label="fixed rule: gap ≥5 kg")
    ax.axhline(0, color=OI["black"], lw=0.6)
    ax.set_xlabel("Threshold probability for re-weighing")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve: triage for re-weighing", loc="left")
    ax.legend(frameon=False, fontsize=6.4, loc="lower left")
    ax = axes[1]
    short = {"admission record": "Admission record", "imaging model": "Imaging model",
             "rule: imaging if gap>=5 kg": "Rule: gap ≥5 kg",
             "rule: imaging if gap>=10 kg": "Rule: gap ≥10 kg",
             "rule: imaging if gap>=5 and low uncertainty": "Rule: gap ≥5 kg, low uncertainty",
             "Devine IBW": "Devine IBW"}
    names = [short.get(r["strategy"], r["strategy"]) for r in dose_rows]
    ypos = np.arange(len(names))
    ax.barh(ypos - 0.18, [r["pct_gt10"] for r in dose_rows], height=0.34,
            color=OI["blue"], label=">10% dose error")
    ax.barh(ypos + 0.18, [r["pct_gt20"] for r in dose_rows], height=0.34,
            color=OI["orange"], label=">20% dose error")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("Patients (%)")
    ax.set_title("Dose error by weight source (ABW)", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.6)
    for name in ("figS3_decision_curve",):
        fig.savefig(FIGURE_DIR / f"{name}.png", facecolor="white")
        fig.savefig(FIGURE_DIR / f"{name}.svg", facecolor="white")
        ioi = Path("Insights_into_Imaging/figures")
        if ioi.exists():
            fig.savefig(ioi / f"{name}.png", facecolor="white")
            fig.savefig(ioi / f"{name}.svg", facecolor="white")
    plt.close(fig)

    out = {"n": int(len(y)), "prevalence_unreliable_5kg": prev5,
           "prevalence_unreliable_10kg": prev10,
           "net_benefit_fixed_rules": nb, "net_benefit_fixed_rule_ci": ci,
           "net_benefit_treat_all": nb_all,
           "reused_calibrated_curves": {"source": "flagging_model_v2.json",
                                        "model": nb_existing["model"],
                                        "single_var": nb_existing["single_var"]},
           "dose_strategies": dose_rows,
           "note": "net benefit for the fitted model and the calibrated single variable are "
                   "reused from flagging_model_v2.json; fixed-rule and strategy comparisons "
                   "are computed here"}
    save_json(TABLE_DIR / "decision_curve_analysis.json", out)
    pd.DataFrame(dose_rows).to_csv(TABLE_DIR / "dose_strategy_comparison.csv", index=False)
    for r in dose_rows:
        log.info("%-52s n=%4d >10%%=%5.1f >20%%=%5.1f median=%.1f%%",
                 r["strategy"], r["n"], r["pct_gt10"], r["pct_gt20"], r["median_abs_pct"])
    for k in rules:
        log.info("%s: %s", k, json.dumps(ci[k], ensure_ascii=False))


if __name__ == "__main__":
    main()
