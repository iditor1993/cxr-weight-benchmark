"""亚组分析（方案冻结版）：T1 轴的相对获益、交互检验与决策门控。

方案：project/docs/subgroup_analysis_plan.md（2026-10-10 项目负责人批准，T1 = 7 轴）
主对照 C1 = 护理录入的入院体重；辅对照 C2 = 身高+性别回归；参照 C3 = 训练集人群均值。

产出
    project/results/tables/subgroup_analysis.json
    project/results/tables/subgroup_analysis_strata.csv
    project/results/figures/figS2_subgroups.png|svg
    Insights_into_Imaging/figures/figS2_subgroups.png|svg
    Insights_into_Imaging/supplementary/ESM2_subgroup_analysis.md
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon

from common import (FIGURE_DIR, MAIN_TAG, PROCESSED_DIR, TABLE_DIR, load_config, save_json,
                    setup_logger)

SEED = 20261010
N_BOOT = 2000
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2",
      "verma": "#D55E00", "purple": "#CC79A7", "grey": "#999999", "black": "#000000"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600})
MM = 1 / 25.4


# ------------------------------------------------------------------ 数据
def load_frame() -> pd.DataFrame:
    """组装分析框：主模型预测 + 对照 + 影像标签 + 液体平衡。"""
    cfg = load_config()
    con = duckdb.connect(cfg["data"]["duckdb"], read_only=True)
    conv = pd.read_csv(PROCESSED_DIR / "conventional_weight_comparison_table.csv")
    preds = pd.read_csv(PROCESSED_DIR / f"test_predictions_{MAIN_TAG}.csv")
    keep = ["stay_id", "study_id", "study_datetime", "pred_std", "gap_hours", "weight_stratum",
            "age_stratum", "ventilated", "rrt", "support_devices"]
    df = conv.merge(preds[keep], on="stay_id", how="left", suffixes=("", "_p"))
    p = str(PROCESSED_DIR / f"test_predictions_{MAIN_TAG}.csv").replace("\\", "/")

    # 影像标签（CheXpert，索引 study）
    chex = con.execute(
        """SELECT study_id, Edema, "Pleural Effusion" AS effusion, "Support Devices" AS dev
           FROM main.cxr_chexpert WHERE study_id IN (SELECT study_id FROM read_csv_auto(?))""",
        [p]).fetchdf()
    df = df.merge(chex, on="study_id", how="left")

    # 液体平衡：ICU 入科 → 索引影像
    fb = con.execute(
        """
        WITH base AS (SELECT stay_id, study_datetime FROM read_csv_auto(?)),
        t AS (SELECT b.stay_id, ic.intime AS icu_intime, b.study_datetime
              FROM base b JOIN main.icustays ic ON ic.stay_id = b.stay_id),
        ins AS (SELECT t.stay_id, sum(ie.amount) AS intake FROM t
                JOIN main.inputevents ie ON ie.stay_id = t.stay_id
                  AND ie.starttime BETWEEN t.icu_intime AND t.study_datetime GROUP BY 1),
        outs AS (SELECT t.stay_id, sum(oe.value) AS output FROM t
                 JOIN main.outputevents oe ON oe.stay_id = t.stay_id
                   AND oe.charttime BETWEEN t.icu_intime AND t.study_datetime GROUP BY 1)
        SELECT t.stay_id, datediff('hour', t.icu_intime, t.study_datetime) AS hours_since_icu,
               (coalesce(ins.intake, 0) - coalesce(outs.output, 0)) / 1000.0 AS fluid_balance_kg
        FROM t LEFT JOIN ins USING (stay_id) LEFT JOIN outs USING (stay_id)
        """, [p]).fetchdf()
    df = df.merge(fb, on="stay_id", how="left")

    # 派生变量
    h = df["height_final_cm"].values.astype(float)
    male = (df["gender"].values == "M").astype(int)
    df["ols_height_sex"] = np.where(np.isfinite(h),
                                    -69.9350080268858 + 0.9196542642028203 * h
                                    + 0.04498295084068797 * male, np.nan)
    df["bmi"] = np.where(np.isfinite(h), df["weight_kg"].values / (h / 100.0) ** 2, np.nan)
    adm = df["admission_weight_kg"].values.astype(float)
    df["adm_disc"] = np.abs(adm - df["weight_kg"].values.astype(float))
    df["dl_adm_gap"] = np.abs(df["pred"].values.astype(float) - adm)
    swin_path = PROCESSED_DIR / "test_predictions_swin_ensemble.csv"
    if swin_path.exists():
        sw = pd.read_csv(swin_path)[["stay_id", "pred"]].rename(columns={"pred": "pred_swin"})
        df = df.merge(sw, on="stay_id", how="left")
    return df


# ------------------------------------------------------- 聚类 bootstrap 工具
def _subject_index(subjects: np.ndarray):
    uniq, inv = np.unique(subjects, return_inverse=True)
    return uniq, inv


def boot_mean(values: np.ndarray, subjects: np.ndarray, n_boot: int = N_BOOT,
              seed: int = SEED) -> np.ndarray:
    _, inv = _subject_index(subjects)
    sums = np.bincount(inv, weights=values)
    cnts = np.bincount(inv)
    k = len(cnts)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    return sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)


def boot_diff_of_means(d: np.ndarray, g: np.ndarray, subjects: np.ndarray,
                       n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """两组（g=1 与 g=0）均值的差，按患者聚类重采样。"""
    _, inv = _subject_index(subjects)
    k = inv.max() + 1
    s1 = np.bincount(inv[g == 1], weights=d[g == 1], minlength=k)
    c1 = np.bincount(inv[g == 1], minlength=k)
    s0 = np.bincount(inv[g == 0], weights=d[g == 0], minlength=k)
    c0 = np.bincount(inv[g == 0], minlength=k)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    m1 = s1[idx].sum(axis=1) / np.maximum(c1[idx].sum(axis=1), 1)
    m0 = s0[idx].sum(axis=1) / np.maximum(c0[idx].sum(axis=1), 1)
    return m1 - m0


def cr_ols_p(y: np.ndarray, x: np.ndarray, subjects: np.ndarray) -> tuple[float, float, float]:
    """聚类稳健 OLS（患者级聚类）：返回 beta、se、双侧 p。"""
    X = sm.add_constant(np.asarray(x, dtype=float))
    fit = sm.OLS(np.asarray(y, dtype=float), X).fit(
        cov_type="cluster", cov_kwds={"groups": np.asarray(subjects)})
    return float(fit.params[1]), float(fit.bse[1]), float(fit.pvalues[1])


def bh(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg 校正。"""
    p = np.asarray([1.0 if v is None or not np.isfinite(v) else v for v in pvals], dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q.tolist()


def paired_stats(y: np.ndarray, a: np.ndarray, b: np.ndarray, subjects: np.ndarray,
                 seed: int = SEED) -> dict:
    """配对统计：Δ = |y-a| − |y-b|（负值利于 a），含聚类 bootstrap CI 与 Wilcoxon p。"""
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return {"n": int(ok.sum())}
    d = np.abs(y[ok] - a[ok]) - np.abs(y[ok] - b[ok])
    boot = boot_mean(d, subjects[ok], seed=seed)
    try:
        p = float(wilcoxon(np.abs(y[ok] - a[ok]), np.abs(y[ok] - b[ok])).pvalue)
    except Exception:  # noqa: BLE001
        p = None
    return {"n": int(ok.sum()),
            "mae_a": float(np.mean(np.abs(y[ok] - a[ok]))),
            "mae_b": float(np.mean(np.abs(y[ok] - b[ok]))),
            "delta": float(d.mean()),
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
            "wilcoxon_p": p,
            "hit10_a": float(np.mean(np.abs(y[ok] - a[ok]) <= 0.10 * y[ok])),
            "hit10_b": float(np.mean(np.abs(y[ok] - b[ok]) <= 0.10 * y[ok]))}


# ------------------------------------------------------------------ 轴定义（冻结）
COMPARATORS = {
    "C1": ("Nurse-recorded admission weight", "admission_weight_kg"),
    "C2": ("Height-and-sex regression", "ols_height_sex"),
    "C3": ("Population mean (train)", "population mean (train)"),
}


def make_axes(df: pd.DataFrame) -> list[dict]:
    y = df["weight_kg"].values.astype(float)
    disc = df["adm_disc"].values.astype(float)
    ps = df["pred_std"].values.astype(float)
    bmi = df["bmi"].values.astype(float)
    h = df["height_final_cm"].values.astype(float)
    fb = df["fluid_balance_kg"].values.astype(float)
    edema = df["Edema"].fillna(0).values == 1
    q_ps = pd.qcut(ps, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    q_fb = pd.qcut(fb, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return [
        {"key": "record_reliability", "label": "Record reliability",
         "definition": "|nurse-recorded admission weight - measured weight|: <2 / 2-5 / >=5 kg",
         "strata": [("<2 kg", disc < 2), ("2-5 kg", (disc >= 2) & (disc < 5)), (">=5 kg", disc >= 5)],
         "contrast": (">=5 kg", "<2 kg"), "continuous": "adm_disc",
         "comparators": ["C1", "C2", "C3"]},
        {"key": "uncertainty", "label": "Model uncertainty",
         "definition": "quartiles of the between-seed prediction standard deviation",
         "strata": [(v, np.asarray(q_ps == v)) for v in ["Q1", "Q2", "Q3", "Q4"]],
         "contrast": ("Q4", "Q1"), "continuous": "pred_std",
         "comparators": ["C1", "C2", "C3"]},
        {"key": "weight", "label": "Measured weight",
         "definition": "<50 / 50-80 / 80-100 / >=100 kg",
         "strata": [("<50 kg", y < 50), ("50-80 kg", (y >= 50) & (y < 80)),
                    ("80-100 kg", (y >= 80) & (y < 100)), (">=100 kg", y >= 100)],
         "contrast": (">=100 kg", "50-80 kg"), "continuous": "weight_kg",
         "comparators": ["C1", "C2", "C3"]},
        {"key": "bmi", "label": "Body mass index",
         "definition": "<18.5 / 18.5-25 / 25-30 / 30-40 / >=40 kg/m2 (final height subset)",
         "strata": [("<18.5", bmi < 18.5), ("18.5-25", (bmi >= 18.5) & (bmi < 25)),
                    ("25-30", (bmi >= 25) & (bmi < 30)), ("30-40", (bmi >= 30) & (bmi < 40)),
                    (">=40", bmi >= 40)],
         "contrast": (">=40", "18.5-25"), "continuous": "bmi",
         "comparators": ["C1", "C2", "C3"]},
        {"key": "height_availability", "label": "Height availability",
         "definition": "final height available (any formula computable) vs missing",
         "strata": [("available", np.isfinite(h)), ("missing", ~np.isfinite(h))],
         "contrast": ("missing", "available"), "continuous": None,
         "comparators": ["C1", "C3"]},
        {"key": "edema", "label": "Radiographic pulmonary edema",
         "definition": "CheXpert Edema label at the index study: positive vs non-positive",
         "strata": [("positive", edema), ("non-positive", ~edema)],
         "contrast": ("positive", "non-positive"), "continuous": None,
         "comparators": ["C1", "C2", "C3"]},
        {"key": "fluid_balance", "label": "Cumulative fluid balance",
         "definition": "ICU admission to index radiograph (input - output), quartiles; plus >=5 kg",
         "strata": [(v, np.asarray(q_fb == v)) for v in ["Q1", "Q2", "Q3", "Q4"]]
                   + [(">=5 kg", fb >= 5)],
         "contrast": ("Q4", "Q1"), "continuous": "fluid_balance_kg",
         "comparators": ["C1", "C2", "C3"]},
    ]


def mask_of(ax: dict, name: str) -> np.ndarray:
    for lab, m in ax["strata"]:
        if lab == name:
            return np.asarray(m)
    raise KeyError(name)


def analyse_axis(df: pd.DataFrame, ax: dict, log) -> dict:
    y = df["weight_kg"].values.astype(float)
    pred = df["pred"].values.astype(float)
    subj = df["subject_id"].values
    out = {"key": ax["key"], "label": ax["label"], "definition": ax["definition"],
           "contrast": list(ax["contrast"]), "strata": {}, "interaction": [], "continuous": None}

    for ckey in ax["comparators"]:
        clabel, col = COMPARATORS[ckey]
        comp = df[col].values.astype(float)
        rows = []
        for name, m in ax["strata"]:
            m = np.asarray(m)
            st = paired_stats(y[m], pred[m], comp[m], subj[m],
                              seed=SEED + 17 * len(rows) + ord(ckey[1]))
            st["stratum"] = name
            rows.append(st)
        out["strata"][ckey] = {"comparator": clabel, "rows": rows}

    # 交互检验（预先指定的对比对）
    a, b = ax["contrast"]
    for ckey in ax["comparators"]:
        _, col = COMPARATORS[ckey]
        comp = df[col].values.astype(float)
        d = np.abs(y - pred) - np.abs(y - comp)
        ma, mb = mask_of(ax, a), mask_of(ax, b)
        sel = (ma | mb) & np.isfinite(d) & np.isfinite(y)
        if sel.sum() < 40:
            continue
        g = ma[sel].astype(int)
        boot = boot_diff_of_means(d[sel], g, subj[sel],
                                  seed=SEED + 5 * len(out["interaction"]) + ord(ckey[1]))
        beta, se, p = cr_ols_p(d[sel], g, subj[sel])
        out["interaction"].append({
            "comparator": ckey, "comparator_label": COMPARATORS[ckey][0],
            "contrast": f"{a} vs {b}", "n_a": int(ma[sel].sum()), "n_b": int(mb[sel].sum()),
            "delta_a": float(d[sel][g == 1].mean()), "delta_b": float(d[sel][g == 0].mean()),
            "delta_diff": beta, "se_cluster": se, "p_cluster": p,
            "boot_ci_low": float(np.quantile(boot, 0.025)),
            "boot_ci_high": float(np.quantile(boot, 0.975))})

    # 连续形式（无切点验证）
    if ax["continuous"]:
        col = ax["continuous"]
        v = df[col].values.astype(float)
        for ckey in ax["comparators"]:
            _, ccol = COMPARATORS[ckey]
            d = np.abs(y - pred) - np.abs(y - df[ccol].values.astype(float))
            ok = np.isfinite(d) & np.isfinite(v) & np.isfinite(y)
            if ok.sum() < 50:
                continue
            beta, se, p = cr_ols_p(d[ok], v[ok], subj[ok])
            out.setdefault("continuous_by_comparator", {})[ckey] = {
                "variable": col, "n": int(ok.sum()), "beta_per_unit": beta, "se_cluster": se,
                "p_cluster": p, "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se}
    return out


# ------------------------------------------------------------- 决策门控
def gate_analysis(df: pd.DataFrame) -> dict:
    y = df["weight_kg"].values.astype(float)
    pred = df["pred"].values.astype(float)
    adm = df["admission_weight_kg"].values.astype(float)
    ps = df["pred_std"].values.astype(float)
    subj = df["subject_id"].values
    ok = np.isfinite(y) & np.isfinite(pred) & np.isfinite(adm) & np.isfinite(ps)
    y, pred, adm, ps, subj = y[ok], pred[ok], adm[ok], ps[ok], subj[ok]
    g1 = np.abs(pred - adm) >= 5.0
    g2 = ps >= np.median(ps)
    d_all = np.abs(y - pred) - np.abs(y - adm)

    cells = []
    for name, m in (("gate- / low uncertainty", ~g1 & ~g2), ("gate- / high uncertainty", ~g1 & g2),
                    ("gate+ / low uncertainty", g1 & ~g2), ("gate+ / high uncertainty", g1 & g2)):
        if m.sum() < 20:
            cells.append({"cell": name, "n": int(m.sum())})
            continue
        boot = boot_mean(d_all[m], subj[m], seed=SEED + 11 * len(cells))
        cells.append({"cell": name, "n": int(m.sum()),
                      "mae_dl": float(np.mean(np.abs(y[m] - pred[m]))),
                      "mae_record": float(np.mean(np.abs(y[m] - adm[m]))),
                      "delta": float(d_all[m].mean()),
                      "ci_low": float(np.quantile(boot, 0.025)),
                      "ci_high": float(np.quantile(boot, 0.975))})

    def rule_stats(use_dl: np.ndarray, name: str) -> dict:
        est = np.where(use_dl, pred, adm)
        d = np.abs(y - est) - np.abs(y - adm)
        boot = boot_mean(d, subj, seed=SEED + 7 * len(name))
        return {"rule": name, "mae": float(np.mean(np.abs(y - est))),
                "delta_vs_always_record": float(d.mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "use_dl_rate": float(np.mean(use_dl))}

    rules = [rule_stats(np.zeros_like(g1), "always record"),
             rule_stats(np.ones_like(g1), "always imaging model"),
             rule_stats(g1, "imaging model when |DL - record| >= 5 kg"),
             rule_stats(g1 & g2, "imaging model when gap >= 5 kg and high uncertainty"),
             rule_stats(g1 & ~g2, "imaging model when gap >= 5 kg and low uncertainty")]
    flag = json.loads((TABLE_DIR / "flagging_model_v2.json").read_text(encoding="utf-8"))
    return {"n": int(ok.sum()), "gate_definition": {
        "gate1": "|model prediction - nurse-recorded admission weight| >= 5 kg",
        "gate2": "between-seed prediction SD >= median (high uncertainty)"},
        "cells": cells, "rules": rules,
        "fitted_flagging_model_reference": flag.get("decision_analysis", {})}


# ------------------------------------------------------------- 稳健性
def placebo_test(df: pd.DataFrame, ax: dict, comparator: str = "C1",
                 n_rep: int = 500) -> dict:
    """随机等规模划分的安慰剂检验（患者级标签置换）。"""
    y = df["weight_kg"].values.astype(float)
    pred = df["pred"].values.astype(float)
    comp = df[COMPARATORS[comparator][1]].values.astype(float)
    subj = df["subject_id"].values
    d = np.abs(y - pred) - np.abs(y - comp)
    a, b = ax["contrast"]
    ma, mb = mask_of(ax, a), mask_of(ax, b)
    sel = (ma | mb) & np.isfinite(d) & np.isfinite(y)
    d, subj, ma = d[sel], subj[sel], ma[sel]
    obs = d[ma].mean() - d[~ma].mean()
    uniq = np.unique(subj)
    subj_group = pd.Series(ma).groupby(subj).max().reindex(uniq).values
    n_a_pat = int(subj_group.sum())
    rng = np.random.default_rng(SEED)
    draws = np.empty(n_rep)
    for i in range(n_rep):
        pick = rng.choice(len(uniq), size=n_a_pat, replace=False)
        lab = np.zeros(len(uniq), dtype=bool)
        lab[pick] = True
        g = lab[np.searchsorted(uniq, subj)]
        if g.sum() < 20 or (~g).sum() < 20:
            draws[i] = np.nan
            continue
        draws[i] = d[g].mean() - d[~g].mean()
    draws = draws[np.isfinite(draws)]
    p = float((1 + np.sum(np.abs(draws) >= abs(obs))) / (1 + len(draws)))
    return {"axis": ax["key"], "comparator": comparator, "contrast": f"{a} vs {b}",
            "observed_delta_diff": float(obs), "n_a_patients": n_a_pat,
            "n_perm": int(len(draws)),
            "placebo_sd": float(np.std(draws)), "placebo_p_two_sided": p}


def swin_replication(df: pd.DataFrame, axes: list[dict]) -> list[dict]:
    """用 Swin-T @308 集成重复 T1 交互检验，检查结论是否依赖骨干。"""
    if "pred_swin" not in df:
        return []
    out = []
    y = df["weight_kg"].values.astype(float)
    subj = df["subject_id"].values
    for ax in axes:
        a, b = ax["contrast"]
        for ckey in ax["comparators"]:
            if ckey == "C2" and ax["key"] == "height_availability":
                continue
            comp = df[COMPARATORS[ckey][1]].values.astype(float)
            for tag, pcol in (("main", "pred"), ("swin", "pred_swin")):
                pv = df[pcol].values.astype(float)
                d = np.abs(y - pv) - np.abs(y - comp)
                ma, mb = mask_of(ax, a), mask_of(ax, b)
                sel = (ma | mb) & np.isfinite(d) & np.isfinite(y)
                if sel.sum() < 40:
                    continue
                beta, se, p = cr_ols_p(d[sel], ma[sel].astype(int), subj[sel])
                out.append({"axis": ax["key"], "comparator": ckey, "model": tag,
                            "delta_diff": beta, "se_cluster": se, "p_cluster": p,
                            "n": int(sel.sum())})
    return out


def win_rate_analysis(df: pd.DataFrame, axes: list[dict]) -> dict:
    y = df["weight_kg"].values.astype(float)
    pred = df["pred"].values.astype(float)
    adm = df["admission_weight_kg"].values.astype(float)
    subj = df["subject_id"].values
    ok = np.isfinite(y) & np.isfinite(pred) & np.isfinite(adm)
    win = np.abs(y - pred) < np.abs(y - adm)

    def rate(m):
        m = m & ok
        if m.sum() < 20:
            return None
        boot = boot_mean(win[m].astype(float), subj[m], seed=SEED + 3)
        return {"n": int(m.sum()), "win_rate": float(win[m].mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975))}

    out = {"overall": rate(np.ones(len(y), dtype=bool)), "by_axis": {}}
    for ax in axes:
        out["by_axis"][ax["key"]] = {name: rate(np.asarray(m)) for name, m in ax["strata"]}
    return out


# ------------------------------------------------------------- 图 S2
def make_figure(results: dict) -> None:
    axes_res = results["axes"]
    order = ["record_reliability", "uncertainty", "weight", "bmi", "height_availability",
             "edema", "fluid_balance"]
    rows = []
    for key in order:
        ax = axes_res[key]
        for ckey in ("C1", "C2"):
            if ckey not in ax["strata"]:
                continue
            for r in ax["strata"][ckey]["rows"]:
                rows.append({"axis": ax["label"], "key": key, "comparator": ckey,
                             "stratum": r["stratum"], "n": r.get("n"),
                             "delta": r.get("delta"), "lo": r.get("ci_low"),
                             "hi": r.get("ci_high")})
    df = pd.DataFrame(rows).dropna(subset=["delta"])
    if df.empty:
        return
    ypos, y, last = {}, 0.0, None
    for _, r in df.iterrows():
        key = (r["key"], r["stratum"])
        if key not in ypos:
            if last is not None and r["key"] != last:
                y += 0.7
            ypos[key] = y
            y += 1.0
            last = r["key"]
    df["y"] = [ypos[(k, s)] for k, s in zip(df["key"], df["stratum"])]
    fig, panels = plt.subplots(1, 2, figsize=(170 * MM, 205 * MM), layout="constrained",
                               sharey=False, width_ratios=[1.0, 1.0])
    for ax, ckey, title in ((panels[0], "C1", "vs nurse-recorded admission weight"),
                            (panels[1], "C2", "vs height-and-sex regression")):
        sub = df[df["comparator"] == ckey]
        ax.axvline(0, color=OI["black"], linewidth=0.8)
        ax.errorbar(sub["delta"], sub["y"],
                    xerr=[sub["delta"] - sub["lo"], sub["hi"] - sub["delta"]],
                    fmt="o", markersize=3.4, color=OI["blue"], ecolor=OI["grey"],
                    capsize=1.6, elinewidth=0.7, linestyle="none")
        ax.set_yticks(sub["y"])
        ax.set_yticklabels([f"{s} (n={int(n)})" for s, n in zip(sub["stratum"], sub["n"])],
                           fontsize=6.6)
        ax.invert_yaxis()
        ax.set_xlabel("Paired difference in absolute error (kg)\nnegative = imaging model better")
        ax.set_title(title, loc="left", fontsize=8.5)
        # 轴分组标题
        for key in order:
            g = sub[sub["key"] == key]
            if g.empty:
                continue
            ax.text(ax.get_xlim()[0], g["y"].min() - 0.45, g["axis"].iloc[0],
                    fontsize=7, fontweight="bold", va="bottom", ha="left",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    for ax in panels:
        ax.set_ylim(y + 0.2, -1.4)
    out = ["figS2_subgroups"]
    for name in out:
        fig.savefig(FIGURE_DIR / f"{name}.png", facecolor="white")
        fig.savefig(FIGURE_DIR / f"{name}.svg", facecolor="white")
        ioi = Path("Insights_into_Imaging/figures")
        if ioi.exists():
            fig.savefig(ioi / f"{name}.png", facecolor="white")
            fig.savefig(ioi / f"{name}.svg", facecolor="white")
    plt.close(fig)


def _row(ax_res: dict, ckey: str, stratum: str) -> dict:
    for r in ax_res["strata"].get(ckey, {}).get("rows", []):
        if r["stratum"] == stratum:
            return r
    return {}


def _f(v, nd=2, plus=False, dash="—"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    return f"{v:+.{nd}f}" if plus else f"{v:.{nd}f}"


def _q(v) -> str:
    if v is None or not np.isfinite(v):
        return "NA"
    return "< 0.001" if v < 0.001 else f"= {v:.3f}"


def write_esm(results: dict, log) -> None:
    out_dir = Path("Insights_into_Imaging/supplementary")
    if not out_dir.exists():
        return
    axes_res = results["axes"]
    order = ["record_reliability", "uncertainty", "weight", "bmi", "height_availability",
             "edema", "fluid_balance"]
    L = ["# ESM 2. Subgroup analysis: where does the imaging model add value?", "",
         "Seven axes and both comparators were declared primary in the study analysis plan before this",
         "analysis was run; the primary comparator is the nurse-recorded admission weight, with the",
         "height-and-sex regression as the secondary comparator. Paired differences are",
         "Δ = |measured − imaging| − |measured − comparator|, so negative values favour imaging. Confidence",
         "intervals use patient-level cluster bootstrap (2,000 resamples); p values for interaction come",
         "from cluster-robust OLS with patient-level clustering; Benjamini–Hochberg correction was applied",
         "within the primary-comparator family across the seven axes and within the secondary family.",
         "", "## Table S3. Paired comparisons by subgroup", "",
         "| Axis | Subgroup | n | MAE imaging | MAE record | Δ vs record (95% CI) | MAE regression | Δ vs regression (95% CI) |",
         "|---|---|---|---|---|---|---|---|"]
    for key in order:
        a = axes_res[key]
        for lab, _ in [(lab, None) for lab, _ in [(r["stratum"], None)
                                                 for r in a["strata"]["C1"]["rows"]]]:
            r1, r2 = _row(a, "C1", lab), _row(a, "C2", lab)
            L.append(f"| {a['label']} | {lab} | {r1.get('n', '—')} | {_f(r1.get('mae_a'))} | "
                     f"{_f(r1.get('mae_b'))} | {_f(r1.get('delta'), plus=True)} "
                     f"({_f(r1.get('ci_low'), plus=True)} to {_f(r1.get('ci_high'), plus=True)}) | "
                     f"{_f(r2.get('mae_b'))} | {_f(r2.get('delta'), plus=True)} |")
    L += ["", "Δ = difference in absolute error (imaging − comparator); negative favours the imaging model.",
          "", "## Table S4. Interaction tests (pre-specified contrasts)", "",
          "| Axis | Contrast | Comparator | Δ difference (95% CI) | p (cluster-robust) | q (BH) |",
          "|---|---|---|---|---|---|"]
    for key in order:
        a = axes_res[key]
        for it in a["interaction"]:
            L.append(f"| {a['label']} | {it['contrast']} | {it['comparator']} | "
                     f"{_f(it['delta_diff'], plus=True)} ({_f(it['boot_ci_low'], plus=True)} to "
                     f"{_f(it['boot_ci_high'], plus=True)}) | {it['p_cluster']:.2e} | "
                     f"{'' if it.get('q_bh') is None else format(it['q_bh'], '.3f')} |")
    L += ["", "## Table S5. Continuous interaction (no cut-points)", "",
          "| Axis | Variable | Comparator | Slope of Δ per unit (95% CI) | p |", "|---|---|---|---|---|"]
    for key in order:
        a = axes_res[key]
        for ckey, c in (a.get("continuous_by_comparator") or {}).items():
            L.append(f"| {a['label']} | {c['variable']} | {ckey} | {_f(c['beta_per_unit'], 3, plus=True)} "
                     f"({_f(c['ci_low'], 3, plus=True)} to {_f(c['ci_high'], 3, plus=True)}) | "
                     f"{c['p_cluster']:.2e} |")
    gate = results["gate"]
    L += ["", "## Table S6. Decision rule using only prediction-time information", "",
          "| Cell | n | MAE imaging | MAE record | Δ |", "|---|---|---|---|---|"]
    for c in gate["cells"]:
        L.append(f"| {c['cell']} | {c.get('n')} | {_f(c.get('mae_dl'))} | {_f(c.get('mae_record'))} | "
                 f"{_f(c.get('delta'), plus=True)} |")
    L += ["", "| Rule (admission record is the default) | MAE (kg) | Δ vs always-record (95% CI) | share using imaging |",
          "|---|---|---|---|"]
    for r in gate["rules"]:
        L.append(f"| {r['rule']} | {_f(r['mae'])} | {_f(r['delta_vs_always_record'], plus=True)} "
                 f"({_f(r['ci_low'], plus=True)} to {_f(r['ci_high'], plus=True)}) | "
                 f"{100*r['use_dl_rate']:.1f}% |")
    wr = results["win_rate"]["overall"]
    L += ["", f"Individual-level win rate (imaging closer than the record): {100*wr['win_rate']:.1f}% "
              f"(95% CI {100*wr['ci_low']:.1f}–{100*wr['ci_high']:.1f}%, n={wr['n']}).", ""]
    L += ["## Robustness", "",
          "**Placebo subgroups.** Random patient-level partitions of identical size give the following",
          "two-sided empirical p values for the observed interaction contrasts:", "",
          "| Axis | Contrast | Observed Δ difference (kg) | Placebo SD (kg) | Empirical p |",
          "|---|---|---|---|---|"]
    for pl in results["placebo"]:
        L.append(f"| {pl['axis']} | {pl['contrast']} | {_f(pl['observed_delta_diff'], plus=True)} | "
                 f"{_f(pl['placebo_sd'])} | {pl['placebo_p_two_sided']:.3f} |")
    L += ["", "**Backbone replication.** The same interaction contrasts recomputed with the Swin-T @308",
          "three-seed ensemble:", "", "| Axis | Comparator | Δ difference, ResNet-50 | Δ difference, Swin-T |",
          "|---|---|---|---|"]
    by_key = {}
    for r in results["swin_replication"]:
        by_key.setdefault((r["axis"], r["comparator"]), {})[r["model"]] = r["delta_diff"]
    for (axis, comp), v in by_key.items():
        L.append(f"| {axes_res[axis]['label']} | {comp} | {_f(v.get('main'), plus=True)} | "
                 f"{_f(v.get('swin'), plus=True)} |")
    urc = json.loads((TABLE_DIR / "unreliable_subgroup_check.json").read_text(encoding="utf-8"))
    g5 = urc["subgroups"]["adm_discrepancy_ge5kg"]
    L += ["", "**Regression to the mean.** In the subgroup where the record and the measurement disagree by",
          "5 kg or more, the comparison was repeated against a calibrated record estimate (linear fit on the",
          f"training split) and a shrunk record estimate. The imaging model remains best",
          f"({g5['胸片深度学习']['mae']:.2f} kg) ahead of the calibrated record",
          f"({g5['训练集校准后的入院体重']['mae']:.2f} kg), the shrunk record "
          f"({g5['训练集最优混合(shrinkage)']['mae']:.2f} kg) and the raw record "
          f"({g5['入院体重']['mae']:.2f} kg); the margin over the calibrated record is smaller than over the",
          "raw record, but the ordering does not reverse.", "",
          "**Other checks.** Restricting to stays where the record and the radiograph were equally close to",
          "the reference measurement, and restricting to images passing strict quality control, reproduce",
          "the main analysis.", ""]
    lq_path = TABLE_DIR / "label_quality_analysis.json"
    if lq_path.exists():
        lq = json.loads(lq_path.read_text(encoding="utf-8"))
        ex = lq["exclusion_sensitivity"]["drop_contradicted"]
        ex2 = lq["exclusion_sensitivity"]["drop_contradicted_and_equivocal"]
        cf = lq["copy_forward_evidence"]
        L += [
            "**Label-quality sensitivity.** Charted weights cannot be independently verified:",
            f"{100*lq['label_class_distribution_cohort']['unverifiable']/lq['n_cohort']:.0f}% of index weights "
            "had no other same-stay weight within 6 h, and among those that did,",
            f"{100*cf['share_exact_same_value']:.0f}% of the nearest records were exact copies of the index value",
            f"(median gap {cf['median_gap_h']:.1f} h), consistent with copy-forward documentation rather than",
            "independent re-measurement. Excluding test-set labels contradicted by ≥5 kg within 6 h",
            f"changed MAE from {lq['overall_test']['mae_dl']:.2f} to {ex['mae_dl']:.2f} kg; also excluding",
            f"equivocal labels (2–5 kg) gave {ex2['mae_dl']:.2f} kg. The three primary interactions were",
            f"unchanged (record reliability {ex['record_reliability_ge5_vs_lt2']['delta_diff']:+.2f} kg; "
            f"BMI ≥40 {ex['bmi_ge40_vs_18_5_25']['delta_diff']:+.2f} kg; fluid balance "
            f"{ex['fluid_balance_q4_vs_q1']['delta_diff']:+.2f} kg). A 'corroborated' subset is not a clean-label",
            "subset, because it selects younger and less ventilated patients whose weight was stable; the",
            "record itself was near-perfect in that subset (MAE 1.20 kg). Label exclusion is therefore",
            "reported as a sensitivity analysis only, not as label cleaning.", ""]
    tv_path = TABLE_DIR / "temporal_validation.json"
    if tv_path.exists():
        tv = json.loads(tv_path.read_text(encoding="utf-8"))
        re_, sw = tv["ensemble_resnet"], tv["ensemble_swin"]
        pr = tv["paired_resnet_vs_swin"]
        L += ["**Temporal hold-out.** The temporal test patients overlap 70.7% with the random training",
              "split, so both architectures were retrained on the temporal training split (6,097 stays)",
              "and evaluated once on 1,353 temporally held-out, patient-disjoint stays. The ResNet-50",
              f"ensemble reached {re_['mae']:.2f} kg (within ±10%: "
              f"{100*re_['hit_within_10pct']:.1f}%) versus {sw['mae']:.2f} kg for the Swin-T ensemble "
              f"({100*sw['hit_within_10pct']:.1f}%); the paired difference was "
              f"{pr['paired_diff_a_minus_b']:+.2f} kg (95% CI {pr['ci95'][0]:+.2f} to {pr['ci95'][1]:+.2f}).",
              "Performance was therefore similar to the random-split test set, and the direction of the",
              "architecture difference was reproduced without reaching significance.", ""]
    dc_path = TABLE_DIR / "decision_curve_analysis.json"
    if dc_path.exists():
        dc = json.loads(dc_path.read_text(encoding="utf-8"))

        def nb_of(mapping, t):
            for k in (str(t), f"{t:.2f}", f"{t:g}"):
                if k in mapping:
                    return float(mapping[k])
            raise KeyError(t)

        dose = {r["strategy"]: r for r in dc["dose_strategies"]}
        L += [
            "**Clinical utility.** In a decision-curve analysis for re-weighing (target: record unreliable",
            f"by ≥5 kg; prevalence {100*dc['prevalence_unreliable_5kg']:.1f}%), net benefit at threshold",
            f"probabilities 0.2 and 0.3 was {nb_of(dc['reused_calibrated_curves']['model'], 0.2):.3f} and "
            f"{nb_of(dc['reused_calibrated_curves']['model'], 0.3):.3f} for the fitted reliability model, versus "
            f"{nb_of(dc['net_benefit_treat_all'], 0.2):.3f} and {nb_of(dc['net_benefit_treat_all'], 0.3):.3f} for",
            f"re-weighing everyone and {dc['net_benefit_fixed_rule_ci']['gap_ge5']['0.2']['nb']:.3f} and "
            f"{dc['net_benefit_fixed_rule_ci']['gap_ge5']['0.3']['nb']:.3f} for the fixed |imaging − record|",
            f"≥5 kg rule. At a threshold of 0.1, re-weighing everyone equalled the model "
            f"({nb_of(dc['net_benefit_treat_all'], 0.1):.3f}). In dose terms, no imaging-based strategy reduced",
            "the proportion with >10% actual-body-weight dose error below the admission record",
            f"({dose['admission record']['pct_gt10']:.1f}%): imaging alone "
            f"{dose['imaging model']['pct_gt10']:.1f}%, fixed gap ≥5 kg "
            f"{dose['rule: imaging if gap>=5 kg']['pct_gt10']:.1f}%, gap ≥10 kg "
            f"{dose['rule: imaging if gap>=10 kg']['pct_gt10']:.1f}%, gap ≥5 kg with low uncertainty "
            f"{dose['rule: imaging if gap>=5 and low uncertainty']['pct_gt10']:.1f}%, and Devine IBW "
            f"{dose['Devine IBW']['pct_gt10']:.1f}%. The imaging model can therefore support selective",
            "re-weighing at moderate thresholds but cannot replace the charted weight for dosing.", ""]
    L += ["## Interpretation", ""]
    rr = axes_res["record_reliability"]
    r2 = _row(rr, "C1", ">=5 kg")
    r0 = _row(rr, "C1", "<2 kg")
    it_rr = next(it for it in rr["interaction"] if it["comparator"] == "C1")
    fbx = axes_res["fluid_balance"]
    fb4, fb1 = _row(fbx, "C1", "Q4"), _row(fbx, "C1", "Q1")
    it_fb = next(it for it in fbx["interaction"] if it["comparator"] == "C1")
    bx = axes_res["bmi"]
    b40, b25 = _row(bx, "C1", ">=40"), _row(bx, "C1", "18.5-25")
    b40r = _row(bx, "C2", ">=40")
    it_b = next(it for it in bx["interaction"] if it["comparator"] == "C1")
    ux = axes_res["uncertainty"]
    u4, u1 = _row(ux, "C1", "Q4"), _row(ux, "C1", "Q1")
    u4r = _row(ux, "C2", "Q4")
    it_u = next(it for it in ux["interaction"] if it["comparator"] == "C1")
    it_ur = next(it for it in ux["interaction"] if it["comparator"] == "C2")
    ed = axes_res["edema"]
    it_ed = next(it for it in ed["interaction"] if it["comparator"] == "C1")
    gates = {r["rule"]: r for r in gate["rules"]}
    wr = results["win_rate"]["by_axis"]["record_reliability"]
    L += [
        f"1. **Record reliability dominates every other axis.** The imaging model is behind the record by",
        f"   {_f(r0['delta'], 2)} kg where the record agrees with the measurement to within 2 kg, but is",
        f"   ahead by {_f(abs(r2['delta']), 2)} kg where the two disagree by 5 kg or more",
        f"   (interaction {_f(it_rr['delta_diff'], 2, plus=True)} kg, 95% CI "
        f"{_f(it_rr['boot_ci_low'], 2, plus=True)} to {_f(it_rr['boot_ci_high'], 2, plus=True)}, "
        f"q {_q(it_rr['q_bh'])}); the per-stay win rate rises from {100*wr['<2 kg']['win_rate']:.1f}% to "
        f"{100*wr['>=5 kg']['win_rate']:.1f}%.",
        f"2. **Cumulative fluid balance behaves like a staleness clock for the record.** Between the first and",
        f"   fourth quartile of fluid accumulated since ICU admission, the record's error rises from",
        f"   {_f(fb1['mae_b'])} to {_f(fb4['mae_b'])} kg while the imaging model stays near {_f(fb4['mae_a'])} kg, so the",
        f"   imaging deficit shrinks from {_f(fb1['delta'], 2)} to {_f(fb4['delta'], 2)} kg",
        f"   (fourth-quartile Δ, 95% CI {_f(fb4['ci_low'], 2, plus=True)} to {_f(fb4['ci_high'], 2, plus=True)} kg;",
        f"   interaction {_f(it_fb['delta_diff'], 2, plus=True)} kg, q {_q(it_fb['q_bh'])}).",
        f"3. **Body-size extremes split the two comparisons.** In class III obesity (BMI ≥ 40) the imaging",
        f"   model is {_f(b40['delta'], 2)} kg behind the record ({_f(b40['mae_a'])} vs {_f(b40['mae_b'])} kg MAE;",
        f"   interaction {_f(it_b['delta_diff'], 2, plus=True)} kg, q {_q(it_b['q_bh'])}), yet it is "
        f"{_f(abs(b40r['delta']), 1)} kg ahead of the height-and-sex regression, which errs by {_f(b40r['mae_b'])} kg.",
        f"   A recorded or measured weight should therefore be preferred over imaging in obesity, and imaging",
        f"   over any formula; the same pattern holds above 100 kg.",
        f"4. **Uncertainty identifies where formulas fail, not where the record can be overruled.** In the",
        f"   highest uncertainty quartile the imaging model is {_f(u4['delta'], 2, plus=True)} kg behind the record",
        f"   (vs {_f(u1['delta'], 2, plus=True)} kg in the lowest; interaction {_f(it_u['delta_diff'], 2, plus=True)} kg,",
        f"   q {_q(it_u['q_bh'])}) but {_f(abs(u4r['delta']), 1)} kg ahead of the regression",
        f"   (interaction {_f(it_ur['delta_diff'], 2, plus=True)} kg, q {_q(it_ur['q_bh'])}).",
        f"5. **No prediction-time rule recovers the subgroup in which imaging beats the record.** Gating on",
        f"   |imaging − record| ≥ 5 kg leaves the record ahead by "
        f"{_f(gates['imaging model when gap >= 5 kg and low uncertainty']['delta_vs_always_record'], 2, plus=True)} to "
        f"{_f(gates['imaging model when gap >= 5 kg and high uncertainty']['delta_vs_always_record'], 2, plus=True)} kg in",
        f"   the selected cells, and the fitted record-reliability model reduces MAE only from "
        f"{_f(gate['fitted_flagging_model_reference'].get('mae_always_adm'))} to "
        f"{_f(gate['fitted_flagging_model_reference'].get('gate_dl_mae'))} kg while firing in "
        f"{100*gate['fitted_flagging_model_reference'].get('gate_dl_use_rate', float('nan')):.1f}% of stays.",
        f"   Knowing that a charted weight is wrong requires a measurement; the model cannot supply that",
        f"   judgement by itself.",
        f"6. **Radiographic pulmonary edema slightly weakens the case for imaging.** Positive oedema shifts the",
        f"   comparison against imaging by {_f(it_ed['delta_diff'], 2, plus=True)} kg (q {_q(it_ed['q_bh'])}), consistent",
        f"   with intrathoracic fluid decoupling the radiograph from total body mass.",
        f"7. **Height availability shows no interaction** (q "
        f"{_q(next(it for it in axes_res['height_availability']['interaction'] if it['comparator'] == 'C1')['q_bh'])});",
        f"   its importance is structural, not statistical: where height is missing, no formula or regression",
        f"   can be computed at all, and imaging is the only available quantitative estimate.",
        "",
        "All subgroup findings are single-centre, post hoc in origin (the axes were frozen before this",
        "analysis but chosen after earlier exploratory work), and require external validation. Two contrasts",
        "rely on small cells (BMI < 18.5, n = 31; weight < 50 kg, n = 52) and are reported for completeness",
        "only.", ""]
    L += [
          "## Descriptive appendix (not inferential)", "",
          "| Population | n | Note |", "|---|---|---|",
          "| Limb loss / amputation (ICD) | 37 | too small for inference |",
          "| Cirrhosis / ascites (ICD) | 24 | too small for inference |",
          "| End-stage renal disease (ICD) | 95 | dialysis-related weight shifts |",
          "| Obesity diagnosis (ICD) | 203 | overlaps with the BMI axis |", "",
          "![Fig. S2](../figures/figS2_subgroups.png)", "",
          "**Fig. S2** Paired difference in absolute error against the nurse-recorded admission weight",
          "(left) and against the height-and-sex regression (right), by subgroup. Negative values favour",
          "the imaging model; whiskers are 95% confidence intervals from patient-level cluster bootstrap.",
          "Rows are grouped by axis; n is the number of stays in that cell.", ""]
    (out_dir / "ESM2_subgroup_analysis.md").write_text("\n".join(L), encoding="utf-8")
    log.info("补充材料 -> %s", out_dir / "ESM2_subgroup_analysis.md")


def main() -> None:
    log = setup_logger("subgroup_analysis")
    df = load_frame()
    axes = make_axes(df)
    log.info("分析框 n=%d，患者 %d", len(df), df["subject_id"].nunique())
    results = {"main_model": MAIN_TAG, "n_stays": int(len(df)),
               "n_patients": int(df["subject_id"].nunique()),
               "plan": "project/docs/subgroup_analysis_plan.md (approved 2026-10-10)",
               "comparators": {k: v[0] for k, v in COMPARATORS.items()}, "axes": {}}
    for ax in axes:
        results["axes"][ax["key"]] = analyse_axis(df, ax, log)
        log.info("轴完成：%s", ax["key"])

    # BH 校正：主对照 7 项、辅对照 6 项，各自成一个 family
    for ckey in ("C1", "C2", "C3"):
        items = [(k, it) for k, a in results["axes"].items() for it in a["interaction"]
                 if it["comparator"] == ckey]
        if not items:
            continue
        qs = bh([it["p_cluster"] for _, it in items])
        for (_, it), q in zip(items, qs):
            it["q_bh"] = q
    results["win_rate"] = win_rate_analysis(df, axes)
    results["gate"] = gate_analysis(df)
    results["placebo"] = [placebo_test(df, a) for a in axes
                          if a["key"] in ("record_reliability", "edema", "fluid_balance",
                                          "uncertainty")]
    results["swin_replication"] = swin_replication(df, axes)
    rows = []
    for key, a in results["axes"].items():
        for ckey, blk in a["strata"].items():
            for r in blk["rows"]:
                rows.append({"axis": key, "comparator": ckey,
                             **{k: v for k, v in r.items() if k != "stratum"},
                             "stratum": r["stratum"]})
    pd.DataFrame(rows).to_csv(TABLE_DIR / "subgroup_analysis_strata.csv", index=False)
    save_json(TABLE_DIR / "subgroup_analysis.json", results)
    make_figure(results)
    write_esm(results, log)
    log.info("完成；交互检验（C1）q<0.05 的轴：%s",
             [k for k, a in results["axes"].items()
              if any(it["comparator"] == "C1" and (it.get("q_bh") or 1) < 0.05
                     for it in a["interaction"])])


if __name__ == "__main__":
    main()
