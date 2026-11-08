"""无可信体重记录人群：胸片深度学习 vs 公式法/护理记录的配对评估。

人群
    拍片 ±24 h 内没有床秤实测体重的 ICU 住院（只有护理录入的入院体重，n=4,656）。
    其中能在 ±96 h 内找到一次床秤实测者进入**可评估子集**（n≈556），以该实测值作为替代参考。

对照
    1) 护理录入的入院体重（临床实际会用到的记录）
    2) 同队列训练的身高+性别 OLS（最强"无体重"对照）
    3) 人群均值 / 性别均值（训练集）
    4-8) IBW 公式（Devine/Hume/Robinson/Miller/Hamwi）
    9) BMI=25 x 身高²

产出
    project/results/tables/no_reliable_record_analysis.json
    project/results/figures/no_reliable_record_analysis.png
"""

from __future__ import annotations

import json

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FIGURE_DIR, MAIN_TAG, PROCESSED_DIR, TABLE_DIR, load_config, save_json,
                    setup_logger)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
SEED = 20261012


def ibw_formulas(male: np.ndarray, cm: np.ndarray) -> dict[str, np.ndarray]:
    inch = np.asarray(cm, dtype=float) / 2.54
    out = {}
    out["Devine IBW"] = np.where(male, 50 + 2.3 * (inch - 60), 45.5 + 2.3 * (inch - 60))
    out["Hume IBW / ARDSNet PBW"] = np.where(male, 50 + 0.91 * (cm - 152.4), 45.5 + 0.91 * (cm - 152.4))
    out["Robinson IBW"] = np.where(male, 52 + 1.9 * (inch - 60), 49 + 1.7 * (inch - 60))
    out["Miller IBW"] = np.where(male, 56.2 + 1.41 * (inch - 60), 53.1 + 1.41 * (inch - 60))
    out["Hamwi IBW"] = np.where(male, 48 + 2.7 * (inch - 60), 45.5 + 2.2 * (inch - 60))
    out["BMI=25 x height²"] = 25.0 * (np.asarray(cm) / 100.0) ** 2
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return {"n": 0}
    err = p - y
    return {
        "n": int(len(y)),
        "mae": float(np.abs(err).mean()),
        "median_ae": float(np.median(np.abs(err))),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "mape_pct": float((np.abs(err) / y).mean() * 100),
        "hit_within_10pct": float((np.abs(err) / y <= 0.10).mean()),
        "hit_within_10kg": float((np.abs(err) <= 10).mean()),
        "hit_within_5kg": float((np.abs(err) <= 5).mean()),
    }


def boot_ci(y, p, subjects, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": y, "p": p, "g": subjects}).dropna()
    groups = list(df.groupby("g", sort=False))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[i][1] for i in idx])
        vals.append(np.abs(s["p"] - s["y"]).mean())
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def paired_diff(dl, other, y, subjects, n_boot=2000, seed=SEED):
    """MAE(DL) − MAE(other) 的患者级聚类 bootstrap（负值 = DL 更好）。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": y, "dl": dl, "o": other, "g": subjects}).dropna()
    groups = list(df.groupby("g", sort=False))
    base = float(np.abs(df["dl"] - df["y"]).mean() - np.abs(df["o"] - df["y"]).mean())
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[i][1] for i in idx])
        vals.append(np.abs(s["dl"] - s["y"]).mean() - np.abs(s["o"] - s["y"]).mean())
    return base, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def reference_drift(con, train_stays: list[int], log) -> dict:
    """用队列 A 的多次床秤实测估计"延迟参考"的噪声：相隔 24–96 h 的两次实测体重差的绝对值。"""
    con.register("tr", pd.DataFrame({"stay_id": train_stays}))
    q = con.execute("""
        WITH w AS (SELECT stay_id, charttime, valuenum FROM main.chartevents
                   WHERE itemid = 224639 AND valuenum BETWEEN 20 AND 350
                     AND stay_id IN (SELECT stay_id FROM tr))
        SELECT a.valuenum AS w1, b.valuenum AS w2,
               abs(date_diff('minute', a.charttime, b.charttime)) / 60.0 AS gap_h
        FROM w a JOIN w b ON a.stay_id = b.stay_id AND b.charttime > a.charttime
        WHERE abs(date_diff('minute', a.charttime, b.charttime)) / 60.0 BETWEEN 24 AND 96
    """).fetchdf()
    d = (q["w1"] - q["w2"]).abs()
    out = {"n_pairs": int(len(q)),
           "mean_abs_change": float(d.mean()) if len(q) else None,
           "median_abs_change": float(d.median()) if len(q) else None,
           "p90_abs_change": float(d.quantile(0.90)) if len(q) else None}
    log.info("延迟参考噪声（队列 A 内 24–96 h 两次实测体重差）：%s", json.dumps(out, ensure_ascii=False))
    return out


def main() -> None:
    log = setup_logger("no_reliable_record_analysis")
    cfg = load_config()
    con = duckdb.connect(cfg["data"]["duckdb"], read_only=True)
    con.execute("PRAGMA threads=8")

    coh = pd.read_csv(PROCESSED_DIR / "cohort_nrr_index.csv")
    pred = pd.read_csv(PROCESSED_DIR / f"preds_nrr_{MAIN_TAG}.csv")[["dicom_id", "pred", "pred_std"]]
    df = coh.merge(pred, on="dicom_id", how="inner")
    df["male"] = (df["gender"] == "M").astype(float)
    log.info("合并后 %d 行（有替代参考 %d）", len(df), df["ref_kg"].notna().sum())

    # ---------- 护理录入体重的记录时间（用于时间可比性分析）----------
    con.register("coh", df[["subject_id", "stay_id"]])
    nt = con.execute("""
        SELECT stay_id, avg(CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218
                                 ELSE valuenum END) AS nurse_chart_kg,
               min(charttime) AS nurse_t_min, max(charttime) AS nurse_t_max,
               count(1) AS nurse_n
        FROM main.chartevents
        WHERE itemid IN (226512, 226531) AND valuenum IS NOT NULL
          AND stay_id IN (SELECT stay_id FROM coh)
        GROUP BY 1
    """).fetchdf()
    df = df.drop(columns=[c for c in ("nurse_chart_kg",) if c in df]).merge(nt, on="stay_id", how="left")
    df["ref_dt"] = pd.to_datetime(df["ref_charttime"])
    df["nurse_t_mid"] = pd.to_datetime(df["nurse_t_max"]) - (
        pd.to_datetime(df["nurse_t_max"]) - pd.to_datetime(df["nurse_t_min"])) / 2
    df["gap_film_ref_h"] = df["ref_gap_hours"].abs()
    df["gap_nurse_ref_h"] = (df["nurse_t_mid"] - df["ref_dt"]).dt.total_seconds().abs() / 3600.0
    log.info("时间距离：影像↔参考 中位 %.1f h；护理记录↔参考 中位 %.1f h",
             df["gap_film_ref_h"].median(), df["gap_nurse_ref_h"].median())

    # ---------- 训练集统计量（人群均值、性别均值、身高性别 OLS）----------
    a = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    a = a.merge(splits[["subject_id", "split_random"]], on="subject_id", how="left")
    hgt = con.execute("SELECT stay_id, avg(height) AS height_icu FROM mimiciv_derived.height GROUP BY 1").fetchdf()
    sex = con.execute("SELECT subject_id, gender, anchor_age FROM main.patients").fetchdf()
    a = a.merge(hgt, on="stay_id", how="left").merge(sex, on="subject_id", how="left")
    a["male"] = (a["gender"] == "M").astype(float)
    tr = a[a["split_random"] == "train"]
    pop_mean = float(tr["weight_kg"].mean())
    sex_mean = tr.groupby("male")["weight_kg"].mean().to_dict()
    ols_tr = tr.dropna(subset=["height_icu"])
    X = np.column_stack([np.ones(len(ols_tr)), ols_tr["height_icu"].values, ols_tr["male"].values])
    beta = np.linalg.lstsq(X, ols_tr["weight_kg"].values, rcond=None)[0]
    log.info("训练集：n=%d，人群均值 %.1f kg；OLS(身高+性别) 系数 %s（n=%d）",
             len(tr), pop_mean, np.round(beta, 3).tolist(), len(ols_tr))

    # ---------- 对照与指标 ----------
    df["pop_mean"] = pop_mean
    df["sex_mean"] = df["male"].map(sex_mean)
    hf = df["height_final_cm"].values
    df["ols_height_sex"] = beta[0] + beta[1] * hf + beta[2] * df["male"].values
    for k, v in ibw_formulas(df["male"].values, hf).items():
        df[k] = v

    evaluable = df[df["ref_kg"].notna()].copy()
    y = evaluable["ref_kg"].values
    subjects = evaluable["subject_id"].values
    methods = ["pred", "nurse_adm_kg", "ols_height_sex", "sex_mean", "pop_mean",
               "Devine IBW", "Hume IBW / ARDSNet PBW", "Robinson IBW", "Miller IBW",
               "Hamwi IBW", "BMI=25 x height²"]
    labels = {"pred": "胸片深度学习（本次）", "nurse_adm_kg": "护理录入的入院体重",
              "ols_height_sex": "身高+性别回归（同队列训练）", "sex_mean": "性别均值",
              "pop_mean": "人群均值", "BMI=25 x height²": "BMI=25 × 身高²"}

    res = {"n_evaluable": int(len(evaluable)),
           "n_stays_no_reliable_record": int(len(coh)),
           "ref_gap_hours_median": float(evaluable["ref_gap_hours"].abs().median()),
           "ref_gap_hours_iqr": [float(evaluable["ref_gap_hours"].abs().quantile(q)) for q in (0.25, 0.75)],
           "frac_reference_after_film": float((evaluable["ref_gap_hours"] > 0).mean()),
           "time_distance_h_median": {
               "film_to_reference": float(evaluable["gap_film_ref_h"].median()),
               "nurse_record_to_reference": float(evaluable["gap_nurse_ref_h"].median())},
           "methods": {}, "paired_vs_dl": {}}
    for m in methods:
        p = evaluable[m].values
        met = metrics(y, p)
        met["mae_ci95"] = boot_ci(y, p, subjects) if met["n"] > 10 else None
        res["methods"][labels.get(m, m)] = met
        log.info("%-28s n=%4d MAE=%.2f  95%%CI=[%.2f,%.2f]  ±10%%=%.1f%%  ±10kg=%.1f%%",
                 labels.get(m, m), met["n"], met["mae"],
                 met["mae_ci95"][0] if met["mae_ci95"] else np.nan,
                 met["mae_ci95"][1] if met["mae_ci95"] else np.nan,
                 100 * met["hit_within_10pct"], 100 * met["hit_within_10kg"])
        if m != "pred" and met["n"] > 10:
            d, lo, hi = paired_diff(evaluable["pred"].values, p, y, subjects)
            ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(evaluable["pred"].values)
            res["paired_vs_dl"][labels.get(m, m)] = {"diff": d, "ci95": [lo, hi],
                                                     "n": int(ok.sum())}
            log.info("    配对（DL − %s）= %+.2f kg [%.2f, %.2f]", labels.get(m, m), d, lo, hi)

    # ---------- 敏感性：参考更近（<=48 h）----------
    near = evaluable[evaluable["ref_gap_hours"].abs() <= 48]
    res["sensitivity_48h"] = {"n": int(len(near)),
                              "methods": {labels.get(m, m): metrics(near["ref_kg"].values, near[m].values)
                                          for m in methods}}
    log.info("敏感性（参考 ≤48 h，n=%d）：DL MAE=%.2f，护理记录 MAE=%.2f",
             len(near), res["sensitivity_48h"]["methods"][labels["pred"]]["mae"],
             res["sensitivity_48h"]["methods"][labels["nurse_adm_kg"]]["mae"])

    # ---------- 敏感性：时间可比（护理记录离参考不近于影像）----------
    matched = evaluable[evaluable["gap_nurse_ref_h"] >= evaluable["gap_film_ref_h"]]
    res["sensitivity_time_matched"] = {
        "n": int(len(matched)),
        "definition": "护理记录与参考实测的时间距离 >= 影像与参考实测的时间距离",
        "methods": {labels.get(m, m): metrics(matched["ref_kg"].values, matched[m].values)
                    for m in ["pred", "nurse_adm_kg", "ols_height_sex", "pop_mean", "Devine IBW"]}}
    if len(matched) > 20:
        d, lo, hi = paired_diff(matched["pred"].values, matched["nurse_adm_kg"].values,
                                matched["ref_kg"].values, matched["subject_id"].values)
        res["sensitivity_time_matched"]["paired_dl_minus_nurse"] = {"diff": d, "ci95": [lo, hi]}
    log.info("敏感性（时间可比，n=%d）：DL MAE=%.2f，护理记录 MAE=%.2f",
             len(matched), res["sensitivity_time_matched"]["methods"][labels["pred"]]["mae"],
             res["sensitivity_time_matched"]["methods"][labels["nurse_adm_kg"]]["mae"])

    # ---------- 敏感性：仅有身高者（IBW/OLS 可比子集）----------
    with_h = evaluable[evaluable["height_final_cm"].notna()]
    res["sensitivity_with_height"] = {"n": int(len(with_h)),
                                      "methods": {labels.get(m, m): metrics(with_h["ref_kg"].values, with_h[m].values)
                                                  for m in ["pred", "nurse_adm_kg", "ols_height_sex", "Devine IBW"]}}
    res["reference_drift"] = reference_drift(con, tr["stay_id"].tolist(), log)
    con.close()

    save_json(TABLE_DIR / "no_reliable_record_analysis.json", res)

    # ---------- 图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    order = ["pred", "nurse_adm_kg", "ols_height_sex", "pop_mean", "sex_mean",
             "BMI=25 x height²", "Miller IBW", "Hamwi IBW", "Robinson IBW",
             "Hume IBW / ARDSNet PBW", "Devine IBW"]
    names = [labels.get(m, m) for m in order]
    maes = [res["methods"][n]["mae"] for n in names]
    colors = ["#c0392b" if m == "pred" else ("#2980b9" if m == "nurse_adm_kg" else "#95a5a6") for m in order]
    axes[0, 0].barh(range(len(order)), maes, color=colors)
    axes[0, 0].set_yticks(range(len(order)))
    axes[0, 0].set_yticklabels(names, fontsize=8)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("MAE (kg)")
    axes[0, 0].set_title(f"无可信体重记录人群：各方法误差（n={res['n_evaluable']}）")
    for i, v in enumerate(maes):
        axes[0, 0].text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=8)

    keys = [k for k in res["paired_vs_dl"] if res["paired_vs_dl"][k]["n"] > 30]
    d = [res["paired_vs_dl"][k]["diff"] for k in keys]
    lo = [res["paired_vs_dl"][k]["ci95"][0] for k in keys]
    hi = [res["paired_vs_dl"][k]["ci95"][1] for k in keys]
    axes[0, 1].errorbar(d, range(len(keys)), xerr=[np.array(d) - np.array(lo), np.array(hi) - np.array(d)],
                        fmt="o", color="#2c3e50", ecolor="#7f8c8d", capsize=3)
    axes[0, 1].axvline(0, color="black", lw=1)
    axes[0, 1].set_yticks(range(len(keys)))
    axes[0, 1].set_yticklabels(keys, fontsize=8)
    axes[0, 1].set_xlabel("配对 MAE 差（深度学习 − 该方法，kg）")
    axes[0, 1].set_title("配对差异（负值 = 深度学习更准）")

    axes[1, 0].scatter(evaluable["ref_kg"], evaluable["pred"], s=12, alpha=0.6,
                       c=np.where(evaluable["ref_gap_hours"] > 0, "#e67e22", "#2980b9"))
    lim = [30, 200]
    axes[1, 0].plot(lim, lim, ":", color="gray")
    axes[1, 0].set_xlim(lim), axes[1, 0].set_ylim(lim)
    axes[1, 0].set_xlabel("替代参考：床秤实测 (kg)"), axes[1, 0].set_ylabel("深度学习预测 (kg)")
    axes[1, 0].set_title("散点（橙 = 参考在拍片之后，蓝 = 之前）")

    axes[1, 1].hist(evaluable["ref_gap_hours"], bins=24, color="#8e44ad", alpha=0.8)
    axes[1, 1].axvline(0, color="black", lw=1)
    axes[1, 1].set_xlabel("参考实测相对拍片的时间（小时，负 = 更早）")
    axes[1, 1].set_ylabel("例数")
    axes[1, 1].set_title(f"替代参考的时间分布（中位 {res['ref_gap_hours_median']:.0f} h）")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "no_reliable_record_analysis.png", dpi=200)
    plt.close(fig)
    log.info("图 -> %s", FIGURE_DIR / "no_reliable_record_analysis.png")


if __name__ == "__main__":
    main()
