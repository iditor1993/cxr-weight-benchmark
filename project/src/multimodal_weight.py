"""多模态体重估计：床旁胸片 + 入院首日"人人可得"的表格数据。

问题：纯影像 MAE 8.15 kg，打不过护理录入的入院体重（4.58 kg）。
本分析检验：加入入院时几乎人人可得的少量变量（年龄、性别、首日生命体征、精简常规化验、
可选身高），能否在保持可得性的前提下把误差压下来，并给出不同"变量预算"下的组合结果。

组合（每档只加少量变量）
    S0  影像单独（pred）
    S1  + 年龄、性别（2 项）
    S2  + 首日生命体征 7 项（心率/收缩压/舒张压/平均压/呼吸/体温/SpO2 均值）
    S3  + 精简化验 9 项（肌酐/尿素氮/钠/钾/血糖/血红蛋白/白细胞/血小板/白蛋白）
    S4  + 身高（可选模块，仅 57–76% 可得）
    T3 纯表格（S3 去掉影像）——衡量影像的边际贡献

模型：HistGradientBoosting（原生处理缺失），与标记模型保持同一套超参数；
所有结果在患者级切分的同一测试集（n=1,303）上评价，配对 bootstrap 与影像单独比较。

产出
    project/results/tables/multimodal_weight.json
    project/results/figures/multimodal_weight.png
"""

from __future__ import annotations

import json

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from common import FIGURE_DIR, MAIN_TAG, PROCESSED_DIR, TABLE_DIR, load_config, save_json, setup_logger

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
SEED = 20261012

VITALS = ["heart_rate_mean", "sbp_mean", "dbp_mean", "mbp_mean", "resp_rate_mean",
          "temperature_mean", "spo2_mean"]
LABS = ["creatinine_max", "bun_max", "sodium_min", "potassium_max", "glucose_max",
        "hemoglobin_min", "wbc_max", "platelets_min", "albumin_min"]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {"n": int(len(y)), "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())), "bias": float(err.mean()),
            "hit_within_10pct": float((np.abs(err) / y <= 0.10).mean()),
            "hit_within_5kg": float((np.abs(err) <= 5).mean()),
            "hit_within_10kg": float((np.abs(err) <= 10).mean())}


def boot_mae_ci(y, p, subjects, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": y, "p": p, "g": subjects})
    groups = list(df.groupby("g", sort=False))
    v = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[i][1] for i in idx])
        v.append(np.abs(s["p"] - s["y"]).mean())
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def paired_diff(a, b, y, subjects, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": y, "a": a, "b": b, "g": subjects}).dropna()
    groups = list(df.groupby("g", sort=False))
    base = float(np.abs(df["a"] - df["y"]).mean() - np.abs(df["b"] - df["y"]).mean())
    v = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[i][1] for i in idx])
        v.append(np.abs(s["a"] - s["y"]).mean() - np.abs(s["b"] - s["y"]).mean())
    return base, float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def main() -> None:
    log = setup_logger("multimodal_weight")
    cfg = load_config()
    con = duckdb.connect(cfg["data"]["duckdb"], read_only=True)
    con.execute("PRAGMA threads=8")

    base = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    ens = pd.read_csv(PROCESSED_DIR / f"preds_all_{MAIN_TAG}.csv")[
        ["dicom_id", "pred", "pred_std"]]
    df = base.merge(splits, on="subject_id", how="left").merge(ens, on="dicom_id", how="left")

    p = str(PROCESSED_DIR / "cohort_a_index.csv").replace("\\", "/")
    vit = con.execute(
        f"SELECT stay_id, {', '.join(VITALS)} FROM mimiciv_derived.first_day_vitalsign "
        "WHERE stay_id IN (SELECT stay_id FROM read_csv_auto(?))", [p]).fetchdf()
    lab = con.execute(
        f"SELECT stay_id, {', '.join(LABS)} FROM mimiciv_derived.first_day_lab "
        "WHERE stay_id IN (SELECT stay_id FROM read_csv_auto(?))", [p]).fetchdf()
    hgt = con.execute(
        "SELECT stay_id, avg(height) AS height_cm FROM mimiciv_derived.height "
        "WHERE stay_id IN (SELECT stay_id FROM read_csv_auto(?)) GROUP BY 1", [p]).fetchdf()
    sex = con.execute("SELECT subject_id, gender, anchor_age FROM main.patients").fetchdf()
    df = (df.merge(vit, on="stay_id", how="left").merge(lab, on="stay_id", how="left")
            .merge(hgt, on="stay_id", how="left").merge(sex, on="subject_id", how="left"))
    df["male"] = (df["gender"] == "M").astype(float)
    df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce")

    log.info("队列 %d 行；测试集 %d", len(df), int((df["split_random"] == "test").sum()))
    tr = df["split_random"] == "train"
    va = df["split_random"] == "val"
    te = df["split_random"] == "test"

    sets = {
        "S0 影像单独": ["pred"],
        "S1 +年龄性别": ["pred", "anchor_age", "male"],
        "S2 +生命体征": ["pred", "anchor_age", "male"] + VITALS,
        "S3 +精简化验": ["pred", "anchor_age", "male"] + VITALS + LABS,
        "S4 +身高(可选)": ["pred", "anchor_age", "male"] + VITALS + LABS + ["height_cm"],
        "T3 纯表格(无影像)": ["anchor_age", "male"] + VITALS + LABS,
    }

    y_tr, y_te = df.loc[tr, "weight_kg"].values, df.loc[te, "weight_kg"].values
    subj_te = df.loc[te, "subject_id"].values
    res = {"n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
           "availability_test": {}, "models": {}, "paired_vs_image_only": {},
           "strata": {}, "notes": []}

    for c in ["anchor_age", "male"] + VITALS + LABS + ["height_cm"]:
        res["availability_test"][c] = float(df.loc[te, c].notna().mean())
    res["availability_test"]["pred"] = float(df.loc[te, "pred"].notna().mean())

    preds_te = {}
    for tag, cols in sets.items():
        gb = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_depth=4, min_samples_leaf=30,
            l2_regularization=1.0, random_state=SEED)
        gb.fit(df.loc[tr, cols], y_tr)
        p_te = gb.predict(df.loc[te, cols])
        preds_te[tag] = p_te
        met = metrics(y_te, p_te)
        met["mae_ci95"] = boot_mae_ci(y_te, p_te, subj_te)
        res["models"][tag] = met
        log.info("%-18s n_vars=%2d MAE=%.2f [%.2f,%.2f] ±10%%=%.1f%% ±5kg=%.1f%%",
                 tag, len(cols), met["mae"], met["mae_ci95"][0], met["mae_ci95"][1],
                 100 * met["hit_within_10pct"], 100 * met["hit_within_5kg"])

    img_only = df.loc[te, "pred"].values
    for tag, p_te in preds_te.items():
        d, lo, hi = paired_diff(p_te, img_only, y_te, subj_te)
        res["paired_vs_image_only"][tag] = {"diff": d, "ci95": [lo, hi]}
        if tag != "S0 影像单独":
            log.info("  配对（%s − 影像单独）= %+.2f kg [%.2f, %.2f]", tag, d, lo, hi)

    # 分体重层：看增益来自哪里
    strata = pd.cut(df.loc[te, "weight_kg"], [0, 50, 80, 100, 400],
                    labels=["<50", "50-80", "80-100", "≥100"])
    for tag in ["S0 影像单独", "S3 +精简化验", "S4 +身高(可选)"]:
        res["strata"][tag] = {
            str(s): metrics(y_te[strata == s], preds_te[tag][(strata == s).values])
            for s in strata.cat.categories}
    log.info("分层 MAE（<50/50-80/80-100/≥100）：影像 %s | S3 %s | S4 %s",
             [round(v["mae"], 1) for v in res["strata"]["S0 影像单独"].values()],
             [round(v["mae"], 1) for v in res["strata"]["S3 +精简化验"].values()],
             [round(v["mae"], 1) for v in res["strata"]["S4 +身高(可选)"].values()])

    # 影像最不确定的亚组里，表格数据是否有帮助
    ps_te = df.loc[te, "pred_std"].values
    q = pd.Series(pd.qcut(ps_te, 4, labels=["Q1 最确定", "Q2", "Q3", "Q4 最不确定"]))
    res["by_image_uncertainty"] = {}
    for s in q.cat.categories:
        m = (q == s).values
        res["by_image_uncertainty"][str(s)] = {
            "n": int(m.sum()),
            "image_only": metrics(y_te[m], img_only[m]),
            "S3": metrics(y_te[m], preds_te["S3 +精简化验"][m]),
            "S4": metrics(y_te[m], preds_te["S4 +身高(可选)"][m])}
    log.info("按影像不确定性四分位：Q4（最不确定）影像 MAE=%.2f，S3=%.2f，S4=%.2f",
             res["by_image_uncertainty"]["Q4 最不确定"]["image_only"]["mae"],
             res["by_image_uncertainty"]["Q4 最不确定"]["S3"]["mae"],
             res["by_image_uncertainty"]["Q4 最不确定"]["S4"]["mae"])

    # 时间可得性敏感性：索引影像在入 ICU 24 h 之后的病例
    sd = pd.to_datetime(df["study_datetime"])
    intime = con.execute(
        "SELECT stay_id, intime FROM main.icustays WHERE stay_id IN "
        "(SELECT stay_id FROM read_csv_auto(?))", [p]).fetchdf()
    df2 = df.merge(intime, on="stay_id", how="left")
    gap_h = (pd.to_datetime(df2["study_datetime"]) - pd.to_datetime(df2["intime"])).dt.total_seconds() / 3600
    late = (df2["split_random"] == "test") & (gap_h >= 24)
    res["sensitivity_late_images"] = {
        "n": int(late.sum()),
        "share_of_test": float(late.sum() / te.sum()),
        "image_only": metrics(df2.loc[late, "weight_kg"].values, df2.loc[late, "pred"].values),
        "S3": metrics(df2.loc[late, "weight_kg"].values,
                      preds_te["S3 +精简化验"][late[te.values].values])}
    log.info("敏感性（影像在入 ICU ≥24 h，n=%d）：影像 MAE=%.2f，S3 MAE=%.2f",
             res["sensitivity_late_images"]["n"],
             res["sensitivity_late_images"]["image_only"]["mae"],
             res["sensitivity_late_images"]["S3"]["mae"])

    # 与护理记录 / 常规方法的对照（同一测试集）
    conv = pd.read_csv(PROCESSED_DIR / "conventional_weight_comparison_table.csv")
    conv = conv.merge(df.loc[te, ["stay_id"]].assign(_t=1), on="stay_id", how="inner")
    conv = conv[conv["admission_weight_kg"].notna()]
    cl = conv["stay_id"].isin(df.loc[te, "stay_id"]).values
    if cl.sum() > 100:
        yc = conv.loc[cl, "weight_kg"].values
        res["comparator_nurse_record"] = metrics(yc, conv.loc[cl, "admission_weight_kg"].values)
        log.info("（对照）护理记录的入院体重在同一测试集 MAE=%.2f（n=%d）",
                 res["comparator_nurse_record"]["mae"], res["comparator_nurse_record"]["n"])
    con.close()

    save_json(TABLE_DIR / "multimodal_weight.json", res)

    # ---------- 图 ----------
    tags = list(sets.keys())
    names = [f"{t}\n({len(sets[t])} 个变量)" for t in tags]
    maes = [res["models"][t]["mae"] for t in tags]
    los = [res["models"][t]["mae_ci95"][0] for t in tags]
    his = [res["models"][t]["mae_ci95"][1] for t in tags]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    axes[0].bar(range(len(tags)), maes, color=["#c0392b", "#e67e22", "#f1c40f", "#27ae60",
                                               "#2980b9", "#7f8c8d"][:len(tags)])
    axes[0].errorbar(range(len(tags)), maes,
                     yerr=[np.array(maes) - np.array(los), np.array(his) - np.array(maes)],
                     fmt="none", ecolor="black", capsize=4)
    if "comparator_nurse_record" in res:
        axes[0].axhline(res["comparator_nurse_record"]["mae"], color="green", ls=":",
                        label=f"护理录入入院体重（{res['comparator_nurse_record']['mae']:.2f} kg）")
        axes[0].legend(fontsize=8)
    axes[0].set_xticks(range(len(tags)))
    axes[0].set_xticklabels(names, fontsize=7.5)
    axes[0].set_ylabel("测试集 MAE (kg)")
    axes[0].set_title("影像 + 入院常规数据的组合结果（n=1,303）")
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.12, f"{v:.2f}", ha="center", fontsize=8)

    keys = [t for t in tags if t != "S0 影像单独"]
    d = [res["paired_vs_image_only"][k]["diff"] for k in keys]
    lo = [res["paired_vs_image_only"][k]["ci95"][0] for k in keys]
    hi = [res["paired_vs_image_only"][k]["ci95"][1] for k in keys]
    axes[1].errorbar(d, range(len(keys)),
                     xerr=[np.array(d) - np.array(lo), np.array(hi) - np.array(d)],
                     fmt="o", color="#2c3e50", ecolor="#7f8c8d", capsize=3)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_yticks(range(len(keys)))
    axes[1].set_yticklabels(keys, fontsize=8)
    axes[1].set_xlabel("配对 MAE 差（组合 − 影像单独，kg）")
    axes[1].set_title("相对纯影像的改善（负值 = 加入表格数据更好）")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "multimodal_weight.png", dpi=200)
    plt.close(fig)
    log.info("图 -> %s", FIGURE_DIR / "multimodal_weight.png")


if __name__ == "__main__":
    main()
