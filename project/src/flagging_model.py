"""用影像预测 + 不确定性构建"入院体重记录不可靠"的前瞻式标记模型。

标签：|入院体重 − 床秤实测| ≥ 5 kg（仅用 admissions 记录与影像，不使用实测体重）
特征消融：
  R  = 记录类特征（入院体重、既往记录差、身高、性别、年龄）
  R+P= +影像点估计（集成预测、异方差 mu）
  R+U= +不确定性（3 种子方差、异方差 sigma）
  R+P+U = 全部
模型：Logistic 回归 与 HistGradientBoosting；患者级切分；测试集 AUROC/AUPRC + 聚类 bootstrap CI + 工作点
"""

from __future__ import annotations

import json

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, load_config, save_json, setup_logger

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def bootstrap_auroc(y, s, subjects, n_boot=500, seed=20261012):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": y, "s": s, "g": subjects})
    groups = list(df.groupby("g", sort=False))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        sub = pd.concat([groups[i][1] for i in idx])
        if sub["y"].nunique() < 2:
            continue
        vals.append(roc_auc_score(sub["y"], sub["s"]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def operating_points(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    out = {}
    for target in (0.80, 0.90):
        i = np.where(tpr >= target)[0]
        if len(i):
            j = i[0]
            flagged = s >= thr[j]
            tp = int(((y == 1) & flagged).sum())
            fp = int(((y == 0) & flagged).sum())
            out[f"sens{int(target*100)}"] = {
                "threshold": float(thr[j]), "sensitivity": float(tpr[j]),
                "specificity": float(1 - fpr[j]), "referral_rate": float(flagged.mean()),
                "ppv": float(tp / max(tp + fp, 1)),
                "n_needed_to_reweigh_per_detected": float((tp + fp) / max(tp, 1))}
    for target in (0.80,):
        i = np.where((1 - fpr) >= target)[0]
        if len(i):
            j = i[np.argmax(tpr[i])]
            flagged = s >= thr[j]
            tp = int(((y == 1) & flagged).sum())
            fp = int(((y == 0) & flagged).sum())
            out[f"spec{int(target*100)}"] = {
                "threshold": float(thr[j]), "sensitivity": float(tpr[j]),
                "specificity": float(1 - fpr[j]), "referral_rate": float(flagged.mean()),
                "ppv": float(tp / max(tp + fp, 1)),
                "n_needed_to_reweigh_per_detected": float((tp + fp) / max(tp, 1))}
    return out


def main() -> None:
    log = setup_logger("flagging_model")
    cfg = load_config()
    con = duckdb.connect(cfg["data"]["duckdb"], read_only=True)
    con.execute("PRAGMA threads=8")

    base = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    base = base.merge(splits, on="subject_id", how="left")
    ens = pd.read_csv(PROCESSED_DIR / "preds_all_swin_ensemble.csv")[["dicom_id", "pred", "pred_std"]]
    het = pd.read_csv(PROCESSED_DIR / "preds_index_hetero.csv")[["dicom_id", "hetero_mu", "hetero_sigma"]]
    df = base.merge(ens, on="dicom_id", how="left").merge(het, on="dicom_id", how="left")

    chart = con.execute("""
        WITH w AS (
            SELECT subject_id, stay_id, charttime, itemid,
                   CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END AS kg
            FROM main.chartevents
            WHERE itemid IN (224639, 226512, 226531) AND valuenum IS NOT NULL)
        SELECT s.stay_id,
               (SELECT avg(w.kg) FROM w WHERE w.stay_id = s.stay_id
                  AND w.itemid IN (226512,226531) AND w.kg BETWEEN 20 AND 350) AS adm,
               (SELECT w.kg FROM w WHERE w.subject_id = s.subject_id AND w.charttime < s.intime
                  AND (w.stay_id IS NULL OR w.stay_id <> s.stay_id) AND w.kg BETWEEN 20 AND 350
                  ORDER BY w.charttime DESC LIMIT 1) AS prior,
               (SELECT date_diff('day', w.charttime, s.intime) FROM w
                  WHERE w.subject_id = s.subject_id AND w.charttime < s.intime
                  AND (w.stay_id IS NULL OR w.stay_id <> s.stay_id) AND w.kg BETWEEN 20 AND 350
                  ORDER BY w.charttime DESC LIMIT 1) AS prior_gap_days
        FROM main.icustays s
        WHERE s.stay_id IN (SELECT stay_id FROM read_csv_auto(?))
    """, [str(PROCESSED_DIR / "cohort_a_index.csv").replace("\\", "/")]).fetchdf()
    height = con.execute("SELECT stay_id, avg(height) AS height_icu FROM mimiciv_derived.height GROUP BY 1").fetchdf()
    sex = con.execute("SELECT subject_id, gender, anchor_age FROM main.patients").fetchdf()
    df = df.merge(chart, on="stay_id", how="left").merge(height, on="stay_id", how="left").merge(sex, on="subject_id", how="left")

    df["label"] = (np.abs(df["adm"] - df["weight_kg"]) >= 5).astype(float)
    df.loc[df["adm"].isna(), "label"] = np.nan
    df["d_dl_adm"] = (df["pred"] - df["adm"]).abs()
    df["d_het_adm"] = (df["hetero_mu"] - df["adm"]).abs()
    df["d_prior_adm"] = (df["prior"] - df["adm"]).abs()
    df["ratio_dl"] = df["d_dl_adm"] / df["adm"]
    df["ratio_sigma"] = df["hetero_sigma"] / df["adm"]
    df["adm_dev_mean"] = (df["adm"] - 83.8).abs()
    df["male"] = (df["gender"] == "M").astype(float)
    df["has_prior"] = df["prior"].notna().astype(float)
    df["has_height"] = df["height_icu"].notna().astype(float)

    feats_R = ["adm", "d_prior_adm", "has_prior", "prior_gap_days", "height_icu", "has_height",
               "male", "anchor_age", "adm_dev_mean"]
    feats_P = ["pred", "hetero_mu", "d_dl_adm", "d_het_adm", "ratio_dl"]
    feats_U = ["pred_std", "hetero_sigma", "ratio_sigma"]
    sets = {"R": feats_R, "R+P": feats_R + feats_P, "R+U": feats_R + feats_U,
            "R+P+U": feats_R + feats_P + feats_U}

    tr = (df["split_random"] == "train") & df["label"].notna()
    va = (df["split_random"] == "val") & df["label"].notna()
    te = (df["split_random"] == "test") & df["label"].notna()
    log.info("train=%d val=%d test=%d prevalence(test)=%.3f", tr.sum(), va.sum(), te.sum(),
             df.loc[te, "label"].mean())
    ytr, yva, yte = df.loc[tr, "label"].values, df.loc[va, "label"].values, df.loc[te, "label"].values

    results = {"n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
               "prevalence_test": float(yte.mean()), "models": {}}

    # 单变量基线
    base_score = df.loc[te, "d_dl_adm"].values
    results["models"]["|DL−入院体重|（单变量基线）"] = {
        "auroc": float(roc_auc_score(yte, base_score)),
        "auprc": float(average_precision_score(yte, base_score)),
        "auroc_ci95": bootstrap_auroc(yte, base_score, df.loc[te, "subject_id"].values),
        "operating_points": operating_points(yte, base_score)}

    for tag, cols in sets.items():
        Xtr, Xva, Xte = df.loc[tr, cols], df.loc[va, cols], df.loc[te, cols]
        # 逻辑回归（标准化 + 中位数填补）
        med = Xtr.median()
        lr = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, C=1.0))])
        lr.fit(Xtr.fillna(med), ytr)
        p_va = lr.predict_proba(Xva.fillna(med))[:, 1]
        p_te = lr.predict_proba(Xte.fillna(med))[:, 1]
        # 梯度提升（原生处理缺失）
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                            l2_regularization=1.0, random_state=20261012)
        gb.fit(Xtr, ytr)
        g_va = gb.predict_proba(Xva)[:, 1]
        g_te = gb.predict_proba(Xte)[:, 1]
        results["models"][f"LR {tag}"] = {
            "auroc": float(roc_auc_score(yte, p_te)),
            "auprc": float(average_precision_score(yte, p_te)),
            "auroc_ci95": bootstrap_auroc(yte, p_te, df.loc[te, "subject_id"].values),
            "brier": float(brier_score_loss(yte, p_te)),
            "val_auroc": float(roc_auc_score(yva, p_va)),
            "operating_points": operating_points(yte, p_te)}
        results["models"][f"GB {tag}"] = {
            "auroc": float(roc_auc_score(yte, g_te)),
            "auprc": float(average_precision_score(yte, g_te)),
            "auroc_ci95": bootstrap_auroc(yte, g_te, df.loc[te, "subject_id"].values),
            "brier": float(brier_score_loss(yte, g_te)),
            "val_auroc": float(roc_auc_score(yva, g_va)),
            "operating_points": operating_points(yte, g_te)}
        log.info("%s: LR AUROC=%.3f (val %.3f) | GB AUROC=%.3f (val %.3f)",
                 tag, results["models"][f"LR {tag}"]["auroc"], results["models"][f"LR {tag}"]["val_auroc"],
                 results["models"][f"GB {tag}"]["auroc"], results["models"][f"GB {tag}"]["val_auroc"])

    # 特征重要性（GB，排列重要性）
    from sklearn.inspection import permutation_importance
    Xtr, Xte = df.loc[tr, sets["R+P+U"]], df.loc[te, sets["R+P+U"]]
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                        l2_regularization=1.0, random_state=20261012)
    gb.fit(Xtr, ytr)
    imp = permutation_importance(gb, Xte, yte, n_repeats=10, random_state=20261012,
                                 scoring="roc_auc")
    results["permutation_importance"] = sorted(
        [{"feature": f, "importance": float(m)} for f, m in zip(sets["R+P+U"], imp.importances_mean)],
        key=lambda r: -r["importance"])
    save_json(TABLE_DIR / "flagging_model.json", results)
    for r in results["permutation_importance"][:8]:
        log.info("importance: %-16s %.4f", r["feature"], r["importance"])

    best = max(results["models"].items(), key=lambda kv: kv[1]["auroc"])
    log.info("best model: %s AUROC=%.3f CI=[%.3f,%.3f]", best[0], best[1]["auroc"],
             best[1]["auroc_ci95"][0], best[1]["auroc_ci95"][1])
    for k, v in best[1]["operating_points"].items():
        log.info("  %s: sens=%.2f spec=%.2f referral=%.2f PPV=%.2f 每发现1例坏记录需复称%.1f例",
                 k, v["sensitivity"], v["specificity"], v["referral_rate"], v["ppv"],
                 v["n_needed_to_reweigh_per_detected"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 重新计算曲线（保存分数）
    scores = {}
    for tag, cols in sets.items():
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                            l2_regularization=1.0, random_state=20261012)
        gb.fit(df.loc[tr, cols], ytr)
        scores[tag] = gb.predict_proba(df.loc[te, cols])[:, 1]
    fpr, tpr, _ = roc_curve(yte, base_score)
    axes[0].plot(fpr, tpr, "--", label=f"单变量 |DL−入院| (AUROC={results['models']['|DL−入院体重|（单变量基线）']['auroc']:.3f})")
    for tag, s in scores.items():
        fpr, tpr, _ = roc_curve(yte, s)
        axes[0].plot(fpr, tpr, label=f"GB {tag} (AUROC={roc_auc_score(yte, s):.3f})")
    axes[0].plot([0, 1], [0, 1], ":", color="gray")
    axes[0].set_xlabel("1 − 特异度"); axes[0].set_ylabel("敏感度")
    axes[0].set_title("ROC：标记不可靠入院体重记录"); axes[0].legend(fontsize=8)
    prec, rec_curve, _ = precision_recall_curve(yte, scores["R+P+U"])
    axes[1].plot(rec_curve, prec, label="GB R+P+U")
    axes[1].axhline(yte.mean(), color="gray", ls=":", label=f"患病率 {yte.mean():.2f}")
    axes[1].set_xlabel("召回率"); axes[1].set_ylabel("精确率")
    axes[1].set_title("PR 曲线"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "flagging_model.png", dpi=200)
    plt.close(fig)
    log.info("figure -> %s", FIGURE_DIR / "flagging_model.png")


if __name__ == "__main__":
    main()
