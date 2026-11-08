"""v2：前瞻式"入院体重记录不可靠"标记模型（改进版）。

相对 v1（flagging_model.py）的改动
  1) 特征：新增记录合理性/重复性特征（整数与 5 kg 取整、与 Devine IBW 的一致性、
     身高换算 BMI 偏离、与既往记录"原样抄录"检测、OMR 门诊体重、入院到入 ICU 时长、
     影像预测的 z 分数（用 3 种子离散度与异方差 sigma 归一化）等。
  2) 模型：LR / HistGB / ExtraTrees / 三者秩平均集成 + 校验集 isotonic 校准。
  3) 评价：AUROC/AUPRC/Brier、患者级聚类 bootstrap CI、工作点（敏感度/特异度）、
     决策曲线净收益、以及"触发复称"闸门式混合估计的 MAE 曲线。
  4) 标签敏感性：主要标签 |入院体重 − 床秤实测| ≥ 5 kg，次要标签 ≥ 10 kg。

所有特征均为"入 ICU 当时可得"，不使用床秤实测体重本身，也不使用同次住院的
后续每日体重（避免标签泄漏）。
"""

from __future__ import annotations

import json

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import FIGURE_DIR, MAIN_TAG, PROCESSED_DIR, TABLE_DIR, load_config, save_json, setup_logger

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261012


def devine_ibw(sex_male: np.ndarray, height_cm: np.ndarray) -> np.ndarray:
    inches = np.asarray(height_cm, dtype=float) / 2.54
    base = np.where(np.asarray(sex_male) == 1, 50.0, 45.5)
    return base + 2.3 * (inches - 60.0)


def subject_bootstrap_ci(y, score, subjects, n_boot=500, seed=SEED):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(score), "g": np.asarray(subjects)})
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
            out[f"sens{int(target * 100)}"] = {
                "threshold": float(thr[j]), "sensitivity": float(tpr[j]),
                "specificity": float(1 - fpr[j]), "referral_rate": float(flagged.mean()),
                "ppv": float(tp / max(tp + fp, 1)),
                "n_needed_to_reweigh_per_detected": float((tp + fp) / max(tp, 1))}
    i = np.where((1 - fpr) >= 0.80)[0]
    if len(i):
        j = i[np.argmax(tpr[i])]
        flagged = s >= thr[j]
        tp = int(((y == 1) & flagged).sum())
        fp = int(((y == 0) & flagged).sum())
        out["spec80"] = {
            "threshold": float(thr[j]), "sensitivity": float(tpr[j]),
            "specificity": float(1 - fpr[j]), "referral_rate": float(flagged.mean()),
            "ppv": float(tp / max(tp + fp, 1)),
            "n_needed_to_reweigh_per_detected": float((tp + fp) / max(tp, 1))}
    return out


def net_benefit(y, p, thresholds):
    n = len(y)
    prev = y.mean()
    out = {}
    for t in thresholds:
        flagged = p >= t
        tp = float(((y == 1) & flagged).sum())
        fp = float(((y == 0) & flagged).sum())
        out[round(float(t), 3)] = float(tp / n - fp / n * t / (1 - t))
    return out


def paired_mae_diff(y_true, a, b, subjects, n_boot=1000, seed=SEED):
    """MAE(a) − MAE(b) 的患者级聚类 bootstrap CI。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": y_true, "a": a, "b": b, "g": subjects})
    groups = list(df.groupby("g", sort=False))
    base = float(np.abs(df["a"] - df["y"]).mean() - np.abs(df["b"] - df["y"]).mean())
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        sub = pd.concat([groups[i][1] for i in idx])
        vals.append(float(np.abs(sub["a"] - sub["y"]).mean() - np.abs(sub["b"] - sub["y"]).mean()))
    return base, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def build_frame(log):
    cfg = load_config()
    con = duckdb.connect(cfg["data"]["duckdb"], read_only=True)
    con.execute("PRAGMA threads=8")

    base = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    base = base.merge(splits, on="subject_id", how="left")
    ens = pd.read_csv(PROCESSED_DIR / f"preds_all_{MAIN_TAG}.csv")[["dicom_id", "pred", "pred_std"]]
    het = pd.read_csv(PROCESSED_DIR / "preds_index_hetero.csv")[["dicom_id", "hetero_mu", "hetero_sigma"]]
    df = base.merge(ens, on="dicom_id", how="left").merge(het, on="dicom_id", how="left")

    idx_csv = str(PROCESSED_DIR / "cohort_a_index.csv").replace("\\", "/")
    chart = con.execute("""
        WITH s AS (
            SELECT subject_id, stay_id, intime FROM main.icustays
            WHERE stay_id IN (SELECT stay_id FROM read_csv_auto(?))),
        w AS (
            SELECT subject_id, stay_id, charttime, itemid,
                   CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END AS kg
            FROM main.chartevents
            WHERE itemid IN (224639, 226512, 226531) AND valuenum IS NOT NULL),
        adm AS (
            SELECT s.stay_id, avg(w.kg) AS adm
            FROM s JOIN w ON w.stay_id = s.stay_id AND w.itemid IN (226512, 226531)
            WHERE w.kg BETWEEN 20 AND 350 GROUP BY 1),
        pri AS (
            SELECT s.stay_id, w.kg, w.charttime,
                   date_diff('day', w.charttime, s.intime) AS gap_days,
                   row_number() OVER (PARTITION BY s.stay_id ORDER BY w.charttime DESC) AS rn,
                   count(1) OVER (PARTITION BY s.stay_id) AS n_prior
            FROM s JOIN w ON w.subject_id = s.subject_id
              AND w.charttime < s.intime
              AND (w.stay_id IS NULL OR w.stay_id <> s.stay_id)
            WHERE w.kg BETWEEN 20 AND 350)
        SELECT adm.stay_id, adm.adm,
               pri.kg AS prior, pri.gap_days AS prior_gap_days, pri.n_prior AS prior_n
        FROM adm LEFT JOIN pri ON pri.stay_id = adm.stay_id AND pri.rn = 1
    """, [idx_csv]).fetchdf()

    omr = con.execute("""
        WITH s AS (
            SELECT subject_id, stay_id, intime FROM main.icustays
            WHERE stay_id IN (SELECT stay_id FROM read_csv_auto(?))),
        o AS (
            SELECT s.stay_id, lower(o.result_name) AS nm,
                   try_cast(o.result_value AS DOUBLE) AS val,
                   date_diff('day', CAST(o.chartdate AS TIMESTAMP), s.intime) AS gap_days,
                   row_number() OVER (PARTITION BY s.stay_id, lower(o.result_name)
                                      ORDER BY o.chartdate DESC) AS rn
            FROM s JOIN main.omr o ON o.subject_id = s.subject_id
            WHERE CAST(o.chartdate AS TIMESTAMP) < s.intime
              AND (lower(o.result_name) LIKE '%weight%' OR lower(o.result_name) LIKE '%bmi%'))
        SELECT stay_id,
               max(CASE WHEN nm LIKE '%weight%' AND rn = 1 AND val BETWEEN 20 AND 350
                        THEN val END) AS omr_weight,
               max(CASE WHEN nm LIKE '%weight%' AND rn = 1 AND val BETWEEN 20 AND 350
                        THEN gap_days END) AS omr_gap_days,
               max(CASE WHEN nm LIKE '%bmi%' AND rn = 1 AND val BETWEEN 10 AND 80
                        THEN val END) AS omr_bmi
        FROM o GROUP BY 1
    """, [idx_csv]).fetchdf()

    height = con.execute(
        "SELECT stay_id, avg(height) AS height_icu FROM mimiciv_derived.height GROUP BY 1").fetchdf()
    sex = con.execute("SELECT subject_id, gender, anchor_age FROM main.patients").fetchdf()
    losd = con.execute("""
        SELECT s.stay_id,
               date_diff('hour', a.admittime, s.intime) / 24.0 AS hosp_days_before_icu
        FROM main.icustays s JOIN main.admissions a ON a.hadm_id = s.hadm_id
        WHERE s.stay_id IN (SELECT stay_id FROM read_csv_auto(?))
    """, [idx_csv]).fetchdf()
    con.close()

    df = (df.merge(chart, on="stay_id", how="left").merge(omr, on="stay_id", how="left")
            .merge(height, on="stay_id", how="left").merge(sex, on="subject_id", how="left")
            .merge(losd, on="stay_id", how="left"))
    log.info("frame: %d rows, adm 可用 %d, 既往 %d, 门诊 %d, 身高 %d", len(df),
             df["adm"].notna().sum(), df["prior"].notna().sum(),
             df["omr_weight"].notna().sum(), df["height_icu"].notna().sum())
    return df


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["label5"] = (np.abs(d["adm"] - d["weight_kg"]) >= 5).astype(float)
    d["label10"] = (np.abs(d["adm"] - d["weight_kg"]) >= 10).astype(float)
    for col in ("label5", "label10"):
        d.loc[d["adm"].isna(), col] = np.nan

    h = d["height_icu"]
    male = (d["gender"] == "M").astype(float)
    d["male"] = male
    d["has_height"] = h.notna().astype(float)
    d["has_prior"] = d["prior"].notna().astype(float)
    d["has_omr"] = d["omr_weight"].notna().astype(float)

    # —— 记录本身的合理性（入 ICU 当时可得）——
    d["adm_dev_mean"] = (d["adm"] - 83.8).abs()
    d["adm_round5"] = (np.mod(d["adm"], 5) < 1e-6).astype(float)
    d["adm_is_int"] = (np.mod(d["adm"], 1) < 1e-6).astype(float)
    d["adm_frac"] = d["adm"] - np.floor(d["adm"])
    d["bmi_adm"] = d["adm"] / (h / 100.0) ** 2
    d["bmi_adm_dev"] = (d["bmi_adm"] - 27.0).abs()
    d["ibw_devine"] = devine_ibw(male.values, h.values)
    d["d_adm_ibw"] = d["adm"] - d["ibw_devine"]
    d["ratio_adm_ibw"] = d["adm"] / d["ibw_devine"]
    d["bmi_omr_dev"] = (d["omr_bmi"] - 27.0).abs()
    d["d_adm_omr"] = (d["adm"] - d["omr_weight"]).abs()
    d["d_omr_implied"] = (d["adm"] - d["omr_bmi"] * (h / 100.0) ** 2).abs()
    d["d_prior_adm"] = (d["adm"] - d["prior"]).abs()
    d["ratio_prior_adm"] = d["prior"] / d["adm"]
    d["adm_copied_prior"] = (d["d_prior_adm"] < 0.05).astype(float)
    d["icu_surgical"] = d["first_careunit"].str.contains("Surgical", na=False).astype(float)
    d["icu_cardiac"] = d["first_careunit"].str.contains("Cardiac", na=False).astype(float)
    d["icu_neuro"] = d["first_careunit"].str.contains("Neuro", na=False).astype(float)

    # —— 影像侧：点估计 + 归一化分歧度 ——
    d["d_dl_adm"] = (d["pred"] - d["adm"]).abs()
    d["diff_dl_adm"] = d["pred"] - d["adm"]
    d["d_het_adm"] = (d["hetero_mu"] - d["adm"]).abs()
    d["ratio_dl"] = d["d_dl_adm"] / d["adm"]
    d["d_bmi_dl_adm"] = d["d_dl_adm"] / (h / 100.0) ** 2
    d["z_seed"] = d["d_dl_adm"] / d["pred_std"].clip(lower=0.5)
    d["z_het"] = d["d_het_adm"] / d["hetero_sigma"].clip(lower=0.5)
    d["z_comb"] = d["d_dl_adm"] / np.sqrt(d["pred_std"] ** 2 + d["hetero_sigma"] ** 2)
    d["pred_std_rel"] = d["pred_std"] / d["pred"]
    d["ratio_sigma"] = d["hetero_sigma"] / d["adm"]
    d["sigma_mu_ratio"] = d["hetero_sigma"] / d["hetero_mu"]
    d["d_dl_prior"] = (d["pred"] - d["prior"]).abs()
    d["d_dl_omr"] = (d["pred"] - d["omr_weight"]).abs()
    d["prior_agrees_adm"] = (d["d_prior_adm"] < d["d_dl_adm"]).astype(float)
    return d


UNIT_FEATS = ["icu_surgical", "icu_cardiac", "icu_neuro"]


FEATS_OLD = ["adm", "d_prior_adm", "has_prior", "prior_gap_days", "height_icu", "has_height",
             "male", "anchor_age", "adm_dev_mean",
             "pred", "hetero_mu", "d_dl_adm", "d_het_adm", "ratio_dl",
             "pred_std", "hetero_sigma", "ratio_sigma"]

FEATS_NEW = FEATS_OLD + [
    "adm_round5", "adm_is_int", "adm_frac", "bmi_adm", "bmi_adm_dev", "ibw_devine",
    "d_adm_ibw", "ratio_adm_ibw", "has_omr", "omr_weight", "omr_gap_days", "omr_bmi",
    "bmi_omr_dev", "d_adm_omr", "d_omr_implied", "ratio_prior_adm", "adm_copied_prior",
    "prior_n", "hosp_days_before_icu", "icu_surgical", "icu_cardiac", "icu_neuro",
    "diff_dl_adm", "d_bmi_dl_adm", "z_seed", "z_het", "z_comb", "pred_std_rel",
    "sigma_mu_ratio", "d_dl_prior", "d_dl_omr", "prior_agrees_adm"]

FEATS_NO_UNIT = [c for c in FEATS_NEW if c not in UNIT_FEATS]


def fit_models(Xtr, ytr, Xva):
    med = Xtr.median()
    models = {}
    lr = Pipeline([("sc", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=3000, C=0.5))])
    lr.fit(Xtr.fillna(med), ytr)
    models["LR"] = (lr, med)
    gb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.04, max_depth=4, min_samples_leaf=30,
        l2_regularization=1.0, random_state=SEED)
    gb.fit(Xtr, ytr)
    models["GB"] = (gb, None)
    et = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5, max_features="sqrt",
                              n_jobs=-1, random_state=SEED)
    et.fit(Xtr.fillna(med), ytr)
    models["ET"] = (et, med)
    return models


def predict_all(models, X):
    p = {}
    for name, (m, med) in models.items():
        p[name] = m.predict_proba(X.fillna(med) if med is not None else X)[:, 1]
    p["ENS"] = np.mean([rankdata(p[k]) / (len(X) + 1) for k in ("LR", "GB", "ET")], axis=0)
    return p


def main() -> None:
    log = setup_logger("flagging_model_v2")
    df = featurize(build_frame(log))

    tr = (df["split_random"] == "train") & df["label5"].notna()
    va = (df["split_random"] == "val") & df["label5"].notna()
    te = (df["split_random"] == "test") & df["label5"].notna()
    log.info("n: train=%d val=%d test=%d | 测试集患病率(>=5kg)=%.3f (>=10kg)=%.3f",
             tr.sum(), va.sum(), te.sum(), df.loc[te, "label5"].mean(), df.loc[te, "label10"].mean())

    results = {"n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
               "prevalence_test_label5": float(df.loc[te, "label5"].mean()),
               "prevalence_test_label10": float(df.loc[te, "label10"].mean()),
               "models": {}, "label10_check": {}, "decision_analysis": {}, "net_benefit": {}}

    ytr, yva, yte = df.loc[tr, "label5"].values, df.loc[va, "label5"].values, df.loc[te, "label5"].values
    subj_te = df.loc[te, "subject_id"].values

    # 单变量基线：|DL − 入院体重|
    base_te = df.loc[te, "d_dl_adm"].values
    results["models"]["单变量 |DL−入院体重|"] = {
        "auroc": float(roc_auc_score(yte, base_te)),
        "auprc": float(average_precision_score(yte, base_te)),
        "auroc_ci95": subject_bootstrap_ci(yte, base_te, subj_te),
        "operating_points": operating_points(yte, base_te)}

    spec_sets = {"v1 特征集 (GB)": FEATS_OLD, "v2 全特征集": FEATS_NEW,
                 "v2 去病区代理": FEATS_NO_UNIT}
    scores_test = {}
    val_scores = {}
    for tag, cols in spec_sets.items():
        models = fit_models(df.loc[tr, cols], ytr, df.loc[va, cols])
        p_va = predict_all(models, df.loc[va, cols])
        p_te = predict_all(models, df.loc[te, cols])
        for name in p_va:
            key = f"{name} [{tag}]"
            val_auroc = float(roc_auc_score(yva, p_va[name]))
            results["models"][key] = {
                "auroc": float(roc_auc_score(yte, p_te[name])),
                "auprc": float(average_precision_score(yte, p_te[name])),
                "auroc_ci95": subject_bootstrap_ci(yte, p_te[name], subj_te),
                "brier": float(brier_score_loss(yte, p_te[name])),
                "val_auroc": val_auroc,
                "operating_points": operating_points(yte, p_te[name])}
            log.info("%-28s AUROC=%.3f (val %.3f) AUPRC=%.3f", key,
                     results["models"][key]["auroc"], val_auroc, results["models"][key]["auprc"])
            if tag == "v2 全特征集":
                scores_test[name] = p_te[name]
        val_scores[tag] = p_va

    best_key = max((k for k in results["models"] if "v2 全特征集" in k),
                   key=lambda k: results["models"][k]["val_auroc"])
    log.info("按校验集 AUROC 选出的最佳模型：%s", best_key)
    results["selected_model"] = best_key

    # —— 病区代理检查：标签患病率是否被病区身份主导 ——
    unit_stats = {}
    for u in UNIT_FEATS:
        m = df.loc[te, u] == 1
        unit_stats[u] = {"n_test": int(m.sum()),
                         "prevalence": float(df.loc[te][m]["label5"].mean()) if m.sum() else None}
    unit_stats["overall"] = {"n_test": int(len(df.loc[te])),
                             "prevalence": float(df.loc[te, "label5"].mean())}
    if df.loc[te, "icu_cardiac"].nunique() > 1:
        unit_stats["auroc_icu_cardiac_alone"] = float(
            roc_auc_score(yte, df.loc[te, "icu_cardiac"].values))
    results["unit_proxy_check"] = unit_stats
    log.info("病区代理检查：%s", json.dumps(unit_stats, ensure_ascii=False))

    best_name = best_key.split(" [")[0]
    p_val = val_scores["v2 全特征集"][best_name]
    p_test = scores_test[best_name]
    p_base_test = df.loc[te, "d_dl_adm"].values
    p_base_val = df.loc[va, "d_dl_adm"].values

    # isotonic 校准（在校验集上拟合）
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val, yva)
    p_test_iso = iso.predict(p_test)
    p_val_iso = iso.predict(p_val)
    iso_base = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_base.fit(p_base_val, yva)
    p_base_test_iso = iso_base.predict(p_base_test)
    results["models"][f"{best_key} + isotonic 校准"] = {
        "auroc": float(roc_auc_score(yte, p_test_iso)),
        "auprc": float(average_precision_score(yte, p_test_iso)),
        "brier": float(brier_score_loss(yte, p_test_iso)),
        "auroc_ci95": subject_bootstrap_ci(yte, p_test_iso, subj_te),
        "operating_points": operating_points(yte, p_test_iso)}
    log.info("isotonic 校准后：AUROC=%.3f Brier=%.3f（未校准 Brier=%.3f）",
             results["models"][f"{best_key} + isotonic 校准"]["auroc"],
             results["models"][f"{best_key} + isotonic 校准"]["brier"],
             results["models"][best_key]["brier"])

    # 校准斜率 / 截距（logit 尺度）
    def calib_slope(y, p):
        eps = 1e-6
        z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
        lr = LogisticRegression(penalty=None, max_iter=2000).fit(z, y)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])

    results["calibration"] = {"before": dict(zip(("slope", "intercept"), calib_slope(yte, p_test))),
                             "after": dict(zip(("slope", "intercept"), calib_slope(yte, p_test_iso)))}

    # —— 排列重要性（ET，测试集，AUROC 下降）——
    et_model, et_med = fit_models(df.loc[tr, FEATS_NEW], ytr, df.loc[va, FEATS_NEW])["ET"]
    Xte_new = df.loc[te, FEATS_NEW]
    rng = np.random.default_rng(SEED)
    base_auc = roc_auc_score(yte, et_model.predict_proba(Xte_new.fillna(et_med))[:, 1])
    imp = []
    for col in FEATS_NEW:
        drops = []
        for _ in range(8):
            perm = Xte_new.copy()
            perm[col] = rng.permutation(perm[col].values)
            drops.append(base_auc - roc_auc_score(
                yte, et_model.predict_proba(perm.fillna(et_med))[:, 1]))
        imp.append({"feature": col, "auc_drop": float(np.mean(drops))})
    results["permutation_importance"] = sorted(imp, key=lambda r: -r["auc_drop"])
    log.info("重要性前 10：%s", ", ".join(
        f"{r['feature']}={r['auc_drop']:.3f}" for r in results["permutation_importance"][:10]))

    # —— 标签敏感性：|Δ|>=10 kg（同样特征集，重新训练）——
    ytr10, yva10, yte10 = (df.loc[tr, "label10"].values, df.loc[va, "label10"].values,
                           df.loc[te, "label10"].values)
    m10 = fit_models(df.loc[tr, FEATS_NEW], ytr10, df.loc[va, FEATS_NEW])
    p10_te = predict_all(m10, df.loc[te, FEATS_NEW])["ENS"]
    results["label10_check"] = {
        "prevalence_test": float(yte10.mean()),
        "auroc": float(roc_auc_score(yte10, p10_te)),
        "auprc": float(average_precision_score(yte10, p10_te)),
        "auroc_ci95": subject_bootstrap_ci(yte10, p10_te, subj_te)}
    log.info("次要标签 |Δ|>=10 kg：AUROC=%.3f AUPRC=%.3f（患病率 %.3f）",
             results["label10_check"]["auroc"], results["label10_check"]["auprc"],
             results["label10_check"]["prevalence_test"])

    # —— 决策分析：闸门式混合估计 ——
    adm_all, dl_all = df.loc[te, "adm"].values, df.loc[te, "pred"].values
    wt = df.loc[te, "weight_kg"].values
    adm_val, dl_val, wt_val = df.loc[va, "adm"].values, df.loc[va, "pred"].values, df.loc[va, "weight_kg"].values
    mae_adm = float(np.abs(adm_all - wt).mean())
    mae_dl = float(np.abs(dl_all - wt).mean())
    alpha = 0.958  # 训练集最优收缩权重（见 unreliable_subgroup_check.json）
    shrink = alpha * adm_all + (1 - alpha) * dl_all
    blend50 = 0.5 * adm_all + 0.5 * dl_all
    thr_grid = np.round(np.arange(0.05, 0.96, 0.01), 3)

    def gate_curve(p, adm_arr, dl_arr, wt_arr, blend_arr=None):
        mae, rate, mae_b = [], [], []
        for t in thr_grid:
            use_dl = p >= t
            mae.append(float(np.abs(np.where(use_dl, dl_arr, adm_arr) - wt_arr).mean()))
            rate.append(float(use_dl.mean()))
            if blend_arr is not None:
                mae_b.append(float(np.abs(np.where(use_dl, blend_arr, adm_arr) - wt_arr).mean()))
        return mae, rate, mae_b

    val_mae, val_rate, _ = gate_curve(p_val_iso, adm_val, dl_val, wt_val)
    b = int(np.argmin(val_mae))
    thr_star = float(thr_grid[b])
    te_mae, te_rate, te_mae_b = gate_curve(p_test_iso, adm_all, dl_all, wt, blend_arr=blend50)
    i_star = int(np.argmin(np.abs(thr_grid - thr_star)))
    mae_gate, rate_gate, mae_gate50 = te_mae[i_star], te_rate[i_star], te_mae_b[i_star]
    rate_gate50 = rate_gate
    d1, lo1, hi1 = paired_mae_diff(wt, np.where(p_test_iso >= thr_star, dl_all, adm_all), adm_all, subj_te)
    d2, lo2, hi2 = paired_mae_diff(wt, np.where(p_test_iso >= thr_star, blend50, adm_all), adm_all, subj_te)
    results["decision_analysis"] = {
        "mae_always_adm": mae_adm, "mae_always_dl": mae_dl,
        "mae_shrinkage_0.958": float(np.abs(shrink - wt).mean()),
        "mae_blend_0.5": float(np.abs(blend50 - wt).mean()),
        "threshold_selected_on_val": thr_star,
        "gate_dl_mae": mae_gate, "gate_dl_use_rate": rate_gate,
        "gate_dl_vs_adm_diff": d1, "gate_dl_vs_adm_ci95": [lo1, hi1],
        "gate_blend50_mae": mae_gate50, "gate_blend50_use_rate": rate_gate50,
        "gate_blend50_vs_adm_diff": d2, "gate_blend50_vs_adm_ci95": [lo2, hi2],
        "gate_curve_val": {"thresholds": [float(t) for t in thr_grid],
                           "mae": [float(v) for v in val_mae],
                           "use_rate": [float(v) for v in val_rate]},
        "gate_curve_test": {"thresholds": [float(t) for t in thr_grid],
                            "mae": [float(v) for v in te_mae],
                            "use_rate": [float(v) for v in te_rate]}}
    log.info("闸门估计：always-adm MAE=%.2f | always-DL=%.2f | 闸门(%.2f)=%.2f (触发率 %.0f%%)",
             mae_adm, mae_dl, thr_star, mae_gate, 100 * rate_gate)

    # —— 决策曲线净收益 ——
    nb_thr = np.round(np.arange(0.05, 0.81, 0.05), 2)
    results["net_benefit"] = {
        "thresholds": [float(t) for t in nb_thr],
        "model": net_benefit(yte, p_test_iso, nb_thr),
        "single_var": net_benefit(yte, p_base_test_iso, nb_thr),
        "admit_all_nb": {round(float(t), 3): float(yte.mean() - (1 - yte.mean()) * t / (1 - t))
                         for t in nb_thr}}

    save_json(TABLE_DIR / "flagging_model_v2.json", results)

    # —— 图 ——
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    order = ["单变量 |DL−入院体重|", f"LR [v2 全特征集]", f"GB [v2 全特征集]",
             f"ET [v2 全特征集]", f"ENS [v2 全特征集]", f"GB [v1 特征集 (GB)]"]
    for k in order:
        if k not in results["models"]:
            continue
        s = p_base_test if k.startswith("单变量") else (
            scores_test["GB"] if k.startswith("GB [v1") else scores_test[k.split(" [")[0]])
        fpr, tpr, _ = roc_curve(yte, s)
        axes[0, 0].plot(fpr, tpr, label=f"{k} ({results['models'][k]['auroc']:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], ":", color="gray")
    axes[0, 0].set_xlabel("1 − 特异度"), axes[0, 0].set_ylabel("敏感度")
    axes[0, 0].set_title("ROC：标记不可靠的入院体重记录")
    axes[0, 0].legend(fontsize=7)

    bins = np.quantile(p_test_iso, np.linspace(0, 1, 11))
    bins[-1] += 1e-9
    idx = np.clip(np.digitize(p_test_iso, bins) - 1, 0, 9)
    obs = [yte[idx == i].mean() if (idx == i).sum() else np.nan for i in range(10)]
    pred_bin = [p_test_iso[idx == i].mean() if (idx == i).sum() else np.nan for i in range(10)]
    axes[0, 1].plot([0, 1], [0, 1], ":", color="gray")
    axes[0, 1].plot(pred_bin, obs, "o-")
    axes[0, 1].set_xlabel("预测概率（十分位）"), axes[0, 1].set_ylabel("实际比例")
    axes[0, 1].set_title(f"校准曲线（isotonic 后，Brier={results['models'][f'{best_key} + isotonic 校准']['brier']:.3f}）")

    axes[1, 0].plot(nb_thr, [results["net_benefit"]["model"][round(float(t), 3)] for t in nb_thr],
                    "o-", label="标记模型")
    axes[1, 0].plot(nb_thr, [results["net_benefit"]["single_var"][round(float(t), 3)] for t in nb_thr],
                    "s--", label="|DL−入院体重|")
    axes[1, 0].plot(nb_thr, [results["net_benefit"]["admit_all_nb"][round(float(t), 3)] for t in nb_thr],
                    ":", label="全部按不可靠处理")
    axes[1, 0].axhline(0, color="black", lw=0.8)
    axes[1, 0].set_xlabel("阈值概率"), axes[1, 0].set_ylabel("净收益")
    axes[1, 0].set_title("决策曲线：是否触发复称"); axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(thr_grid, results["decision_analysis"]["gate_curve_test"]["mae"], "-",
                    label="测试集：闸门混合估计")
    axes[1, 1].plot(thr_grid, results["decision_analysis"]["gate_curve_val"]["mae"], "--",
                    label="校验集")
    axes[1, 1].axhline(mae_adm, color="green", ls=":", label=f"全部用入院体重（{mae_adm:.2f} kg）")
    axes[1, 1].axhline(mae_dl, color="red", ls=":", label=f"全部用深度学习（{mae_dl:.2f} kg）")
    axes[1, 1].axvline(thr_star, color="gray", lw=0.8)
    axes[1, 1].set_xlabel("触发复称的概率阈值"), axes[1, 1].set_ylabel("MAE (kg)")
    axes[1, 1].set_title("闸门式混合估计的 MAE 曲线"); axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "flagging_model_v2.png", dpi=200)
    plt.close(fig)
    log.info("图 -> %s", FIGURE_DIR / "flagging_model_v2.png")
    log.info("AUROC 排序：%s", json.dumps(
        {k: round(v["auroc"], 3) for k, v in
         sorted(results["models"].items(), key=lambda kv: -kv[1]["auroc"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
