"""早融合检验：Swin 影像特征向量 + 入院常规表格数据 → MLP。

目的：确认"加入生命体征/化验没有帮助"不是简单模型（HistGB + 点预测）造成的假阴性。
比较三种输入：
    E0 = 768 维影像特征（冻结骨干，全局池化）
    E1 = E0 + 年龄/性别/首日生命体征/精简化验
    E2 = E1 + 身高（含缺失指示）
每个配置训练 3 个种子，取校验集最优轮次，测试集报告 MAE；配对 bootstrap 比较 E1/E2 与 E0。

产出
    project/results/tables/multimodal_embed_mlp.json
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import FEATURE_DIR, PROCESSED_DIR, TABLE_DIR, load_config, save_json, setup_logger

SEED = 20261012
VITALS = ["heart_rate_mean", "sbp_mean", "dbp_mean", "mbp_mean", "resp_rate_mean",
          "temperature_mean", "spo2_mean"]
LABS = ["creatinine_max", "bun_max", "sodium_min", "potassium_max", "glucose_max",
        "hemoglobin_min", "wbc_max", "platelets_min", "albumin_min"]
TAB = ["anchor_age", "male"] + VITALS + LABS


def metrics(y, p):
    err = p - y
    return {"n": int(len(y)), "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())), "bias": float(err.mean()),
            "hit_within_10pct": float((np.abs(err) / y <= 0.10).mean()),
            "hit_within_5kg": float((np.abs(err) <= 5).mean())}


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


class MLP(nn.Module):
    def __init__(self, d_in, hidden=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_once(Xtr, ytr, Xva, yva, Xte, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.L1Loss()
    mu, sd = float(ytr.mean()), float(ytr.std())

    def t(a):
        return torch.tensor(a, dtype=torch.float32, device=dev)

    Xtr_t, ytr_t = t(Xtr), t((ytr - mu) / sd)
    Xva_t, Xte_t = t(Xva), t(Xte)
    n = len(Xtr_t)
    best, best_state, patience = 1e9, None, 0
    for epoch in range(80):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).cpu().numpy() * sd + mu
        mae = float(np.abs(pv - yva).mean())
        if mae < best - 1e-4:
            best, best_state, patience = mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 15:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pte = model(Xte_t).cpu().numpy() * sd + mu
    return pte, best


def main() -> None:
    log = setup_logger("multimodal_embed_mlp")
    cfg = load_config()
    import duckdb
    con = duckdb.connect(cfg["data"]["duckdb"], read_only=True)
    con.execute("PRAGMA threads=8")

    feats = pd.read_csv(PROCESSED_DIR / "swin_features_index.csv")
    emb = np.load(FEATURE_DIR / "swin_features_global.npy").astype(np.float32)
    assert len(feats) == len(emb), (len(feats), len(emb))
    feats = feats.reset_index(drop=True)

    base = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    bb = (base[["dicom_id", "subject_id", "stay_id", "weight_kg", "study_datetime"]]
          .rename(columns={"weight_kg": "weight_kg_base"}))
    df = (feats.drop(columns=["weight_kg"]).merge(bb, on="dicom_id", how="left")
          .rename(columns={"weight_kg_base": "weight_kg"}))
    df = df.merge(splits, on="subject_id", how="left")

    p = str(PROCESSED_DIR / "cohort_a_index.csv").replace("\\", "/")
    vit = con.execute(f"SELECT stay_id, {', '.join(VITALS)} FROM mimiciv_derived.first_day_vitalsign "
                      "WHERE stay_id IN (SELECT stay_id FROM read_csv_auto(?))", [p]).fetchdf()
    lab = con.execute(f"SELECT stay_id, {', '.join(LABS)} FROM mimiciv_derived.first_day_lab "
                      "WHERE stay_id IN (SELECT stay_id FROM read_csv_auto(?))", [p]).fetchdf()
    hgt = con.execute("SELECT stay_id, avg(height) AS height_cm FROM mimiciv_derived.height "
                      "WHERE stay_id IN (SELECT stay_id FROM read_csv_auto(?)) GROUP BY 1", [p]).fetchdf()
    sex = con.execute("SELECT subject_id, gender, anchor_age FROM main.patients").fetchdf()
    df = (df.merge(vit, on="stay_id", how="left").merge(lab, on="stay_id", how="left")
            .merge(hgt, on="stay_id", how="left").merge(sex, on="subject_id", how="left"))
    df["male"] = (df["gender"] == "M").astype(float)
    df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
    con.close()

    tr_m = (df["split_random"] == "train").values
    va_m = (df["split_random"] == "val").values
    te_m = (df["split_random"] == "test").values
    assert tr_m.sum() and va_m.sum() and te_m.sum()
    log.info("train/val/test = %d/%d/%d", tr_m.sum(), va_m.sum(), te_m.sum())

    def tab_matrix(cols, add_missing=False):
        X = df[cols].astype(float).copy()
        if add_missing:
            for c in cols:
                X[f"{c}_miss"] = df[c].isna().astype(float)
        med = X[tr_m].median()
        X = X.fillna(med)
        mu, sd = X[tr_m].mean(), X[tr_m].std().replace(0, 1)
        return ((X - mu) / sd).values.astype(np.float32)

    X_img = emb
    X_tab = tab_matrix(TAB)
    X_tab_h = tab_matrix(TAB + ["height_cm"], add_missing=True)
    y = df["weight_kg"].values.astype(np.float32)
    subj = df["subject_id"].values
    configs = {"E0 影像特征": [X_img], "E1 +常规表格": [X_img, X_tab],
               "E2 +表格+身高": [X_img, X_tab_h]}

    res = {"models": {}, "paired": {}}
    preds = {}
    for tag, mats in configs.items():
        X = np.concatenate(mats, axis=1)
        ps = []
        for s in (1, 2, 3):
            pte, vmae = train_once(X[tr_m], y[tr_m], X[va_m], y[va_m], X[te_m], SEED + s)
            ps.append(pte)
            log.info("%s seed%d：val MAE=%.2f，test MAE=%.2f", tag, s, vmae,
                     float(np.abs(pte - y[te_m]).mean()))
        p = np.mean(ps, axis=0)
        preds[tag] = p
        res["models"][tag] = metrics(y[te_m], p)
        res["models"][tag]["n_features"] = int(X.shape[1])
        res["models"][tag]["seed_maes"] = [float(np.abs(x - y[te_m]).mean()) for x in ps]
        log.info("%-14s 3 种子平均：MAE=%.2f（%d 维输入）", tag,
                 res["models"][tag]["mae"], X.shape[1])

    for tag in ("E1 +常规表格", "E2 +表格+身高"):
        d, lo, hi = paired_diff(preds[tag], preds["E0 影像特征"], y[te_m], subj[te_m])
        res["paired"][tag] = {"diff": d, "ci95": [lo, hi]}
        log.info("配对（%s − E0）= %+.2f kg [%.2f, %.2f]", tag, d, lo, hi)
    save_json(TABLE_DIR / "multimodal_embed_mlp.json", res)
    log.info("结果 -> %s", TABLE_DIR / "multimodal_embed_mlp.json")


if __name__ == "__main__":
    main()
