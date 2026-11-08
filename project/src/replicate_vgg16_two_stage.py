"""复刻 Ichikawa/Sugimori 2023（Radiol Phys Technol 16:127-134）的两阶段体重估计设计。

原文规格（来自其开放获取的同组论文 Sci Rep 2021 与本篇摘要）
    - 骨干：VGG16，ImageNet 预训练，**卷积层冻结**，只训练新增全连接头
      （flatten → dense(ReLU) → dense(1)）
    - 输入：224 × 224（保持长宽比 + 零填充）
    - 损失 MSE；Adam，初始 lr 0.001，每 3 个 epoch 降为 1/10；最多 40 epoch；batch 32
    - 无数据增强；k 折交叉验证（原文 k=5）
    - 2023 版两阶段：① 性别分类模型 → ② 性别特异体重回归

本脚本的适配
    - 数据换成 MIMIC-IV ICU 床旁 AP 胸片队列 A（8,767 张索引影像，患者级切分）
    - 特征只提取一次（冻结），在训练集内做 5 折交叉验证，测试集用 5 折模型集成
    - 同时给出"按预测性别路由"与"按已知性别"两种结果，并加一个"不分性别单模型"对照

用法
    python project/src/replicate_vgg16_two_stage.py            # 全流程
    python project/src/replicate_vgg16_two_stage.py --skip-extract   # 复用已提取特征

产出
    project/data/features/vgg16_feat_224.npy / .csv        （冻结特征，float16）
    project/results/tables/vgg16_two_stage.json
    project/results/figures/vgg16_two_stage.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from common import (FEATURE_DIR, FIGURE_DIR, MAIN_LABEL, MAIN_TAG, PROCESSED_DIR, TABLE_DIR,
                    save_json, setup_logger)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261012
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CacheSet(Dataset):
    def __init__(self, df: pd.DataFrame, cache: Path, size: int = 224):
        self.df = df.reset_index(drop=True)
        self.cache, self.size = cache, size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = np.load(self.cache / f"{row['dicom_id']}.npy").astype(np.float32) / 255.0
        t = torch.from_numpy(img).unsqueeze(0)
        if t.shape[-1] != self.size:
            t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(self.size, self.size),
                                                mode="bilinear", align_corners=False).squeeze(0)
        t = t.repeat(3, 1, 1).numpy()
        t = (t - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        return torch.from_numpy(t).float(), i


def extract_features(df: pd.DataFrame, cache: Path, out_npy: Path, out_csv: Path, log) -> None:
    from torchvision.models import VGG16_Weights, vgg16

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    feats = model.features.to(dev).eval()
    for p in feats.parameters():
        p.requires_grad_(False)
    ds = CacheSet(df, cache)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    outs = []
    with torch.no_grad():
        for k, (x, _) in enumerate(loader):
            f = feats(x.to(dev, non_blocking=True))
            outs.append(f.flatten(1).half().cpu().numpy())
            if (k + 1) % 20 == 0:
                log.info("  extract %d/%d batches", k + 1, len(loader))
    arr = np.concatenate(outs, axis=0)
    log.info("特征矩阵 %s (%.0f MB)", arr.shape, arr.nbytes / 1024 ** 2)
    np.save(out_npy, arr)
    df[["dicom_id", "subject_id", "stay_id", "weight_kg"]].to_csv(out_csv, index=False)


class Head(nn.Module):
    """原文头部：flatten → dense(ReLU) → dense(out)。"""

    def __init__(self, d_in: int, hidden: int = 512, out: int = 1, p_drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Dropout(p_drop) if p_drop else nn.Identity(),
                                 nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def train_head(Xtr, ytr, Xva, yva, out_dim: int, task: str, log=None, dropout: float = 0.0,
               max_epochs: int = 40, batch: int = 32, lr: float = 1e-3, seed: int = SEED):
    """按原文超参训练（Adam 1e-3，每 3 epoch ×0.1，最多 40 epoch），校验集早停。"""
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=dev)
    if task == "cls":
        ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
        yva_np = yva
        lossf = nn.CrossEntropyLoss()
    else:
        mu, sd = float(np.mean(ytr)), float(np.std(ytr))
        ytr_t = torch.tensor((ytr - mu) / sd, dtype=torch.float32, device=dev)
        lossf = nn.MSELoss()

    model = Head(Xtr.shape[1], 512, out_dim, p_drop=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
    n = len(Xtr_t)
    best, best_state, patience = math.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            if task == "cls":
                loss = lossf(out, ytr_t[idx])
            else:
                loss = lossf(out.squeeze(-1), ytr_t[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            out = model(Xva_t)
            if task == "cls":
                va = -float((out.argmax(1).cpu().numpy() == yva_np).mean())   # 越大越好 → 取负
            else:
                pv = out.squeeze(-1).cpu().numpy() * sd + mu
                va = float(np.abs(pv - yva).mean())
        if va < best - 1e-4:
            best, best_state, patience = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 10:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, (mu, sd) if task == "reg" else None, -best if task == "cls" else best


def predict_head(model, X, task: str, norm=None) -> np.ndarray:
    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32, device=dev))
    if task == "cls":
        return out.argmax(1).cpu().numpy()
    p = out.squeeze(-1).cpu().numpy()
    return p * norm[1] + norm[0]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    err = p - y
    return {"n": int(len(y)), "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())), "bias": float(err.mean()),
            "hit_within_10pct": float((np.abs(err) / y <= 0.10).mean()),
            "hit_within_5kg": float((np.abs(err) <= 5).mean()),
            "hit_within_10kg": float((np.abs(err) <= 10).mean())}


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--cache", default="index_320")
    args = ap.parse_args()

    log = setup_logger("replicate_vgg16_two_stage")
    feat_npy = FEATURE_DIR / "vgg16_feat_224.npy"
    feat_csv = PROCESSED_DIR / "vgg16_feat_224_index.csv"

    coh = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    coh = coh.merge(splits[["subject_id", "split_random"]], on="subject_id", how="left")
    coh = coh[coh["dicom_id"].map(lambda d: (FEATURE_DIR / args.cache / f"{d}.npy").exists())]
    coh = coh.reset_index(drop=True)
    log.info("队列 %d 张索引影像（train %d / val %d / test %d）", len(coh),
             (coh.split_random == "train").sum(), (coh.split_random == "val").sum(),
             (coh.split_random == "test").sum())

    if not args.skip_extract or not feat_npy.exists():
        extract_features(coh, FEATURE_DIR / args.cache, feat_npy, feat_csv, log)
    else:
        log.info("复用已提取特征 %s", feat_npy)

    X = np.load(feat_npy).astype(np.float32)
    order = pd.read_csv(feat_csv)
    import duckdb

    from common import load_config

    con = duckdb.connect(load_config()["data"]["duckdb"], read_only=True)
    sex_df = con.execute("SELECT subject_id, gender FROM main.patients").fetchdf()
    con.close()
    df = (order.merge(coh[["dicom_id", "subject_id", "split_random"]], on="dicom_id", how="left",
                      suffixes=("", "_c"))
              .merge(sex_df, on="subject_id", how="left"))
    df["male"] = (df["gender"] == "M").astype(int)
    y = df["weight_kg"].values.astype(np.float32)
    sex = df["male"].values.astype(int)
    subj = df["subject_id"].values
    tr = (df["split_random"] == "train").values
    te = (df["split_random"] == "test").values
    log.info("特征 %s | train %d / test %d", X.shape, tr.sum(), te.sum())

    # 训练集内 5 折（患者级）
    from common import stable_fraction

    fold = np.array([int(stable_fraction(str(s), SEED) * 5) for s in subj])
    res = {"n": {"train": int(tr.sum()), "test": int(te.sum())},
           "design": "VGG16-224 冻结特征 + flatten→dense(ReLU)→dense；5 折集成；"
                     "Adam 1e-3（每 3 epoch ×0.1），MSE，batch 32，最多 40 epoch，无增强",
           "models": {}}

    te_idx = np.where(te)[0]
    n_folds = 5
    y_te, subj_te = y[te_idx], subj[te_idx]

    # 性别分类（一次，供所有配置共用）
    pred_sex = np.zeros((n_folds, te.sum()), dtype=int)
    fold_acc, fold_auc = [], []
    for k in range(n_folds):
        fit, val = tr & (fold != k), tr & (fold == k)
        m_sex, _, _ = train_head(X[fit], sex[fit], X[val], sex[val], 2, "cls")
        ps = predict_head(m_sex, X[te_idx], "cls")
        pred_sex[k] = ps
        fold_acc.append(float((ps == sex[te_idx]).mean()))
        try:
            fold_auc.append(float(roc_auc_score(sex[te_idx], ps)))
        except ValueError:
            fold_auc.append(float("nan"))
    log.info("性别分类：acc=%.3f（原文体检数据 0.992）", float(np.mean(fold_acc)))

    # 三种配置：原文设定 / +dropout / +特征标准化+dropout
    configs = [
        ("A 原文设定（原始特征、无 dropout）", dict(std=False, dropout=0.0)),
        ("B 头部加 dropout 0.5", dict(std=False, dropout=0.5)),
        # 逐维标准化会让低方差维度放大噪声并导致训练塌缩，这里改为"全局标量标准化 + 低学习率"
        ("C 全局标准化 + 低学习率 + dropout", dict(std="global", dropout=0.5, lr=1e-4)),
    ]
    out_by_cfg: dict[str, dict[str, np.ndarray]] = {}
    for cfg_name, opt in configs:
        pr = np.zeros((n_folds, te.sum()))
        pk = np.zeros((n_folds, te.sum()))
        psin = np.zeros((n_folds, te.sum()))
        for k in range(n_folds):
            fit, val = tr & (fold != k), tr & (fold == k)
            if opt["std"] == "global":          # 全局标量标准化（仅用本折训练集统计量）
                g_mu, g_sd = float(X[fit].mean()), float(X[fit].std()) + 1e-6
                mu_f = np.full((1, X.shape[1]), g_mu, dtype=np.float32)
                sd_f = np.full((1, X.shape[1]), g_sd, dtype=np.float32)
            else:
                mu_f, sd_f = np.zeros((1, X.shape[1]), dtype=np.float32), np.ones((1, X.shape[1]),
                                                                                 dtype=np.float32)

            def norm_apply(idx):
                return (X[idx] - mu_f) / sd_f

            p_by_pred, p_by_known = np.zeros(te.sum()), np.zeros(te.sum())
            for s in (0, 1):
                fit_s, val_s = fit & (sex == s), val & (sex == s)
                lr_cfg = opt.get("lr", 1e-3)
                m_reg, norm, _ = train_head(norm_apply(fit_s), y[fit_s], norm_apply(val_s),
                                            y[val_s], 1, "reg", dropout=opt["dropout"], lr=lr_cfg)
                p_all = predict_head(m_reg, norm_apply(te_idx), "reg", norm)
                p_by_pred[pred_sex[k] == s] = p_all[pred_sex[k] == s]
                p_by_known[sex[te_idx] == s] = p_all[sex[te_idx] == s]
            m_all, norm_all, _ = train_head(norm_apply(fit), y[fit], norm_apply(val), y[val], 1,
                                            "reg", dropout=opt["dropout"], lr=opt.get("lr", 1e-3))
            pr[k], pk[k] = p_by_pred, p_by_known
            psin[k] = predict_head(m_all, norm_apply(te_idx), "reg", norm_all)
        out_by_cfg[cfg_name] = {"routed": pr, "known": pk, "single": psin}
        for tag, arr in (("预测性别路由", pr), ("已知性别", pk), ("单模型不分性别", psin)):
            res["models"][f"{cfg_name} / {tag}"] = metrics(y_te, arr.mean(axis=0))
        log.info("%s：两阶段(已知性别) MAE=%.2f kg | 单模型 %.2f kg", cfg_name,
                 res["models"][f"{cfg_name} / 已知性别"]["mae"],
                 res["models"][f"{cfg_name} / 单模型不分性别"]["mae"])

    best_cfg = min(configs, key=lambda c: res["models"][f"{c[0]} / 已知性别"]["mae"])[0]
    res["best_config"] = best_cfg
    out = {"VGG16 两阶段（已知性别）": out_by_cfg[best_cfg]["known"].mean(axis=0)}

    res["sex_classifier"] = {"accuracy_mean": float(np.mean(fold_acc)),
                             "accuracy_per_fold": fold_acc,
                             "auroc_mean": float(np.nanmean(fold_auc))}
    log.info("最优配置：%s", best_cfg)

    # 与本项目主模型 / 常规方法对比
    main = pd.read_csv(PROCESSED_DIR / f"test_predictions_{MAIN_TAG}.csv")
    main = main[["dicom_id", "pred", "weight_kg"]]
    cmp_df = pd.DataFrame({"dicom_id": df.loc[te_idx, "dicom_id"].values, "y": y_te,
                           "vgg": out["VGG16 两阶段（已知性别）"]}).merge(main, on="dicom_id", how="left")
    if cmp_df["pred"].notna().all():
        res["models"][MAIN_LABEL] = metrics(
            cmp_df["y"].values, cmp_df["pred"].values)
        d, lo, hi = paired_diff(cmp_df["vgg"].values, cmp_df["pred"].values,
                                cmp_df["y"].values, subj_te)
        res["paired_vgg_vs_main"] = {"diff": d, "ci95": [lo, hi]}
        log.info("配对（VGG16 − 主模型）= %+.2f kg [%.2f, %.2f]", d, lo, hi)
    conv = pd.read_csv(PROCESSED_DIR / "conventional_weight_comparison_table.csv")
    key = pd.DataFrame({"stay_id": df.loc[te_idx, "stay_id"].values,
                        "vgg": out["VGG16 两阶段（已知性别）"]})
    conv = conv.merge(key, on="stay_id", how="inner")
    for col, label in (("admission_weight_kg", "护理录入入院体重"),
                       ("OLS 身高+性别", "身高+性别回归"), ("Devine IBW", "Devine IBW")):
        if col == "OLS 身高+性别":
            conv[col] = (-69.9350080268858 + 0.9196542642028203 * conv["height_final_cm"]
                         + 0.04498295084068797 * (conv["gender"] == "M").astype(float))
        res["models"][label] = metrics(conv["weight_kg"].values, conv[col].values)
        d, lo, hi = paired_diff(conv["vgg"].values, conv[col].values, conv["weight_kg"].values,
                                conv["subject_id"].values)
        res.setdefault("paired_vgg_vs_comparators", {})[label] = {"diff": d, "ci95": [lo, hi]}
        log.info("配对（复刻 VGG16 − %s）= %+.2f kg [%.2f, %.2f]", label, d, lo, hi)
    save_json(TABLE_DIR / "vgg16_two_stage.json", res)

    # 图
    order_lbl = [f"{best_cfg} / 预测性别路由", f"{best_cfg} / 已知性别",
                 f"{best_cfg} / 单模型不分性别",
                 MAIN_LABEL, "护理录入入院体重", "身高+性别回归", "Devine IBW"]
    vals = [res["models"][k]["mae"] for k in order_lbl if k in res["models"]]
    lbls = [k for k in order_lbl if k in res["models"]]
    colors = ["#e67e22", "#d35400", "#f1c40f", "#c0392b", "#2980b9", "#95a5a6", "#7f8c8d"][:len(lbls)]
    lbls = [l.replace(best_cfg + " / ", "") if l.startswith(best_cfg) else l for l in lbls]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(lbls)), vals, color=colors)
    ax.set_yticks(range(len(lbls)))
    ax.set_yticklabels(lbls, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("测试集 MAE (kg)")
    ax.set_title(f"复刻 Ichikawa/Sugimori 2023 设计在 MIMIC ICU 队列上的表现（n={res['n']['test']}）")
    for i, v in enumerate(vals):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vgg16_two_stage.png", dpi=200)
    plt.close(fig)
    log.info("图 -> %s", FIGURE_DIR / "vgg16_two_stage.png")


if __name__ == "__main__":
    main()
