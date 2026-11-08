"""能否用胸片预测值标记"不可靠的入院体重记录"（不知道实测体重的前提下）。

标签：|入院体重 - 床秤实测| >= 5 kg
评分（仅使用影像预测 + 病历记录，不使用实测体重）：
  - |DL 预测 - 入院体重|
  - |既往记录 - 入院体重|（有既往记录时）
  - DL 预测本身偏离人群均值的程度（对照）
评估：AUROC、AUPRC、在不同阈值下的敏感度/特异度/需重新称重比例
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from common import PROCESSED_DIR, TABLE_DIR, save_json, setup_logger


def main() -> None:
    log = setup_logger("flag_unreliable_records")
    df = pd.read_csv(PROCESSED_DIR / "unreliable_subgroup_check_table.csv")
    adm = df["admission_weight_kg"].values.astype(float)
    y = df["weight_kg"].values.astype(float)
    dl = df["pred"].values.astype(float)
    prior = df["charted_last_weight_kg"].values.astype(float)
    ok = np.isfinite(adm) & np.isfinite(dl)
    label = (np.abs(adm - y) >= 5).astype(int)
    scores = {
        "|DL预测 - 入院体重|": np.abs(dl - adm),
        "|DL预测 - 入院体重|/入院体重": np.abs(dl - adm) / np.maximum(adm, 1e-6),
        "|既往记录 - 入院体重|": np.abs(prior - adm),
        "入院体重偏离100kg的程度": np.abs(adm - 100.0),
        "入院体重偏离人群均值(83.8kg)": np.abs(adm - 83.8),
    }
    out = {"n": int(ok.sum()), "prevalence": float(label[ok].mean())}
    for name, s in scores.items():
        m = ok & np.isfinite(s)
        if m.sum() < 50 or len(np.unique(label[m])) < 2:
            continue
        auroc = float(roc_auc_score(label[m], s[m]))
        auprc = float(average_precision_score(label[m], s[m]))
        fpr, tpr, thr = roc_curve(label[m], s[m])
        # 选定阈值：敏感度>=0.8 时最小转诊率
        idx = np.where(tpr >= 0.80)[0]
        rec = {"n": int(m.sum()), "auroc": auroc, "auprc": auprc}
        if len(idx):
            i = idx[0]
            rec["at_sens80"] = {"threshold": float(thr[i]), "sensitivity": float(tpr[i]),
                                "specificity": float(1 - fpr[i]),
                                "referral_rate": float((s[m] >= thr[i]).mean())}
        out[name] = rec
        log.info("%-32s n=%4d AUROC=%.3f AUPRC=%.3f (prevalence=%.2f)%s", name,
                 rec["n"], auroc, auprc, out["prevalence"],
                 f" | sens>=80%: spec={rec['at_sens80']['specificity']:.2f}, "
                 f"referral={100*rec['at_sens80']['referral_rate']:.0f}%" if "at_sens80" in rec else "")
    save_json(TABLE_DIR / "flag_unreliable_records.json", out)


if __name__ == "__main__":
    main()
