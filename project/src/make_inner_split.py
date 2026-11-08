"""从原始训练集切出 inner-train / inner-val，保留原验证集与测试集。

用途：模型搜索阶段只在 inner-train 上训练、inner-val 上选择；原验证集只做最终确认，
测试集始终封存。患者级切分，并按体重分层抽样 15%。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import PROCESSED_DIR, TABLE_DIR, save_json, setup_logger

SEED = 20261027
VAL_FRAC = 0.15


def main() -> None:
    log = setup_logger("make_inner_split")
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    cohort = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")
    subj_w = (cohort.groupby("subject_id")["weight_kg"].mean().rename("mean_weight").reset_index())
    sp = splits.merge(subj_w, on="subject_id", how="left")
    sp["wbin"] = pd.cut(sp["mean_weight"], [0, 50, 80, 100, 1000],
                        labels=["<50", "50-80", "80-100", ">=100"])
    rng = np.random.default_rng(SEED)
    sp["split_inner"] = sp["split_random"].map({"val": "confirm_val", "test": "test"})
    train = sp[sp["split_random"] == "train"].copy()
    val_subjects = set()
    for _, g in train.groupby("wbin", observed=True):
        k = max(1, int(round(VAL_FRAC * len(g))))
        val_subjects.update(rng.choice(g["subject_id"].values, size=k, replace=False).tolist())
    train["split_inner"] = np.where(train["subject_id"].isin(val_subjects), "inner_val", "inner_train")
    sp.loc[train.index, "split_inner"] = train["split_inner"]
    out = sp[["subject_id", "split_random", "split_inner"]].copy()
    out.to_csv(PROCESSED_DIR / "splits_inner.csv", index=False)

    stays = cohort.merge(out, on="subject_id", how="left")
    stats = {}
    for s in ("inner_train", "inner_val", "confirm_val", "test"):
        g = stays[stays["split_inner"] == s]
        stats[s] = {"n_stays": int(len(g)), "n_patients": int(g["subject_id"].nunique()),
                    "weight_median": float(g["weight_kg"].median()),
                    "share_lt50": float((g["weight_kg"] < 50).mean()),
                    "share_ge100": float((g["weight_kg"] >= 100).mean())}
        log.info("%-12s stays=%4d patients=%4d median=%.1f kg", s, len(g),
                 g["subject_id"].nunique(), g["weight_kg"].median())
    save_json(TABLE_DIR / "inner_split_stats.json",
              {"seed": SEED, "val_fraction": VAL_FRAC, "splits": stats})


if __name__ == "__main__":
    main()
