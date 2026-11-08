"""评价：单模型/多种子集成在测试集上的性能、成功标准 S1–S3、亚组与 bootstrap CI。"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import FEATURE_DIR, PROCESSED_DIR, TABLE_DIR, load_config, save_json, setup_logger
from dataset import WeightCXR
from metrics import bootstrap_ci_arrays, regression_metrics
from models import build_model, normalization_for


def predict(model, loader, device, target_mean: float, target_std: float,
            head: str = "point") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, trues, sigmas = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            out = model(x).squeeze(-1)
            if head == "hetero" and out.ndim > 1:
                mu, logvar = out[..., 0], out[..., 1]
                sigma = torch.exp(0.5 * logvar)
                sigmas.append(sigma.float().cpu().numpy() * abs(target_std))
            else:
                mu = out[..., 0] if out.ndim > 1 else out
            preds.append(mu.float().cpu().numpy() * target_std + target_mean)
            trues.append(batch["weight"].numpy())
    sigma_arr = np.concatenate(sigmas) if sigmas else np.zeros(len(np.concatenate(preds)))
    return np.concatenate(preds), np.concatenate(trues), sigma_arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--split", default="random")
    ap.add_argument("--head", default="point", choices=["point", "hetero"])
    args = ap.parse_args()

    log = setup_logger(f"evaluate_{args.tag}")
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = normalization_for(args.backbone)
    ds = WeightCXR("test", split_kind=args.split, size=args.size, train=False, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    log.info("test images: %d", len(ds))

    all_preds = []
    target_mean, target_std = None, None
    for ckpt_path in args.checkpoints:
        model = build_model(args.backbone, out_dim=2 if args.head == "hetero" else 1,
                            img_size=args.size).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        target_mean = float(ckpt.get("target_mean", 0.0))
        target_std = float(ckpt.get("target_std", 1.0))
        pred, true, sigma = predict(model, loader, device, target_mean, target_std, args.head)
        if args.head == "hetero":
            all_sigma = sigma
        all_preds.append(pred)
        log.info("checkpoint %s -> MAE %.3f", Path(ckpt_path).name,
                 float(np.mean(np.abs(true - pred))))
    preds = np.vstack(all_preds)
    y_pred = preds.mean(axis=0)
    y_pred_std = preds.std(axis=0) if len(all_preds) > 1 else np.zeros_like(y_pred)

    meta = ds.df.reset_index(drop=True).copy()
    meta["pred"] = y_pred
    meta["pred_std"] = y_pred_std
    if args.head == "hetero":
        meta["pred_sigma"] = all_sigma
    meta["abs_err"] = np.abs(meta["weight_kg"] - meta["pred"])
    meta["rel_err"] = meta["abs_err"] / meta["weight_kg"]

    # 上下文变量
    con = duckdb.connect(cfg["data"]["duckdb"], read_only=True)
    sex = con.execute("""
        SELECT p.subject_id, p.gender, p.anchor_age, min(a.race) AS race
        FROM main.patients p
        LEFT JOIN main.admissions a USING (subject_id)
        GROUP BY 1,2,3
    """).fetchdf()
    height = con.execute("SELECT stay_id, avg(height) AS height_cm FROM mimiciv_derived.height GROUP BY 1").fetchdf()
    vent = con.execute("""SELECT DISTINCT stay_id, 1 AS ventilated FROM mimiciv_derived.ventilation
                          WHERE ventilation_status='InvasiveVent'""").fetchdf()
    rrt = con.execute("SELECT DISTINCT stay_id, 1 AS rrt FROM mimiciv_derived.rrt").fetchdf()
    chex = con.execute("""SELECT subject_id, study_id, "Support Devices" AS support_devices
                          FROM main.cxr_chexpert""").fetchdf()
    meta = (meta.merge(sex, on="subject_id", how="left")
                .merge(height, on="stay_id", how="left")
                .merge(vent, on="stay_id", how="left")
                .merge(rrt, on="stay_id", how="left")
                .merge(chex, on=["subject_id", "study_id"], how="left"))
    meta["ventilated"] = meta["ventilated"].fillna(0).astype(int)
    meta["rrt"] = meta["rrt"].fillna(0).astype(int)
    # CheXpert 标签：1=阳性，0=阴性，-1=不确定，NaN=未提及（按阴性处理，需在报告中说明）
    meta["support_devices"] = meta["support_devices"].fillna(0).clip(lower=0).astype(int)
    meta["bmi"] = meta["weight_kg"] / (meta["height_cm"] / 100.0) ** 2
    meta["weight_stratum"] = pd.cut(meta["weight_kg"], [0, 50, 80, 100, 1000],
                                    labels=["<50", "50-80", "80-100", ">=100"])
    meta["age_stratum"] = pd.cut(meta["anchor_age"], [0, 50, 65, 80, 200],
                                 labels=["<50", "50-65", "65-80", ">=80"])
    meta.to_csv(PROCESSED_DIR / f"test_predictions_{args.tag}.csv", index=False)

    overall = regression_metrics(meta["weight_kg"].values, meta["pred"].values)
    pt, lo, hi = bootstrap_ci_arrays(meta["weight_kg"].values, meta["pred"].values,
                                     meta["subject_id"].values, "mae")
    overall["mae_ci95"] = [pt, lo, hi]
    pt, lo, hi = bootstrap_ci_arrays(meta["weight_kg"].values, meta["pred"].values,
                                     meta["subject_id"].values, "hit10pct")
    overall["hit10_ci95"] = [pt, lo, hi]
    overall["mean_pred_std"] = float(meta["pred_std"].mean())

    subgroups = {}
    for name, col in (("sex", "gender"), ("weight_stratum", "weight_stratum"),
                      ("age_stratum", "age_stratum"), ("care_unit", "first_careunit"),
                      ("ventilated", "ventilated"), ("rrt", "rrt"),
                      ("support_devices", "support_devices"), ("race", "race")):
        rows = []
        for key, g in meta.groupby(col, dropna=False):
            if len(g) < 20:
                continue
            m = regression_metrics(g["weight_kg"].values, g["pred"].values)
            rows.append({"group": str(key), "n": len(g), "mae": m["mae"], "rmse": m["rmse"],
                         "hit_within_10pct": m["hit_within_10pct"],
                         "hit_within_5kg": m["hit_within_5kg"], "bias": m["bias"]})
        subgroups[name] = rows

    s1 = overall["hit_within_10pct"] >= cfg["success_criteria"]["s1_hit_rate_within_10pct"]
    s2 = (overall["mae"] <= cfg["success_criteria"]["s2_mae_max_kg"]
          and overall["mae_ci95"][2] < cfg["success_criteria"]["s2_ci_upper_max_kg"])
    extreme = meta[meta["weight_stratum"].isin(["<50", ">=100"])]
    extreme_hits = {str(k): float(np.mean(np.abs(g["weight_kg"] - g["pred"]) <= 0.10 * g["weight_kg"]))
                    for k, g in extreme.groupby("weight_stratum") if len(g) >= 20}
    s3 = all(v >= cfg["success_criteria"]["s3_extreme_subgroup_hit_min"] for v in extreme_hits.values())
    success = {"S1_hit10_ge80pct": bool(s1), "S2_mae_le5_and_ci_lt6": bool(s2),
               "S3_extreme_subgroups_ge65pct": bool(s3), "S3_extreme_hit_rates": extreme_hits,
               "all_passed": bool(s1 and s2 and s3)}

    payload = {"tag": args.tag, "checkpoints": args.checkpoints, "n_test": int(len(meta)),
               "overall": overall, "success_criteria": success, "subgroups": subgroups}
    save_json(TABLE_DIR / f"evaluation_{args.tag}.json", payload)
    log.info("overall: %s", json.dumps(overall, ensure_ascii=False))
    log.info("success: %s", json.dumps(success, ensure_ascii=False))
    for name, rows in subgroups.items():
        log.info("subgroup %s: %s", name, json.dumps(rows, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
