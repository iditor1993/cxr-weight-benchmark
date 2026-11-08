"""在指定切分（train/val/test）上做单模型或多种子集成推理，输出逐例预测与指标。

用途：模型搜索阶段只在 train/val 内比较候选方案，避免反复使用测试集。

用法：
  python project/src/predict_split.py --split val --backbone resnet50 --size 308 \
      --checkpoints a.pt b.pt --out val_preds_a_b.csv
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import MAIN_TAG, PROCESSED_DIR, TABLE_DIR, save_json, setup_logger
from dataset import WeightCXR
from metrics import regression_metrics
from models import build_model, normalization_for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val",
                    help="切分标签（test/val/train/inner_val/confirm_val 等）")
    ap.add_argument("--split-kind", default="random", choices=["random", "temporal"])
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--split-col", default=None)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--size", type=int, default=308)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="输出 CSV 文件名（写入 project/data/processed）")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    log = setup_logger(f"predict_{args.split}_{args.out.replace('.csv', '')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = normalization_for(args.backbone)
    ds = WeightCXR(args.split, split_kind=args.split_kind, size=args.size, train=False,
                   mean=mean, std=std, split_file=args.split_file, split_col=args.split_col)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    log.info("split=%s n=%d checkpoints=%d", args.split, len(ds), len(args.checkpoints))

    preds, meta = [], None
    for ckpt_path in args.checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(args.backbone, img_size=args.size).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        t_mean = float(ckpt.get("target_mean", 0.0))
        t_std = float(ckpt.get("target_std", 1.0))
        p, rows = [], []
        with torch.no_grad():
            for batch in loader:
                out = model(batch["image"].to(device)).squeeze(-1).float().cpu().numpy()
                p.append(out * t_std + t_mean)
                if meta is None:
                    rows.append(pd.DataFrame({
                        "subject_id": batch["subject_id"].numpy(),
                        "stay_id": batch["stay_id"].numpy(),
                        "dicom_id": batch["dicom_id"],
                        "weight_kg": batch["weight"].numpy()}))
        if meta is None:
            meta = pd.concat(rows, ignore_index=True)
        preds.append(np.concatenate(p))
        log.info("predicted with %s", ckpt_path.split("/")[-1])

    arr = np.vstack(preds)
    meta["pred"] = arr.mean(axis=0)
    meta["pred_std"] = arr.std(axis=0) if len(preds) > 1 else 0.0
    m = regression_metrics(meta["weight_kg"].values, meta["pred"].values)
    meta.to_csv(PROCESSED_DIR / args.out, index=False)
    payload = {"split": args.split, "n": int(len(meta)), "checkpoints": args.checkpoints,
               "metrics": m, "per_seed_mae": [float(np.mean(np.abs(meta["weight_kg"].values - p)))
                                              for p in preds]}
    save_json(TABLE_DIR / f"prediction_{args.out.replace('.csv', '')}.json", payload)
    log.info("MAE=%.3f kg (n=%d) -> %s", m["mae"], len(meta), args.out)
    print(json.dumps(payload["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
