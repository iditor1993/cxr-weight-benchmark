"""对任意队列影像集做集成推理，输出逐影像预测（用于纵向/构念效度分析）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common import FEATURE_DIR, PROCESSED_DIR, load_config, setup_logger
from models import build_model, normalization_for


class ImageSet(Dataset):
    def __init__(self, df: pd.DataFrame, size: int, mean: float, std: float, cache_dir: Path):
        self.df = df.reset_index(drop=True)
        self.size, self.mean, self.std, self.cache_dir = size, mean, std, cache_dir
        self.df = self.df[self.df["dicom_id"].map(lambda d: (cache_dir / f"{d}.npy").exists())].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = np.load(self.cache_dir / f"{row['dicom_id']}.npy")
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        if img.shape[-1] != self.size:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(self.size, self.size),
                                                  mode="bilinear", align_corners=False).squeeze(0)
        img = (img - self.mean) / self.std
        return img.repeat(3, 1, 1), i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="cohort_a_all_images.csv")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--backbone", default="swin_t")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="缓存目录名（默认按 cohort 名推断）")
    args = ap.parse_args()

    log = setup_logger(f"predict_all_{Path(args.out).stem}")
    cfg = load_config()
    df = pd.read_csv(PROCESSED_DIR / args.cohort)
    mean, std = normalization_for(args.backbone)
    cache = FEATURE_DIR / args.cache if args.cache else (
        FEATURE_DIR / ("all_320" if "all" in args.cohort else "index_320"))
    ds = ImageSet(df, args.size, mean, std, cache)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = []
    for ckpt_path in args.checkpoints:
        model = build_model(args.backbone, img_size=args.size).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
        model.eval()
        p = []
        target_mean = float(torch.load(ckpt_path, map_location="cpu", weights_only=False).get("target_mean", 0.0))
        target_std = float(torch.load(ckpt_path, map_location="cpu", weights_only=False).get("target_std", 1.0))
        with torch.no_grad():
            for x, _ in loader:
                p.append(model(x.to(device)).squeeze(-1).float().cpu().numpy() * target_std + target_mean)
        preds.append(np.concatenate(p))
        log.info("predicted with %s", Path(ckpt_path).name)
    arr = np.vstack(preds)
    out = ds.df.copy()
    out["pred"] = arr.mean(axis=0)
    out["pred_std"] = arr.std(axis=0) if len(preds) > 1 else 0.0
    out.to_csv(PROCESSED_DIR / args.out, index=False)
    log.info("saved %d rows -> %s", len(out), args.out)


if __name__ == "__main__":
    main()
