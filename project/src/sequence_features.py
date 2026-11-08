"""用冻结主模型提取全部配对影像的嵌入特征（供序列注意力模型使用）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import FEATURE_DIR, PROCESSED_DIR, load_config, setup_logger
from models import build_model, normalization_for
from predict_all import ImageSet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="swin_t")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cohort", default="cohort_a_all_images.csv")
    ap.add_argument("--out", default="sequence_features.npy")
    args = ap.parse_args()

    log = setup_logger("sequence_features")
    df = pd.read_csv(PROCESSED_DIR / args.cohort)
    mean, std = normalization_for(args.backbone)
    cache = FEATURE_DIR / ("all_320" if "all" in args.cohort else "index_320")
    ds = ImageSet(df, args.size, mean, std, cache)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, img_size=args.size).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    model.eval()
    feats = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            f = model.forward_head(model.forward_features(x), pre_logits=True)
            feats.append(f.float().cpu().numpy().astype(np.float16))
            if i % 50 == 0:
                log.info("batch %d/%d", i, len(loader))
    arr = np.concatenate(feats, axis=0)
    np.save(FEATURE_DIR / args.out, arr)
    ds.df.to_csv(PROCESSED_DIR / "sequence_features_index.csv", index=False)
    log.info("features: %s shape=%s", args.out, arr.shape)


if __name__ == "__main__":
    main()
