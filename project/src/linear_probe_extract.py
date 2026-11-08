"""提取冻结 Swin-T 特征：全局池化（768-d）与 stage-3 空间特征（20×20×384）。"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import FEATURE_DIR, PROCESSED_DIR, save_json, setup_logger
from dataset import WeightCXR
from models import build_model, normalization_for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="linear_probe")
    ap.add_argument("--out", default="swin_features")
    args = ap.parse_args()
    log = setup_logger(f"{args.tag}_extract")
    device = torch.device("cuda")
    ckpt = torch.load(FEATURE_DIR / "checkpoints" / "swin_t_308_s1.pt",
                      map_location=device, weights_only=False)
    model = build_model("swin_t", img_size=308).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    target = model.layers[-2].blocks[-1]
    captured = {}

    def hook(_, __, out):
        captured["feat"] = out

    h = target.register_forward_hook(hook)
    mean, std = normalization_for("swin_t")
    ds = WeightCXR("train", split_kind="random", size=308, train=False, mean=mean, std=std)
    # 用全部索引影像：三个 split 的并集
    cohort = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    all_df = cohort.merge(splits, on="subject_id", how="left")
    all_df["split_random"] = all_df["split_random"].fillna("train")
    all_df = all_df[all_df["dicom_id"].map(
        lambda d: (FEATURE_DIR / "index_320" / f"{d}.npy").exists())].reset_index(drop=True)
    ds.df = all_df[["dicom_id", "subject_id", "stay_id", "weight_kg"]].copy()
    log.info("extracting features for %d images", len(ds))

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    globals_, spatials, weights, dicoms = [], [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for k, batch in enumerate(loader, 1):
            x = batch["image"].to(device)
            _ = model(x)
            fmap = captured["feat"]                      # (B,H,W,C)
            b, hh, ww, cc = fmap.shape
            spatials.append(fmap.permute(0, 3, 1, 2).reshape(b, cc, hh * ww).half().cpu().numpy())
            g = model.forward_head(model.forward_features(x), pre_logits=True)
            globals_.append(g.half().cpu().numpy())
            weights.append(batch["weight"].numpy())
            dicoms.extend(batch["dicom_id"])
            if k % 20 == 0:
                log.info("batch %d/%d (%.1f min)", k, len(loader), (time.perf_counter() - t0) / 60)
    h.remove()
    sp = np.concatenate(spatials, axis=0)                # (N,384,400)
    gl = np.concatenate(globals_, axis=0)                # (N,768)
    y = np.concatenate(weights)
    np.save(FEATURE_DIR / f"{args.out}_global.npy", gl)
    np.save(FEATURE_DIR / f"{args.out}_spatial.npy", sp)
    pd.DataFrame({"dicom_id": dicoms, "weight_kg": y}).to_csv(
        PROCESSED_DIR / f"{args.out}_index.csv", index=False)
    log.info("saved: global %s, spatial %s", gl.shape, sp.shape)


if __name__ == "__main__":
    main()
