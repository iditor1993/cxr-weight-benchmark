"""对全部索引影像计算异方差模型的 mu 与 sigma（供标记模型使用）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import FEATURE_DIR, PROCESSED_DIR, setup_logger
from models import build_model, normalization_for
from predict_all import ImageSet


def main() -> None:
    log = setup_logger("predict_hetero_all")
    device = torch.device("cuda")
    ckpt = torch.load(FEATURE_DIR / "checkpoints" / "swin_t_308_hetero_s1.pt",
                      map_location=device, weights_only=False)
    model = build_model("swin_t", out_dim=2, img_size=308).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tm, ts = float(ckpt.get("target_mean", 0.0)), float(ckpt.get("target_std", 1.0))
    mean, std = normalization_for("swin_t")
    df = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv")
    ds = ImageSet(df, 308, mean, std, FEATURE_DIR / "index_320")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    mus, sigmas = [], []
    with torch.no_grad():
        for k, (x, _) in enumerate(loader, 1):
            out = model(x.to(device))
            mu = out[:, 0].float().cpu().numpy() * ts + tm
            sigma = torch.exp(0.5 * out[:, 1]).float().cpu().numpy() * abs(ts)
            mus.append(mu)
            sigmas.append(sigma)
            if k % 40 == 0:
                log.info("batch %d/%d", k, len(loader))
    out = ds.df[["subject_id", "stay_id", "dicom_id"]].copy()
    out["hetero_mu"] = np.concatenate(mus)
    out["hetero_sigma"] = np.concatenate(sigmas)
    out.to_csv(PROCESSED_DIR / "preds_index_hetero.csv", index=False)
    log.info("saved %d rows -> preds_index_hetero.csv; sigma mean=%.2f",
             len(out), out["hetero_sigma"].mean())


if __name__ == "__main__":
    main()
