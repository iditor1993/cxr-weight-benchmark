"""诊断 RISE 图为何接近均匀：检查掩膜预测的方差与重要性图的空间方差（5 张 × 1000 掩膜）。"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF

from attribution_pilot3_rise import rise_map_streaming
from common import FEATURE_DIR, TABLE_DIR, save_json, setup_logger
from dataset import WeightCXR
from models import build_model, normalization_for

log = setup_logger("rise_pred_stats")
device = torch.device("cuda")
ckpt = torch.load(FEATURE_DIR / "checkpoints" / "swin_t_308_s1.pt", map_location=device,
                  weights_only=False)
model = build_model("swin_t", img_size=308).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
tm, ts = float(ckpt.get("target_mean", 0.0)), float(ckpt.get("target_std", 1.0))
mean, std = normalization_for("swin_t")
ds = WeightCXR("test", split_kind="random", size=308, train=False, mean=mean, std=std)
rng = np.random.default_rng(20261008)
rows = []
for i in range(5):
    s = ds[i]
    x = s["image"].unsqueeze(0).to(device)
    x_blur = TF.gaussian_blur(x, kernel_size=61, sigma=20.0)
    imp, preds, dt = rise_map_streaming(model, x, x_blur, tm, ts, 1000, 16, 308, rng, chunk=256)
    fi = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)
    rows.append({
        "idx": i, "pred_mean": float(preds.mean()), "pred_std": float(preds.std()),
        "pred_range": float(preds.max() - preds.min()),
        "map_std_after_norm": float(fi.std()),
        "map_top10pct_mass": float(np.sort(fi.ravel())[-int(0.1 * fi.size):].sum() / fi.sum()),
        "seconds": dt,
    })
    log.info("img %d: pred mean=%.2f std=%.2f range=%.2f | map std=%.4f top10%%=%.3f",
             i, rows[-1]["pred_mean"], rows[-1]["pred_std"], rows[-1]["pred_range"],
             rows[-1]["map_std_after_norm"], rows[-1]["map_top10pct_mass"])
save_json(TABLE_DIR / "rise_pred_stats.json", {"rows": rows})
