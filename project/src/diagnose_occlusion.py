"""诊断遮挡实验：查看三种条件下模型预测值本身的分布（而不只是 MAE）。"""

from __future__ import annotations

import numpy as np
import torch

from attention_occlusion import gradcam
from common import FEATURE_DIR, TABLE_DIR, save_json, setup_logger
from dataset import WeightCXR
from models import build_model, normalization_for

log = setup_logger("diagnose_occlusion")
backbone, size, n = "swin_t", 308, 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(FEATURE_DIR / "checkpoints" / "swin_t_308_s1.pt", map_location=device,
                  weights_only=False)
model = build_model(backbone, img_size=size).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
target_mean = float(ckpt.get("target_mean", 0.0))
target_std = float(ckpt.get("target_std", 1.0))
mean, std = normalization_for(backbone)
ds = WeightCXR("test", split_kind="random", size=size, train=False, mean=mean, std=std)
rng = np.random.default_rng(20261004)
idxs = rng.choice(len(ds), size=n, replace=False)

rows = []
for i in idxs:
    s = ds[int(i)]
    x = s["image"].unsqueeze(0).to(device)
    y = float(s["weight"])
    cam = gradcam(model, x, backbone, stage=-2)[0]
    mask = cam >= np.quantile(cam, 0.80)
    sy, sx = int(rng.integers(0, cam.shape[0])), int(rng.integers(0, cam.shape[1]))
    shift = np.roll(np.roll(mask, sy, axis=0), sx, axis=1)
    preds = {}
    for name, m in (("base", None), ("salient", mask), ("shifted", shift)):
        xx = x.clone()
        if m is not None:
            xx[0, :, torch.from_numpy(m).to(device)] = 0.0
        with torch.no_grad():
            p = model(xx).squeeze(-1)
            p = float(p[..., 0] if p.ndim > 0 else p)
            preds[name] = p * target_std + target_mean
    rows.append({"true": y, **preds})

import pandas as pd
df = pd.DataFrame(rows)
out = {"n": n}
for name in ("base", "salient", "shifted"):
    out[name] = {
        "pred_mean": float(df[name].mean()), "pred_median": float(df[name].median()),
        "pred_p05": float(df[name].quantile(0.05)), "pred_p95": float(df[name].quantile(0.95)),
        "mae": float(np.mean(np.abs(df["true"] - df[name]))),
        "frac_pred_lt20kg": float((df[name] < 20).mean()),
        "frac_pred_gt150kg": float((df[name] > 150).mean()),
    }
out["true_weight_mean"] = float(df["true"].mean())
save_json(TABLE_DIR / "occlusion_diagnosis.json", out)
log.info("true weight mean: %.1f", out["true_weight_mean"])
for name in ("base", "salient", "shifted"):
    log.info("%s: pred mean=%.1f median=%.1f p05=%.1f p95=%.1f MAE=%.1f frac<20kg=%.2f frac>150kg=%.2f",
             name, out[name]["pred_mean"], out[name]["pred_median"], out[name]["pred_p05"],
             out[name]["pred_p95"], out[name]["mae"], out[name]["frac_pred_lt20kg"],
             out[name]["frac_pred_gt150kg"])
