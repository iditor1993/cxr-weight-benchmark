"""修复后 CAM 的目视与定量验证（输出对比图到 results/figures/_debug）。"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attention_occlusion import gradcam
from common import FEATURE_DIR, FIGURE_DIR, setup_logger
from dataset import WeightCXR
from models import build_model, normalization_for

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

log = setup_logger("test_cam_fix")
backbone, size = "swin_t", 308
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(FEATURE_DIR / "checkpoints" / "swin_t_308_s1.pt", map_location=device,
                  weights_only=False)
model = build_model(backbone, img_size=size).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
mean, std = normalization_for(backbone)
ds = WeightCXR("test", split_kind="random", size=size, train=False, mean=mean, std=std)

out_dir = FIGURE_DIR / "_debug"
out_dir.mkdir(parents=True, exist_ok=True)
rows = 4
fig, axes = plt.subplots(rows, 4, figsize=(14, 15))
for r in range(rows):
    sample = ds[r]
    x = sample["image"].unsqueeze(0).to(device)
    cam_last = gradcam(model, x, backbone, stage=-1)[0]
    cam_mid = gradcam(model, x, backbone, stage=-2)[0]
    mask = cam_last >= np.quantile(cam_last, 0.80)
    img = sample["image"][0].numpy()
    axes[r, 0].imshow(img, cmap="gray"); axes[r, 0].set_title("原图" if r == 0 else "")
    axes[r, 1].imshow(img, cmap="gray"); axes[r, 1].imshow(cam_last, cmap="jet", alpha=0.45)
    axes[r, 1].set_title("CAM 最后stage" if r == 0 else "")
    axes[r, 2].imshow(img, cmap="gray"); axes[r, 2].imshow(cam_mid, cmap="jet", alpha=0.45)
    axes[r, 2].set_title("CAM 倒数第2 stage" if r == 0 else "")
    axes[r, 3].imshow(img, cmap="gray"); axes[r, 3].imshow(mask, cmap="gray", alpha=0.5)
    axes[r, 3].set_title("遮挡掩膜(top20%)" if r == 0 else "")
    for c in range(4):
        axes[r, c].axis("off")
    top10 = cam_last[cam_last >= np.quantile(cam_last, 0.9)].sum() / (cam_last.sum() + 1e-8)
    log.info("row %d: cam_last std=%.4f top10%%mass=%.3f mask_frac=%.3f",
             r, float(cam_last.std()), float(top10), float(mask.mean()))
fig.tight_layout()
fig.savefig(out_dir / "cam_fix_check.png", dpi=150)
plt.close(fig)
log.info("saved -> %s", out_dir / "cam_fix_check.png")
