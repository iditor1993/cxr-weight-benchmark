"""胸片解剖分割模型（torchxrayvision chestx_anatomy）冒烟测试与可视化。"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import torchxrayvision as xrv
import torchxrayvision.baseline_models.chestx_anatomy as ca
from common import FEATURE_DIR, FIGURE_DIR, setup_logger

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

log = setup_logger("anatomy_smoke")
device = torch.device("cuda")
model = ca.UNetResNet50(weights=True).to(device).eval()
log.info("model: %s | input_resolution=%s | n_classes=%d", model, model.input_resolution, len(ca.TARGETS))

CLASSES = ["right lung", "left lung", "cardiomediastinum", "left hemidiaphragm",
           "right hemidiaphragm", "stomach", "liver", "spine"]
idx = {c: ca.TARGETS.index(c) for c in CLASSES if c in ca.TARGETS}
log.info("class indices: %s", idx)

files = sorted((FEATURE_DIR / "index_320").glob("*.npy"))[:4]
fig, axes = plt.subplots(len(files), 4, figsize=(13, 3.3 * len(files)))
for r, f in enumerate(files):
    img = np.load(f).astype(np.float32)  # 0-255
    x = torch.from_numpy(img)[None, None] / 255.0 * 2048 - 1024   # xrv 归一化
    res = model.input_resolution
    x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False).to(device)
    with torch.no_grad():
        out = model(x)
    logits = out[0] if isinstance(out, (list, tuple)) else out
    logits = logits[0] if logits.ndim == 4 else logits
    prob = torch.sigmoid(logits)
    masks = (prob > 0.5).float().cpu().numpy()
    log.info("%s -> logits %s; positive fractions: %s", f.stem[:8], tuple(logits.shape),
             {c: round(float(masks[idx[c]].mean()), 3) for c in idx})
    axes[r, 0].imshow(img, cmap="gray"); axes[r, 0].set_title("原图" if r == 0 else "")
    overlay = np.zeros((*masks.shape[1:], 3), dtype=np.float32)
    overlay[..., 0] = np.clip(masks[idx["right lung"]] + 0.6 * masks[idx["left lung"]], 0, 1)
    overlay[..., 1] = np.clip(masks[idx["cardiomediastinum"]], 0, 1)
    overlay[..., 2] = np.clip(masks[idx["left hemidiaphragm"]] + masks[idx["right hemidiaphragm"]], 0, 1)
    axes[r, 1].imshow(img, cmap="gray", extent=(0, masks.shape[1], masks.shape[0], 0))
    axes[r, 1].imshow(overlay, alpha=0.45)
    axes[r, 1].set_title("肺(红)/纵隔(绿)/膈(蓝)" if r == 0 else "")
    combined = np.clip(masks[idx["right lung"]] + masks[idx["left lung"]] +
                       masks[idx["cardiomediastinum"]] +
                       masks[idx["left hemidiaphragm"]] + masks[idx["right hemidiaphragm"]], 0, 1)
    axes[r, 2].imshow(img, cmap="gray", extent=(0, masks.shape[1], masks.shape[0], 0))
    axes[r, 2].imshow(combined, cmap="viridis", alpha=0.45)
    axes[r, 2].set_title("合并区域" if r == 0 else "")
    axes[r, 3].imshow(img, cmap="gray"); axes[r, 3].axis("off")
    for c in range(4):
        axes[r, c].axis("off")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "_debug" / "anatomy_segmentation_smoke.png", dpi=150)
plt.close(fig)
log.info("saved -> %s", FIGURE_DIR / "_debug" / "anatomy_segmentation_smoke.png")
