"""诊断 Swin GradCAM：检查特征图布局、梯度形状与 CAM 统计。"""

from __future__ import annotations

import numpy as np
import torch

from common import FEATURE_DIR, setup_logger
from dataset import WeightCXR
from models import build_model, normalization_for

log = setup_logger("diagnose_cam")

backbone = "swin_t"
size = 308
ckpt_path = FEATURE_DIR / "checkpoints" / "swin_t_308_s1.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = build_model(backbone, img_size=size).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

ds = WeightCXR("test", split_kind="random", size=size, train=False,
               mean=normalization_for(backbone)[0], std=normalization_for(backbone)[1])
sample = ds[0]
x = sample["image"].unsqueeze(0).to(device)

layer = model.layers[-1].blocks[-1]
captured = {}


def fwd_hook(_, __, out):
    captured["act"] = out


def bwd_hook(_, grad_in, grad_out):
    captured["grad"] = grad_out[0]


h1 = layer.register_forward_hook(fwd_hook)
h2 = layer.register_full_backward_hook(bwd_hook)
model.zero_grad(set_to_none=True)
out = model(x).squeeze(-1)
out.sum().backward()
h1.remove()
h2.remove()

act = captured["act"].detach()
grad = captured["grad"].detach()
log.info("input shape: %s", tuple(x.shape))
log.info("activation shape: %s", tuple(act.shape))
log.info("gradient shape: %s", tuple(grad.shape))

# 判断通道维：与分类头输入维度 768 对齐
head_in = None
if hasattr(model, "head") and hasattr(model.head, "in_features"):
    head_in = model.head.in_features
log.info("head input features: %s", head_in)
for dim, size_ in enumerate(act.shape):
    log.info("act dim %d size=%d %s", dim, size_, "<-- matches head_in" if size_ == head_in else "")

# 复现当前实现（错误路径）
try:
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam_bad = (weights * act).sum(dim=1, keepdim=True)
    log.info("current-impl cam shape before interpolate: %s", tuple(cam_bad.shape))
    cam_up = torch.nn.functional.interpolate(
        cam_bad, size=x.shape[-2:], mode="bilinear", align_corners=False)
    log.info("current-impl cam shape after interpolate: %s", tuple(cam_up.shape))
    arr = cam_up[0, 0].detach().cpu().numpy()
    log.info("current-impl cam stats: min=%.4f max=%.4f std=%.4f",
             float(arr.min()), float(arr.max()), float(arr.std()))
except Exception as exc:  # noqa: BLE001
    log.info("current-impl failed: %s: %s", type(exc).__name__, exc)

# 正确路径（NHWC -> NCHW）
if head_in is not None and act.shape[-1] == head_in:
    a = act.permute(0, 3, 1, 2)          # (B,C,H,W)
    g = grad.permute(0, 3, 1, 2)
else:
    a, g = act, grad
w = g.mean(dim=(2, 3), keepdim=True)
cam = torch.relu((w * a).sum(dim=1, keepdim=True))
log.info("fixed cam shape: %s", tuple(cam.shape))
cam_up = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                                         align_corners=False)
arr = cam_up[0, 0].detach().cpu().numpy()
q = np.quantile(arr, 0.99)
mask = arr >= np.quantile(arr, 0.80)
log.info("fixed cam stats: min=%.4f max=%.4f std=%.4f; top10%% mass=%.3f; mask cols=%d rows=%d",
         float(arr.min()), float(arr.max()), float(arr.std()),
         float(arr[arr >= np.quantile(arr, 0.9)].sum() / (arr.sum() + 1e-8)),
         int(mask.any(axis=0).sum()), int(mask.any(axis=1).sum()))
