"""注意力热图 + 高显著区域遮挡实验（研究方案第 10 节去偏设计）。

实现说明：MIMIC-CXR 无器械分割标注，按方案的退化规则执行：
  (1) 器械阳性/阴性亚组性能（见 evaluate.py）；
  (2) 高显著区域（模型梯度 CAM 前 20% 像素）遮挡 vs 随机区域遮挡的 MAE 变化。
器械弱监督 CAM 掩膜与 100 张人工核对列为待人工完成项。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import FEATURE_DIR, FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, save_json, setup_logger
from dataset import WeightCXR
from models import build_model, normalization_for

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def target_layer(model, backbone: str, stage: int = -1):
    if backbone == "resnet50":
        return model.layer4[-1]
    if backbone == "densenet121":
        return model.features[-1]
    if backbone == "swin_t":
        return model.layers[stage].blocks[-1]
    raise ValueError(backbone)


def gradcam(model, x: torch.Tensor, backbone: str, stage: int = -1) -> np.ndarray:
    """GradCAM，自动兼容 NCHW（CNN）与 NHWC（Swin/Transformer）特征图。

    修复记录（2026-10-06）：原实现将 Swin 的 (B,H,W,C) 特征误按 (B,C,H,W) 归约，
    导致 CAM 退化为常数/竖条纹；现按通道维自动判向。
    """
    feats = {}
    layer = target_layer(model, backbone, stage)

    def fwd_hook(_, __, out):
        feats["act"] = out

    def bwd_hook(_, grad_in, grad_out):
        feats["grad"] = grad_out[0]

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        out = model(x)
        score = out.squeeze(-1) if out.ndim > 1 else out
        score.sum().backward()
        act = feats["act"].detach()
        grad = feats["grad"].detach()
        if act.ndim == 4:
            # 判向规则：方形输入下，空间两维相等。
            # NHWC: dim1==dim2；NCHW: dim2==dim3
            if act.shape[1] == act.shape[2] and act.shape[3] != act.shape[1]:
                act = act.permute(0, 3, 1, 2)
                grad = grad.permute(0, 3, 1, 2)
            elif act.shape[2] != act.shape[3] and act.shape[1] != act.shape[2]:
                raise RuntimeError(f"无法判定特征图布局: {tuple(act.shape)}")
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
        else:  # transformer tokens: (B, N, C)
            weights = grad.mean(dim=1, keepdim=True)
            cam = (weights * act).sum(dim=-1, keepdim=True).transpose(1, 2)
            side = int(round(cam.shape[-1] ** 0.5))
            if side * side == cam.shape[-1]:
                cam = cam.reshape(cam.shape[0], 1, side, side)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                                              align_corners=False)
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        if float(cam.std()) < 1e-6:
            raise RuntimeError("CAM 退化为常数：请检查目标层/通道布局")
        return cam.detach().cpu().numpy()[:, 0]
    finally:
        h1.remove()
        h2.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="swin_t")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="random")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stage", type=int, default=-2,
                    help="Swin stage 索引：-1=最后 stage（10×10），-2=倒数第二（20×20，默认）")
    args = ap.parse_args()

    log = setup_logger(f"attention_{args.tag}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = normalization_for(args.backbone)
    ds = WeightCXR("test", split_kind=args.split, size=args.size, train=False, mean=mean, std=std)
    model = build_model(args.backbone, img_size=args.size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    target_mean = float(ckpt.get("target_mean", 0.0))
    target_std = float(ckpt.get("target_std", 1.0))
    model.eval()

    rng = np.random.default_rng(20261004)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    base_err, occ_err, rand_err, shift_err, device_frac = [], [], [], [], []
    shift_deltas = []
    grid_saved = 0
    for i in idxs:
        sample = ds[int(i)]
        x = sample["image"].unsqueeze(0).to(device)
        y = float(sample["weight"])
        with torch.no_grad():
            p = model(x).squeeze(-1)
            p = float(p[..., 0] if p.ndim > 0 else p) * target_std + target_mean
        cam = gradcam(model, x, args.backbone, stage=args.stage)[0]
        thr = np.quantile(cam, 0.80)
        mask = cam >= thr
        x_occ = x.clone()
        x_occ[0, :, torch.from_numpy(mask).to(device)] = 0.0
        with torch.no_grad():
            p_occ = model(x_occ).squeeze(-1)
            p_occ = float(p_occ[..., 0] if p_occ.ndim > 0 else p_occ) * target_std + target_mean
        # 对照 1：同形状掩膜随机平移（更干净的"位置对照"）
        sy, sx = int(rng.integers(0, cam.shape[0])), int(rng.integers(0, cam.shape[1]))
        shift_mask = np.roll(np.roll(mask, sy, axis=0), sx, axis=1)
        x_shift = x.clone()
        x_shift[0, :, torch.from_numpy(shift_mask).to(device)] = 0.0
        with torch.no_grad():
            p_shift = model(x_shift).squeeze(-1)
            p_shift = float(p_shift[..., 0] if p_shift.ndim > 0 else p_shift) * target_std + target_mean
        # 对照 2：随机像素遮挡（会引入高频伪影，仅作参考）
        rand_mask = rng.random(cam.shape) >= 0.80
        x_rand = x.clone()
        x_rand[0, :, torch.from_numpy(rand_mask).to(device)] = 0.0
        with torch.no_grad():
            p_rand = model(x_rand).squeeze(-1)
            p_rand = float(p_rand[..., 0] if p_rand.ndim > 0 else p_rand) * target_std + target_mean
        base_err.append(abs(p - y))
        occ_err.append(abs(p_occ - y))
        rand_err.append(abs(p_rand - y))
        shift_err.append(abs(p_shift - y))
        shift_deltas.append(abs(p_shift - y) - abs(p_occ - y))
        device_frac.append(float(mask.mean()))
        if grid_saved < 8:
            fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
            axes[0].imshow(sample["image"][0].numpy(), cmap="gray")
            axes[0].set_title("原图")
            axes[1].imshow(sample["image"][0].numpy(), cmap="gray")
            axes[1].imshow(cam, cmap="jet", alpha=0.45)
            _stage_name = "最后 stage（10×10）" if args.stage == -1 else "倒数第 2 stage（20×20）"
            axes[1].set_title(f"CAM（{_stage_name} 上采样）")
            axes[2].imshow(x_occ[0, 0].detach().cpu().numpy(), cmap="gray")
            axes[2].set_title("遮挡高显著区")
            axes[3].imshow(x_shift[0, 0].detach().cpu().numpy(), cmap="gray")
            axes[3].set_title("遮挡等形状随机平移区")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / f"fig_attention_camfix_{args.tag}_{grid_saved}.png", dpi=160)
            plt.close(fig)
            grid_saved += 1
    out = {
        "n": len(idxs),
        "cam_stage": args.stage,
        "base_mae": float(np.mean(base_err)),
        "occlusion_mae": float(np.mean(occ_err)),
        "shifted_occlusion_mae": float(np.mean(shift_err)),
        "random_occlusion_mae": float(np.mean(rand_err)),
        "delta_occlusion": float(np.mean(occ_err) - np.mean(base_err)),
        "delta_shifted": float(np.mean(shift_err) - np.mean(base_err)),
        "delta_random": float(np.mean(rand_err) - np.mean(base_err)),
        "delta_shifted_minus_salient": float(np.mean(shift_err) - np.mean(occ_err)),
        "n_shift_worse_than_salient": int(np.sum(np.array(shift_deltas) > 0)),
        "device_mask_fraction_mean": float(np.mean(device_frac)),
        "note": "CAM 已修复 NHWC/NCHW 布局错误（2026-10-06）；等形状随机平移作为位置对照；"
                "随机像素遮挡会引入伪影，仅作参考。器械 CAM 人工核对已按 WAIVER 豁免。",
    }
    save_json(TABLE_DIR / f"attention_occlusion_{args.tag}.json", out)
    log.info(out)


if __name__ == "__main__":
    main()
