"""模型工厂：timm 骨干（ImageNet）与胸片域预训练模型（RadDINO）。"""

from __future__ import annotations

import torch
import torch.nn as nn

TIMM_NAMES = {
    "resnet50": "resnet50.a1_in1k",
    "densenet121": "densenet121.ra_in1k",
    "vit_b_16": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "swin_t": "swin_tiny_patch4_window7_224.ms_in1k",
    "convnext_t": "convnext_tiny.fb_in1k",
}

NORMALIZATION = {
    "rad_dino": (0.5307, 0.2583),
    "default": (0.5, 0.5),
}


class RadDinoRegressor(nn.Module):
    """RadDINO（DINOv2 胸片版）骨干 + 线性回归头。"""

    def __init__(self, out_dim: int = 1, freeze_backbone: bool = False):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(
            "microsoft/rad-dino", local_files_only=True)
        hidden = int(self.backbone.config.hidden_size)
        self.head = nn.Linear(hidden, out_dim)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        return self.head(pooled)


def build_model(backbone: str, out_dim: int = 1, pretrained: bool = True,
                freeze_backbone: bool = False, img_size: int | None = None) -> nn.Module:
    backbone = backbone.lower()
    if backbone in ("rad_dino", "raddino"):
        return RadDinoRegressor(out_dim=out_dim, freeze_backbone=freeze_backbone)

    import timm

    if backbone not in TIMM_NAMES:
        raise ValueError(f"unknown backbone: {backbone}")
    kwargs = {}
    if backbone == "swin_t":
        # Swin 需要在创建时指定图像尺寸；取最接近 320 且能被 patch(4)×window(7) 整除的 308
        kwargs["img_size"] = int(img_size or 308)
    return timm.create_model(TIMM_NAMES[backbone], pretrained=pretrained,
                             num_classes=out_dim, **kwargs)


def normalization_for(backbone: str) -> tuple[float, float]:
    return NORMALIZATION.get(backbone.lower(), NORMALIZATION["default"])


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
