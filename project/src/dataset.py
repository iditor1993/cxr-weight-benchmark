"""PyTorch 数据集：读取 320px npy 缓存 + 队列标签 + 切分。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import FEATURE_DIR, PROCESSED_DIR


class WeightCXR(Dataset):
    def __init__(
        self,
        split: str,
        split_kind: str = "random",
        size: int = 320,
        feature_set: str = "index",
        train: bool = False,
        cache_dir: Path | None = None,
        mean: float = 0.5,
        std: float = 0.5,
        split_file: str | None = None,
        split_col: str | None = None,
        augment_profile: str = "default",
    ):
        self.size = size
        self.train = train
        self.augment_profile = augment_profile
        self.mean, self.std = mean, std
        cohort = pd.read_csv(PROCESSED_DIR / "cohort_a_index.csv"
                             if feature_set == "index" else
                             PROCESSED_DIR / "cohort_a_all_images.csv")
        split_path = PROCESSED_DIR / (split_file or (
            "splits_random.csv" if split_kind == "random" else "splits_temporal.csv"))
        splits = pd.read_csv(split_path)
        split_col = split_col or ("split_random" if split_kind == "random" else "split_temporal")
        self.df = cohort.merge(splits[["subject_id", split_col]], on="subject_id", how="inner")
        self.df = self.df[self.df[split_col] == split].reset_index(drop=True)
        # 缓存固定为 320 px；其他尺寸在读取时缩放
        if cache_dir is not None:
            self.cache_dir = cache_dir
        else:
            cand_320 = FEATURE_DIR / f"{feature_set}_320"
            cand_native = FEATURE_DIR / f"{feature_set}_{size}"
            self.cache_dir = cand_320 if cand_320.exists() else cand_native
        self.df = self.df[self.df["dicom_id"].map(
            lambda d: (self.cache_dir / f"{d}.npy").exists())].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = np.load(self.cache_dir / f"{row['dicom_id']}.npy")
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        if img.shape[-1] != self.size:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(self.size, self.size),
                mode="bilinear", align_corners=False).squeeze(0)
        if self.train:
            img = self._augment(img)
        img = (img - self.mean) / self.std
        img = img.repeat(3, 1, 1)
        return {
            "image": img,
            "weight": torch.tensor(float(row["weight_kg"]), dtype=torch.float32),
            "subject_id": int(row["subject_id"]),
            "stay_id": int(row["stay_id"]),
            "dicom_id": row["dicom_id"],
        }

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        # 随机亮度/对比度
        if torch.rand(1).item() < 0.8:
            img = img * (0.9 + 0.2 * torch.rand(1).item())
            img = img + (torch.rand(1).item() - 0.5) * 0.1
            img = img.clamp(0, 1)
        # 随机平移/缩放（保持解剖语义，不做水平翻转）
        if self.augment_profile != "geom_off" and torch.rand(1).item() < 0.5:
            import torchvision.transforms.functional as TF
            angle = float((torch.rand(1).item() - 0.5) * 20.0)
            scale = 0.9 + 0.2 * torch.rand(1).item()
            img = TF.affine(img, angle=angle, translate=[0, 0], scale=scale, shear=0)
        # 随机遮挡
        if torch.rand(1).item() < 0.3:
            h, w = img.shape[-2:]
            ch, cw = int(h * 0.15), int(w * 0.15)
            y = torch.randint(0, h - ch, (1,)).item()
            x = torch.randint(0, w - cw, (1,)).item()
            img = img.clone()
            img[..., y:y + ch, x:x + cw] = 0.0
        return img
