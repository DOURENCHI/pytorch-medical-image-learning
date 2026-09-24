from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class XrayDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.mask_paths = sorted(self.mask_dir.glob("*.png"))

        print("找到图像:", len(self.image_paths))
        print("找到标签:", len(self.mask_paths))

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("图像数量和 Mask 数量不一致！")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # =========================
        # 读取 X-ray
        # =========================

        image = Image.open(image_path)
        image = np.array(image).astype(np.float32)

        # =========================
        # Resize X-ray
        # =========================

        image = Image.fromarray(image)

        image = image.resize(
            (512, 512),
            resample=Image.Resampling.BILINEAR
        )

        image = np.array(image).astype(np.float32)

        # =========================
        # Resize 后归一化
        # =========================

        image = image - image.min()

        if image.max() > 0:
            image = image / image.max()

        # =========================
        # 读取 Mask
        # =========================

        mask = Image.open(mask_path).convert("L")

        mask = np.array(mask).astype(np.float32)

        mask = mask / 255.0

        # =========================
        # Resize Mask
        # =========================

        mask = Image.fromarray(mask)

        mask = mask.resize(
            (512, 512),
            resample=Image.Resampling.NEAREST
        )

        mask = np.array(mask).astype(np.float32)

        # =========================
        # 转 Tensor
        # =========================

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # [H,W]
        # ↓
        # [C,H,W]

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return image, mask