from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PatchSegDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        split_file = self.root / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        self.ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        image = Image.open(self.root / "images" / f"{sid}.png").convert("RGB")
        mask = Image.open(self.root / "masks" / f"{sid}.png")

        image_arr = np.asarray(image).astype("float32") / 255.0
        mask_arr = np.asarray(mask).astype("int64")

        image_t = torch.from_numpy(image_arr).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_arr)
        return image_t, mask_t

