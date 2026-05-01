from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DeepfakeFrameDataset(Dataset):
    """Frame-level dataset for DeepGuard."""

    def __init__(self, csv_path: str | Path, transform: Optional[Callable] = None):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.transform = transform

        required = {"image_path", "label_rf", "label_method"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {self.csv_path}: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        image_path = Path(row["image_path"])

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label_rf": torch.tensor(float(row["label_rf"]), dtype=torch.float32),
            "label_method": torch.tensor(int(row["label_method"]), dtype=torch.long),
            "image_path": str(image_path),
        }
