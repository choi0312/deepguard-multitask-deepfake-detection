from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

REQUIRED_COLUMNS = {"image_path", "video_id", "label_rf", "label_method"}


class DeepfakeFrameDataset(Dataset):
    """Frame-level dataset for real/fake and manipulation-method classification.

    CSV schema:
        image_path, video_id, label_rf, label_method, method_name, split
    """

    def __init__(self, csv_path: str | Path, transform=None, image_root: str | Path | None = None):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {self.csv_path}: {sorted(missing)}")

        self.transform = transform
        self.image_root = Path(image_root) if image_root is not None else None

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        if p.is_absolute():
            return p
        if self.image_root is not None:
            return self.image_root / p
        return p

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self._resolve_path(str(row["image_path"]))
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label_rf = int(row["label_rf"])
        label_method = int(row["label_method"])
        label_5cls = 0 if label_rf == 0 else label_method + 1

        return {
            "image": image,
            "label_rf": torch.tensor(label_rf, dtype=torch.float32),
            "label_method": torch.tensor(label_method, dtype=torch.long),
            "label_5cls": torch.tensor(label_5cls, dtype=torch.long),
            "image_path": str(img_path),
            "video_id": str(row["video_id"]),
        }
