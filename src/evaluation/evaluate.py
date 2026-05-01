from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DeepfakeFrameDataset
from src.data.transforms import build_transforms
from src.models.multitask import DeepGuardMultiTask
from src.utils.config import load_config
from src.utils.metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = cfg["data"][f"{args.split}_csv"]
    dataset = DeepfakeFrameDataset(
        csv_path,
        transform=build_transforms(int(cfg["data"]["image_size"]), train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 2)),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DeepGuardMultiTask(**cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rf_targets = []
    rf_probs = []
    method_targets = []
    method_preds = []

    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["image"].to(device)
            outputs = model(images)

            rf_targets.append(batch["label_rf"].numpy())
            rf_probs.append(torch.sigmoid(outputs["rf_logit"]).cpu().numpy())
            method_targets.append(batch["label_method"].numpy())
            method_preds.append(torch.argmax(outputs["method_logits"], dim=1).cpu().numpy())

    metrics = compute_metrics(
        rf_targets=np.concatenate(rf_targets).astype(int),
        rf_probs=np.concatenate(rf_probs),
        method_targets=np.concatenate(method_targets),
        method_preds=np.concatenate(method_preds),
    )

    output_dir = Path(cfg["project"]["output_dir"]) / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_metrics.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
