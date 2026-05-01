from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DeepfakeFrameDataset
from src.data.transforms import build_transforms
from src.models.multitask import DeepGuardMultiTask
from src.utils.config import load_config
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    bce_loss: nn.Module,
    ce_loss: nn.Module,
    device: torch.device,
    lambda_method: float,
    scaler: GradScaler | None = None,
    gradient_clip_norm: float = 1.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    rf_targets = []
    rf_probs = []
    method_targets = []
    method_preds = []

    iterator = tqdm(loader, leave=False)

    for batch in iterator:
        images = batch["image"].to(device)
        y_rf = batch["label_rf"].to(device)
        y_method = batch["label_method"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            use_amp = scaler is not None and device.type == "cuda"

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss_rf = bce_loss(outputs["rf_logit"], y_rf)

                fake_mask = y_method >= 0
                if fake_mask.any():
                    loss_method = ce_loss(
                        outputs["method_logits"][fake_mask],
                        y_method[fake_mask],
                    )
                else:
                    loss_method = torch.tensor(0.0, device=device)

                loss = loss_rf + lambda_method * loss_method

            if is_train:
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()

        losses.append(float(loss.detach().cpu()))

        prob = torch.sigmoid(outputs["rf_logit"]).detach().cpu().numpy()
        pred_method = torch.argmax(outputs["method_logits"], dim=1).detach().cpu().numpy()

        rf_targets.append(y_rf.detach().cpu().numpy())
        rf_probs.append(prob)
        method_targets.append(y_method.detach().cpu().numpy())
        method_preds.append(pred_method)

    rf_targets_np = np.concatenate(rf_targets).astype(int)
    rf_probs_np = np.concatenate(rf_probs)
    method_targets_np = np.concatenate(method_targets)
    method_preds_np = np.concatenate(method_preds)

    metrics = compute_metrics(
        rf_targets=rf_targets_np,
        rf_probs=rf_probs_np,
        method_targets=method_targets_np,
        method_preds=method_preds_np,
    )
    metrics["loss"] = float(np.mean(losses))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"].get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg["project"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(cfg["data"]["image_size"])

    train_dataset = DeepfakeFrameDataset(
        cfg["data"]["train_csv"],
        transform=build_transforms(image_size, train=True),
    )
    val_dataset = DeepfakeFrameDataset(
        cfg["data"]["val_csv"],
        transform=build_transforms(image_size, train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 2)),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 2)),
        pin_memory=True,
    )

    model = DeepGuardMultiTask(**cfg["model"]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    use_amp = bool(cfg["training"].get("use_amp", True)) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    best_val_f1 = -1.0
    patience = int(cfg["training"].get("patience", 5))
    bad_epochs = 0
    history = []

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        print(f"\nEpoch {epoch}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            bce_loss=bce_loss,
            ce_loss=ce_loss,
            device=device,
            lambda_method=float(cfg["training"]["lambda_method"]),
            scaler=scaler,
            gradient_clip_norm=float(cfg["training"].get("gradient_clip_norm", 1.0)),
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            bce_loss=bce_loss,
            ce_loss=ce_loss,
            device=device,
            lambda_method=float(cfg["training"]["lambda_method"]),
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        print("train:", train_metrics)
        print("val:", val_metrics)

        val_f1 = val_metrics["task1_f1"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            bad_epochs = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "best_val_f1": best_val_f1,
            }
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            print(f"Saved best checkpoint: {checkpoint_dir / 'best.pt'}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break

    with (metrics_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
