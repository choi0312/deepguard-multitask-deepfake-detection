from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DeepfakeFrameDataset
from src.data.transforms import train_transform, eval_transform
from src.models.multitask import build_multitask_model
from src.training.losses import MultiTaskLoss
from src.utils.config import load_config, save_config, get_output_dir
from src.utils.logger import CsvLogger, save_json
from src.utils.metrics import binary_metrics, method_metrics_fake_only, joint_5class_from_multitask, multiclass_metrics
from src.utils.seed import set_seed
from src.utils.system import get_device, system_info


def compute_class_weights(train_csv: str, num_methods: int, device: torch.device, use_pos_weight: bool, use_method_weight: bool):
    df = pd.read_csv(train_csv)
    pos_weight = None
    method_weight = None

    if use_pos_weight:
        positives = float((df["label_rf"] == 1).sum())
        negatives = float((df["label_rf"] == 0).sum())
        if positives > 0:
            pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)

    if use_method_weight:
        fake_df = df[df["label_method"] >= 0]
        counts = fake_df["label_method"].value_counts().reindex(range(num_methods), fill_value=0).astype(float).values
        counts = torch.tensor(counts, dtype=torch.float32, device=device)
        if torch.all(counts > 0):
            inv = counts.sum() / (num_methods * counts)
            method_weight = inv / inv.mean()

    return pos_weight, method_weight


def build_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_ds = DeepfakeFrameDataset(data_cfg["train_csv"], transform=train_transform(data_cfg["image_size"]))
    val_ds = DeepfakeFrameDataset(data_cfg["val_csv"], transform=eval_transform(data_cfg["image_size"]))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def build_optimizer(cfg: Dict, model: torch.nn.Module):
    name = cfg["train"].get("optimizer", "adamw").lower()
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"].get("weight_decay", 0.0))
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(cfg: Dict, optimizer, steps_per_epoch: int):
    scheduler_name = cfg["train"].get("scheduler", "cosine")
    if scheduler_name == "none":
        return None
    epochs = int(cfg["train"]["epochs"])
    min_lr = float(cfg["train"].get("min_lr", 1e-6))
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, cfg, epoch: int):
    model.train()
    running = {"loss": 0.0, "loss_rf": 0.0, "loss_method": 0.0}
    total_samples = 0
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    grad_clip_norm = cfg["train"].get("grad_clip_norm", None)

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        y_rf = batch["label_rf"].to(device, non_blocking=True)
        y_method = batch["label_method"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            out = model(images)
            losses = criterion(out["logits_rf"], out["logits_method"], y_rf, y_method)
            loss = losses["loss"]

        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_samples += bs
        for k in running:
            running[k] += float(losses[k]) * bs
        pbar.set_postfix(loss=running["loss"] / max(total_samples, 1))

    return {k: v / max(total_samples, 1) for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, cfg) -> Dict[str, float]:
    model.eval()
    running = {"loss": 0.0, "loss_rf": 0.0, "loss_method": 0.0}
    total_samples = 0
    y_rf_true, y_rf_prob, y_method_true, y_method_pred, y_5_true = [], [], [], [], []
    threshold = float(cfg["inference"].get("threshold", 0.5))

    for batch in tqdm(loader, desc="Validate", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        y_rf = batch["label_rf"].to(device, non_blocking=True)
        y_method = batch["label_method"].to(device, non_blocking=True)
        out = model(images)
        losses = criterion(out["logits_rf"], out["logits_method"], y_rf, y_method)

        prob_rf = torch.sigmoid(out["logits_rf"]).detach().cpu().numpy()
        pred_method = out["logits_method"].argmax(dim=1).detach().cpu().numpy()

        bs = images.size(0)
        total_samples += bs
        for k in running:
            running[k] += float(losses[k]) * bs

        y_rf_true.extend(y_rf.cpu().numpy().astype(int).tolist())
        y_rf_prob.extend(prob_rf.tolist())
        y_method_true.extend(y_method.cpu().numpy().astype(int).tolist())
        y_method_pred.extend(pred_method.tolist())
        y_5_true.extend(batch["label_5cls"].cpu().numpy().astype(int).tolist())

    rf = binary_metrics(y_rf_true, y_rf_prob, threshold=threshold)
    method = method_metrics_fake_only(y_method_true, y_method_pred)
    joint_pred = joint_5class_from_multitask(y_rf_prob, y_method_pred, threshold=threshold)
    joint = multiclass_metrics(y_5_true, joint_pred)

    metrics = {k: v / max(total_samples, 1) for k, v in running.items()}
    metrics.update({f"rf_{k}": v for k, v in rf.items()})
    metrics.update(method)
    metrics.update({f"joint_{k}": v for k, v in joint.items()})
    metrics["composite"] = metrics["rf_f1"] + 0.5 * metrics["method_macro_f1"]
    return metrics


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: Dict[str, float], cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
        "config": cfg,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepGuard multi-task model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"].get("seed", 42)))
    device = get_device()
    output_dir = get_output_dir(cfg)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.resolved.yaml")
    save_json(system_info(), output_dir / "logs" / "system_info.json")

    train_loader, val_loader = build_loaders(cfg)
    model = build_multitask_model(cfg).to(device)

    pos_weight, method_weight = compute_class_weights(
        cfg["data"]["train_csv"],
        num_methods=int(cfg["model"].get("num_methods", 4)),
        device=device,
        use_pos_weight=bool(cfg["loss"].get("use_pos_weight", True)),
        use_method_weight=bool(cfg["loss"].get("use_method_class_weight", True)),
    )

    criterion = MultiTaskLoss(
        lambda_method=float(cfg["loss"].get("lambda_method", 1.0)),
        pos_weight=pos_weight,
        method_weight=method_weight,
        method_loss=cfg["loss"].get("method_loss", "ce"),
        focal_gamma=float(cfg["loss"].get("focal_gamma", 1.5)),
        label_smoothing=float(cfg["loss"].get("label_smoothing", 0.0)),
    ).to(device)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and bool(cfg["train"].get("amp", True))))

    fieldnames = [
        "epoch", "lr", "train_loss", "train_loss_rf", "train_loss_method",
        "val_loss", "val_loss_rf", "val_loss_method", "val_rf_acc", "val_rf_f1",
        "val_method_acc", "val_method_macro_f1", "val_joint_acc", "val_joint_macro_f1", "val_composite",
    ]
    logger = CsvLogger(output_dir / "logs" / "train_log.csv", fieldnames=fieldnames)

    selection_metric = cfg["train"].get("selection_metric", "composite")
    if selection_metric == "composite":
        best_key = "composite"
    elif selection_metric == "rf_f1":
        best_key = "rf_f1"
    else:
        best_key = selection_metric

    best_score = -math.inf
    best_metrics = {}
    patience = int(cfg["train"].get("early_stopping_patience", 5))
    stale_epochs = 0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, cfg, epoch)
        val_m = validate(model, val_loader, criterion, device, cfg)
        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_m["loss"],
            "train_loss_rf": train_m["loss_rf"],
            "train_loss_method": train_m["loss_method"],
            "val_loss": val_m["loss"],
            "val_loss_rf": val_m["loss_rf"],
            "val_loss_method": val_m["loss_method"],
            "val_rf_acc": val_m["rf_acc"],
            "val_rf_f1": val_m["rf_f1"],
            "val_method_acc": val_m["method_acc"],
            "val_method_macro_f1": val_m["method_macro_f1"],
            "val_joint_acc": val_m["joint_acc"],
            "val_joint_macro_f1": val_m["joint_macro_f1"],
            "val_composite": val_m["composite"],
        }
        logger.log(row)
        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['train_loss']:.4f}/{row['val_loss']:.4f} | "
            f"RF F1={row['val_rf_f1']:.4f} | "
            f"Method Macro-F1={row['val_method_macro_f1']:.4f} | "
            f"Joint Acc={row['val_joint_acc']:.4f}"
        )

        current_score = val_m.get(best_key, -math.inf)
        save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, epoch, val_m, cfg)

        if current_score > best_score:
            best_score = current_score
            best_metrics = val_m
            stale_epochs = 0
            save_checkpoint(output_dir / "checkpoints" / "best.pt", model, optimizer, scheduler, epoch, val_m, cfg)
            print(f"  -> New best checkpoint saved: {best_key}={best_score:.4f}")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping triggered after {patience} stale epochs.")
                break

    save_json({"best_score": best_score, "best_metric_key": best_key, "best_metrics": best_metrics}, output_dir / "logs" / "summary.json")
    print(f"Training finished. Best {best_key}: {best_score:.4f}")


if __name__ == "__main__":
    main()
