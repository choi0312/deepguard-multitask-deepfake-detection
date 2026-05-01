from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DeepfakeFrameDataset
from src.data.transforms import eval_transform
from src.models.multitask import build_multitask_model
from src.utils.config import load_config, get_output_dir
from src.utils.logger import save_json
from src.utils.metrics import (
    binary_metrics,
    method_metrics_fake_only,
    multiclass_metrics,
    joint_5class_from_multitask,
    classification_report_dict,
    confusion_matrix_array,
)
from src.utils.system import get_device


def plot_confusion_matrix(cm: np.ndarray, labels, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def run_eval(cfg, checkpoint_path: str, split: str):
    device = get_device()
    split_csv = cfg["data"][f"{split}_csv"]
    ds = DeepfakeFrameDataset(split_csv, transform=eval_transform(cfg["data"]["image_size"]))
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
    )

    model = build_multitask_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    y_rf_true, y_rf_prob, y_method_true, y_method_pred, y_5_true = [], [], [], [], []
    threshold = float(cfg["inference"].get("threshold", 0.5))

    for batch in tqdm(loader, desc=f"Evaluate {split}"):
        images = batch["image"].to(device, non_blocking=True)
        out = model(images)
        prob_rf = torch.sigmoid(out["logits_rf"]).detach().cpu().numpy()
        prob_method = torch.softmax(out["logits_method"], dim=1).detach().cpu().numpy()
        pred_method = prob_method.argmax(axis=1)

        for i in range(len(prob_rf)):
            row = {
                "image_path": batch["image_path"][i],
                "video_id": batch["video_id"][i],
                "label_rf": int(batch["label_rf"][i].item()),
                "label_method": int(batch["label_method"][i].item()),
                "label_5cls": int(batch["label_5cls"][i].item()),
                "prob_fake": float(prob_rf[i]),
                "pred_rf": int(prob_rf[i] >= threshold),
                "pred_method": int(pred_method[i]),
            }
            for j, name in enumerate(cfg["data"]["method_names"]):
                row[f"prob_{name}"] = float(prob_method[i, j])
            rows.append(row)

        y_rf_true.extend(batch["label_rf"].cpu().numpy().astype(int).tolist())
        y_rf_prob.extend(prob_rf.tolist())
        y_method_true.extend(batch["label_method"].cpu().numpy().astype(int).tolist())
        y_method_pred.extend(pred_method.tolist())
        y_5_true.extend(batch["label_5cls"].cpu().numpy().astype(int).tolist())

    y_joint_pred = joint_5class_from_multitask(y_rf_prob, y_method_pred, threshold=threshold)
    fake_mask = np.asarray(y_method_true) >= 0

    metrics = {}
    metrics.update({f"rf_{k}": v for k, v in binary_metrics(y_rf_true, y_rf_prob, threshold=threshold).items()})
    metrics.update(method_metrics_fake_only(y_method_true, y_method_pred))
    metrics.update({f"joint_{k}": v for k, v in multiclass_metrics(y_5_true, y_joint_pred).items()})

    out_dir = get_output_dir(cfg) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(out_dir / f"{split}_frame_predictions.csv", index=False)
    save_json(metrics, out_dir / f"{split}_metrics.json")

    if fake_mask.sum() > 0:
        method_labels = cfg["data"]["method_names"]
        cm_method = confusion_matrix_array(np.asarray(y_method_true)[fake_mask], np.asarray(y_method_pred)[fake_mask], labels=list(range(len(method_labels))))
        plot_confusion_matrix(cm_method, method_labels, out_dir / f"{split}_method_confusion_matrix.png", "Method Confusion Matrix")
        report = classification_report_dict(np.asarray(y_method_true)[fake_mask], np.asarray(y_method_pred)[fake_mask], target_names=method_labels)
        save_json(report, out_dir / f"{split}_method_classification_report.json")

    class_labels = cfg["data"]["class_names_5cls"]
    cm_joint = confusion_matrix_array(y_5_true, y_joint_pred, labels=list(range(len(class_labels))))
    plot_confusion_matrix(cm_joint, class_labels, out_dir / f"{split}_joint_confusion_matrix.png", "Joint 5-class Confusion Matrix")

    print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepGuard checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_eval(cfg, args.checkpoint, args.split)


if __name__ == "__main__":
    main()
