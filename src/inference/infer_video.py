from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.transforms import build_transforms
from src.models.multitask import DeepGuardMultiTask
from src.utils.config import load_config


def sample_video_frames(video_path: str | Path, frame_interval: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames: List[Image.Image] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

        frame_idx += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames sampled from video: {video_path}")

    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DeepGuardMultiTask(**cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    frames = sample_video_frames(
        args.video_path,
        frame_interval=int(cfg["inference"].get("frame_interval", 10)),
    )

    transform = build_transforms(int(cfg["data"]["image_size"]), train=False)

    fake_probs = []
    method_probs = []

    with torch.no_grad():
        for frame in frames:
            tensor = transform(frame).unsqueeze(0).to(device)
            outputs = model(tensor)

            fake_probs.append(torch.sigmoid(outputs["rf_logit"])[0].item())
            method_probs.append(torch.softmax(outputs["method_logits"], dim=1)[0].cpu().numpy())

    fake_probability = float(np.mean(fake_probs))
    avg_method_probs = np.mean(np.stack(method_probs), axis=0)

    method_names = cfg["classes"]["methods"]
    method_idx = int(avg_method_probs.argmax())

    result = {
        "video_path": args.video_path,
        "num_frames": len(frames),
        "fake_probability": fake_probability,
        "is_fake": bool(fake_probability >= float(cfg["inference"]["threshold"])),
        "predicted_method": method_names[method_idx],
        "method_probabilities": {
            name: float(prob) for name, prob in zip(method_names, avg_method_probs)
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
