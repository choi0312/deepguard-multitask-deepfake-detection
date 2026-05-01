from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.preprocess import extract_frames
from src.data.transforms import eval_transform
from src.models.multitask import build_multitask_model
from src.utils.config import load_config
from src.utils.system import get_device


@torch.no_grad()
def predict_video(cfg, checkpoint: str, video_path: str, max_frames: int | None = None, frame_stride: int | None = None):
    device = get_device()
    model = build_multitask_model(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    max_frames = max_frames or int(cfg["inference"].get("max_frames", 32))
    frame_stride = frame_stride or int(cfg["inference"].get("frame_stride", 10))
    threshold = float(cfg["inference"].get("threshold", 0.5))
    transform = eval_transform(cfg["data"]["image_size"])

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = extract_frames(video_path, tmpdir, stride=frame_stride, max_frames=max_frames)
        if not frame_paths:
            raise RuntimeError(f"No frames were extracted from {video_path}")

        fake_probs = []
        method_probs = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            out = model(x)
            fake_probs.append(float(torch.sigmoid(out["logits_rf"])[0].item()))
            method_probs.append(torch.softmax(out["logits_method"], dim=1)[0].cpu().numpy())

    fake_prob = float(np.mean(fake_probs))
    avg_method_probs = np.mean(np.stack(method_probs, axis=0), axis=0)
    method_idx = int(avg_method_probs.argmax())
    method_name = cfg["data"]["method_names"][method_idx]

    result = {
        "video_path": str(video_path),
        "num_frames_used": len(fake_probs),
        "fake_probability": fake_prob,
        "predicted_is_fake": bool(fake_prob >= threshold),
        "predicted_method": method_name if fake_prob >= threshold else "N/A",
        "method_probabilities": {name: float(avg_method_probs[i]) for i, name in enumerate(cfg["data"]["method_names"])},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepGuard video inference.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = predict_video(cfg, args.checkpoint, args.video_path, args.max_frames, args.frame_stride)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
