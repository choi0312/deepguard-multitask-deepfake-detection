from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.data.transforms import eval_transform
from src.models.multitask import build_multitask_model
from src.utils.config import load_config
from src.utils.system import get_device


@torch.no_grad()
def predict_image(cfg, checkpoint: str, image_path: str):
    device = get_device()
    model = build_multitask_model(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = eval_transform(cfg["data"]["image_size"])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    out = model(x)
    prob_fake = float(torch.sigmoid(out["logits_rf"])[0].item())
    method_probs = torch.softmax(out["logits_method"], dim=1)[0].cpu().numpy()
    method_idx = int(method_probs.argmax())
    method_name = cfg["data"]["method_names"][method_idx]
    threshold = float(cfg["inference"].get("threshold", 0.5))

    result = {
        "image_path": str(image_path),
        "fake_probability": prob_fake,
        "predicted_is_fake": bool(prob_fake >= threshold),
        "predicted_method": method_name if prob_fake >= threshold else "N/A",
        "method_probabilities": {name: float(method_probs[i]) for i, name in enumerate(cfg["data"]["method_names"])},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepGuard image inference.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = predict_image(cfg, args.checkpoint, args.image_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
