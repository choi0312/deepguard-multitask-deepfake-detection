from __future__ import annotations

import argparse
import json

import torch
from PIL import Image

from src.data.transforms import build_transforms
from src.models.multitask import DeepGuardMultiTask
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DeepGuardMultiTask(**cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transforms(int(cfg["data"]["image_size"]), train=False)
    image = Image.open(args.image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        fake_prob = torch.sigmoid(outputs["rf_logit"])[0].item()
        method_probs = torch.softmax(outputs["method_logits"], dim=1)[0].cpu().numpy()

    method_names = cfg["classes"]["methods"]
    method_idx = int(method_probs.argmax())

    result = {
        "image_path": args.image_path,
        "fake_probability": float(fake_prob),
        "is_fake": bool(fake_prob >= float(cfg["inference"]["threshold"])),
        "predicted_method": method_names[method_idx],
        "method_probabilities": {
            name: float(prob) for name, prob in zip(method_names, method_probs)
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
