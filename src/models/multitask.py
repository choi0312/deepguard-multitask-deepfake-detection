from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import timm


class DeepGuardMultiTask(nn.Module):
    """Multi-task deepfake detector.

    Outputs:
        logits_rf: shape (B,), binary real/fake logit. Higher means fake.
        logits_method: shape (B, num_methods), fake manipulation method logits.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        num_methods: int = 4,
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.shared_projection = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.rf_head = nn.Linear(hidden_dim, 1)
        self.method_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_methods),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        z = self.shared_projection(features)
        logits_rf = self.rf_head(z).squeeze(1)
        logits_method = self.method_head(z)
        return {"logits_rf": logits_rf, "logits_method": logits_method, "features": z}


def build_multitask_model(cfg: Dict) -> DeepGuardMultiTask:
    model_cfg = cfg["model"]
    return DeepGuardMultiTask(
        backbone=model_cfg.get("backbone", "resnet34"),
        num_methods=int(model_cfg.get("num_methods", 4)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
    )
