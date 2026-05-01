from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class DeepGuardMultiTask(nn.Module):
    """Shared-backbone multi-task model for deepfake detection."""

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        num_methods: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        if backbone != "resnet34":
            raise ValueError(
                f"Unsupported backbone: {backbone}. Currently only resnet34 is supported."
            )

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet34(weights=weights)

        feature_dim = net.fc.in_features
        net.fc = nn.Identity()

        self.backbone = net
        self.binary_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )
        self.method_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_methods),
        )

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        rf_logit = self.binary_head(features).squeeze(1)
        method_logits = self.method_head(features)

        return {
            "rf_logit": rf_logit,
            "method_logits": method_logits,
            "features": features,
        }
