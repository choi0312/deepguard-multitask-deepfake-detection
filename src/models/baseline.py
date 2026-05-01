from __future__ import annotations

import torch
from torch import nn
import timm


class FiveClassBaseline(nn.Module):
    """Single-head baseline for 5-class classification.

    Classes: Real + four fake manipulation methods.
    """

    def __init__(self, backbone: str = "resnet34", num_classes: int = 5, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.classifier(feat)
