from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


class FocalCrossEntropyLoss(nn.Module):
    """Focal-style cross entropy for difficult method-class samples."""

    def __init__(self, gamma: float = 1.5, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """BCE real/fake loss + fake-only method classification loss."""

    def __init__(
        self,
        lambda_method: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
        method_weight: Optional[torch.Tensor] = None,
        method_loss: str = "ce",
        focal_gamma: float = 1.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_method = lambda_method
        self.rf_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if method_loss == "focal_ce":
            self.method_loss = FocalCrossEntropyLoss(
                gamma=focal_gamma,
                weight=method_weight,
                label_smoothing=label_smoothing,
            )
        else:
            self.method_loss = nn.CrossEntropyLoss(weight=method_weight, label_smoothing=label_smoothing)

    def forward(
        self,
        logits_rf: torch.Tensor,
        logits_method: torch.Tensor,
        label_rf: torch.Tensor,
        label_method: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        label_rf = label_rf.float()
        loss_rf = self.rf_loss(logits_rf, label_rf)

        fake_mask = label_method >= 0
        if fake_mask.any():
            loss_method = self.method_loss(logits_method[fake_mask], label_method[fake_mask])
        else:
            loss_method = logits_method.sum() * 0.0

        total = loss_rf + self.lambda_method * loss_method
        return {"loss": total, "loss_rf": loss_rf.detach(), "loss_method": loss_method.detach()}
