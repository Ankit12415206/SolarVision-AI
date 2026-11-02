"""
Module: losses
Description: Numerically stable Dice and BCE loss functions for imbalanced segmentation
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Optional

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary semantic segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        """Initialize DiceLoss.

        Args:
            smooth: Laplace smoothing constant to prevent division by zero. Defaults to 1.0.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: Raw prediction tensor of shape (B, 1, H, W).
            targets: Binary ground truth tensor of shape (B, 1, H, W) with values {0.0, 1.0}.

        Returns:
            torch.Tensor: Scalar Dice loss.
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy and Soft Dice Loss.

    Designed specifically for extreme defect class imbalances in solar cell EL imagery.
    Applies positive weighting to the BCE component to emphasize sparse defect pixels,
    capping extreme weights to maintain strict numerical stability.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = 15.0,
        max_pos_weight: float = 50.0,
        smooth: float = 1.0,
    ) -> None:
        """Initialize BCEDiceLoss.

        Args:
            bce_weight: Weight assigned to BCE loss component. Defaults to 0.5.
            dice_weight: Weight assigned to Dice loss component. Defaults to 0.5.
            pos_weight: Positive weight for BCE loss. If raw dataset imbalance
                        (e.g., 1939.69) is passed, it is capped to max_pos_weight
                        to prevent gradient instability. Defaults to 15.0.
            max_pos_weight: Maximum allowable positive weight. Defaults to 50.0.
            smooth: Smoothing constant for Dice loss. Defaults to 1.0.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth)

        if pos_weight is not None:
            # Prevent gradient explosion from raw extreme imbalance values
            effective_pos_weight = min(float(pos_weight), max_pos_weight)
            weight_tensor = torch.tensor([effective_pos_weight], dtype=torch.float32)
            self.bce_loss: nn.Module = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined BCE and Dice loss.

        Args:
            logits: Predicted logits of shape (B, 1, H, W).
            targets: Binary targets of shape (B, 1, H, W).

        Returns:
            torch.Tensor: Weighted combined loss scalar.
        """
        # Ensure target matches logits device and dtype
        targets = targets.to(dtype=logits.dtype, device=logits.device)
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
