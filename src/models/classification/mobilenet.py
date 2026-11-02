"""
Module: mobilenet
Description: MobileNetV2 transfer learning model for Solar Panel Defect Classification
Author: SolarVision AI Team
Created: 2026-11-02
"""

import warnings
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class SolarMobileNetV2(nn.Module):
    """MobileNetV2 transfer learning architecture for solar panel classification.

    Supports fully frozen backbone (linear probe), partially unfrozen backbone
    (fine-tuning the last N inverted residual blocks), or fully unfrozen training.
    """

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_blocks: int = 0,
        dropout: float = 0.3,
    ) -> None:
        """Initialize SolarMobileNetV2.

        Args:
            num_classes: Number of defect classes. Defaults to 6.
            pretrained: Whether to load ImageNet pre-trained weights. Defaults to True.
            freeze_backbone: If True, freezes backbone weights.
            unfreeze_blocks: Number of final inverted residual blocks to keep unfrozen
                             when freeze_backbone is True. Defaults to 0.
            dropout: Dropout probability in the classifier head. Defaults to 0.3.
        """
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        try:
            base_model = mobilenet_v2(weights=weights)
        except Exception as exc:
            warnings.warn(
                f"Failed to load pretrained weights: {exc}. Initializing randomly."
            )
            base_model = mobilenet_v2(weights=None)

        self.features = base_model.features

        # Backbone freezing logic
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

            if unfreeze_blocks > 0:
                # Features in mobilenet_v2 has 19 sequential sub-modules (0 to 18)
                # Unfreeze the last `unfreeze_blocks` layers
                num_layers = len(self.features)
                for i in range(max(0, num_layers - unfreeze_blocks), num_layers):
                    for param in self.features[i].parameters():
                        param.requires_grad = True

        in_features = base_model.classifier[1].in_features  # 1280
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes).
        """
        feat = self.features(x)
        pooled = self.pool(feat)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits
