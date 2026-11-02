"""
Module: classification
Description: Classification model architectures and factory constructors
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Any, Dict, Optional, Union

import torch.nn as nn

from src.models.classification.custom_cnn import SolarCustomCNN
from src.models.classification.mobilenet import SolarMobileNetV2


def build_classification_model(
    model_name: str = "custom_cnn",
    num_classes: int = 6,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    unfreeze_blocks: int = 0,
    dropout: float = 0.3,
    **kwargs: Any,
) -> nn.Module:
    """Build and return a classification model instance.

    Args:
        model_name: Identifier ('custom_cnn', 'mobilenet_v2').
        num_classes: Number of target categories. Defaults to 6.
        pretrained: For pretrained transfer learning models. Defaults to True.
        freeze_backbone: Whether to freeze backbone weights.
        unfreeze_blocks: Number of final blocks to unfreeze.
        dropout: Head dropout rate.
        **kwargs: Additional model parameters.

    Returns:
        nn.Module: Instantiated classification model.

    Raises:
        ValueError: If model_name is unrecognized.
    """
    name = model_name.lower().strip()
    if name in ["custom_cnn", "baseline", "cnn"]:
        return SolarCustomCNN(
            num_classes=num_classes,
            dropout=dropout,
            **kwargs,
        )
    elif name in ["mobilenet_v2", "mobilenet"]:
        return SolarMobileNetV2(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            unfreeze_blocks=unfreeze_blocks,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown classification model '{model_name}'. "
            f"Available: ['custom_cnn', 'mobilenet_v2']"
        )


__all__ = [
    "SolarCustomCNN",
    "SolarMobileNetV2",
    "build_classification_model",
]
