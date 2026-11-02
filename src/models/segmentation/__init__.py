"""
Module: segmentation
Description: Segmentation model architectures, loss functions, and constructors
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Any

import torch.nn as nn

from src.models.segmentation.losses import BCEDiceLoss, DiceLoss
from src.models.segmentation.unet import SolarLightweightUNet


def build_segmentation_model(
    model_name: str = "unet",
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 16,
    **kwargs: Any,
) -> nn.Module:
    """Build and return a segmentation model.

    Args:
        model_name: Model identifier ('unet', 'lightweight_unet'). Defaults to 'unet'.
        in_channels: Input channels (1 for grayscale EL images). Defaults to 1.
        out_channels: Output logits channels (1 for binary defect masks). Defaults to 1.
        base_channels: Initial channel multiplier. Defaults to 16.
        **kwargs: Additional parameters.

    Returns:
        nn.Module: Instantiated segmentation model.
    """
    return SolarLightweightUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    )


__all__ = [
    "SolarLightweightUNet",
    "DiceLoss",
    "BCEDiceLoss",
    "build_segmentation_model",
]
