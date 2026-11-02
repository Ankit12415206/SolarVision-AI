"""
Module: detection
Description: Object detection model architectures and factory constructors
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Any, Dict

import torch.nn as nn

from src.models.detection.yolo_lightweight import SolarYOLOv8n


def build_detection_model(
    model_name: str = "yolov8n",
    num_classes: int = 2,
    in_channels: int = 3,
    base_channels: int = 16,
    **kwargs: Any,
) -> nn.Module:
    """Build and return an object detection model.

    Args:
        model_name: Model identifier ('yolov8n', 'yolo'). Defaults to 'yolov8n'.
        num_classes: Number of detection target classes. Defaults to 2.
        in_channels: Input image channel depth. Defaults to 3.
        base_channels: Channel width scale factor. Defaults to 16.
        **kwargs: Additional model parameters.

    Returns:
        nn.Module: Instantiated detection model.
    """
    return SolarYOLOv8n(
        num_classes=num_classes,
        in_channels=in_channels,
        base_channels=base_channels,
    )


__all__ = [
    "SolarYOLOv8n",
    "build_detection_model",
]
