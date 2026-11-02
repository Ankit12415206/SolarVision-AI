"""
Module: models
Description: Deep learning architectures for Classification, Detection, and Segmentation
Author: SolarVision AI Team
Created: 2026-11-02
"""

from src.models.classification import (
    SolarCustomCNN,
    SolarMobileNetV2,
    build_classification_model,
)
from src.models.detection import (
    SolarYOLOv8n,
    build_detection_model,
)
from src.models.segmentation import (
    BCEDiceLoss,
    DiceLoss,
    SolarLightweightUNet,
    build_segmentation_model,
)

__all__ = [
    "SolarCustomCNN",
    "SolarMobileNetV2",
    "build_classification_model",
    "SolarYOLOv8n",
    "build_detection_model",
    "SolarLightweightUNet",
    "DiceLoss",
    "BCEDiceLoss",
    "build_segmentation_model",
]
