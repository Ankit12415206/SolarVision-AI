"""
Module: inference
Description: Unified inference module for SolarVision AI
Author: SolarVision AI Team
Created: 2026-11-02
"""

from src.inference.pipeline import SolarVisionInferencePipeline
from src.inference.types import BoundingBox, InferenceResult, InspectionType
from src.inference.visualizer import (
    annotate_classification,
    annotate_detection,
    annotate_segmentation,
)

__all__ = [
    "SolarVisionInferencePipeline",
    "InspectionType",
    "BoundingBox",
    "InferenceResult",
    "annotate_classification",
    "annotate_detection",
    "annotate_segmentation",
]
