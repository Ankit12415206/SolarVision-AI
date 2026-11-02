"""
Module: training
Description: Training infrastructure, trainer classes, and callbacks
Author: SolarVision AI Team
Created: 2026-11-02
"""

from src.training.callbacks import EarlyStopping
from src.training.detection_trainer import DetectionTrainer
from src.training.segmentation_trainer import SegmentationTrainer
from src.training.trainer import ClassificationTrainer

__all__ = [
    "EarlyStopping",
    "ClassificationTrainer",
    "DetectionTrainer",
    "SegmentationTrainer",
]
