"""
Module: data
Description: Data loading, preprocessing pipelines, and datasets for SolarVision AI
Author: SolarVision AI Team
Created: 2026-11-02
"""

from src.data.classification import (
    CLASS_TO_IDX,
    CLASSIFICATION_CLASSES,
    IDX_TO_CLASS,
    SolarClassificationDataset,
    create_classification_splits,
    get_classification_dataloaders,
    scan_classification_images,
)
from src.data.detection import (
    DETECTION_CLASSES,
    SolarDetectionDataset,
    detection_collate_fn,
    get_detection_dataloaders,
    parse_yolo_label,
    verify_and_manifest_detection_data,
)
from src.data.segmentation import (
    SolarSegmentationDataset,
    create_segmentation_splits,
    get_segmentation_dataloaders,
    scan_segmentation_data,
)
from src.data.transforms import (
    DualSegmentationTransform,
    get_classification_transforms,
)

__all__ = [
    "CLASSIFICATION_CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "SolarClassificationDataset",
    "create_classification_splits",
    "get_classification_dataloaders",
    "scan_classification_images",
    "DETECTION_CLASSES",
    "SolarDetectionDataset",
    "detection_collate_fn",
    "get_detection_dataloaders",
    "parse_yolo_label",
    "verify_and_manifest_detection_data",
    "SolarSegmentationDataset",
    "create_segmentation_splits",
    "get_segmentation_dataloaders",
    "scan_segmentation_data",
    "DualSegmentationTransform",
    "get_classification_transforms",
]
