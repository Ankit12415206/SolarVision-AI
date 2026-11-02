"""
Module: types
Description: Data structures, result models, and enums for unified inference pipeline
Author: SolarVision AI Team
Created: 2026-11-02
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


class InspectionType(str, Enum):
    """Enumeration of supported solar panel inspection types."""

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"

    @classmethod
    def from_str(cls, value: str) -> "InspectionType":
        """Parse string to InspectionType enum with aliases.

        Args:
            value: Input string (e.g., 'classification', 'panel_condition', 'detection', 'el_defect').

        Returns:
            InspectionType: Matching enum member.

        Raises:
            ValueError: If value is not a recognized inspection type.
        """
        val_clean = value.strip().lower()
        if val_clean in ("classification", "panel_condition", "condition", "cls"):
            return cls.CLASSIFICATION
        elif val_clean in ("detection", "aerial_soiling", "soiling", "det", "yolo"):
            return cls.DETECTION
        elif val_clean in ("segmentation", "el_defect", "microcrack", "seg", "unet"):
            return cls.SEGMENTATION
        raise ValueError(
            f"Unsupported inspection type '{value}'. Supported types: "
            f"'{cls.CLASSIFICATION.value}', '{cls.DETECTION.value}', '{cls.SEGMENTATION.value}'."
        )


@dataclass
class BoundingBox:
    """Bounding box detection result."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert bounding box to dictionary."""
        return {
            "x1": round(self.x1, 4),
            "y1": round(self.y1, 4),
            "x2": round(self.x2, 4),
            "y2": round(self.y2, 4),
            "score": round(self.score, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class InferenceResult:
    """Standardized result schema for all SolarVision AI inspection modalities."""

    inspection_type: InspectionType
    model_name: str
    model_version: str
    processing_time_ms: float
    image_metadata: Dict[str, Any]

    # Classification attributes
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    class_probabilities: Optional[Dict[str, float]] = None

    # Detection attributes
    boxes: Optional[List[BoundingBox]] = None

    # Segmentation attributes
    mask: Optional[np.ndarray] = None
    defect_ratio: Optional[float] = None
    threshold: Optional[float] = None

    # Visualization
    annotated_image: Optional[Image.Image] = None

    def to_dict(self, include_mask_array: bool = False) -> Dict[str, Any]:
        """Convert result to JSON-serializable dictionary.

        Args:
            include_mask_array: If True, includes full 2D mask as nested list.
                                Defaults to False to keep dictionary lightweight.

        Returns:
            Dict[str, Any]: Dictionary representation of inference result.
        """
        data: Dict[str, Any] = {
            "inspection_type": self.inspection_type.value,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "image_metadata": self.image_metadata,
        }

        if self.inspection_type == InspectionType.CLASSIFICATION:
            data.update(
                {
                    "predicted_class": self.predicted_class,
                    "confidence": (
                        round(self.confidence, 4) if self.confidence else None
                    ),
                    "class_probabilities": (
                        {k: round(v, 4) for k, v in self.class_probabilities.items()}
                        if self.class_probabilities
                        else None
                    ),
                }
            )

        elif self.inspection_type == InspectionType.DETECTION:
            data.update(
                {
                    "num_detections": len(self.boxes) if self.boxes else 0,
                    "boxes": [b.to_dict() for b in self.boxes] if self.boxes else [],
                }
            )

        elif self.inspection_type == InspectionType.SEGMENTATION:
            data.update(
                {
                    "defect_ratio": (
                        round(self.defect_ratio, 6)
                        if self.defect_ratio is not None
                        else None
                    ),
                    "threshold": self.threshold,
                    "has_defect": (self.defect_ratio or 0.0) > 0.0,
                }
            )
            if include_mask_array and self.mask is not None:
                data["mask"] = self.mask.tolist()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceResult":
        """Reconstruct InferenceResult from a dictionary representation.

        Args:
            data: Dictionary representation.

        Returns:
            InferenceResult: Reconstructed instance.
        """
        itype = InspectionType.from_str(data["inspection_type"])
        boxes = None
        if "boxes" in data and data["boxes"] is not None:
            boxes = [
                BoundingBox(
                    x1=float(b["x1"]),
                    y1=float(b["y1"]),
                    x2=float(b["x2"]),
                    y2=float(b["y2"]),
                    score=float(b["score"]),
                    class_id=int(b["class_id"]),
                    class_name=str(b["class_name"]),
                )
                for b in data["boxes"]
            ]

        mask = None
        if "mask" in data and data["mask"] is not None:
            mask = np.array(data["mask"], dtype=np.uint8)

        return cls(
            inspection_type=itype,
            model_name=data.get("model_name", "SolarVision"),
            model_version=data.get("model_version", "v1.0"),
            processing_time_ms=float(data.get("processing_time_ms", 0.0)),
            image_metadata=data.get("image_metadata", {}),
            predicted_class=data.get("predicted_class"),
            confidence=(
                float(data["confidence"])
                if data.get("confidence") is not None
                else None
            ),
            class_probabilities=data.get("class_probabilities"),
            boxes=boxes,
            mask=mask,
            defect_ratio=(
                float(data["defect_ratio"])
                if data.get("defect_ratio") is not None
                else None
            ),
            threshold=(
                float(data["threshold"]) if data.get("threshold") is not None else None
            ),
        )
