"""
Module: test_inference
Description: Unit tests for unified inference pipeline, result schemas, and visualizers
Author: SolarVision AI Team
Created: 2026-11-02
"""

import numpy as np
import pytest
from PIL import Image

from src.inference.pipeline import SolarVisionInferencePipeline
from src.inference.types import BoundingBox, InferenceResult, InspectionType
from src.inference.visualizer import (
    annotate_classification,
    annotate_detection,
    annotate_segmentation,
)


class TestInspectionType:
    """Test suite for InspectionType enum and string parser."""

    def test_from_str_valid_exact(self) -> None:
        """Verify exact string matching."""
        assert (
            InspectionType.from_str("classification") == InspectionType.CLASSIFICATION
        )
        assert InspectionType.from_str("detection") == InspectionType.DETECTION
        assert InspectionType.from_str("segmentation") == InspectionType.SEGMENTATION

    def test_from_str_aliases(self) -> None:
        """Verify aliases for inspection types."""
        assert InspectionType.from_str("cls") == InspectionType.CLASSIFICATION
        assert (
            InspectionType.from_str("panel_condition") == InspectionType.CLASSIFICATION
        )
        assert InspectionType.from_str("soiling") == InspectionType.DETECTION
        assert InspectionType.from_str("yolo") == InspectionType.DETECTION
        assert InspectionType.from_str("microcrack") == InspectionType.SEGMENTATION
        assert InspectionType.from_str("unet") == InspectionType.SEGMENTATION

    def test_from_str_invalid(self) -> None:
        """Verify unknown string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported inspection type"):
            InspectionType.from_str("thermal_infrared")


class TestBoundingBox:
    """Test suite for BoundingBox data structure."""

    def test_bounding_box_to_dict(self) -> None:
        """Verify BoundingBox dictionary serialization."""
        box = BoundingBox(
            x1=0.12345,
            y1=0.23456,
            x2=0.56789,
            y2=0.67891,
            score=0.91234,
            class_id=1,
            class_name="bird-droppings",
        )
        data = box.to_dict()
        assert data["x1"] == 0.1235
        assert data["y1"] == 0.2346
        assert data["x2"] == 0.5679
        assert data["y2"] == 0.6789
        assert data["score"] == 0.9123
        assert data["class_id"] == 1
        assert data["class_name"] == "bird-droppings"


class TestInferenceResult:
    """Test suite for standardized InferenceResult schema."""

    def test_classification_result_dict(self) -> None:
        """Verify classification result dictionary conversion."""
        res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=45.678,
            image_metadata={"width": 640, "height": 480},
            predicted_class="Clean",
            confidence=0.98765,
            class_probabilities={"Clean": 0.98765, "Dusty": 0.01235},
        )
        d = res.to_dict()
        assert d["inspection_type"] == "classification"
        assert d["predicted_class"] == "Clean"
        assert d["confidence"] == 0.9877
        assert d["processing_time_ms"] == 45.68

    def test_detection_result_dict(self) -> None:
        """Verify detection result dictionary conversion."""
        box = BoundingBox(0.1, 0.2, 0.3, 0.4, 0.85, 0, "Dust")
        res = InferenceResult(
            inspection_type=InspectionType.DETECTION,
            model_name="SolarYOLOv8n",
            model_version="v1.1",
            processing_time_ms=62.5,
            image_metadata={"width": 640, "height": 640},
            boxes=[box],
        )
        d = res.to_dict()
        assert d["num_detections"] == 1
        assert len(d["boxes"]) == 1
        assert d["boxes"][0]["class_name"] == "Dust"

    def test_segmentation_result_dict(self) -> None:
        """Verify segmentation result dictionary conversion."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:4, 2:4] = 1
        res = InferenceResult(
            inspection_type=InspectionType.SEGMENTATION,
            model_name="SolarLightweightUNet",
            model_version="v1.0",
            processing_time_ms=28.4,
            image_metadata={"width": 10, "height": 10},
            mask=mask,
            defect_ratio=0.04,
            threshold=0.70,
        )
        d = res.to_dict(include_mask_array=False)
        assert d["defect_ratio"] == 0.04
        assert d["has_defect"] is True
        assert "mask" not in d

        d_full = res.to_dict(include_mask_array=True)
        assert "mask" in d_full
        assert len(d_full["mask"]) == 10


class TestVisualizers:
    """Test suite for inference visualizer functions."""

    def test_annotate_classification(self) -> None:
        """Verify classification annotation banner."""
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        annotated = annotate_classification(
            image=img,
            predicted_class="Clean",
            confidence=0.95,
            class_probabilities={"Clean": 0.95, "Dusty": 0.05},
        )
        assert isinstance(annotated, Image.Image)
        assert annotated.size == (200, 260)  # 200 width, 200 + 60 height

    def test_annotate_detection(self) -> None:
        """Verify detection bounding box annotation."""
        img = Image.new("RGB", (300, 300), color=(100, 100, 100))
        boxes = [
            BoundingBox(0.1, 0.1, 0.5, 0.5, 0.88, 0, "Dust"),
        ]
        annotated = annotate_detection(image=img, boxes=boxes)
        assert isinstance(annotated, Image.Image)
        assert annotated.size == (300, 300)

    def test_annotate_segmentation(self) -> None:
        """Verify segmentation multi-panel composite image."""
        img = Image.new("L", (100, 100), color=50)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:50, 30:50] = 1
        composite = annotate_segmentation(image=img, mask=mask)
        assert isinstance(annotated_composite := composite, Image.Image)
        assert annotated_composite.size == (300, 130)  # 3 panels wide, 100 + 30 banner


class TestPipelineValidation:
    """Test suite for pipeline input validation and caching."""

    def test_validate_pil_image(self) -> None:
        """Verify PIL Image loading and metadata extraction."""
        pipeline = SolarVisionInferencePipeline(device="cpu")
        img = Image.new("RGB", (120, 80), color=(255, 0, 0))
        validated_img, meta = pipeline._validate_and_load_image(img)
        assert validated_img.size == (120, 80)
        assert meta["width"] == 120
        assert meta["height"] == 80
        assert meta["source"] == "memory"

    def test_validate_numpy_array(self) -> None:
        """Verify NumPy array loading (2D and 3D)."""
        pipeline = SolarVisionInferencePipeline(device="cpu")
        arr_2d = np.ones((50, 50), dtype=np.uint8) * 128
        img_2d, _ = pipeline._validate_and_load_image(arr_2d)
        assert img_2d.size == (50, 50)
        assert img_2d.mode == "L"

        arr_3d = np.ones((60, 40, 3), dtype=np.uint8) * 200
        img_3d, _ = pipeline._validate_and_load_image(arr_3d)
        assert img_3d.size == (40, 60)
        assert img_3d.mode == "RGB"

    def test_validate_invalid_inputs(self) -> None:
        """Verify invalid input types and dimensions raise ValueError."""
        pipeline = SolarVisionInferencePipeline(device="cpu")
        with pytest.raises(FileNotFoundError):
            pipeline._validate_and_load_image("non_existent_file_xyz123.jpg")

        with pytest.raises(ValueError, match="Unsupported image input type"):
            pipeline._validate_and_load_image(12345)  # type: ignore

        with pytest.raises(ValueError, match="Input NumPy array is empty"):
            pipeline._validate_and_load_image(np.array([]))

        with pytest.raises(ValueError, match="Unsupported array dimensions"):
            pipeline._validate_and_load_image(np.ones((10, 10, 10, 10)))

    def test_status_and_unload(self) -> None:
        """Verify get_status and unload_models operations."""
        pipeline = SolarVisionInferencePipeline(device="cpu")
        status = pipeline.get_status()
        assert status["device"] == "cpu"
        assert len(status["loaded_models"]) == 0

        pipeline.unload_models()
        assert len(pipeline._models) == 0
