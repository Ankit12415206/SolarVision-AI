"""
Module: test_report_generator
Description: Unit tests for ReportLab PDF report generation utilities
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src.inference.types import BoundingBox, InferenceResult, InspectionType
from src.utils.report_generator import (
    _pil_to_reportlab_image,
    generate_inspection_pdf,
    generate_inspection_pdf_bytes,
)


class TestPDFReportGenerator:
    """Test suite for PDF report generation."""

    def test_classification_pdf_generation(self, tmp_path: Path) -> None:
        """Verify generating a classification inspection PDF report."""
        annotated_img = Image.new("RGB", (320, 240), color=(30, 40, 50))
        result = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=88.5,
            image_metadata={"width": 640, "height": 480, "source": "panel_01.jpg"},
            predicted_class="Dusty",
            confidence=0.895,
            class_probabilities={
                "Clean": 0.05,
                "Dusty": 0.895,
                "Bird-drop": 0.03,
                "Electrical-damage": 0.01,
                "Physical-Damage": 0.01,
                "Snow-Covered": 0.005,
            },
            annotated_image=annotated_img,
        )

        pdf_path = tmp_path / "test_classification_report.pdf"
        out_file = generate_inspection_pdf(
            result=result,
            output_path=pdf_path,
            inspection_id=42,
            image_name="panel_01.jpg",
            notes="Field scan line 4",
        )

        assert out_file.exists()
        assert out_file.stat().st_size > 2000

        with open(out_file, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-"

    def test_detection_pdf_generation(self, tmp_path: Path) -> None:
        """Verify generating a detection inspection PDF report with bounding boxes."""
        annotated_img = Image.new("RGB", (640, 640), color=(40, 40, 40))
        boxes = [
            BoundingBox(0.15, 0.20, 0.35, 0.45, 0.88, 0, "Dust"),
            BoundingBox(0.50, 0.60, 0.70, 0.80, 0.75, 1, "bird-droppings"),
        ]
        result = InferenceResult(
            inspection_type=InspectionType.DETECTION,
            model_name="SolarYOLOv8n",
            model_version="v1.1",
            processing_time_ms=135.2,
            image_metadata={"width": 640, "height": 640, "source": "aerial_field.jpg"},
            boxes=boxes,
            annotated_image=annotated_img,
        )

        pdf_bytes = generate_inspection_pdf_bytes(
            result=result,
            inspection_id=101,
            image_name="aerial_field.jpg",
        )
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 2000
        assert pdf_bytes.startswith(b"%PDF-")

    def test_segmentation_pdf_generation(self, tmp_path: Path) -> None:
        """Verify generating a segmentation inspection PDF report."""
        annotated_img = Image.new("RGB", (768, 256), color=(20, 25, 30))
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:80, 50:80] = 1

        result = InferenceResult(
            inspection_type=InspectionType.SEGMENTATION,
            model_name="SolarLightweightUNet",
            model_version="v1.0",
            processing_time_ms=64.0,
            image_metadata={"width": 256, "height": 256, "source": "el_crack.png"},
            mask=mask,
            defect_ratio=0.0137,
            threshold=0.70,
            annotated_image=annotated_img,
        )

        pdf_path = tmp_path / "test_segmentation_report.pdf"
        out_file = generate_inspection_pdf(
            result=result,
            output_path=pdf_path,
            inspection_id=77,
            image_name="el_crack.png",
        )

        assert out_file.exists()
        assert out_file.stat().st_size > 2000

    def test_inspection_id_string_in_pdf(self) -> None:
        """Verify report generation with standardized unique Inspection ID string."""
        res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=75.0,
            image_metadata={"width": 224, "height": 224, "source": "panel.jpg"},
            predicted_class="Clean",
            confidence=0.98,
        )
        pdf_bytes = generate_inspection_pdf_bytes(
            result=res,
            inspection_id="SV-2026-000001",
            image_name="panel.jpg",
        )
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF-")

    def test_pil_to_reportlab_image_scaling(self) -> None:
        """Verify aspect ratio preservation during image scaling."""
        img = Image.new("RGB", (1000, 500), color=(100, 150, 200))
        flowable = _pil_to_reportlab_image(img, max_width=400, max_height=300)
        assert flowable.drawWidth == 400
        assert flowable.drawHeight == 200  # 1000x500 is 2:1 ratio
