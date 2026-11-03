"""
Module: test_report_generator
Description: Unit tests for ReportLab PDF report generation utilities
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import numpy as np
from PIL import Image

from reportlab.lib import colors
from reportlab.platypus import Paragraph, Table

from src.inference.types import BoundingBox, InferenceResult, InspectionType
from src.utils.report_generator import (
    _pil_to_reportlab_image,
    build_pdf_story,
    generate_inspection_pdf,
    generate_inspection_pdf_bytes,
    get_table_header_style,
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

    def test_table_header_style_specification(self) -> None:
        """Verify dedicated white bold ParagraphStyle properties for PDF table headers."""
        style = get_table_header_style()
        assert style.fontName == "Helvetica-Bold"
        assert style.fontSize == 8.5
        assert style.textColor == colors.HexColor("#FFFFFF")

    def test_all_table_headers_use_dedicated_white_style(self) -> None:
        """Verify classification, detection, and segmentation table headers use white bold style."""
        # 1. Classification
        res_cls = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=50.0,
            image_metadata={"width": 224, "height": 224},
            predicted_class="Clean",
            confidence=0.751,
            class_probabilities={"Clean": 0.751, "Dusty": 0.249},
        )
        story_cls = build_pdf_story(res_cls)
        tables_cls = [elem for elem in story_cls if isinstance(elem, Table)]
        assert len(tables_cls) >= 2
        prob_table = tables_cls[-1]
        for header_cell in prob_table._cellvalues[0]:
            assert isinstance(header_cell, Paragraph)
            assert header_cell.style.fontName == "Helvetica-Bold"
            assert header_cell.style.fontSize == 8.5
            assert header_cell.style.textColor == colors.HexColor("#FFFFFF")

        # 2. Detection
        res_det = InferenceResult(
            inspection_type=InspectionType.DETECTION,
            model_name="SolarYOLOv8n",
            model_version="v1.1",
            processing_time_ms=60.0,
            image_metadata={"width": 640, "height": 640},
            boxes=[BoundingBox(0.1, 0.1, 0.5, 0.5, 0.85, 0, "Dust")],
        )
        story_det = build_pdf_story(res_det)
        tables_det = [elem for elem in story_det if isinstance(elem, Table)]
        assert len(tables_det) >= 2
        det_table = tables_det[-1]
        for header_cell in det_table._cellvalues[0]:
            assert isinstance(header_cell, Paragraph)
            assert header_cell.style.fontName == "Helvetica-Bold"
            assert header_cell.style.fontSize == 8.5
            assert header_cell.style.textColor == colors.HexColor("#FFFFFF")

        # 3. Segmentation
        res_seg = InferenceResult(
            inspection_type=InspectionType.SEGMENTATION,
            model_name="SolarLightweightUNet",
            model_version="v1.0",
            processing_time_ms=40.0,
            image_metadata={"width": 256, "height": 256},
            defect_ratio=0.012,
            threshold=0.70,
        )
        story_seg = build_pdf_story(res_seg)
        tables_seg = [elem for elem in story_seg if isinstance(elem, Table)]
        assert len(tables_seg) >= 2
        seg_table = tables_seg[-1]
        for header_cell in seg_table._cellvalues[0]:
            assert isinstance(header_cell, Paragraph)
            assert header_cell.style.fontName == "Helvetica-Bold"
            assert header_cell.style.fontSize == 8.5
            assert header_cell.style.textColor == colors.HexColor("#FFFFFF")

    def test_classification_confidence_terminology_separation(self) -> None:
        """Verify classification reports cleanly separate Predicted Finding from Model Confidence."""
        res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=45.0,
            image_metadata={"width": 224, "height": 224},
            predicted_class="Clean",
            confidence=0.751,
            class_probabilities={"Clean": 0.751, "Dusty": 0.249},
        )
        story = build_pdf_story(res)
        meta_table = [elem for elem in story if isinstance(elem, Table)][0]

        # Extract cell text
        text_grid = []
        for row in meta_table._cellvalues:
            row_texts = [cell.text if isinstance(cell, Paragraph) else str(cell) for cell in row]
            text_grid.append(row_texts)

        flattened = " ".join(" ".join(r) for r in text_grid)
        assert "Predicted Finding:" in flattened
        assert "Clean" in flattened
        assert "Model Confidence:" in flattened
        assert "75.1%" in flattened
        assert "Confidence Basis:" in flattened
        assert "not physical certainty" in flattened

    def test_detection_and_segmentation_retain_appropriate_metrics(self) -> None:
        """Verify detection and segmentation retain proper domain metrics without inventing confidence."""
        # Detection
        res_det = InferenceResult(
            inspection_type=InspectionType.DETECTION,
            model_name="SolarYOLOv8n",
            model_version="v1.1",
            processing_time_ms=60.0,
            image_metadata={"width": 640, "height": 640},
            boxes=[BoundingBox(0.1, 0.1, 0.5, 0.5, 0.85, 0, "Dust")],
        )
        story_det = build_pdf_story(res_det)
        meta_det = [elem for elem in story_det if isinstance(elem, Table)][0]
        flattened_det = " ".join(
            " ".join(cell.text for cell in row if isinstance(cell, Paragraph))
            for row in meta_det._cellvalues
        )
        assert "1 detected object(s)" in flattened_det
        assert "Model Confidence:" not in flattened_det

        # Segmentation
        res_seg = InferenceResult(
            inspection_type=InspectionType.SEGMENTATION,
            model_name="SolarLightweightUNet",
            model_version="v1.0",
            processing_time_ms=40.0,
            image_metadata={"width": 256, "height": 256},
            defect_ratio=0.0125,
            threshold=0.70,
        )
        story_seg = build_pdf_story(res_seg)
        meta_seg = [elem for elem in story_seg if isinstance(elem, Table)][0]
        flattened_seg = " ".join(
            " ".join(cell.text for cell in row if isinstance(cell, Paragraph))
            for row in meta_seg._cellvalues
        )
        assert "1.2500% active defect area" in flattened_seg
        assert "Model Confidence:" not in flattened_seg

    def test_image_fallback_to_original_when_annotated_missing(self) -> None:
        """Verify original image is embedded when annotated overlay is absent."""
        orig_img = Image.new("RGB", (300, 200), color=(120, 180, 240))
        res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=50.0,
            image_metadata={"width": 300, "height": 200},
            predicted_class="Clean",
            confidence=0.95,
            annotated_image=None,
        )
        story = build_pdf_story(res, original_image=orig_img)
        # Check that caption says Original Inspection Asset
        caption_paras = [
            elem.text for elem in story
            if isinstance(elem, Paragraph) and "Original Inspection Asset" in elem.text
        ]
        assert len(caption_paras) == 1

    def test_image_fallback_to_metadata_source_file(self, tmp_path: Path) -> None:
        """Verify image is loaded from disk if source metadata exists and memory image is None."""
        disk_img = Image.new("RGB", (200, 200), color=(80, 160, 200))
        img_file = tmp_path / "saved_asset.jpg"
        disk_img.save(img_file)

        res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=50.0,
            image_metadata={"width": 200, "height": 200, "source": str(img_file)},
            predicted_class="Clean",
            confidence=0.90,
            annotated_image=None,
        )
        story = build_pdf_story(res, original_image=None)
        caption_paras = [
            elem.text for elem in story
            if isinstance(elem, Paragraph) and "Inspection Asset" in elem.text
        ]
        assert len(caption_paras) == 1

    def test_image_unavailable_notice_rendered(self) -> None:
        """Verify clear 'Inspection Image Unavailable' notice is rendered when no image is available."""
        res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=50.0,
            image_metadata={"width": 0, "height": 0, "source": "non_existent_file.jpg"},
            predicted_class="Clean",
            confidence=0.90,
            annotated_image=None,
        )
        story = build_pdf_story(res, original_image=None)
        # Check that the fallback box contains the notice
        all_paras = []
        for elem in story:
            if isinstance(elem, Paragraph):
                all_paras.append(elem.text)
            elif isinstance(elem, Table):
                for row in elem._cellvalues:
                    for cell in row:
                        if isinstance(cell, Paragraph):
                            all_paras.append(cell.text)

        joined_text = " ".join(all_paras)
        assert "Inspection Image Unavailable" in joined_text
        assert "not retained in active memory" in joined_text
