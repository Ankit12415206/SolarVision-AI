"""
Module: test_inference_pipeline
Description: Integration tests for SolarVision AI unified inference pipeline and CLI
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from scripts.run_inference import main
from src.inference import (
    BoundingBox,
    InferenceResult,
    InspectionType,
    SolarVisionInferencePipeline,
)

SAMPLE_CLS_IMAGE = Path("data/raw/classification/Bird-drop/10.JPG")
SAMPLE_DET_IMAGE = Path(
    "data/raw/dust/aerial_soiling/test/images/ROI_102_jpg.rf.fa9f2d998fd0b9874bc2a3a58394e73e.jpg"
)
SAMPLE_SEG_IMAGE = Path(
    "data/raw/microcracks/msps/test/images/First_101_20210705143317_5.png"
)


class TestInferencePipelineIntegration:
    """Integration test suite executing inference on real model checkpoints."""

    @pytest.fixture(scope="module")
    @classmethod
    def pipeline(cls) -> SolarVisionInferencePipeline:
        """Instantiate single warm pipeline instance for tests."""
        return SolarVisionInferencePipeline(device="cpu")

    def test_classification_inference(
        self,
        pipeline: SolarVisionInferencePipeline,
    ) -> None:
        """Verify end-to-end classification inference on real test sample."""
        assert SAMPLE_CLS_IMAGE.exists(), f"Missing sample: {SAMPLE_CLS_IMAGE}"
        result = pipeline.predict(
            image_input=SAMPLE_CLS_IMAGE,
            inspection_type=InspectionType.CLASSIFICATION,
        )

        assert isinstance(result, InferenceResult)
        assert result.inspection_type == InspectionType.CLASSIFICATION
        assert result.predicted_class in [
            "Bird-drop",
            "Clean",
            "Dusty",
            "Electrical-damage",
            "Physical-Damage",
            "Snow-Covered",
        ]
        assert result.confidence is not None and 0.0 <= result.confidence <= 1.0
        assert result.class_probabilities is not None
        assert len(result.class_probabilities) == 6
        assert np.isclose(sum(result.class_probabilities.values()), 1.0, atol=1e-3)
        assert result.annotated_image is not None
        assert isinstance(result.annotated_image, Image.Image)
        assert result.processing_time_ms > 0

    def test_detection_inference(
        self,
        pipeline: SolarVisionInferencePipeline,
    ) -> None:
        """Verify end-to-end detection inference on real test sample."""
        assert SAMPLE_DET_IMAGE.exists(), f"Missing sample: {SAMPLE_DET_IMAGE}"
        result = pipeline.predict(
            image_input=SAMPLE_DET_IMAGE,
            inspection_type=InspectionType.DETECTION,
            conf_threshold=0.20,
        )

        assert isinstance(result, InferenceResult)
        assert result.inspection_type == InspectionType.DETECTION
        assert result.boxes is not None
        for b in result.boxes:
            assert isinstance(b, BoundingBox)
            assert 0.0 <= b.x1 <= 1.0
            assert 0.0 <= b.y1 <= 1.0
            assert 0.0 <= b.x2 <= 1.0
            assert 0.0 <= b.y2 <= 1.0
            assert 0.0 <= b.score <= 1.0
            assert b.class_name in ["Dust", "bird-droppings"]

        assert result.annotated_image is not None
        assert isinstance(result.annotated_image, Image.Image)

    def test_segmentation_inference(
        self,
        pipeline: SolarVisionInferencePipeline,
    ) -> None:
        """Verify end-to-end segmentation inference on real test sample."""
        assert SAMPLE_SEG_IMAGE.exists(), f"Missing sample: {SAMPLE_SEG_IMAGE}"
        result = pipeline.predict(
            image_input=SAMPLE_SEG_IMAGE,
            inspection_type=InspectionType.SEGMENTATION,
            threshold=0.70,
        )

        assert isinstance(result, InferenceResult)
        assert result.inspection_type == InspectionType.SEGMENTATION
        assert result.mask is not None
        assert result.mask.ndim == 2
        assert set(np.unique(result.mask)).issubset({0, 1})
        assert result.defect_ratio is not None and 0.0 <= result.defect_ratio <= 1.0
        assert result.threshold == 0.70
        assert result.annotated_image is not None
        assert isinstance(result.annotated_image, Image.Image)

    def test_model_caching(
        self,
        pipeline: SolarVisionInferencePipeline,
    ) -> None:
        """Verify models are cached warm and not reloaded on every call."""
        initial_models = set(pipeline._models.keys())
        # Repeat classification prediction
        _ = pipeline.predict(SAMPLE_CLS_IMAGE, InspectionType.CLASSIFICATION)
        post_models = set(pipeline._models.keys())
        assert initial_models == post_models

    def test_batch_inference(
        self,
        pipeline: SolarVisionInferencePipeline,
    ) -> None:
        """Verify batch inference execution."""
        batch = [SAMPLE_CLS_IMAGE, SAMPLE_CLS_IMAGE]
        results = pipeline.predict_batch(
            batch,
            inspection_type=InspectionType.CLASSIFICATION,
        )
        assert len(results) == 2
        for r in results:
            assert r.inspection_type == InspectionType.CLASSIFICATION

    def test_cli_runner(self, tmp_path: Path) -> None:
        """Verify CLI script execution for all modalities."""
        runner = CliRunner()

        # Classification CLI
        res_cls = runner.invoke(
            main,
            [
                "--image",
                str(SAMPLE_CLS_IMAGE),
                "--type",
                "classification",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert res_cls.exit_code == 0
        assert "Model:            SolarMobileNetV2" in res_cls.output

        # Segmentation CLI
        res_seg = runner.invoke(
            main,
            [
                "--image",
                str(SAMPLE_SEG_IMAGE),
                "--type",
                "segmentation",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert res_seg.exit_code == 0
        assert "Model:            SolarLightweightUNet" in res_seg.output
