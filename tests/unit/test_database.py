"""
Module: test_database
Description: Unit tests for SQLite inspection database manager and analytics
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import pytest

from src.database.manager import InspectionDatabaseManager
from src.inference.types import BoundingBox, InferenceResult, InspectionType


@pytest.fixture
def db_manager(tmp_path: Path) -> InspectionDatabaseManager:
    """Provide isolated SQLite database manager instance in temporary folder."""
    db_file = tmp_path / "test_inspections.db"
    return InspectionDatabaseManager(db_path=db_file)


class TestInspectionDatabaseManager:
    """Test suite for InspectionDatabaseManager operations."""

    def test_save_and_retrieve_classification(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify saving and querying a classification inspection."""
        res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=85.2,
            image_metadata={"width": 640, "height": 480, "source": "test_panel.jpg"},
            predicted_class="Dusty",
            confidence=0.92,
            class_probabilities={"Dusty": 0.92, "Clean": 0.08},
        )

        rec_id = db_manager.save_inspection(
            result=res,
            image_name="test_panel.jpg",
            notes="Zone A solar farm",
        )
        assert rec_id > 0

        record = db_manager.get_inspection_by_id(rec_id)
        assert record is not None
        assert record["image_name"] == "test_panel.jpg"
        assert record["inspection_type"] == "classification"
        assert record["predicted_class"] == "Dusty"
        assert record["confidence"] == 0.92
        assert record["has_defect"] == 1
        assert record["notes"] == "Zone A solar farm"

    def test_save_and_retrieve_detection(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify saving and querying a detection inspection."""
        boxes = [
            BoundingBox(0.1, 0.2, 0.3, 0.4, 0.88, 0, "Dust"),
        ]
        res = InferenceResult(
            inspection_type=InspectionType.DETECTION,
            model_name="SolarYOLOv8n",
            model_version="v1.1",
            processing_time_ms=120.0,
            image_metadata={"width": 640, "height": 640, "source": "drone_aerial.jpg"},
            boxes=boxes,
        )

        rec_id = db_manager.save_inspection(res, image_name="drone_aerial.jpg")
        assert rec_id > 0

        record = db_manager.get_inspection_by_id(rec_id)
        assert record is not None
        assert record["inspection_type"] == "detection"
        assert record["num_detections"] == 1
        assert record["has_defect"] == 1

    def test_save_and_retrieve_segmentation(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify saving and querying a segmentation inspection."""
        res = InferenceResult(
            inspection_type=InspectionType.SEGMENTATION,
            model_name="SolarLightweightUNet",
            model_version="v1.0",
            processing_time_ms=55.4,
            image_metadata={"width": 256, "height": 256, "source": "el_sample.png"},
            defect_ratio=0.0035,
            threshold=0.70,
        )

        rec_id = db_manager.save_inspection(res, image_name="el_sample.png")
        assert rec_id > 0

        record = db_manager.get_inspection_by_id(rec_id)
        assert record is not None
        assert record["inspection_type"] == "segmentation"
        assert record["defect_ratio"] == 0.0035
        assert record["has_defect"] == 1

    def test_filtering_and_search(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify filtering by type, defect, and search keywords."""
        # 1. Clean panel
        clean_res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=50.0,
            image_metadata={"width": 100, "height": 100, "source": "clean_panel.jpg"},
            predicted_class="Clean",
            confidence=0.99,
        )
        db_manager.save_inspection(clean_res, image_name="clean_panel.jpg")

        # 2. Defect panel
        defect_res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=52.0,
            image_metadata={"width": 100, "height": 100, "source": "broken_panel.jpg"},
            predicted_class="Physical-Damage",
            confidence=0.89,
        )
        db_manager.save_inspection(
            defect_res, image_name="broken_panel.jpg", notes="Severe hail damage"
        )

        # Query all
        all_recs = db_manager.get_inspections()
        assert len(all_recs) == 2

        # Query only defects
        defects = db_manager.get_inspections(has_defect=True)
        assert len(defects) == 1
        assert defects[0]["image_name"] == "broken_panel.jpg"

        # Query only clean
        cleans = db_manager.get_inspections(has_defect=False)
        assert len(cleans) == 1
        assert cleans[0]["image_name"] == "clean_panel.jpg"

        # Search keyword
        search_res = db_manager.get_inspections(search_query="hail")
        assert len(search_res) == 1
        assert search_res[0]["image_name"] == "broken_panel.jpg"

    def test_delete_inspection(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify record deletion."""
        res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=50.0,
            image_metadata={"width": 100, "height": 100, "source": "p.jpg"},
            predicted_class="Clean",
            confidence=0.99,
        )
        rec_id = db_manager.save_inspection(res, image_name="p.jpg")
        assert db_manager.delete_inspection(rec_id) is True
        assert db_manager.get_inspection_by_id(rec_id) is None
        assert db_manager.delete_inspection(99999) is False

    def test_analytics_summary(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify aggregated analytics calculation."""
        # Empty summary
        empty_summary = db_manager.get_analytics_summary()
        assert empty_summary["total_inspections"] == 0
        assert empty_summary["defect_rate_pct"] == 0.0

        # Insert 1 clean and 1 defect
        r1 = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=100.0,
            image_metadata={"width": 100, "height": 100, "source": "p1.jpg"},
            predicted_class="Clean",
            confidence=0.95,
        )
        r2 = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=200.0,
            image_metadata={"width": 100, "height": 100, "source": "p2.jpg"},
            predicted_class="Dusty",
            confidence=0.91,
        )
        db_manager.save_inspection(r1, image_name="p1.jpg")
        db_manager.save_inspection(r2, image_name="p2.jpg")

        summary = db_manager.get_analytics_summary()
        assert summary["total_inspections"] == 2
        assert summary["defect_count"] == 1
        assert summary["clean_count"] == 1
        assert summary["defect_rate_pct"] == 50.0
        assert summary["avg_latency_ms"] == 150.0
        assert summary["classification_distribution"]["Dusty"] == 1

    def test_export_to_dataframe(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify exporting history to pandas DataFrame."""
        df_empty = db_manager.export_to_dataframe()
        assert len(df_empty) == 0

        r = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=60.0,
            image_metadata={"width": 100, "height": 100, "source": "p.jpg"},
            predicted_class="Clean",
            confidence=0.95,
        )
        db_manager.save_inspection(r, image_name="p.jpg")
        df = db_manager.export_to_dataframe()
        assert len(df) == 1
        assert "timestamp" in df.columns
        assert "predicted_class" in df.columns
