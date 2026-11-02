"""
Module: test_app_integration
Description: Integration tests for Streamlit application components, charts, and workflow
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import plotly.graph_objects as go

from app.components.charts import (
    plot_class_probabilities,
    plot_daily_trend,
    plot_defect_distribution,
    plot_modality_breakdown,
)
from app.main import get_db_manager, get_inference_pipeline
from src.database.manager import InspectionDatabaseManager
from src.inference import InspectionType, SolarVisionInferencePipeline


class TestAppComponentsAndCharts:
    """Test suite verifying Streamlit components and Plotly charts."""

    def test_singletons_initialization(self) -> None:
        """Verify pipeline and database singletons initialize successfully."""
        pipeline = get_inference_pipeline()
        assert isinstance(pipeline, SolarVisionInferencePipeline)
        assert pipeline.device.type == "cpu"

        db_mgr = get_db_manager()
        assert isinstance(db_mgr, InspectionDatabaseManager)
        assert db_mgr.db_path.exists()

    def test_class_probabilities_chart(self) -> None:
        """Verify class probabilities chart generates valid Plotly Figure."""
        probs = {
            "Clean": 0.05,
            "Dusty": 0.70,
            "Bird-drop": 0.15,
            "Electrical-damage": 0.02,
            "Physical-Damage": 0.05,
            "Snow-Covered": 0.03,
        }
        fig = plot_class_probabilities(probs)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].orientation == "h"

    def test_defect_distribution_chart(self) -> None:
        """Verify defect distribution donut chart."""
        analytics = {"clean_count": 8, "defect_count": 12}
        fig = plot_defect_distribution(analytics)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "pie"

    def test_modality_breakdown_chart(self) -> None:
        """Verify modality breakdown bar chart."""
        analytics = {
            "inspections_by_type": {
                "classification": 10,
                "detection": 5,
                "segmentation": 8,
            }
        }
        fig = plot_modality_breakdown(analytics)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "bar"

    def test_daily_trend_chart(self) -> None:
        """Verify daily trend line chart."""
        analytics = {
            "daily_trend": [
                {"date": "2026-11-01", "count": 5},
                {"date": "2026-11-02", "count": 14},
            ]
        }
        fig = plot_daily_trend(analytics)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "scatter"


class TestFullDashboardWorkflow:
    """Test suite verifying end-to-end integration between pipeline and database."""

    def test_end_to_end_inspection_and_persistence(self, tmp_path: Path) -> None:
        """Verify inference execution followed by persistence and analytics query."""
        test_db = tmp_path / "workflow_test.db"
        db_mgr = InspectionDatabaseManager(db_path=test_db)
        pipeline = SolarVisionInferencePipeline(device="cpu")

        sample_img = Path("data/raw/classification/Clean/Clean (10).JPG")
        assert sample_img.exists()

        # 1. Run live classification inference
        result = pipeline.predict(
            sample_img, inspection_type=InspectionType.CLASSIFICATION
        )
        assert result.predicted_class is not None

        # 2. Persist to database
        rec_id = db_mgr.save_inspection(
            result=result,
            image_name="Clean (10).JPG",
            image_path=str(sample_img),
            notes="Automated E2E test inspection",
        )
        assert rec_id == 1

        # 3. Retrieve record from database
        stored = db_mgr.get_inspection_by_id(rec_id)
        assert stored is not None
        assert stored["predicted_class"] == result.predicted_class
        assert stored["model_name"] == "SolarMobileNetV2"
        assert stored["notes"] == "Automated E2E test inspection"

        # 4. Analytics reflect the new entry
        analytics = db_mgr.get_analytics_summary()
        assert analytics["total_inspections"] == 1
        assert analytics["inspections_by_type"]["classification"] == 1
