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
from app.components.styles import apply_custom_theme
from app.main import get_db_manager, get_inference_pipeline
from app.views import (
    render_batch_page,
    render_history_page,
    render_inspection_page,
    render_overview_page,
)
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

    def test_theme_aware_charts(self) -> None:
        """Verify charts render in both dark and light modes with proper styling."""
        probs = {"Clean": 0.8, "Dusty": 0.2}
        analytics = {
            "clean_count": 5,
            "defect_count": 2,
            "inspections_by_type": {"classification": 7},
            "daily_trend": [{"date": "2026-11-01", "count": 7}],
        }
        for theme in ["dark", "light"]:
            fig_probs = plot_class_probabilities(probs, theme=theme)
            fig_donut = plot_defect_distribution(analytics, theme=theme)
            fig_bar = plot_modality_breakdown(analytics, theme=theme)
            fig_trend = plot_daily_trend(analytics, theme=theme)

            assert isinstance(fig_probs, go.Figure)
            assert isinstance(fig_donut, go.Figure)
            assert isinstance(fig_bar, go.Figure)
            assert isinstance(fig_trend, go.Figure)

    def test_views_exported_callable(self) -> None:
        """Verify all 4 UI views are properly exported and callable."""
        assert callable(render_overview_page)
        assert callable(render_inspection_page)
        assert callable(render_batch_page)
        assert callable(render_history_page)

    def test_apply_custom_theme_injection(self, monkeypatch) -> None:
        """Verify CSS injection runs for both dark and light modes."""
        markdown_calls = []
        monkeypatch.setattr(
            "streamlit.markdown",
            lambda content, **kwargs: markdown_calls.append(content),
        )
        apply_custom_theme("dark")
        apply_custom_theme("light")
        assert len(markdown_calls) == 2
        assert "--sv-bg-main" in markdown_calls[0]
        assert "--sv-bg-main" in markdown_calls[1]
        assert "#0B0F19" in markdown_calls[0]
        assert "#F8FAFC" in markdown_calls[1]


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
