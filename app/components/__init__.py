"""
Module: components
Description: Streamlit UI components, themes, and interactive charts
Author: SolarVision AI Team
Created: 2026-11-02
"""

from app.components.charts import (
    plot_class_probabilities,
    plot_daily_trend,
    plot_defect_distribution,
    plot_modality_breakdown,
)
from app.components.styles import (
    THEME_PALETTES,
    apply_custom_theme,
    render_disclaimer_notice,
    render_kpi_card,
    render_model_transparency,
)

__all__ = [
    "THEME_PALETTES",
    "apply_custom_theme",
    "render_disclaimer_notice",
    "render_kpi_card",
    "render_model_transparency",
    "plot_class_probabilities",
    "plot_defect_distribution",
    "plot_modality_breakdown",
    "plot_daily_trend",
]
