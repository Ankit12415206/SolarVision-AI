"""
Module: pages
Description: Streamlit multi-page interface definitions for SolarVision AI
Author: SolarVision AI Team
Created: 2026-11-02
"""

from app.pages.batch import render_batch_page
from app.pages.history import render_history_page
from app.pages.inspection import render_inspection_page

__all__ = [
    "render_inspection_page",
    "render_batch_page",
    "render_history_page",
]
