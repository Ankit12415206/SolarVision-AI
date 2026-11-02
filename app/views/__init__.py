"""
Module: views
Description: Streamlit UI views for overview, live inspection, batch processing, and history
Author: SolarVision AI Team
Created: 2026-11-02
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.views.batch import render_batch_page
from app.views.history import render_history_page
from app.views.inspection import render_inspection_page
from app.views.overview import render_overview_page

__all__ = [
    "render_overview_page",
    "render_inspection_page",
    "render_batch_page",
    "render_history_page",
]
