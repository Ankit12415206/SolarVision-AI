"""
Module: main
Description: Streamlit entrypoint and multi-page coordinator for SolarVision AI
Author: SolarVision AI Team
Created: 2026-11-02
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.components.styles import apply_custom_theme
from app.pages.batch import render_batch_page
from app.pages.history import render_history_page
from app.pages.inspection import render_inspection_page
from src.database.manager import DEFAULT_DB_PATH, InspectionDatabaseManager
from src.inference import SolarVisionInferencePipeline

# Page setup
st.set_page_config(
    page_title="SolarVision AI — Defect Inspection",
    page_icon="🔆",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_inference_pipeline() -> SolarVisionInferencePipeline:
    """Create and cache warm inference pipeline singleton."""
    return SolarVisionInferencePipeline(device="cpu", auto_load=False)


@st.cache_resource
def get_db_manager() -> InspectionDatabaseManager:
    """Create and cache SQLite database manager singleton."""
    return InspectionDatabaseManager(db_path=DEFAULT_DB_PATH)


def main() -> None:
    """Main execution function for SolarVision AI Streamlit Dashboard."""
    apply_custom_theme()

    pipeline = get_inference_pipeline()
    db_manager = get_db_manager()

    # Sidebar Navigation & Diagnostics
    with st.sidebar:
        st.markdown("## 🔆 SolarVision AI")
        st.caption("Automated Drone & EL Defect Inspection")
        st.markdown("---")

        page_choice = st.radio(
            "Navigation",
            options=[
                "🔍 Live Inspection",
                "📁 Batch Inspection",
                "📊 History & Analytics",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("### 🖥️ Engine Status")
        status = pipeline.get_status()
        st.text(f"Device: {status['device'].upper()}")

        loaded = status.get("loaded_models", [])
        if loaded:
            st.success(f"Cached in RAM: {', '.join(loaded)}")
            if st.button(
                "🧹 Free Cached Models", help="Releases model weights from RAM"
            ):
                pipeline.unload_models()
                st.info("Models unloaded.")
                st.rerun()
        else:
            st.info("Models cached on-demand (Lazy)")

        st.markdown("---")
        st.caption("SolarVision AI v1.0 — Intel Arc / CPU Safe")

    # Header Banner
    st.markdown(
        """
        <div class="brand-header">
            <div>
                <h1 class="brand-title">SolarVision AI 🔆</h1>
                <p class="brand-tagline">Deep Learning Defect Inspection & Aerial Survey Analytics for Solar Photovoltaics</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Page Routing
    if page_choice == "🔍 Live Inspection":
        render_inspection_page(pipeline=pipeline, db_manager=db_manager)
    elif page_choice == "📁 Batch Inspection":
        render_batch_page(pipeline=pipeline, db_manager=db_manager)
    elif page_choice == "📊 History & Analytics":
        render_history_page(db_manager=db_manager)


if __name__ == "__main__":
    main()
