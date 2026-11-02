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
from app.views.batch import render_batch_page
from app.views.history import render_history_page
from app.views.inspection import render_inspection_page
from app.views.overview import render_overview_page
from src.database.manager import DEFAULT_DB_PATH, InspectionDatabaseManager
from src.inference import SolarVisionInferencePipeline

# Page setup
st.set_page_config(
    page_title="SolarVision AI — Industrial Solar Defect Inspection",
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
    # Initialize session theme
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"

    apply_custom_theme(theme=st.session_state["theme"])

    pipeline = get_inference_pipeline()
    db_manager = get_db_manager()

    nav_options = [
        ":material/dashboard: Overview",
        ":material/image_search: Live Inspection",
        ":material/batch_prediction: Batch Inspection",
        ":material/history: History & Analytics",
    ]

    # Handle programmatic navigation (from quick-action buttons)
    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = nav_options[0]

    current_idx = 0
    if st.session_state["nav_selection"] in nav_options:
        current_idx = nav_options.index(st.session_state["nav_selection"])

    # Sidebar Navigation & Platform Controls
    with st.sidebar:
        st.markdown("### 🔆 SolarVision AI")
        st.caption("Fleet Defect Inspection & Aerial Analytics")
        st.markdown("---")

        selected_page = st.radio(
            "Navigation Menu",
            options=nav_options,
            index=current_idx,
            label_visibility="collapsed",
        )
        st.session_state["nav_selection"] = selected_page

        st.markdown("---")
        st.markdown("#### :material/memory: Inference Engine")
        status = pipeline.get_status()
        st.caption(f"Device: **{status['device'].upper()} (CPU-Optimized)**")

        loaded = status.get("loaded_models", [])
        if loaded:
            st.success(f"Cached in RAM: {', '.join(loaded)}")
            if st.button(
                ":material/cleaning_services: Free Cached Models",
                help="Releases model weights from RAM to reduce memory footprint",
                use_container_width=True,
            ):
                pipeline.unload_models()
                st.info("RAM cache cleared.")
                st.rerun()
        else:
            st.caption("Models loaded on-demand (lazy initialization)")

        st.markdown("---")
        st.markdown("#### :material/palette: Appearance")
        theme_choice = st.radio(
            "Theme Mode",
            options=[":material/dark_mode: Dark", ":material/light_mode: Light"],
            index=0 if st.session_state["theme"] == "dark" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )

        new_theme = "dark" if "Dark" in theme_choice else "light"
        if new_theme != st.session_state["theme"]:
            st.session_state["theme"] = new_theme
            st.rerun()

        st.markdown("---")
        st.caption("SolarVision AI Enterprise v1.1.0 • Academic & Commercial Edition")

    # Enterprise Top Brand Header
    theme = st.session_state["theme"]
    badge_label = "Dark Mode" if theme == "dark" else "Light Mode"
    st.markdown(
        f"""
        <div class="brand-header">
            <div class="brand-header-left">
                <h1 class="brand-title">
                    🔆 SolarVision AI
                    <span class="badge-info">Fleet Edition</span>
                </h1>
                <p class="brand-tagline">
                    Automated Deep Learning Defect Inspection & Aerial Survey Analytics for Solar Photovoltaics
                </p>
            </div>
            <div>
                <span class="badge-clean">{badge_label} Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Page Routing
    if "Overview" in selected_page:
        render_overview_page(db_manager=db_manager, theme=theme)
    elif "Live Inspection" in selected_page:
        render_inspection_page(pipeline=pipeline, db_manager=db_manager, theme=theme)
    elif "Batch Inspection" in selected_page:
        render_batch_page(pipeline=pipeline, db_manager=db_manager, theme=theme)
    elif "History" in selected_page:
        render_history_page(db_manager=db_manager, theme=theme)


if __name__ == "__main__":
    main()
