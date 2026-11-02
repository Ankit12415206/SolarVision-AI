"""
Module: overview
Description: Executive Overview dashboard with fleet KPIs, trends, and quick actions
Author: SolarVision AI Team
Created: 2026-11-02
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path for cloud deployment
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from app.components.charts import (
    plot_defect_distribution,
    plot_modality_breakdown,
)
from app.components.styles import render_disclaimer_notice
from src.database.manager import DEFAULT_DB_PATH, InspectionDatabaseManager


def render_overview_page(
    db_manager: InspectionDatabaseManager,
    theme: str = "dark",
) -> None:
    """Render the executive Overview dashboard with fleet KPIs and quick navigation.

    Args:
        db_manager: SQLite database manager instance.
        theme: Active theme ('dark' or 'light').
    """
    st.markdown("### :material/dashboard: Executive Fleet Inspection Overview")
    st.caption(
        "Automated multi-task computer vision diagnostic platform for solar photovoltaic (PV) "
        "aerial drone surveys and electroluminescence (EL) cell quality inspection."
    )

    # Core Capabilities Intro Banner
    with st.expander(":material/info: Overview of Three Deep Learning Inspection Modalities", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown(
                "**1. Panel Condition Classification**  \n"
                "*Architecture:* `SolarMobileNetV2`  \n"
                "*Input:* $224 \\times 224$ RGB Drone Photo  \n"
                "*Task:* 6-class surface condition categorization (Clean, Dusty, "
                "Bird-drop, Physical Damage, Electrical Damage, Snow)."
            )
        with col_c2:
            st.markdown(
                "**2. Aerial Soiling Object Detection**  \n"
                "*Architecture:* `SolarYOLOv8n`  \n"
                "*Input:* $640 \\times 640$ RGB Drone Survey  \n"
                "*Task:* Localized bounding boxes identifying discrete dust patches "
                "and avian soiling deposits across field arrays."
            )
        with col_c3:
            st.markdown(
                "**3. EL Defect Segmentation**  \n"
                "*Architecture:* `SolarLightweightUNet`  \n"
                "*Input:* $256 \\times 256$ Grayscale EL Scan  \n"
                "*Task:* Pixel-level binary mask segmentation highlighting internal "
                "silicon wafer microcracks and cell fractures."
            )

    analytics: Dict[str, Any] = db_manager.get_analytics_summary()
    total = analytics.get("total_inspections", 0)
    defects = analytics.get("defect_count", 0)
    clean = analytics.get("clean_count", 0)
    demos = analytics.get("demo_count", 0)
    rate = analytics.get("defect_rate_pct", 0.0)
    avg_lat = analytics.get("avg_latency_ms", 0.0)

    # Executive KPI Cards Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Total Inspections",
            value=f"{total:,}",
            delta=f"{demos} demo runs" if demos > 0 else None,
            delta_color="off",
            help="Total number of solar inspection records logged in SQLite database.",
        )
    with k2:
        st.metric(
            label="Defects Identified",
            value=f"{defects:,}",
            delta=f"{clean:,} Nominal" if total > 0 else None,
            delta_color="off",
            help="Panels identified with soiling, bird-drops, physical damage, or cracks.",
        )
    with k3:
        st.metric(
            label="Defect Rate",
            value=f"{rate:.1f}%",
            delta="Fleet Anomaly Ratio" if total > 0 else None,
            delta_color="inverse",
            help="Percentage of inspected assets flagged with defects.",
        )
    with k4:
        st.metric(
            label="Avg Model Latency",
            value=f"{avg_lat:.1f} ms" if total > 0 else "0.0 ms",
            help="Mean CPU inference response time across all vision modalities.",
        )

    st.markdown("---")

    # Evaluator Guide Card
    st.markdown("#### :material/school: Evaluator Quick-Start Guide (No Dataset Upload Required)")
    st.markdown(
        """
        SolarVision AI is pre-packaged with curated benchmark samples for academic demonstration:
        1. **Select Live Inspection** from the sidebar or click the button below.
        2. Choose any inspection modality (Classification, Detection, or Segmentation).
        3. In **Step 2**, pick an evaluator demo preset from the **Demo Preset Selector** dropdown.
        4. Click **Execute Model Inference** — the AI will process the sample on CPU, display
           visual comparison overlays, assign a unique traceable **Inspection ID** (e.g. `SV-2026-000001`),
           and let you compile a professional ReportLab PDF audit report.
        """
    )

    st.markdown("---")

    # Quick Action Cards
    st.markdown("#### :material/bolt: Quick Operations")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown(
            "**Live Single-Image Inspection**  \n"
            "Run real-time defect analysis on individual aerial RGB or EL images with demo presets."
        )
        if st.button(
            ":material/image_search: Launch Live Inspection",
            use_container_width=True,
            type="primary",
        ):
            st.session_state["nav_selection"] = ":material/image_search: Live Inspection"
            st.rerun()

    with q2:
        st.markdown(
            "**Sequential Batch Survey**  \n"
            "Process multi-image drone surveys with memory-safe CPU queuing."
        )
        if st.button(
            ":material/batch_prediction: Launch Batch Survey",
            use_container_width=True,
        ):
            st.session_state["nav_selection"] = ":material/batch_prediction: Batch Inspection"
            st.rerun()

    with q3:
        st.markdown(
            "**Fleet History & PDF Reports**  \n"
            "Search audit records, analyze historical trends, and export PDFs."
        )
        if st.button(
            ":material/history: View Inspection Logs",
            use_container_width=True,
        ):
            st.session_state["nav_selection"] = ":material/history: History & Analytics"
            st.rerun()

    st.markdown("---")

    # Analytics Visualizations
    if total > 0:
        st.markdown("#### :material/analytics: Inspection Analytics")
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(
                plot_modality_breakdown(analytics, theme=theme),
                use_container_width=True,
            )
        with ch2:
            st.plotly_chart(
                plot_defect_distribution(analytics, theme=theme),
                use_container_width=True,
            )

        # Recent Activity Feed
        st.markdown("---")
        st.markdown("#### :material/schedule: Recent Inspection Activity")
        recent_records = db_manager.get_inspections(limit=6)
        if recent_records:
            feed_rows = []
            for r in recent_records:
                finding = (
                    r["predicted_class"]
                    if r["inspection_type"] == "classification"
                    else (
                        f"{r['num_detections']} detections"
                        if r["inspection_type"] == "detection"
                        else f"{(r['defect_ratio'] or 0.0) * 100:.3f}% crack"
                    )
                )
                feed_rows.append(
                    {
                        "Inspection ID": r.get("inspection_id") or f"#{r['id']}",
                        "Timestamp": r["timestamp"][:19].replace("T", " "),
                        "Image File": r["image_name"],
                        "Type": "Demo" if r.get("is_demo") else "Upload",
                        "Modality": r["inspection_type"].capitalize(),
                        "Model": r["model_name"],
                        "Finding": finding,
                        "Status": "Defect" if r["has_defect"] else "Normal",
                        "Latency": f"{r['processing_time_ms']:.1f} ms",
                    }
                )
            df_feed = pd.DataFrame(feed_rows)
            st.dataframe(df_feed, use_container_width=True, hide_index=True)
    else:
        # High quality empty state
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-title">No Inspection Data in SQLite Fleet Database Yet</div>
                <div class="empty-desc">
                    Get started by running a live inspection on a sample solar panel image,
                    or upload a batch of drone survey images. All inference records are
                    persisted in local SQLite for auditability.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_disclaimer_notice()


if __name__ == "__main__":
    from app.components.styles import apply_custom_theme

    apply_custom_theme()
    db = InspectionDatabaseManager(db_path=DEFAULT_DB_PATH)
    render_overview_page(db)
