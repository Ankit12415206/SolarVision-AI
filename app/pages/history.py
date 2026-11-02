"""
Module: history
Description: Inspection history, search filters, and analytics dashboard page for Streamlit
Author: SolarVision AI Team
Created: 2026-11-02
"""

import json

import pandas as pd
import streamlit as st

from app.components.charts import (
    plot_daily_trend,
    plot_defect_distribution,
    plot_modality_breakdown,
)
from src.database.manager import InspectionDatabaseManager


def render_history_page(db_manager: InspectionDatabaseManager) -> None:
    """Render the Searchable Inspection History & Analytics dashboard page."""
    st.markdown("### 📊 Inspection History & Fleet Analytics")
    st.write(
        "Explore persistent solar inspection records stored in SQLite, analyze defect trends, "
        "filter historical records, and export reports."
    )

    analytics = db_manager.get_analytics_summary()

    # KPI Metrics
    total = analytics.get("total_inspections", 0)
    defect_cnt = analytics.get("defect_count", 0)
    rate = analytics.get("defect_rate_pct", 0.0)
    avg_lat = analytics.get("avg_latency_ms", 0.0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Inspections", total)
    k2.metric("Defects Identified", defect_cnt)
    k3.metric("Defect Rate", f"{rate:.1f}%")
    k4.metric("Avg Latency", f"{avg_lat:.1f} ms")

    st.markdown("---")

    # Analytics Charts
    if total > 0:
        st.markdown("#### 📈 Analytics & Trends")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                plot_defect_distribution(analytics), use_container_width=True
            )
        with c2:
            st.plotly_chart(
                plot_modality_breakdown(analytics), use_container_width=True
            )

        st.plotly_chart(plot_daily_trend(analytics), use_container_width=True)
    else:
        st.info(
            "ℹ️ No inspection records found yet. Run inspections from the Live or Batch pages to populate analytics."
        )

    st.markdown("---")
    st.markdown("#### 🔎 Search & Filter Inspection Logs")

    # Search & Filter Controls
    f1, f2, f3 = st.columns([1.5, 1.0, 1.0])
    with f1:
        search_kw = st.text_input(
            "Search by Image Name or Notes",
            placeholder="Type keyword...",
            key="history_search_input",
        )
    with f2:
        type_filter = st.selectbox(
            "Filter by Modality",
            options=["All Modalities", "classification", "detection", "segmentation"],
        )
    with f3:
        status_filter = st.selectbox(
            "Filter by Condition",
            options=["All Statuses", "Defects Only", "Clean / Normal Only"],
        )

    # Convert filter options
    itype_param = None if type_filter == "All Modalities" else type_filter
    has_defect_param = None
    if status_filter == "Defects Only":
        has_defect_param = True
    elif status_filter == "Clean / Normal Only":
        has_defect_param = False

    records = db_manager.get_inspections(
        limit=200,
        inspection_type=itype_param,
        has_defect=has_defect_param,
        search_query=search_kw if search_kw.strip() else None,
    )

    if records:
        display_rows = []
        for r in records:
            display_rows.append(
                {
                    "ID": r["id"],
                    "Timestamp": r["timestamp"][:19].replace("T", " "),
                    "Image": r["image_name"],
                    "Modality": r["inspection_type"],
                    "Model": r["model_name"],
                    "Finding": (
                        r["predicted_class"]
                        if r["inspection_type"] == "classification"
                        else (
                            f"{r['num_detections']} detections"
                            if r["inspection_type"] == "detection"
                            else f"{round((r['defect_ratio'] or 0.0) * 100, 3)}% crack"
                        )
                    ),
                    "Defect": "⚠️ Defect" if r["has_defect"] else "✅ Clean",
                    "Latency (ms)": round(r["processing_time_ms"], 1),
                    "Notes": r["notes"] or "",
                }
            )

        df_display = pd.DataFrame(display_rows)
        st.dataframe(df_display, use_container_width=True)

        col_dl1, col_dl2 = st.columns([2, 1])
        with col_dl1:
            full_df = db_manager.export_to_dataframe()
            csv_data = full_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Full Inspection Database (CSV)",
                data=csv_data,
                file_name="solarvision_fleet_inspection_history.csv",
                mime="text/csv",
            )

        # Inspect Single Record Details
        with st.expander("🔍 View Inspection Record Details & JSON Payload"):
            record_id = st.number_input(
                "Enter Record ID to Inspect",
                min_value=1,
                max_value=max(r["id"] for r in records),
                value=records[0]["id"],
                step=1,
            )
            rec = db_manager.get_inspection_by_id(int(record_id))
            if rec:
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write(f"**Image:** `{rec['image_name']}`")
                    st.write(f"**Modality:** `{rec['inspection_type']}`")
                    st.write(
                        f"**Model:** `{rec['model_name']} ({rec['model_version']})`"
                    )
                    st.write(f"**Timestamp:** `{rec['timestamp']}`")
                with col_d2:
                    st.write(
                        f"**Has Defect:** `{'YES' if rec['has_defect'] else 'NO'}`"
                    )
                    st.write(f"**Latency:** `{rec['processing_time_ms']:.1f} ms`")
                    st.write(f"**Notes:** `{rec['notes'] or 'None'}`")

                if rec["details_json"]:
                    try:
                        st.json(json.loads(rec["details_json"]))
                    except Exception:
                        st.text(rec["details_json"])

                if st.button(f"🗑️ Delete Record #{rec['id']}", type="secondary"):
                    db_manager.delete_inspection(rec["id"])
                    st.success(f"Record #{rec['id']} deleted. Refreshing...")
                    st.rerun()
    else:
        st.info("No matching inspection records found for the applied filter.")
