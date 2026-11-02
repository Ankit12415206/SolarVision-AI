"""
Module: history
Description: Inspection history, search filters, and analytics dashboard page for Streamlit
Author: SolarVision AI Team
Created: 2026-11-02
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path for cloud deployment
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from app.components.charts import (
    plot_daily_trend,
    plot_defect_distribution,
    plot_modality_breakdown,
)
from app.components.styles import render_disclaimer_notice
from src.database.manager import DEFAULT_DB_PATH, InspectionDatabaseManager
from src.inference.types import InferenceResult
from src.utils.report_generator import generate_inspection_pdf_bytes


def render_history_page(
    db_manager: InspectionDatabaseManager,
    theme: str = "dark",
) -> None:
    """Render the Searchable Inspection History & Fleet Analytics page.

    Args:
        db_manager: SQLite database manager.
        theme: Active theme ('dark' or 'light').
    """
    st.markdown("### :material/history: Fleet Inspection History & Audit Analytics")
    st.caption(
        "Audit persistent inspection records stored in SQLite, search by unique Inspection ID, "
        "differentiate demo runs from user uploads, and compile verified PDF compliance reports."
    )

    analytics = db_manager.get_analytics_summary()

    # KPI Metrics
    total = analytics.get("total_inspections", 0)
    defect_cnt = analytics.get("defect_count", 0)
    rate = analytics.get("defect_rate_pct", 0.0)
    avg_lat = analytics.get("avg_latency_ms", 0.0)
    demos = analytics.get("demo_count", 0)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Fleet Records",
            value=f"{total:,}",
            delta=f"{demos} Demo • {total - demos} Real" if total > 0 else None,
            delta_color="off",
        )
    with k2:
        st.metric(label="Defects Identified", value=f"{defect_cnt:,}")
    with k3:
        st.metric(
            label="Overall Defect Rate",
            value=f"{rate:.1f}%",
            delta_color="inverse",
        )
    with k4:
        st.metric(
            label="Average Latency",
            value=f"{avg_lat:.1f} ms" if total > 0 else "0.0 ms",
        )

    st.markdown("---")

    # Analytics Charts
    if total > 0:
        st.markdown("#### :material/monitoring: Fleet Trends & Distributions")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                plot_defect_distribution(analytics, theme=theme),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                plot_modality_breakdown(analytics, theme=theme),
                use_container_width=True,
            )

        st.plotly_chart(
            plot_daily_trend(analytics, theme=theme),
            use_container_width=True,
        )
        st.markdown("---")

    # Search & Filter Controls
    st.markdown("#### :material/filter_alt: Search & Filter Audit Records")
    f1, f2, f3, f4 = st.columns([1.5, 1.0, 1.0, 1.0])
    with f1:
        search_kw = st.text_input(
            "Search by ID, Image Name, or Notes",
            placeholder="e.g. SV-2026-000001, Clean, demo...",
            key="history_search_input",
        )
    with f2:
        type_filter = st.selectbox(
            "Filter Modality",
            options=["All Modalities", "classification", "detection", "segmentation"],
        )
    with f3:
        status_filter = st.selectbox(
            "Filter Condition",
            options=["All Statuses", "Defects Only", "Clean / Normal Only"],
        )
    with f4:
        demo_filter = st.selectbox(
            "Filter Source",
            options=["All Inspections", "User Uploads Only", "Demo Presets Only"],
        )

    # Filter parameter resolution
    itype_param = None if type_filter == "All Modalities" else type_filter
    has_defect_param = None
    if status_filter == "Defects Only":
        has_defect_param = True
    elif status_filter == "Clean / Normal Only":
        has_defect_param = False

    demo_param = None
    if demo_filter == "Demo Presets Only":
        demo_param = True
    elif demo_filter == "User Uploads Only":
        demo_param = False

    records = db_manager.get_inspections(
        limit=200,
        inspection_type=itype_param,
        has_defect=has_defect_param,
        search_query=search_kw.strip() if search_kw.strip() else None,
        is_demo=demo_param,
    )

    if records:
        display_rows = []
        for r in records:
            finding = (
                r["predicted_class"]
                if r["inspection_type"] == "classification"
                else (
                    f"{r['num_detections']} detections"
                    if r["inspection_type"] == "detection"
                    else f"{(r['defect_ratio'] or 0.0) * 100:.3f}% crack"
                )
            )
            display_rows.append(
                {
                    "Inspection ID": r.get("inspection_id") or f"#{r['id']}",
                    "Timestamp": r["timestamp"][:19].replace("T", " "),
                    "Image Name": r["image_name"],
                    "Source": "Demo" if r.get("is_demo") else "Upload",
                    "Modality": r["inspection_type"].capitalize(),
                    "Model": r["model_name"],
                    "Finding": finding,
                    "Condition": "Defect" if r["has_defect"] else "Normal",
                    "Latency (ms)": round(r["processing_time_ms"], 1),
                    "Notes": r["notes"] or "",
                }
            )

        df_display = pd.DataFrame(display_rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        col_dl1, _ = st.columns([2, 1])
        with col_dl1:
            full_df = db_manager.export_to_dataframe()
            csv_data = full_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=":material/download: Export Complete Audit Database with Inspection IDs (CSV)",
                data=csv_data,
                file_name="solarvision_fleet_inspection_history.csv",
                mime="text/csv",
            )

        # Inspect Single Record Details
        st.markdown("---")
        with st.expander(":material/find_in_page: Retrieve Detailed Record & PDF Audit Report", expanded=False):
            # Offer selection dropdown or direct lookup
            id_options = [
                f"{r.get('inspection_id') or r['id']} — {r['image_name']} ({r['inspection_type']})"
                for r in records
            ]
            selected_choice = st.selectbox(
                "Select Record to Inspect from Results",
                options=id_options,
                index=0,
            )
            selected_uid = selected_choice.split(" — ")[0].strip()

            rec = db_manager.get_inspection_by_id(selected_uid)
            if rec:
                st.markdown(f"#### Record Inspection: `{rec.get('inspection_id') or rec['id']}`")
                st.caption("Copy Inspection ID:")
                st.code(rec.get("inspection_id") or str(rec["id"]), language=None)

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write(f"**Image Filename:** `{rec['image_name']}`")
                    st.write(f"**Inspection Modality:** `{rec['inspection_type'].capitalize()}`")
                    st.write(f"**Model Name & Version:** `{rec['model_name']} ({rec['model_version']})`")
                    st.write(f"**Audit Timestamp:** `{rec['timestamp']}`")
                with col_d2:
                    st.write(f"**Source Type:** `{'Evaluator Demo Sample' if rec.get('is_demo') else 'User Upload'}`")
                    st.write(f"**Has Defect Flag:** `{'YES — Defect Identified' if rec['has_defect'] else 'NO — Normal Asset'}`")
                    st.write(f"**Inference Latency:** `{rec['processing_time_ms']:.1f} ms`")
                    st.write(f"**Inspection Notes:** `{rec['notes'] or 'None recorded'}`")

                if rec["details_json"]:
                    st.markdown("**Structured Inference Payload (JSON):**")
                    try:
                        st.json(json.loads(rec["details_json"]))
                    except Exception:
                        st.text(rec["details_json"])

                col_act1, col_act2 = st.columns(2)
                with col_act1:
                    if rec["details_json"]:
                        try:
                            res_obj = InferenceResult.from_dict(
                                json.loads(rec["details_json"])
                            )
                            uid_label = rec.get("inspection_id") or str(rec["id"])
                            pdf_bytes = generate_inspection_pdf_bytes(
                                result=res_obj,
                                inspection_id=uid_label,
                                image_name=rec["image_name"],
                                notes=rec["notes"],
                                timestamp=rec["timestamp"],
                            )
                            st.download_button(
                                label=f":material/picture_as_pdf: Download PDF Report for {uid_label}",
                                data=pdf_bytes,
                                file_name=f"SolarVision_{uid_label}_{Path(rec['image_name']).stem}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        except Exception as exc:
                            st.warning(f"Could not compile PDF: {exc}")

                with col_act2:
                    delete_label = rec.get("inspection_id") or f"Record #{rec['id']}"
                    if st.button(
                        f":material/delete: Delete Record {delete_label}",
                        type="secondary",
                        use_container_width=True,
                    ):
                        db_manager.delete_inspection(rec["id"])
                        st.success(f"{delete_label} removed from database.")
                        st.rerun()
    else:
        st.info("No matching inspection records found for the applied filter and search criteria.")

    render_disclaimer_notice()


if __name__ == "__main__":
    from app.components.styles import apply_custom_theme

    apply_custom_theme()
    db = InspectionDatabaseManager(db_path=DEFAULT_DB_PATH)
    render_history_page(db)
