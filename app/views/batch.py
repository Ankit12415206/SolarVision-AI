"""
Module: batch
Description: Batch multi-image solar panel inspection page with traceable IDs and safe CPU queuing
Author: SolarVision AI Team
Created: 2026-11-02
"""

from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path for cloud deployment
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st
from PIL import Image

from app.components.styles import render_disclaimer_notice
from src.database.manager import (
    DEFAULT_DB_PATH,
    InspectionDatabaseManager,
    format_inspection_id,
)
from src.inference import InspectionType, SolarVisionInferencePipeline


def render_batch_page(
    pipeline: SolarVisionInferencePipeline,
    db_manager: InspectionDatabaseManager,
    theme: str = "dark",
) -> None:
    """Render the Sequential Batch Processing page for multi-asset drone surveys.

    Args:
        pipeline: Active SolarVision inference pipeline.
        db_manager: SQLite database manager.
        theme: Active theme ('dark' or 'light').
    """
    st.markdown("### :material/batch_prediction: Batch Aerial Survey Inspection")
    st.caption(
        "Upload multi-image drone surveys or EL inspection folders. Processing executes "
        "sequentially to respect CPU and memory limits (16 GB RAM laptop friendly) while assigning unique Inspection IDs."
    )

    col1, col2 = st.columns([1.4, 1.0])
    with col1:
        modality_label = st.selectbox(
            "Batch Inspection Modality",
            options=[
                "Panel Condition Classification (6 Classes)",
                "Aerial Soiling Detection (Dust & Bird-drops)",
                "Electroluminescence Defect Segmentation (Microcracks)",
            ],
            key="batch_modality_select",
        )

        if "Classification" in modality_label:
            selected_type = InspectionType.CLASSIFICATION
        elif "Detection" in modality_label:
            selected_type = InspectionType.DETECTION
        else:
            selected_type = InspectionType.SEGMENTATION

    with col2:
        st.write("")
        st.write("")
        auto_save_db = st.checkbox(
            "Auto-log results to SQLite database",
            value=True,
            help="Automatically saves each processed image and inference finding into the audit history.",
        )

    uploaded_files = st.file_uploader(
        "Select Survey Images for Batch Queue",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload multiple survey images (JPG/PNG) to process together sequentially.",
    )

    if not uploaded_files:
        st.info("Select or drag multiple survey images above to start batch processing.")
        render_disclaimer_notice()
        return

    st.markdown(
        f"**Queue Status:** `{len(uploaded_files)}` survey image(s) loaded."
    )

    if st.button(
        ":material/play_arrow: Start Sequential Batch Inspection",
        type="primary",
        use_container_width=True,
    ):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        results_records: List[Dict[str, Any]] = []
        annotated_previews: List[Dict[str, Any]] = []

        total = len(uploaded_files)
        success_count = 0
        error_count = 0
        year = datetime.now(timezone.utc).year

        # Pre-query next available ID sequence from maximum existing record ID
        with db_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM inspections")
            current_seq = int(cursor.fetchone()["max_id"])

        for idx, file in enumerate(uploaded_files, start=1):
            status_text.text(f"Processing ({idx}/{total}): {file.name}...")
            assigned_uid = format_inspection_id(year, current_seq + idx)

            try:
                img = Image.open(file)
                result = pipeline.predict(
                    image_input=img,
                    inspection_type=selected_type,
                )

                has_defect = False
                defect_label = "Clean"
                confidence = None
                det_count = 0
                ratio = 0.0

                if selected_type == InspectionType.CLASSIFICATION:
                    defect_label = result.predicted_class or "Unknown"
                    has_defect = defect_label.lower() != "clean"
                    confidence = result.confidence

                elif selected_type == InspectionType.DETECTION:
                    det_count = len(result.boxes) if result.boxes else 0
                    has_defect = det_count > 0
                    defect_label = (
                        f"{det_count} objects" if has_defect else "No soiling"
                    )

                elif selected_type == InspectionType.SEGMENTATION:
                    ratio = result.defect_ratio or 0.0
                    has_defect = ratio > 0.0
                    defect_label = (
                        f"{ratio * 100:.3f}% crack area"
                        if has_defect
                        else "No crack"
                    )

                if auto_save_db:
                    db_manager.save_inspection(
                        result=result,
                        image_name=file.name,
                        notes="Batch automated survey",
                        is_demo=False,
                        inspection_id=assigned_uid,
                    )

                rec: Dict[str, Any] = {
                    "Inspection ID": assigned_uid,
                    "File Name": file.name,
                    "Modality": selected_type.value.capitalize(),
                    "Model": result.model_name,
                    "Status": "Defect" if has_defect else "Normal",
                    "Finding": defect_label,
                    "Confidence": (
                        f"{confidence * 100:.1f}%" if confidence else "N/A"
                    ),
                    "Latency (ms)": round(result.processing_time_ms, 1),
                    "Error Reason": "None",
                }

                results_records.append(rec)
                if result.annotated_image:
                    annotated_previews.append(
                        {
                            "id": assigned_uid,
                            "name": file.name,
                            "image": result.annotated_image,
                            "status": rec["Status"],
                            "finding": rec["Finding"],
                        }
                    )
                success_count += 1

            except Exception as exc:
                error_count += 1
                results_records.append(
                    {
                        "Inspection ID": assigned_uid,
                        "File Name": file.name,
                        "Modality": selected_type.value.capitalize(),
                        "Model": "N/A",
                        "Status": "Failed",
                        "Finding": "Inference Error",
                        "Confidence": "N/A",
                        "Latency (ms)": 0.0,
                        "Error Reason": str(exc),
                    }
                )

            progress_bar.progress(idx / total)

        status_text.success(
            f"Batch inspection finished! Successfully processed: {success_count}, Failed: {error_count}"
        )
        st.session_state["batch_results"] = results_records
        st.session_state["batch_previews"] = annotated_previews

    # Display results if present
    if "batch_results" in st.session_state:
        df_batch = pd.DataFrame(st.session_state["batch_results"])

        st.markdown("---")
        st.markdown("#### :material/table_chart: Traceable Batch Survey Findings")

        total = len(df_batch)
        defects = len(df_batch[df_batch["Status"] == "Defect"])
        clean = len(df_batch[df_batch["Status"] == "Normal"])
        failed = len(df_batch[df_batch["Status"] == "Failed"])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(label="Processed Assets", value=f"{total}")
        with m2:
            st.metric(
                label="Defects Detected",
                value=f"{defects}",
                delta=f"{(defects / total * 100):.1f}% rate" if total else None,
                delta_color="inverse",
            )
        with m3:
            st.metric(label="Nominal / Clean", value=f"{clean}")
        with m4:
            st.metric(
                label="Failed / Errored",
                value=f"{failed}",
                delta_color="inverse" if failed > 0 else "off",
            )

        st.dataframe(df_batch, use_container_width=True, hide_index=True)

        # Export CSV Button with Inspection IDs
        csv_data = df_batch.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=":material/download: Export Batch Results with Traceable IDs (CSV)",
            data=csv_data,
            file_name="solarvision_batch_inspection_results.csv",
            mime="text/csv",
        )

        # Visual Gallery (Preview up to 6 annotated results)
        previews = st.session_state.get("batch_previews", [])
        if previews:
            st.markdown("---")
            st.markdown("#### :material/gallery_thumbnail: Visual Inspection Gallery")
            cols = st.columns(min(3, len(previews)))
            for i, p in enumerate(previews[:6]):
                with cols[i % len(cols)]:
                    st.caption(
                        f"**{p['id']}** • `{p['name']}`  \n"
                        f"**{p['status']}**: {p['finding']}"
                    )
                    st.image(p["image"], use_container_width=True)

    render_disclaimer_notice()


if __name__ == "__main__":
    from app.components.styles import apply_custom_theme

    apply_custom_theme()
    pipe = SolarVisionInferencePipeline(device="cpu", auto_load=False)
    db = InspectionDatabaseManager(db_path=DEFAULT_DB_PATH)
    render_batch_page(pipe, db)
