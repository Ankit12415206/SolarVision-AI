"""
Module: batch
Description: Batch multi-image solar panel inspection page for Streamlit
Author: SolarVision AI Team
Created: 2026-11-02
"""

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

from src.database.manager import DEFAULT_DB_PATH, InspectionDatabaseManager
from src.inference import InspectionType, SolarVisionInferencePipeline


def render_batch_page(
    pipeline: SolarVisionInferencePipeline,
    db_manager: InspectionDatabaseManager,
) -> None:
    """Render the Sequential Batch Processing page."""
    st.markdown("### 📁 Batch Panel Inspection")
    st.write(
        "Upload a batch of drone survey or EL inspection images. Processing executes "
        "sequentially to respect CPU and memory limits (16 GB RAM laptop friendly)."
    )

    col1, col2 = st.columns([1.2, 1.0])
    with col1:
        modality_label = st.selectbox(
            "Select Inspection Modality for Batch",
            options=[
                "🏷️ Panel Condition Classification (6 Classes)",
                "🎯 Aerial Soiling Detection (Dust & Bird-drops)",
                "🔬 Electroluminescence Defect Segmentation (Microcracks)",
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
        auto_save_db = st.checkbox(
            "Auto-log batch results into database history",
            value=True,
            help="Automatically records each processed image result into SQLite.",
        )

    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload multiple survey images to process together.",
    )

    if not uploaded_files:
        st.info(
            "👆 Please upload one or more image files to initiate batch processing."
        )
        return

    st.write(f"📁 Selected **{len(uploaded_files)}** images for inspection.")

    if st.button("🚀 Process Batch Sequentially", type="primary"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        results_records: List[Dict[str, Any]] = []
        annotated_previews: List[Dict[str, Any]] = []

        total = len(uploaded_files)
        success_count = 0
        error_count = 0

        for idx, file in enumerate(uploaded_files, start=1):
            status_text.text(f"Processing ({idx}/{total}): {file.name}...")
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
                        f"{ratio * 100:.3f}% defect area" if has_defect else "No crack"
                    )

                rec: Dict[str, Any] = {
                    "File Name": file.name,
                    "Modality": selected_type.value,
                    "Model": result.model_name,
                    "Status": "Defect" if has_defect else "Normal",
                    "Finding": defect_label,
                    "Confidence": f"{confidence * 100:.1f}%" if confidence else "N/A",
                    "Latency (ms)": round(result.processing_time_ms, 1),
                }

                if auto_save_db:
                    db_manager.save_inspection(
                        result=result,
                        image_name=file.name,
                        notes="Batch automated inspection",
                    )

                results_records.append(rec)
                if result.annotated_image:
                    annotated_previews.append(
                        {
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
                        "File Name": file.name,
                        "Modality": selected_type.value,
                        "Model": "N/A",
                        "Status": "Failed",
                        "Finding": f"Error: {exc}",
                        "Confidence": "N/A",
                        "Latency (ms)": 0.0,
                    }
                )

            progress_bar.progress(idx / total)

        status_text.text("✅ Batch processing finished!")
        st.session_state["batch_results"] = results_records
        st.session_state["batch_previews"] = annotated_previews

    # Display results if present
    if "batch_results" in st.session_state:
        df_batch = pd.DataFrame(st.session_state["batch_results"])

        st.markdown("---")
        st.markdown("#### 📋 Batch Results Overview")

        total = len(df_batch)
        defects = len(df_batch[df_batch["Status"] == "Defect"])
        clean = len(df_batch[df_batch["Status"] == "Normal"])
        failed = len(df_batch[df_batch["Status"] == "Failed"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Processed Images", total)
        m2.metric("Defects / Soiled", defects)
        m3.metric("Healthy Panels", clean)
        m4.metric("Failed", failed)

        st.dataframe(df_batch, use_container_width=True)

        # Export CSV
        csv_data = df_batch.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Batch Results (CSV)",
            data=csv_data,
            file_name="solarvision_batch_inspection_results.csv",
            mime="text/csv",
        )

        # Visual Gallery (Preview first 6)
        previews = st.session_state.get("batch_previews", [])
        if previews:
            st.markdown("#### 🖼️ Inspection Visual Gallery")
            cols = st.columns(min(3, len(previews)))
            for i, p in enumerate(previews[:6]):
                with cols[i % len(cols)]:
                    st.caption(f"**{p['name']}** — {p['status']}: {p['finding']}")
                    st.image(p["image"], use_container_width=True)


if __name__ == "__main__":
    from app.components.styles import apply_custom_theme

    apply_custom_theme()
    pipe = SolarVisionInferencePipeline(device="cpu", auto_load=False)
    db = InspectionDatabaseManager(db_path=DEFAULT_DB_PATH)
    render_batch_page(pipe, db)
