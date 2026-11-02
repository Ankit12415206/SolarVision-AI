"""
Module: inspection
Description: Live single-image solar panel inspection page for Streamlit
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image

from app.components.charts import plot_class_probabilities
from src.database.manager import InspectionDatabaseManager
from src.inference import InspectionType, SolarVisionInferencePipeline
from src.utils.report_generator import generate_inspection_pdf_bytes

# Preset sample test images
SAMPLE_IMAGES = {
    "Select a preset sample...": None,
    "[Classification] Clean Solar Panel": Path(
        "data/raw/classification/Clean/Clean (10).JPG"
    ),
    "[Classification] Physical Damage Panel": Path(
        "data/raw/classification/Physical-Damage/Physical (10).jpg"
    ),
    "[Classification] Bird-drop Soiled Panel": Path(
        "data/raw/classification/Bird-drop/10.JPG"
    ),
    "[Detection] Aerial Soiled Field Crop": Path(
        "data/raw/dust/aerial_soiling/test/images/ROI_102_jpg.rf.fa9f2d998fd0b9874bc2a3a58394e73e.jpg"
    ),
    "[Segmentation] EL Microcrack Cell": Path(
        "data/raw/microcracks/msps/test/images/First_101_20210705143317_5.png"
    ),
}


def render_inspection_page(
    pipeline: SolarVisionInferencePipeline,
    db_manager: InspectionDatabaseManager,
) -> None:
    """Render the Live Single-Image Inspection page."""
    st.markdown("### 🔍 Live Panel Defect Inspection")
    st.write(
        "Upload a drone-captured RGB photo or electroluminescence (EL) image to run "
        "real-time AI inspection across classification, detection, or segmentation."
    )

    col_setup1, col_setup2 = st.columns([1.2, 1.0])

    with col_setup1:
        modality_label = st.selectbox(
            "Select Inspection Modality",
            options=[
                "🏷️ Panel Condition Classification (6 Classes)",
                "🎯 Aerial Soiling Detection (Dust & Bird-drops)",
                "🔬 Electroluminescence Defect Segmentation (Microcracks)",
            ],
            index=0,
        )

        if "Classification" in modality_label:
            selected_type = InspectionType.CLASSIFICATION
        elif "Detection" in modality_label:
            selected_type = InspectionType.DETECTION
        else:
            selected_type = InspectionType.SEGMENTATION

    with col_setup2:
        preset_choice = st.selectbox(
            "Quick Demo: Load Sample Image",
            options=list(SAMPLE_IMAGES.keys()),
            index=0,
        )

    # Threshold adjustments
    kwargs = {}
    if selected_type == InspectionType.DETECTION:
        with st.expander("⚙️ Detection Threshold Settings", expanded=False):
            conf_thresh = st.slider("Confidence Threshold", 0.05, 0.95, 0.10, 0.05)
            iou_thresh = st.slider("NMS IoU Threshold", 0.10, 0.80, 0.45, 0.05)
            kwargs["conf_threshold"] = conf_thresh
            kwargs["iou_threshold"] = iou_thresh

    elif selected_type == InspectionType.SEGMENTATION:
        with st.expander("⚙️ Segmentation Threshold Settings", expanded=False):
            seg_thresh = st.slider(
                "Binary Probability Threshold", 0.10, 0.95, 0.70, 0.05
            )
            kwargs["threshold"] = seg_thresh

    # Image input: uploaded file takes priority over preset
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Supports standard drone RGB aerial photos and grayscale EL images.",
    )

    image_to_process: Optional[Image.Image] = None
    image_filename = "custom_image.jpg"
    image_file_path: Optional[str] = None

    if uploaded_file is not None:
        try:
            image_to_process = Image.open(uploaded_file)
            image_filename = uploaded_file.name
        except Exception as exc:
            st.error(f"Failed to load uploaded image: {exc}")
    elif preset_choice and SAMPLE_IMAGES[preset_choice] is not None:
        sample_path = SAMPLE_IMAGES[preset_choice]
        if sample_path.exists():
            image_to_process = Image.open(sample_path)
            image_filename = sample_path.name
            image_file_path = str(sample_path)
        else:
            st.info(
                f"ℹ️ Demo preset '{preset_choice}' is not available in this cloud deployment. "
                "Please upload your own panel image below to test real-time inference."
            )

    if image_to_process is None:
        st.info("👆 Please upload an image or select a sample image above to proceed.")
        return

    # Action button
    if st.button("🚀 Run AI Inspection", type="primary", use_container_width=True):
        with st.spinner(f"Running {selected_type.value} inference on CPU..."):
            try:
                result = pipeline.predict(
                    image_input=image_to_process,
                    inspection_type=selected_type,
                    **kwargs,
                )
                st.session_state["latest_result"] = result
                st.session_state["latest_image_name"] = image_filename
                st.session_state["latest_image_path"] = image_file_path
                st.session_state["latest_original_image"] = image_to_process
                st.session_state.pop("saved_rec_id", None)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                return

    # Render results if available in session
    if "latest_result" in st.session_state:
        result = st.session_state["latest_result"]
        img_name = st.session_state.get("latest_image_name", "image.jpg")
        orig_img = st.session_state.get("latest_original_image", image_to_process)

        st.markdown("---")
        st.markdown("#### 📊 Inspection Summary")

        # Top Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", result.model_name)
        c2.metric("Latency", f"{result.processing_time_ms:.1f} ms")
        c3.metric(
            "Dimensions",
            f"{result.image_metadata['width']}x{result.image_metadata['height']}",
        )

        if result.inspection_type == InspectionType.CLASSIFICATION:
            conf_str = f"{result.confidence * 100:.1f}%" if result.confidence else "N/A"
            c4.metric("Predicted Condition", result.predicted_class, delta=conf_str)
        elif result.inspection_type == InspectionType.DETECTION:
            num_dets = len(result.boxes) if result.boxes else 0
            c4.metric("Detections", f"{num_dets} objects")
        elif result.inspection_type == InspectionType.SEGMENTATION:
            pct = (result.defect_ratio or 0.0) * 100.0
            c4.metric(
                "Defect Area",
                f"{pct:.3f}%",
                delta="Microcrack Detected" if pct > 0 else "Normal",
                delta_color="inverse" if pct > 0 else "normal",
            )

        # Image display columns
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("**Original Input**")
            st.image(orig_img, use_container_width=True)

        with col_img2:
            st.markdown("**Annotated Result**")
            if result.annotated_image:
                st.image(result.annotated_image, use_container_width=True)

        # Modality Details
        if (
            result.inspection_type == InspectionType.CLASSIFICATION
            and result.class_probabilities
        ):
            st.plotly_chart(
                plot_class_probabilities(result.class_probabilities),
                use_container_width=True,
            )

        elif result.inspection_type == InspectionType.DETECTION:
            if result.boxes:
                st.markdown("**Detected Bounding Boxes:**")
                box_rows = [
                    {
                        "Class": b.class_name,
                        "Confidence": f"{b.score * 100:.1f}%",
                        "X1": round(b.x1, 3),
                        "Y1": round(b.y1, 3),
                        "X2": round(b.x2, 3),
                        "Y2": round(b.y2, 3),
                    }
                    for b in result.boxes
                ]
                st.dataframe(pd.DataFrame(box_rows), use_container_width=True)
            else:
                st.info(
                    "No soiling or defect objects detected at this confidence threshold."
                )

        elif result.inspection_type == InspectionType.SEGMENTATION:
            st.info(
                f"Defect area represents **{(result.defect_ratio or 0.0)*100:.4f}%** "
                f"of total cell area (Threshold: {result.threshold})."
            )

        # Persistence & Report Export
        st.markdown("---")
        col_save1, col_save2, col_save3 = st.columns([1.5, 1.0, 1.0])
        with col_save1:
            user_notes = st.text_input(
                "Add Inspection Notes (Optional)",
                placeholder="e.g., Array Section 4, Row 12, Routine scan",
                key="inspection_notes_input",
            )
        with col_save2:
            st.write("")
            st.write("")
            if st.button("💾 Save to History", use_container_width=True):
                rec_id = db_manager.save_inspection(
                    result=result,
                    image_name=img_name,
                    image_path=st.session_state.get("latest_image_path"),
                    notes=user_notes,
                )
                st.session_state["saved_rec_id"] = rec_id
                st.success(f"✅ Logged! Record ID: #{rec_id}")

        with col_save3:
            st.write("")
            st.write("")
            pdf_bytes = generate_inspection_pdf_bytes(
                result=result,
                inspection_id=st.session_state.get("saved_rec_id"),
                image_name=img_name,
                original_image=orig_img,
                notes=user_notes,
            )
            st.download_button(
                label="📥 Export PDF Report",
                data=pdf_bytes,
                file_name=f"SolarVision_{result.inspection_type.value}_{Path(img_name).stem}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
