"""
Module: inspection
Description: Live single-image solar panel inspection page with demo mode and traceability
Author: SolarVision AI Team
Created: 2026-11-02
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path for cloud deployment
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st
from PIL import Image

from app.components.charts import plot_class_probabilities
from app.components.styles import (
    render_disclaimer_notice,
    render_model_transparency,
)
from src.database.manager import (
    DEFAULT_DB_PATH,
    InspectionDatabaseManager,
)
from src.inference import InspectionType, SolarVisionInferencePipeline
from src.utils.demo_samples import (
    get_demo_samples_for_modality,
)
from src.utils.report_generator import generate_inspection_pdf_bytes


def render_inspection_page(
    pipeline: SolarVisionInferencePipeline,
    db_manager: InspectionDatabaseManager,
    theme: str = "dark",
) -> None:
    """Render the Live Single-Image Inspection page with 3-step structured workflow.

    Args:
        pipeline: Active SolarVision inference pipeline.
        db_manager: SQLite database manager.
        theme: Active theme ('dark' or 'light').
    """
    col_hdr, col_reset = st.columns([3, 1])
    with col_hdr:
        st.markdown("### :material/image_search: Live Panel Defect Inspection")
        st.caption(
            "Inspect individual aerial drone RGB photos or electroluminescence (EL) imagery "
            "with full traceability, verified demo samples, and academic model transparency."
        )
    with col_reset:
        st.write("")
        if st.button(":material/refresh: Reset / New Inspection", use_container_width=True):
            st.session_state.pop("latest_result", None)
            st.session_state.pop("latest_inspection_id", None)
            st.session_state.pop("latest_image_name", None)
            st.session_state.pop("latest_image_path", None)
            st.session_state.pop("latest_original_image", None)
            st.session_state.pop("saved_rec_id", None)
            st.session_state.pop("latest_is_demo", None)
            st.session_state.pop("live_notes_input", None)
            st.rerun()

    # ---------------- Step 1: Modality Selection ----------------
    st.markdown("#### :material/tune: Step 1: Select Inspection Modality")
    col_mod1, col_mod2 = st.columns([1.5, 1.0])

    with col_mod1:
        modality_label = st.radio(
            "Inspection Task",
            options=[
                "Panel Condition Classification (6 Classes)",
                "Aerial Soiling Detection (Dust & Bird-drops)",
                "Electroluminescence Defect Segmentation (Microcracks)",
            ],
            index=0,
            horizontal=False,
            label_visibility="collapsed",
        )

        if "Classification" in modality_label:
            selected_type = InspectionType.CLASSIFICATION
            modality_desc = (
                "Evaluates entire panel surface into 6 health categories: "
                "Clean, Dusty, Bird-drop, Electrical-damage, Physical-Damage, Snow-Covered."
            )
            model_display_name = "SolarMobileNetV2"
        elif "Detection" in modality_label:
            selected_type = InspectionType.DETECTION
            modality_desc = (
                "Pinpoints localized dust clusters and bird-drop deposits using bounding boxes "
                "via YOLOv8n deep object detector."
            )
            model_display_name = "SolarYOLOv8n"
        else:
            selected_type = InspectionType.SEGMENTATION
            modality_desc = (
                "Computes pixel-precise binary mask of internal microcracks, fractures, "
                "and busbar interruptions in PV silicon cells from grayscale EL scans."
            )
            model_display_name = "SolarLightweightUNet"

        st.info(f"**Modality Objective ({model_display_name}):** {modality_desc}")

    with col_mod2:
        # Filter demo samples specific to this modality
        modality_samples = get_demo_samples_for_modality(selected_type)
        demo_options = ["None (Upload Custom Asset)"] + [
            s.title for s in modality_samples
        ]
        demo_choice = st.selectbox(
            "Evaluator Demo Preset",
            options=demo_options,
            index=0,
            help="Select a pre-bundled evaluation image. No dataset downloads or uploads required.",
        )

    # Collapsible threshold adjustments
    kwargs = {}
    sensitivity_info = "Standard Defaults"
    if selected_type == InspectionType.DETECTION:
        with st.expander(":material/settings: Detection Sensitivity Settings", expanded=False):
            conf_thresh = st.slider(
                "Confidence Threshold",
                min_value=0.05,
                max_value=0.95,
                value=0.10,
                step=0.05,
                help="Minimum confidence for bounding box proposals.",
            )
            iou_thresh = st.slider(
                "NMS IoU Threshold",
                min_value=0.10,
                max_value=0.80,
                value=0.45,
                step=0.05,
                help="Intersection-over-Union threshold for Non-Maximum Suppression.",
            )
            kwargs["conf_threshold"] = conf_thresh
            kwargs["iou_threshold"] = iou_thresh
            sensitivity_info = f"Conf: {conf_thresh:.2f}, IoU: {iou_thresh:.2f}"

    elif selected_type == InspectionType.SEGMENTATION:
        with st.expander(":material/settings: Segmentation Sensitivity Settings", expanded=False):
            seg_thresh = st.slider(
                "Binary Probability Threshold",
                min_value=0.10,
                max_value=0.95,
                value=0.70,
                step=0.05,
                help="Decision cutoff for pixel-level crack mask classification.",
            )
            kwargs["threshold"] = seg_thresh
            sensitivity_info = f"Threshold: {seg_thresh:.2f}"

    st.markdown("---")

    # ---------------- Step 2: Image Selection ----------------
    st.markdown("#### :material/upload_file: Step 2: Image Input & Evaluator Verification")

    image_to_process: Optional[Image.Image] = None
    image_filename = "custom_image.jpg"
    image_file_path: Optional[str] = None
    is_demo = False

    if demo_choice != "None (Upload Custom Asset)":
        # Find matching DemoSample
        selected_sample = next(
            (s for s in modality_samples if s.title == demo_choice), None
        )
        if selected_sample:
            resolved_path = selected_sample.resolve_path()
            if resolved_path and resolved_path.exists():
                try:
                    image_to_process = Image.open(resolved_path)
                    image_filename = resolved_path.name
                    image_file_path = str(resolved_path)
                    is_demo = True
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 0.75rem;">
                            <span class="badge-demo">🔬 EVALUATOR DEMO SAMPLE</span>
                            <span style="margin-left: 0.5rem; font-weight: 600;">{selected_sample.title}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"**Evaluator Note:** {selected_sample.description}  \n"
                        f"**Expected Benchmark Finding:** `{selected_sample.expected_finding}`  \n"
                        f"**Attribution & License:** *{selected_sample.attribution}*"
                    )
                except Exception as exc:
                    st.error(f"Error loading demo sample `{resolved_path}`: {exc}")
            else:
                if "Dataset A" in selected_sample.attribution:
                    st.warning(
                        f"Demo preset `{demo_choice}` is excluded from the public repository bundle "
                        "due to unverified upstream redistribution licensing for Dataset A. "
                        "Please select the verified CC BY 4.0 Aerial Solar Array preset above, "
                        "or upload your own solar panel photo below."
                    )
                else:
                    st.error(
                        f"Demo preset `{demo_choice}` is not available on disk at `{selected_sample.primary_path}`. "
                        "Unrelated images will not be silently substituted. Please upload a panel image below."
                    )

    if image_to_process is None:
        file_hint = (
            "Supported formats: RGB JPG/PNG for Drone Aerial Survey; 8-bit Grayscale PNG/JPG for EL Scan."
        )
        uploaded_file = st.file_uploader(
            "Upload Solar Asset Image",
            type=["jpg", "jpeg", "png"],
            help=file_hint,
        )
        if uploaded_file is not None:
            try:
                image_to_process = Image.open(uploaded_file)
                image_filename = uploaded_file.name
                is_demo = False
            except Exception as exc:
                st.error(f"Failed to decode uploaded image: {exc}")

    if image_to_process is None:
        st.info("Select an Evaluator Demo Preset or upload a panel photo above to begin inspection.")
        render_disclaimer_notice()
        return

    st.markdown("---")

    # ---------------- Step 3: Run & Results ----------------
    st.markdown("#### :material/play_circle: Step 3: Run AI Inspection & View Traceable Findings")

    if st.button(
        ":material/rocket_launch: Execute Model Inference",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(f"Executing {selected_type.value} model ({model_display_name}) on CPU..."):
            try:
                result = pipeline.predict(
                    image_input=image_to_process,
                    inspection_type=selected_type,
                    **kwargs,
                )

                # Allocate unique traceable Inspection ID from database sequence
                assigned_uid = db_manager.get_next_inspection_id()

                st.session_state["latest_result"] = result
                st.session_state["latest_inspection_id"] = assigned_uid
                st.session_state["latest_image_name"] = image_filename
                st.session_state["latest_image_path"] = image_file_path
                st.session_state["latest_original_image"] = image_to_process
                st.session_state["latest_is_demo"] = is_demo
                st.session_state.pop("saved_rec_id", None)
            except Exception as exc:
                st.error(f"Inference execution failed: {exc}")
                return

    # Render results if available in session
    if "latest_result" in st.session_state:
        result = st.session_state["latest_result"]
        img_name = st.session_state.get("latest_image_name", "image.jpg")
        orig_img = st.session_state.get("latest_original_image", image_to_process)
        uid = st.session_state.get("latest_inspection_id", "SV-2026-UNASSIGNED")
        is_demo_run = st.session_state.get("latest_is_demo", False)

        # Prominent Traceable ID Chip
        st.markdown(
            f"""
            <div style="background-color: var(--sv-bg-card); border: 1px solid var(--sv-border); border-radius: 8px; padding: 0.85rem 1.25rem; margin: 1rem 0; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--sv-text-secondary); font-weight: 600;">Traceable Inspection ID</span>
                    <div style="font-size: 1.3rem; font-weight: 700; color: var(--sv-text-primary); font-family: monospace;">{uid}</div>
                </div>
                <div>
                    {'<span class="badge-demo">🔬 DEMO INSPECTION</span>' if is_demo_run else '<span class="badge-info">📸 USER UPLOAD</span>'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Copy Inspection ID:")
        st.code(uid, language=None)

        # Summary Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(label="Model Architecture", value=result.model_name)
        with c2:
            st.metric(
                label="CPU Latency",
                value=f"{result.processing_time_ms:.1f} ms",
                help="CPU execution latency for end-to-end preprocessing, model forward pass, and decoding.",
            )
        with c3:
            st.metric(
                label="Resolution",
                value=f"{result.image_metadata['width']}x{result.image_metadata['height']}",
            )
        with c4:
            if result.inspection_type == InspectionType.CLASSIFICATION:
                conf_str = (
                    f"{result.confidence * 100:.1f}% confidence"
                    if result.confidence
                    else "N/A"
                )
                st.metric(
                    label="Predicted Condition",
                    value=result.predicted_class or "Unknown",
                    delta=conf_str,
                )
            elif result.inspection_type == InspectionType.DETECTION:
                num_dets = len(result.boxes) if result.boxes else 0
                st.metric(
                    label="Detected Deposits",
                    value=f"{num_dets} objects",
                    delta="Defect Present" if num_dets > 0 else "Clean Surface",
                    delta_color="inverse" if num_dets > 0 else "normal",
                )
            elif result.inspection_type == InspectionType.SEGMENTATION:
                pct = (result.defect_ratio or 0.0) * 100.0
                st.metric(
                    label="EL Defect Area",
                    value=f"{pct:.3f}%",
                    delta="Silicon Defect" if pct > 0 else "Nominal Cell",
                    delta_color="inverse" if pct > 0 else "normal",
                )

        # Model Transparency Strip
        dim_str = f"{result.image_metadata['width']}x{result.image_metadata['height']}"
        render_model_transparency(
            model_name=result.model_name,
            modality=result.inspection_type.value.capitalize(),
            latency_ms=result.processing_time_ms,
            resolution=dim_str,
            metric_info=sensitivity_info,
        )

        # Visual Comparison Dual Panels
        st.markdown("##### :material/compare: Visual Analysis Comparison")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.caption(f"**Original Input Imagery ({img_name})**")
            st.image(orig_img, use_container_width=True)

        with col_img2:
            st.caption(f"**Annotated Output ({result.model_name})**")
            if result.annotated_image:
                st.image(result.annotated_image, use_container_width=True)
            else:
                st.info("No annotated visualization returned for this prediction.")

        # Modality Details Breakdown
        if (
            result.inspection_type == InspectionType.CLASSIFICATION
            and result.class_probabilities
        ):
            st.plotly_chart(
                plot_class_probabilities(result.class_probabilities, theme=theme),
                use_container_width=True,
            )
            st.caption(
                "Classification softmax distribution indicates the model's categorical certainty "
                "across the 6 trained classes. High confidence is indicative but does not guarantee ground-truth correctness."
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
                st.success("Clean surface: No discrete soiling or bird-drop deposits detected at this threshold.")

        elif result.inspection_type == InspectionType.SEGMENTATION:
            defect_pct = (result.defect_ratio or 0.0) * 100.0
            if defect_pct > 0:
                st.warning(
                    f"Silicon discontinuity detected: Anomaly occupies **{defect_pct:.4f}%** "
                    f"of photovoltaic silicon surface (Threshold: {result.threshold}). "
                    "Note: Electroluminescence anomalies represent cracks, finger interruptions, "
                    "or cell fractures as labeled in the benchmark annotations."
                )
            else:
                st.success("Nominal EL Scan: No silicon microcrack or fracture anomalies identified.")

        # Persistence & Report Actions
        st.markdown("---")
        st.markdown("##### :material/save: Logging, Traceability & Audit Documentation")
        col_save1, col_save2, col_save3 = st.columns([1.5, 1.0, 1.0])
        with col_save1:
            default_notes = "Evaluator demo run" if is_demo_run else ""
            user_notes = st.text_input(
                "Inspection Notes & Audit Tags",
                value=default_notes,
                placeholder="e.g., Array 03, String 14, Pre-cleaning audit",
                key="live_notes_input",
            )
        with col_save2:
            st.write("")
            st.write("")
            if st.button(
                ":material/bookmark: Save to Fleet Database",
                use_container_width=True,
            ):
                rec_id = db_manager.save_inspection(
                    result=result,
                    image_name=img_name,
                    image_path=st.session_state.get("latest_image_path"),
                    notes=user_notes,
                    is_demo=is_demo_run,
                    inspection_id=uid,
                )
                st.session_state["saved_rec_id"] = rec_id
                st.success(f"Audit recorded under ID `{uid}` (Row #{rec_id})")

        with col_save3:
            st.write("")
            st.write("")
            pdf_bytes = generate_inspection_pdf_bytes(
                result=result,
                inspection_id=uid,
                image_name=img_name,
                original_image=orig_img,
                notes=user_notes,
            )
            st.download_button(
                label=":material/picture_as_pdf: Download PDF Report",
                data=pdf_bytes,
                file_name=f"SolarVision_{uid}_{Path(img_name).stem}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    render_disclaimer_notice()


if __name__ == "__main__":
    from app.components.styles import apply_custom_theme

    apply_custom_theme()
    pipe = SolarVisionInferencePipeline(device="cpu", auto_load=False)
    db = InspectionDatabaseManager(db_path=DEFAULT_DB_PATH)
    render_inspection_page(pipe, db)
