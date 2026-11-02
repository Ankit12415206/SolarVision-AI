"""
Module: report_generator
Description: Professional PDF inspection report generator using ReportLab for SolarVision AI
Author: SolarVision AI Team
Created: 2026-11-02
"""

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Union

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
)
from reportlab.platypus import Image as RLImage
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.inference.types import InferenceResult, InspectionType


def _pil_to_reportlab_image(
    img: PILImage.Image,
    max_width: float = 460,
    max_height: float = 240,
) -> RLImage:
    """Convert a PIL Image into an in-memory ReportLab flowable image preserving aspect ratio.

    Args:
        img: Input PIL Image.
        max_width: Maximum allowed width in points. Defaults to 460.
        max_height: Maximum allowed height in points. Defaults to 240.

    Returns:
        RLImage: ReportLab image flowable.
    """
    buf = io.BytesIO()
    # Save as PNG
    img.save(buf, format="PNG")
    buf.seek(0)

    orig_w, orig_h = img.size
    aspect = orig_h / orig_w if orig_w > 0 else 1.0

    target_w = min(max_width, float(orig_w))
    target_h = target_w * aspect

    if target_h > max_height:
        target_h = max_height
        target_w = target_h / aspect

    return RLImage(buf, width=target_w, height=target_h)


def generate_inspection_pdf_bytes(
    result: InferenceResult,
    inspection_id: Optional[int] = None,
    image_name: Optional[str] = None,
    original_image: Optional[PILImage.Image] = None,
    notes: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> bytes:
    """Generate PDF inspection report and return bytes.

    Args:
        result: The InferenceResult to generate the report for.
        inspection_id: Optional database record primary key ID.
        image_name: Display filename of the inspected image.
        original_image: Optional PIL Image of original input.
        notes: Optional operator inspection notes.
        timestamp: Optional ISO timestamp. Defaults to current UTC time.

    Returns:
        bytes: Raw PDF bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    story: List[Any] = []
    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "BrandTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
    )
    tagline_style = ParagraphStyle(
        "BrandTagline",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748B"),
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748B"),
    )

    now_str = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    record_id_str = f"#{inspection_id}" if inspection_id else "Unsaved Session"
    img_name_str = (
        image_name or Path(result.image_metadata.get("source", "sample.jpg")).name
    )

    # Header Title Banner
    story.append(Paragraph("SolarVision AI 🔆", title_style))
    story.append(
        Paragraph(
            "Automated Solar Photovoltaic Defect Inspection & Aerial Survey Report",
            tagline_style,
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor("#F59E0B"),
            spaceBefore=2,
            spaceAfter=10,
        )
    )

    # Metadata & Results Summary Table
    has_defect = False
    finding_summary = "Normal"
    if result.inspection_type == InspectionType.CLASSIFICATION:
        finding_summary = (
            f"{result.predicted_class} ({(result.confidence or 0)*100:.1f}%)"
        )
        has_defect = (result.predicted_class or "").lower() != "clean"
    elif result.inspection_type == InspectionType.DETECTION:
        num = len(result.boxes) if result.boxes else 0
        finding_summary = f"{num} detected objects"
        has_defect = num > 0
    elif result.inspection_type == InspectionType.SEGMENTATION:
        pct = (result.defect_ratio or 0.0) * 100.0
        finding_summary = f"{pct:.3f}% active defect area"
        has_defect = pct > 0

    status_tag = "DEFECT / SOILED" if has_defect else "CLEAN / NORMAL"
    status_color = (
        colors.HexColor("#EF4444") if has_defect else colors.HexColor("#10B981")
    )

    meta_table_data = [
        [
            Paragraph("<b>Inspection ID:</b>", body_style),
            Paragraph(record_id_str, body_style),
            Paragraph("<b>Date & Time:</b>", body_style),
            Paragraph(now_str, body_style),
        ],
        [
            Paragraph("<b>Inspection Type:</b>", body_style),
            Paragraph(result.inspection_type.value.capitalize(), body_style),
            Paragraph("<b>Input Image:</b>", body_style),
            Paragraph(img_name_str, body_style),
        ],
        [
            Paragraph("<b>Model Architecture:</b>", body_style),
            Paragraph(result.model_name, body_style),
            Paragraph("<b>Model Version:</b>", body_style),
            Paragraph(result.model_version, body_style),
        ],
        [
            Paragraph("<b>Inference Latency:</b>", body_style),
            Paragraph(f"{result.processing_time_ms:.1f} ms (CPU)", body_style),
            Paragraph("<b>Image Resolution:</b>", body_style),
            Paragraph(
                f"{result.image_metadata.get('width', 0)} x {result.image_metadata.get('height', 0)} px",
                body_style,
            ),
        ],
        [
            Paragraph("<b>Inspection Finding:</b>", body_style),
            Paragraph(f"<b>{finding_summary}</b>", body_style),
            Paragraph("<b>Condition Status:</b>", body_style),
            Paragraph(
                f"<font color='{status_color.hexval()}'><b>{status_tag}</b></font>",
                body_style,
            ),
        ],
    ]

    meta_table = Table(meta_table_data, colWidths=[110, 160, 110, 160])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(meta_table)

    if notes:
        story.append(Spacer(1, 6))
        notes_p = Paragraph(f"<b>Operator Notes:</b> {notes}", body_style)
        story.append(notes_p)

    story.append(Spacer(1, 10))

    # Inspection Visuals Section
    story.append(Paragraph("Inspection Visualizations", section_style))

    # Include Annotated Image
    if result.annotated_image is not None:
        rl_annotated = _pil_to_reportlab_image(
            result.annotated_image, max_width=520, max_height=200
        )
        story.append(rl_annotated)
        story.append(Spacer(1, 8))

    # Modality Specific Data Breakdown
    if result.inspection_type == InspectionType.CLASSIFICATION:
        if result.class_probabilities:
            story.append(Paragraph("Class Probability Breakdown", section_style))
            prob_rows: List[List[Any]] = [
                [
                    Paragraph("<b>Class Name</b>", body_style),
                    Paragraph("<b>Probability (%)</b>", body_style),
                    Paragraph("<b>Indicator</b>", body_style),
                ]
            ]
            sorted_probs = sorted(
                result.class_probabilities.items(), key=lambda x: x[1], reverse=True
            )
            for cname, p in sorted_probs:
                is_top = cname == result.predicted_class
                mark = "TOP MATCH" if is_top else ""
                prob_rows.append(
                    [
                        Paragraph(f"<b>{cname}</b>" if is_top else cname, body_style),
                        Paragraph(f"{p * 100:.2f}%", body_style),
                        Paragraph(f"<b>{mark}</b>", body_style),
                    ]
                )

            prob_table = Table(prob_rows, colWidths=[200, 150, 170])
            prob_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(prob_table)

    elif result.inspection_type == InspectionType.DETECTION:
        story.append(Paragraph("Detected Soiling & Defect Objects", section_style))
        if result.boxes:
            det_rows: List[List[Any]] = [
                [
                    Paragraph("<b>#</b>", body_style),
                    Paragraph("<b>Class</b>", body_style),
                    Paragraph("<b>Confidence</b>", body_style),
                    Paragraph(
                        "<b>Normalized Bounding Box [x1, y1, x2, y2]</b>", body_style
                    ),
                ]
            ]
            for idx, b in enumerate(result.boxes[:15], 1):  # Limit table rows to top 15
                bbox_str = f"[{b.x1:.3f}, {b.y1:.3f}, {b.x2:.3f}, {b.y2:.3f}]"
                det_rows.append(
                    [
                        Paragraph(str(idx), body_style),
                        Paragraph(b.class_name, body_style),
                        Paragraph(f"{b.score * 100:.1f}%", body_style),
                        Paragraph(bbox_str, body_style),
                    ]
                )
            det_table = Table(det_rows, colWidths=[30, 130, 100, 260])
            det_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(det_table)
            if len(result.boxes) > 15:
                story.append(
                    Paragraph(
                        f"<i>Showing top 15 of {len(result.boxes)} total detections.</i>",
                        body_style,
                    )
                )
        else:
            story.append(
                Paragraph(
                    "No bounding box detections identified at the applied confidence threshold.",
                    body_style,
                )
            )

    elif result.inspection_type == InspectionType.SEGMENTATION:
        story.append(
            Paragraph("Electroluminescence Segmentation Analysis", section_style)
        )
        seg_rows: List[List[Any]] = [
            [
                Paragraph("<b>Metric</b>", body_style),
                Paragraph("<b>Value / Details</b>", body_style),
            ],
            [
                Paragraph("Defect Area Ratio", body_style),
                Paragraph(
                    f"{(result.defect_ratio or 0.0) * 100:.4f}% of cell", body_style
                ),
            ],
            [
                Paragraph("Decision Threshold", body_style),
                Paragraph(f"{result.threshold} (Validated Optimal)", body_style),
            ],
            [
                Paragraph("Validation Benchmark Dice (F1)", body_style),
                Paragraph("73.54% (MSPS Official Test Split)", body_style),
            ],
            [
                Paragraph("Validation Benchmark IoU (Jaccard)", body_style),
                Paragraph("58.16% (MSPS Official Test Split)", body_style),
            ],
        ]
        seg_table = Table(seg_rows, colWidths=[200, 320])
        seg_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(seg_table)

    # Disclaimer Section (Mandatory academic disclosure)
    story.append(Spacer(1, 14))
    disclaimer_block = [
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#CBD5E1"),
            spaceBefore=4,
            spaceAfter=6,
        ),
        Paragraph(
            "<b>NOTICE & ACADEMIC DISCLAIMER:</b> This automated inspection report was generated by SolarVision AI, "
            "an academic research prototype. Automated deep learning predictions (classification, detection bounding boxes, "
            "and segmentation masks) are indicative screening aids only. They must not be relied upon as the sole basis for "
            "commercial maintenance, insurance claims, or module warranty decisions without secondary on-site verification "
            "by a certified photovoltaic engineer or qualified solar inspection technician.",
            disclaimer_style,
        ),
    ]
    story.append(KeepTogether(disclaimer_block))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def generate_inspection_pdf(
    result: InferenceResult,
    output_path: Union[str, Path],
    inspection_id: Optional[int] = None,
    image_name: Optional[str] = None,
    original_image: Optional[PILImage.Image] = None,
    notes: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """Generate PDF inspection report and write to file.

    Args:
        result: The InferenceResult to generate the report for.
        output_path: Target path for the PDF file.
        inspection_id: Optional database record primary key ID.
        image_name: Display filename of the inspected image.
        original_image: Optional PIL Image of original input.
        notes: Optional operator inspection notes.
        timestamp: Optional ISO timestamp.

    Returns:
        Path: Path to the generated PDF file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = generate_inspection_pdf_bytes(
        result=result,
        inspection_id=inspection_id,
        image_name=image_name,
        original_image=original_image,
        notes=notes,
        timestamp=timestamp,
    )
    with open(out_file, "wb") as f:
        f.write(pdf_bytes)
    return out_file
