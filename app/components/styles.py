"""
Module: styles
Description: Professional enterprise SaaS UI design system and themes for SolarVision AI
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Dict
import streamlit as st

# Enterprise Color Palettes
THEME_PALETTES: Dict[str, Dict[str, str]] = {
    "dark": {
        "bg_main": "#0B0F19",
        "bg_surface": "#131B2E",
        "bg_card": "#1E293B",
        "bg_card_hover": "#243248",
        "border_color": "#334155",
        "border_subtle": "#1E293B",
        "text_primary": "#F8FAFC",
        "text_secondary": "#94A3B8",
        "text_muted": "#64748B",
        "accent_blue": "#3B82F6",
        "accent_teal": "#06B6D4",
        "badge_clean_bg": "#14532D",
        "badge_clean_text": "#86EFAC",
        "badge_defect_bg": "#7F1D1D",
        "badge_defect_text": "#FCA5A5",
        "badge_warning_bg": "#78350F",
        "badge_warning_text": "#FCD34D",
        "badge_info_bg": "#1E3A8A",
        "badge_info_text": "#93C5FD",
        "card_shadow": "0 2px 8px rgba(0, 0, 0, 0.4)",
    },
    "light": {
        "bg_main": "#F8FAFC",
        "bg_surface": "#FFFFFF",
        "bg_card": "#FFFFFF",
        "bg_card_hover": "#F1F5F9",
        "border_color": "#E2E8F0",
        "border_subtle": "#F1F5F9",
        "text_primary": "#0F172A",
        "text_secondary": "#475569",
        "text_muted": "#94A3B8",
        "accent_blue": "#2563EB",
        "accent_teal": "#0891B2",
        "badge_clean_bg": "#DCFCE7",
        "badge_clean_text": "#15803D",
        "badge_defect_bg": "#FEE2E2",
        "badge_defect_text": "#B91C1C",
        "badge_warning_bg": "#FEF3C7",
        "badge_warning_text": "#B45309",
        "badge_info_bg": "#DBEAFE",
        "badge_info_text": "#1D4ED8",
        "card_shadow": "0 2px 6px rgba(0, 0, 0, 0.06)",
    },
}


def apply_custom_theme(theme: str = "dark") -> None:
    """Inject custom CSS styles into Streamlit app for enterprise SaaS design.

    Args:
        theme: Theme name, either 'dark' or 'light'. Defaults to 'dark'.
    """
    mode = "light" if theme.lower() == "light" else "dark"
    p = THEME_PALETTES[mode]

    css = f"""
    <style>
    /* CSS Variables */
    :root {{
        --sv-bg-main: {p["bg_main"]};
        --sv-bg-surface: {p["bg_surface"]};
        --sv-bg-card: {p["bg_card"]};
        --sv-border: {p["border_color"]};
        --sv-text-primary: {p["text_primary"]};
        --sv-text-secondary: {p["text_secondary"]};
        --sv-accent: {p["accent_blue"]};
    }}

    /* Global layout container */
    .stApp {{
        background-color: {p["bg_main"]};
        color: {p["text_primary"]};
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }}

    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {p["bg_surface"]};
        border-right: 1px solid {p["border_color"]};
    }}

    section[data-testid="stSidebar"] hr {{
        margin: 1rem 0;
        border-color: {p["border_color"]};
    }}

    /* Top Brand Bar */
    .brand-header {{
        background-color: {p["bg_card"]};
        border: 1px solid {p["border_color"]};
        border-radius: 10px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: {p["card_shadow"]};
    }}

    .brand-header-left {{
        display: flex;
        flex-direction: column;
    }}

    .brand-title {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {p["text_primary"]};
        margin: 0;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .brand-tagline {{
        font-size: 0.88rem;
        color: {p["text_secondary"]};
        margin-top: 0.25rem;
        margin-bottom: 0;
    }}

    /* SaaS KPI Card */
    .kpi-card {{
        background-color: {p["bg_card"]};
        border: 1px solid {p["border_color"]};
        border-radius: 8px;
        padding: 1.1rem;
        text-align: left;
        box-shadow: {p["card_shadow"]};
        height: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}

    .kpi-card:hover {{
        background-color: {p["bg_card_hover"]};
        border-color: {p["accent_blue"]};
    }}

    .kpi-label {{
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {p["text_secondary"]};
        margin: 0;
    }}

    .kpi-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: {p["text_primary"]};
        line-height: 1.2;
        margin: 0.2rem 0;
    }}

    .kpi-delta {{
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 0.35rem;
    }}

    /* Workflow Panel Card */
    .panel-card {{
        background-color: {p["bg_card"]};
        border: 1px solid {p["border_color"]};
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: {p["card_shadow"]};
    }}

    .panel-header {{
        font-size: 1.05rem;
        font-weight: 600;
        color: {p["text_primary"]};
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid {p["border_color"]};
        padding-bottom: 0.5rem;
    }}

    /* Model Transparency Card */
    .model-info-box {{
        background-color: {p["bg_surface"]};
        border: 1px solid {p["border_color"]};
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin: 0.8rem 0;
        font-size: 0.84rem;
        color: {p["text_secondary"]};
    }}

    .model-info-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 1.25rem;
        align-items: center;
    }}

    .model-info-item {{
        display: flex;
        flex-direction: column;
    }}

    .model-info-key {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {p["text_muted"]};
        font-weight: 600;
    }}

    .model-info-val {{
        font-size: 0.9rem;
        color: {p["text_primary"]};
        font-weight: 600;
    }}

    /* Status Badges */
    .badge-clean {{
        background-color: {p["badge_clean_bg"]};
        color: {p["badge_clean_text"]};
        padding: 0.2rem 0.65rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.02em;
    }}

    .badge-defect {{
        background-color: {p["badge_defect_bg"]};
        color: {p["badge_defect_text"]};
        padding: 0.2rem 0.65rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.02em;
    }}

    .badge-warning {{
        background-color: {p["badge_warning_bg"]};
        color: {p["badge_warning_text"]};
        padding: 0.2rem 0.65rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.02em;
    }}

    .badge-info {{
        background-color: {p["badge_info_bg"]};
        color: {p["badge_info_text"]};
        padding: 0.2rem 0.65rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.02em;
    }}

    .badge-demo {{
        background-color: {p["badge_warning_bg"]};
        color: {p["badge_warning_text"]};
        border: 1px solid {p["badge_warning_text"]};
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }}

    /* Disclaimer box */
    .disclaimer-box {{
        background-color: {p["bg_surface"]};
        border-left: 3px solid {p["accent_blue"]};
        border-radius: 0 6px 6px 0;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: {p["text_secondary"]};
        margin-top: 1.5rem;
    }}

    /* Empty state card */
    .empty-state {{
        background-color: {p["bg_card"]};
        border: 1px dashed {p["border_color"]};
        border-radius: 8px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }}

    .empty-title {{
        font-size: 1.15rem;
        font-weight: 600;
        color: {p["text_primary"]};
        margin-bottom: 0.5rem;
    }}

    .empty-desc {{
        font-size: 0.9rem;
        color: {p["text_secondary"]};
        max-width: 500px;
        margin: 0 auto 1.25rem auto;
    }}

    /* Streamlit native widget contrast tuning */
    div[data-testid="stMetricValue"] {{
        color: {p["text_primary"]};
        font-weight: 700;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {p["text_secondary"]};
        font-size: 0.85rem;
    }}

    /* Button refinement */
    div.stButton > button {{
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.15s ease;
    }}

    div.stDownloadButton > button {{
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: str,
    delta: str = "",
    delta_color: str = "normal",
) -> None:
    """Render a styled KPI metric card in HTML.

    Args:
        label: Metric title.
        value: Main numeric value.
        delta: Optional secondary change or subtitle.
        delta_color: Text color hint: 'normal', 'positive', 'negative'.
    """
    color_map = {
        "normal": "inherit",
        "positive": "#10B981",
        "negative": "#EF4444",
        "warning": "#F59E0B",
    }
    d_color = color_map.get(delta_color, "inherit")

    delta_html = ""
    if delta:
        delta_html = (
            f'<div class="kpi-delta" style="color: {d_color};">{delta}</div>'
        )

    html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_disclaimer_notice() -> None:
    """Render authoritative Academic Prototype & screening disclaimer."""
    html = """
    <div class="disclaimer-box">
        <strong>Academic Prototype:</strong> SolarVision AI provides AI-assisted
        screening of solar panel conditions and potential defects. Predictions
        are experimental and may contain errors. Results should be verified by
        qualified personnel before maintenance or operational decisions.
        <br/><br/>
        <em>Note on Metrics & Terminology:</em> Classification confidence scores
        reflect model certainty over categorical classes, not absolute correctness.
        Electroluminescence (EL) anomaly predictions identify silicon wafer discontinuities
        (such as microcracks, cell fractures, and busbar interruptions) as documented in
        the training annotations. Validation/test benchmark metrics reflect held-out
        offline evaluation and do not guarantee equivalent single-sample live accuracy.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_model_transparency(
    model_name: str,
    modality: str,
    latency_ms: float,
    resolution: str,
    metric_info: str = "",
) -> None:
    """Render a compact model transparency and specification strip.

    Args:
        model_name: Architecture name.
        modality: Vision inspection modality.
        latency_ms: CPU processing time in milliseconds.
        resolution: Image input resolution (WxH).
        metric_info: Additional threshold or confidence descriptor.
    """
    html = f"""
    <div class="model-info-box">
        <div class="model-info-row">
            <div class="model-info-item">
                <span class="model-info-key">Active Model</span>
                <span class="model-info-val">{model_name}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-key">Modality</span>
                <span class="model-info-val">{modality}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-key">CPU Latency</span>
                <span class="model-info-val">{latency_ms:.1f} ms</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-key">Resolution</span>
                <span class="model-info-val">{resolution}</span>
            </div>
            {f'<div class="model-info-item"><span class="model-info-key">Sensitivity / Threshold</span><span class="model-info-val">{metric_info}</span></div>' if metric_info else ''}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
