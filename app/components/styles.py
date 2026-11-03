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
        "input_bg": "#131B2E",
        "input_border": "#334155",
        "input_text": "#F8FAFC",
        "input_placeholder": "#64748B",
        "button_secondary_bg": "#1E293B",
        "button_secondary_border": "#334155",
        "button_secondary_text": "#F8FAFC",
        "table_header_bg": "#131B2E",
        "table_header_text": "#F8FAFC",
        "table_row_bg": "#1E293B",
        "table_row_alt": "#172033",
        "table_text": "#F8FAFC",
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
        "border_color": "#CBD5E1",
        "border_subtle": "#E2E8F0",
        "text_primary": "#0F172A",
        "text_secondary": "#334155",
        "text_muted": "#64748B",
        "accent_blue": "#2563EB",
        "accent_teal": "#0891B2",
        "input_bg": "#FFFFFF",
        "input_border": "#CBD5E1",
        "input_text": "#0F172A",
        "input_placeholder": "#94A3B8",
        "button_secondary_bg": "#FFFFFF",
        "button_secondary_border": "#CBD5E1",
        "button_secondary_text": "#0F172A",
        "table_header_bg": "#F1F5F9",
        "table_header_text": "#0F172A",
        "table_row_bg": "#FFFFFF",
        "table_row_alt": "#F8FAFC",
        "table_text": "#0F172A",
        "badge_clean_bg": "#DCFCE7",
        "badge_clean_text": "#15803D",
        "badge_defect_bg": "#FEE2E2",
        "badge_defect_text": "#B91C1C",
        "badge_warning_bg": "#FEF3C7",
        "badge_warning_text": "#B45309",
        "badge_info_bg": "#DBEAFE",
        "badge_info_text": "#1D4ED8",
        "card_shadow": "0 1px 4px rgba(0, 0, 0, 0.08)",
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
    :root, .stApp {{
        --sv-bg-main: {p["bg_main"]};
        --sv-bg-surface: {p["bg_surface"]};
        --sv-bg-card: {p["bg_card"]};
        --sv-bg-card-hover: {p["bg_card_hover"]};
        --sv-border: {p["border_color"]};
        --sv-border-subtle: {p["border_subtle"]};
        --sv-text-primary: {p["text_primary"]};
        --sv-text-secondary: {p["text_secondary"]};
        --sv-text-muted: {p["text_muted"]};
        --sv-accent: {p["accent_blue"]};
        --sv-accent-teal: {p["accent_teal"]};
        --text-color: {p["text_primary"]};
        --background-color: {p["bg_main"]};
        --secondary-background-color: {p["bg_surface"]};
        --primary-color: {p["accent_blue"]};
    }}

    /* Global layout container */
    .stApp {{
        background-color: {p["bg_main"]} !important;
        color: {p["text_primary"]} !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }}

    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }}

    /* Typography & Core Text */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: {p["text_primary"]} !important;
        font-weight: 700;
        letter-spacing: -0.01em;
    }}

    .stApp p, .stApp .stMarkdown p {{
        color: {p["text_primary"]} !important;
        line-height: 1.6;
    }}

    .stApp strong, .stApp .stMarkdown strong {{
        color: {p["text_primary"]} !important;
    }}

    .stApp em, .stApp .stMarkdown em {{
        color: {p["text_secondary"]};
    }}

    .stApp div[data-testid="stCaptionContainer"] p,
    .stApp .stCaption,
    .stApp small {{
        color: {p["text_secondary"]} !important;
    }}

    .stApp hr {{
        border-color: {p["border_color"]} !important;
        margin: 1.25rem 0;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {p["bg_surface"]} !important;
        border-right: 1px solid {p["border_color"]};
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {{
        color: {p["text_primary"]} !important;
    }}

    section[data-testid="stSidebar"] p {{
        color: {p["text_primary"]} !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p,
    section[data-testid="stSidebar"] small {{
        color: {p["text_secondary"]} !important;
    }}

    section[data-testid="stSidebar"] hr {{
        border-color: {p["border_color"]} !important;
        margin: 1rem 0;
    }}

    /* Radio Controls (Navigation Menu & Theme Selector) */
    div[data-testid="stRadio"] label {{
        color: {p["text_primary"]} !important;
    }}

    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span {{
        color: {p["text_primary"]} !important;
        font-size: 0.92rem;
    }}

    div[data-testid="stRadio"] label:has(input:checked) p,
    div[data-testid="stRadio"] label:has(input:checked) span,
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] p {{
        color: {p["accent_blue"]} !important;
        font-weight: 700 !important;
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
        background-color: {p["bg_card"]};
        border: 1px solid {p["border_color"]};
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin: 0.8rem 0;
        font-size: 0.84rem;
        box-shadow: {p["card_shadow"]};
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
        border: 1px solid {p["badge_clean_text"]};
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
        border: 1px solid {p["badge_defect_text"]};
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
        border: 1px solid {p["badge_warning_text"]};
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
        border: 1px solid {p["badge_info_text"]};
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
        background-color: {p["bg_card"]};
        border: 1px solid {p["border_color"]};
        border-left: 4px solid {p["accent_blue"]};
        border-radius: 0 6px 6px 0;
        padding: 0.85rem 1.15rem;
        font-size: 0.82rem;
        color: {p["text_secondary"]};
        margin-top: 1.5rem;
        box-shadow: {p["card_shadow"]};
    }}

    .disclaimer-box strong {{
        color: {p["text_primary"]} !important;
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

    /* Expanders */
    div[data-testid="stExpander"] {{
        background-color: {p["bg_card"]} !important;
        border: 1px solid {p["border_color"]} !important;
        border-radius: 8px;
        box-shadow: {p["card_shadow"]};
        margin: 0.75rem 0;
    }}

    div[data-testid="stExpander"] details {{
        background-color: {p["bg_card"]} !important;
        border-radius: 8px;
    }}

    div[data-testid="stExpander"] summary {{
        background-color: {p["bg_card"]} !important;
        color: {p["text_primary"]} !important;
        border-radius: 8px;
        padding: 0.6rem 0.85rem;
    }}

    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span {{
        color: {p["text_primary"]} !important;
        font-weight: 600;
    }}

    div[data-testid="stExpander"] summary svg {{
        fill: {p["text_secondary"]} !important;
        color: {p["text_secondary"]} !important;
    }}

    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        background-color: {p["bg_card"]} !important;
        border-top: 1px solid {p["border_color"]};
        padding: 1rem;
    }}

    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] p {{
        color: {p["text_secondary"]} !important;
    }}

    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] strong {{
        color: {p["text_primary"]} !important;
    }}

    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] code {{
        background-color: {p["bg_surface"]} !important;
        color: {p["accent_blue"]} !important;
    }}

    /* Streamlit Metrics */
    div[data-testid="stMetric"] {{
        background-color: {p["bg_card"]} !important;
        border: 1px solid {p["border_color"]} !important;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: {p["card_shadow"]};
    }}

    div[data-testid="stMetricValue"] {{
        color: {p["text_primary"]} !important;
        font-weight: 700;
    }}

    div[data-testid="stMetricValue"] > div {{
        color: {p["text_primary"]} !important;
    }}

    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] label {{
        color: {p["text_secondary"]} !important;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    /* Form Inputs & Selectors */
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] label p,
    div[data-testid="stWidgetLabel"] label span {{
        color: {p["text_primary"]} !important;
        font-weight: 600;
        font-size: 0.88rem;
    }}

    div[data-baseweb="input"] {{
        background-color: {p["input_bg"]} !important;
        border: 1px solid {p["input_border"]} !important;
        border-radius: 6px;
    }}

    div[data-baseweb="input"] input {{
        background-color: transparent !important;
        color: {p["input_text"]} !important;
    }}

    div[data-baseweb="input"] input::placeholder {{
        color: {p["input_placeholder"]} !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: {p["input_bg"]} !important;
        border: 1px solid {p["input_border"]} !important;
        color: {p["input_text"]} !important;
        border-radius: 6px;
    }}

    div[data-baseweb="select"] span {{
        color: {p["input_text"]} !important;
    }}

    div[data-baseweb="select"] svg {{
        fill: {p["text_secondary"]} !important;
    }}

    ul[data-baseweb="menu"] {{
        background-color: {p["bg_card"]} !important;
        border: 1px solid {p["border_color"]} !important;
    }}

    li[data-baseweb="menu-item"] {{
        background-color: {p["bg_card"]} !important;
        color: {p["text_primary"]} !important;
    }}

    li[data-baseweb="menu-item"]:hover {{
        background-color: {p["bg_card_hover"]} !important;
    }}

    /* File Uploader */
    div[data-testid="stFileUploader"] section {{
        background-color: {p["bg_card"]} !important;
        border: 1px dashed {p["border_color"]} !important;
        border-radius: 8px;
    }}

    div[data-testid="stFileUploader"] section p,
    div[data-testid="stFileUploader"] section span,
    div[data-testid="stFileUploader"] section small {{
        color: {p["text_secondary"]} !important;
    }}

    /* Buttons */
    div.stButton > button,
    div.stDownloadButton > button {{
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.15s ease;
    }}

    div.stButton > button[kind="primary"],
    div.stDownloadButton > button[kind="primary"] {{
        background-color: {p["accent_blue"]} !important;
        color: #FFFFFF !important;
        border: 1px solid {p["accent_blue"]} !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
    }}

    div.stButton > button[kind="primary"]:hover,
    div.stDownloadButton > button[kind="primary"]:hover {{
        filter: brightness(1.1);
    }}

    div.stButton > button:not([kind="primary"]),
    div.stDownloadButton > button:not([kind="primary"]) {{
        background-color: {p["button_secondary_bg"]} !important;
        color: {p["button_secondary_text"]} !important;
        border: 1px solid {p["button_secondary_border"]} !important;
        box-shadow: {p["card_shadow"]};
    }}

    div.stButton > button:not([kind="primary"]):hover,
    div.stDownloadButton > button:not([kind="primary"]):hover {{
        background-color: {p["bg_card_hover"]} !important;
        border-color: {p["accent_blue"]} !important;
        color: {p["accent_blue"]} !important;
    }}

    /* Code & JSON display */
    code {{
        background-color: {p["bg_surface"]} !important;
        color: {p["accent_blue"]} !important;
        border: 1px solid {p["border_color"]};
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        font-size: 0.88em;
    }}

    div[data-testid="stCodeBlock"] {{
        background-color: {p["bg_surface"]} !important;
        border: 1px solid {p["border_color"]} !important;
        border-radius: 6px;
    }}

    div[data-testid="stCodeBlock"] pre {{
        color: {p["text_primary"]} !important;
    }}

    div[data-testid="stJson"] {{
        background-color: {p["bg_surface"]} !important;
        border: 1px solid {p["border_color"]} !important;
        border-radius: 6px;
        padding: 0.75rem;
    }}

    /* DataFrames & Tables */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {p["border_color"]} !important;
        border-radius: 6px;
    }}

    table {{
        color: {p["text_primary"]} !important;
        background-color: {p["bg_card"]} !important;
        border-collapse: collapse;
    }}

    th {{
        background-color: {p["table_header_bg"]} !important;
        color: {p["table_header_text"]} !important;
        border-bottom: 2px solid {p["border_color"]} !important;
        font-weight: 600;
    }}

    td {{
        border-bottom: 1px solid {p["border_color"]} !important;
        color: {p["table_text"]} !important;
    }}

    /* Alerts */
    div[data-testid="stAlert"] {{
        border-radius: 6px;
    }}

    div[data-testid="stAlert"] p {{
        color: {p["text_primary"]} !important;
    }}

    div[data-testid="stAlert"] strong {{
        color: {p["text_primary"]} !important;
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
