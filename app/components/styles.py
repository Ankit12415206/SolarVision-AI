"""
Module: styles
Description: Professional solar & technology UI theme and CSS styling for Streamlit
Author: SolarVision AI Team
Created: 2026-11-02
"""

import streamlit as st


def apply_custom_theme() -> None:
    """Inject custom CSS styles into Streamlit app for a sleek, modern UI."""
    css = """
    <style>
    /* Global Container */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Brand Header Banner */
    .brand-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .brand-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .brand-tagline {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }

    /* Metric Card */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-top: 0.2rem;
    }

    /* Badge Tags */
    .badge-clean {
        background-color: #14532D;
        color: #86EFAC;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    .badge-defect {
        background-color: #7F1D1D;
        color: #FCA5A5;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    .badge-warning {
        background-color: #78350F;
        color: #FCD34D;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
