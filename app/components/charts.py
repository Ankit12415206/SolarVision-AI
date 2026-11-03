"""
Module: charts
Description: Plotly interactive charting components for SolarVision AI Streamlit Dashboard
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Any, Dict

import plotly.graph_objects as go

# Brand color constants
PALETTE = {
    "clean": "#10B981",  # Emerald Green
    "dusty": "#F59E0B",  # Amber
    "damage": "#EF4444",  # Crimson Red
    "accent": "#3B82F6",  # Royal Blue
    "teal": "#06B6D4",  # Cyan/Teal
    "purple": "#8B5CF6",  # Violet
}


def _get_theme_styles(theme: str) -> Dict[str, str]:
    """Retrieve theme-specific background and font colors.

    Args:
        theme: Either 'dark' or 'light'.

    Returns:
        Dict with color configuration.
    """
    if theme.lower() == "light":
        return {
            "template": "plotly_white",
            "paper_bgcolor": "#FFFFFF",
            "plot_bgcolor": "#F8FAFC",
            "font_color": "#0F172A",
            "font_secondary": "#334155",
            "grid_color": "#E2E8F0",
            "border_color": "#CBD5E1",
        }
    return {
        "template": "plotly_dark",
        "paper_bgcolor": "#1E293B",
        "plot_bgcolor": "#0F172A",
        "font_color": "#F8FAFC",
        "font_secondary": "#94A3B8",
        "grid_color": "#334155",
        "border_color": "#475569",
    }


def plot_class_probabilities(
    probs: Dict[str, float], theme: str = "dark"
) -> go.Figure:
    """Create horizontal bar chart of predicted class probabilities.

    Args:
        probs: Mapping of defect classes to softmax probabilities.
        theme: 'dark' or 'light'. Defaults to 'dark'.

    Returns:
        go.Figure: Interactive Plotly horizontal bar chart.
    """
    ts = _get_theme_styles(theme)
    sorted_items = sorted(probs.items(), key=lambda x: x[1])
    labels = [k for k, _ in sorted_items]
    values = [v * 100.0 for _, v in sorted_items]

    # Color clean green, damages red/amber
    colors = [
        (
            PALETTE["clean"]
            if lbl.lower() == "clean"
            else (PALETTE["dusty"] if "dust" in lbl.lower() else PALETTE["damage"])
        )
        for lbl in labels
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(color=ts["border_color"], width=1)),
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
            insidetextfont=dict(color="#FFFFFF", size=11),
            outsidetextfont=dict(color=ts["font_color"], size=11),
        )
    )

    fig.update_layout(
        template=ts["template"],
        font=dict(
            color=ts["font_color"],
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        ),
        title=dict(
            text="Class Probability Distribution",
            font=dict(color=ts["font_color"], size=13),
        ),
        xaxis=dict(
            title=dict(text="Probability (%)", font=dict(color=ts["font_color"])),
            range=[0, 100],
            showgrid=True,
            gridcolor=ts["grid_color"],
            color=ts["font_color"],
            tickfont=dict(color=ts["font_color"]),
        ),
        yaxis=dict(
            color=ts["font_color"],
            tickfont=dict(color=ts["font_color"]),
        ),
        plot_bgcolor=ts["plot_bgcolor"],
        paper_bgcolor=ts["paper_bgcolor"],
        margin=dict(l=10, r=20, t=35, b=20),
        height=260,
    )
    return fig


def plot_defect_distribution(
    analytics: Dict[str, Any], theme: str = "dark"
) -> go.Figure:
    """Create donut chart showing Clean vs Defective panel ratios.

    Args:
        analytics: Analytics dictionary from InspectionDatabaseManager.
        theme: 'dark' or 'light'. Defaults to 'dark'.

    Returns:
        go.Figure: Donut chart figure.
    """
    ts = _get_theme_styles(theme)
    clean = analytics.get("clean_count", 0)
    defect = analytics.get("defect_count", 0)

    if clean == 0 and defect == 0:
        clean = 1  # Placeholder empty state

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Healthy / Clean", "Defect / Soiled"],
                values=[clean, defect],
                hole=0.62,
                marker=dict(
                    colors=[PALETTE["clean"], PALETTE["damage"]],
                    line=dict(color=ts["border_color"], width=1),
                ),
                textinfo="label+percent",
                insidetextfont=dict(color="#FFFFFF", size=11),
            )
        ]
    )

    fig.update_layout(
        template=ts["template"],
        font=dict(
            color=ts["font_color"],
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        ),
        title=dict(
            text="Panel Health Ratio",
            font=dict(color=ts["font_color"], size=13),
        ),
        plot_bgcolor=ts["plot_bgcolor"],
        paper_bgcolor=ts["paper_bgcolor"],
        legend=dict(font=dict(color=ts["font_color"])),
        margin=dict(l=20, r=20, t=35, b=20),
        height=270,
    )
    return fig


def plot_modality_breakdown(
    analytics: Dict[str, Any], theme: str = "dark"
) -> go.Figure:
    """Create bar chart of inspections categorized by modality.

    Args:
        analytics: Analytics dictionary.
        theme: 'dark' or 'light'. Defaults to 'dark'.

    Returns:
        go.Figure: Modality breakdown bar chart.
    """
    ts = _get_theme_styles(theme)
    by_type = analytics.get("inspections_by_type", {})
    categories = ["Classification", "Detection", "Segmentation"]
    counts = [
        by_type.get("classification", 0),
        by_type.get("detection", 0),
        by_type.get("segmentation", 0),
    ]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=counts,
            marker=dict(
                color=[PALETTE["accent"], PALETTE["dusty"], PALETTE["purple"]],
                line=dict(color=ts["border_color"], width=1),
            ),
            text=counts,
            textposition="auto",
            insidetextfont=dict(color="#FFFFFF", size=11),
            outsidetextfont=dict(color=ts["font_color"], size=11),
        )
    )

    fig.update_layout(
        template=ts["template"],
        font=dict(
            color=ts["font_color"],
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        ),
        title=dict(
            text="Inspections by Modality",
            font=dict(color=ts["font_color"], size=13),
        ),
        xaxis=dict(
            color=ts["font_color"],
            tickfont=dict(color=ts["font_color"]),
        ),
        yaxis=dict(
            title=dict(text="Inspections", font=dict(color=ts["font_color"])),
            color=ts["font_color"],
            gridcolor=ts["grid_color"],
            tickfont=dict(color=ts["font_color"]),
        ),
        plot_bgcolor=ts["plot_bgcolor"],
        paper_bgcolor=ts["paper_bgcolor"],
        margin=dict(l=10, r=10, t=35, b=20),
        height=270,
    )
    return fig


def plot_daily_trend(analytics: Dict[str, Any], theme: str = "dark") -> go.Figure:
    """Create line chart showing daily inspection volume trend.

    Args:
        analytics: Analytics dictionary with daily_trend list.
        theme: 'dark' or 'light'. Defaults to 'dark'.

    Returns:
        go.Figure: Daily trend line chart.
    """
    ts = _get_theme_styles(theme)
    daily = analytics.get("daily_trend", [])
    dates = [d["date"] for d in daily] or ["Today"]
    counts = [d["count"] for d in daily] or [0]

    fill_rgba = (
        "rgba(37, 99, 235, 0.12)"
        if theme.lower() == "light"
        else "rgba(59, 130, 246, 0.18)"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=counts,
            mode="lines+markers",
            line=dict(color=PALETTE["accent"], width=2.5),
            marker=dict(size=7, color=PALETTE["clean"]),
            fill="tozeroy",
            fillcolor=fill_rgba,
        )
    )

    fig.update_layout(
        template=ts["template"],
        font=dict(
            color=ts["font_color"],
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        ),
        title=dict(
            text="Inspection Volume Trend",
            font=dict(color=ts["font_color"], size=13),
        ),
        xaxis=dict(
            title=dict(text="Date", font=dict(color=ts["font_color"])),
            color=ts["font_color"],
            tickfont=dict(color=ts["font_color"]),
        ),
        yaxis=dict(
            title=dict(text="Inspections", font=dict(color=ts["font_color"])),
            color=ts["font_color"],
            gridcolor=ts["grid_color"],
            tickfont=dict(color=ts["font_color"]),
        ),
        plot_bgcolor=ts["plot_bgcolor"],
        paper_bgcolor=ts["paper_bgcolor"],
        margin=dict(l=10, r=10, t=35, b=20),
        height=270,
    )
    return fig
