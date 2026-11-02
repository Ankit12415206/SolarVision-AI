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
    "clean": "#22C55E",  # Green
    "dusty": "#F59E0B",  # Amber
    "damage": "#EF4444",  # Red
    "accent": "#0EA5E9",  # Sky blue
    "bg_dark": "#1E293B",  # Slate
    "text_light": "#F8FAFC",  # White
}


def plot_class_probabilities(probs: Dict[str, float]) -> go.Figure:
    """Create horizontal bar chart of predicted class probabilities.

    Args:
        probs: Mapping of defect classes to softmax probabilities.

    Returns:
        go.Figure: Interactive Plotly horizontal bar chart.
    """
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
            marker=dict(color=colors, line=dict(color="#334155", width=1)),
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(
            text="Class Probability Distribution",
            font=dict(color=PALETTE["text_light"]),
        ),
        xaxis=dict(
            title="Probability (%)",
            range=[0, 100],
            showgrid=True,
            gridcolor="#334155",
            color=PALETTE["text_light"],
        ),
        yaxis=dict(color=PALETTE["text_light"]),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#1E293B",
        margin=dict(l=10, r=20, t=40, b=20),
        height=260,
    )
    return fig


def plot_defect_distribution(analytics: Dict[str, Any]) -> go.Figure:
    """Create donut chart showing Clean vs Defective panel ratios.

    Args:
        analytics: Analytics dictionary from InspectionDatabaseManager.

    Returns:
        go.Figure: Donut chart figure.
    """
    clean = analytics.get("clean_count", 0)
    defect = analytics.get("defect_count", 0)

    if clean == 0 and defect == 0:
        clean = 1  # Placeholder empty state

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Healthy / Clean", "Defect / Soiled"],
                values=[clean, defect],
                hole=0.6,
                marker=dict(colors=[PALETTE["clean"], PALETTE["damage"]]),
                textinfo="label+percent",
            )
        ]
    )

    fig.update_layout(
        title=dict(text="Panel Health Ratio", font=dict(color=PALETTE["text_light"])),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#1E293B",
        legend=dict(font=dict(color=PALETTE["text_light"])),
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
    )
    return fig


def plot_modality_breakdown(analytics: Dict[str, Any]) -> go.Figure:
    """Create bar chart of inspections categorized by modality.

    Args:
        analytics: Analytics dictionary.

    Returns:
        go.Figure: Modality breakdown bar chart.
    """
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
            marker=dict(color=[PALETTE["accent"], PALETTE["dusty"], "#8B5CF6"]),
            text=counts,
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(
            text="Inspections by Modality", font=dict(color=PALETTE["text_light"])
        ),
        xaxis=dict(color=PALETTE["text_light"]),
        yaxis=dict(
            title="Inspections", color=PALETTE["text_light"], gridcolor="#334155"
        ),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#1E293B",
        margin=dict(l=10, r=10, t=40, b=20),
        height=280,
    )
    return fig


def plot_daily_trend(analytics: Dict[str, Any]) -> go.Figure:
    """Create line chart showing daily inspection volume trend.

    Args:
        analytics: Analytics dictionary with daily_trend list.

    Returns:
        go.Figure: Daily trend line chart.
    """
    daily = analytics.get("daily_trend", [])
    dates = [d["date"] for d in daily] or ["Today"]
    counts = [d["count"] for d in daily] or [0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=counts,
            mode="lines+markers",
            line=dict(color=PALETTE["accent"], width=3),
            marker=dict(size=8, color=PALETTE["clean"]),
            fill="tozeroy",
            fillcolor="rgba(14, 165, 233, 0.15)",
        )
    )

    fig.update_layout(
        title=dict(
            text="Inspection Volume Over Time", font=dict(color=PALETTE["text_light"])
        ),
        xaxis=dict(title="Date", color=PALETTE["text_light"]),
        yaxis=dict(title="Count", color=PALETTE["text_light"], gridcolor="#334155"),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#1E293B",
        margin=dict(l=10, r=10, t=40, b=20),
        height=280,
    )
    return fig
