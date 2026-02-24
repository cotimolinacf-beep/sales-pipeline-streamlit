"""
Plotly chart builders for the dashboard visualizations.
"""
from __future__ import annotations

import plotly.graph_objects as go
from models import PipelineResult, get_stages_for_type


# Color palette
GREEN = "#22c55e"
YELLOW = "#eab308"
RED = "#ef4444"
PRIMARY = "#6366f1"
PRIMARY_LIGHT = "#a5b4fc"
MUTED = "#94a3b8"
BG_CARD = "#ffffff"


def stage_distribution_chart(
    stage_counts: dict[str, int],
    pipeline_type: str,
    total: int,
) -> go.Figure:
    """Horizontal bar chart showing how many conversations reached each stage as their max."""
    if not stage_counts:
        fig = go.Figure()
        fig.update_layout(
            title="Maximo lead alcanzado",
            annotations=[{"text": "No hay datos", "showarrow": False, "font": {"size": 16}}],
        )
        return fig

    stages_order = get_stages_for_type(pipeline_type)
    # Order stages, include any extra stages not in the predefined order
    ordered = [s for s in stages_order if s in stage_counts]
    for s in stage_counts:
        if s not in ordered:
            ordered.append(s)

    labels = list(reversed(ordered))  # Reverse so highest stage is at top
    values = [stage_counts.get(s, 0) for s in labels]
    pcts = [(v / total * 100) if total > 0 else 0 for v in values]

    # Color gradient: later stages get greener
    n = len(labels)
    colors = []
    for i, label in enumerate(labels):
        # i=0 is the highest stage (top), i=n-1 is the lowest
        idx_in_order = ordered.index(label) if label in ordered else 0
        ratio = idx_in_order / max(len(ordered) - 1, 1)
        if ratio >= 0.7:
            colors.append(GREEN)
        elif ratio >= 0.4:
            colors.append(YELLOW)
        else:
            colors.append("#fb923c")  # orange for early stages

    hover_texts = [
        f"<b>{s}</b><br>Conversaciones: {v:,}<br>Porcentaje: {p:.1f}%"
        for s, v, p in zip(labels, values, pcts)
    ]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=colors,
            text=[f"{v} ({p:.1f}%)" for v, p in zip(values, pcts)],
            textposition="outside",
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title={"text": "Maximo lead alcanzado por conversacion", "font": {"size": 16}},
        xaxis_title="Conversaciones",
        height=max(250, len(labels) * 50 + 80),
        margin=dict(l=20, r=80, t=50, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def abandonment_chart(pipeline: PipelineResult) -> go.Figure | None:
    """Bar chart for abandonment reasons."""
    if not pipeline.abandonment_analysis:
        return None
    reasons = pipeline.abandonment_analysis.top_abandonment_reasons
    if not reasons:
        return None

    labels = [r.reason[:60] + "..." if len(r.reason) > 60 else r.reason for r in reasons]
    values = [r.occurrences for r in reasons]
    pcts = [r.percentage for r in reasons]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color="#fb923c",
            text=[f"{v} ({p:.1f}%)" for v, p in zip(values, pcts)],
            textposition="outside",
        )
    )
    fig.update_layout(
        title={
            "text": f"Razones de abandono - {pipeline.pipeline_name}",
            "font": {"size": 14},
        },
        xaxis_title="Ocurrencias",
        height=max(200, len(reasons) * 50 + 80),
        margin=dict(l=20, r=100, t=40, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
