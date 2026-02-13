"""
Plotly chart builders for the dashboard visualizations.
"""

import plotly.graph_objects as go
from models import EvaluationResults, PipelineResult, FunnelGoal, get_stages_for_type


# Color palette
GREEN = "#22c55e"
YELLOW = "#eab308"
RED = "#ef4444"
PRIMARY = "#6366f1"
PRIMARY_LIGHT = "#a5b4fc"
MUTED = "#94a3b8"
BG_CARD = "#ffffff"


def funnel_chart(data: EvaluationResults) -> go.Figure:
    """Global conversion funnel aggregated across all pipelines."""
    sales_order = ["Awareness", "Lead", "MQL", "SQL", "Opportunity", "Customer"]
    service_order = [
        "Ticket creado", "Ticket abierto", "Ticket en proceso", "Ticket resuelto"
    ]
    all_stage_order = sales_order + service_order

    stage_map: dict[str, dict] = {}
    for pipeline in data.pipelines:
        for goal in pipeline.funnel:
            if goal.stage not in stage_map:
                stage_map[goal.stage] = {"success": 0, "failure": 0}
            stage_map[goal.stage]["success"] += goal.success_count
            stage_map[goal.stage]["failure"] += goal.failure_count

    # Order stages
    ordered_stages = []
    for s in all_stage_order:
        if s in stage_map:
            ordered_stages.append(s)
    for s in stage_map:
        if s not in ordered_stages:
            ordered_stages.append(s)

    if not ordered_stages:
        fig = go.Figure()
        fig.update_layout(
            title="Embudo de conversion global",
            annotations=[{"text": "No hay datos", "showarrow": False, "font": {"size": 16}}],
        )
        return fig

    labels = []
    values = []
    colors = []
    hover_texts = []

    for stage in ordered_stages:
        s = stage_map[stage]["success"]
        f = stage_map[stage]["failure"]
        total = s + f
        rate = (s / total * 100) if total > 0 else 0
        labels.append(stage)
        values.append(s)
        colors.append(GREEN if rate >= 50 else RED)
        hover_texts.append(
            f"<b>{stage}</b><br>"
            f"Exitosos: {s:,}<br>"
            f"Fallidos: {f:,}<br>"
            f"Tasa: {rate:.1f}%"
        )

    fig = go.Figure(
        go.Funnel(
            y=labels,
            x=values,
            textinfo="value+percent initial",
            marker={"color": colors, "line": {"width": 1, "color": "white"}},
            hovertext=hover_texts,
            hoverinfo="text",
            connector={"line": {"color": PRIMARY_LIGHT, "width": 1}},
        )
    )
    fig.update_layout(
        title={"text": "Embudo de conversion global", "font": {"size": 18}},
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def pipeline_goals_chart(pipeline: PipelineResult) -> go.Figure:
    """Horizontal bar chart showing success/failure per objective in a pipeline."""
    goals = pipeline.funnel
    if not goals:
        fig = go.Figure()
        fig.update_layout(title=pipeline.pipeline_name)
        return fig

    names = [g.objective_name for g in goals]
    success = [g.success_count for g in goals]
    failure = [g.failure_count for g in goals]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=names,
            x=success,
            name="Exitosos",
            orientation="h",
            marker_color=GREEN,
            text=[f"{s}" for s in success],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            y=names,
            x=failure,
            name="Fallidos",
            orientation="h",
            marker_color=RED,
            text=[f"{f}" for f in failure],
            textposition="inside",
        )
    )
    fig.update_layout(
        barmode="stack",
        title={"text": f"Objetivos - {pipeline.pipeline_name}", "font": {"size": 16}},
        xaxis_title="Conversaciones",
        height=max(200, len(goals) * 60 + 80),
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def keyword_distribution_chart(goal: FunnelGoal) -> go.Figure | None:
    """Horizontal bar chart for keyword distribution of a goal."""
    if not goal.keyword_distribution:
        return None

    kw = goal.keyword_distribution[:7]
    labels = [k.value for k in kw]
    counts = [k.count for k in kw]
    pcts = [k.percentage for k in kw]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=counts,
            orientation="h",
            marker_color=PRIMARY_LIGHT,
            text=[f"{c} ({p:.1f}%)" for c, p in zip(counts, pcts)],
            textposition="outside",
        )
    )
    fig.update_layout(
        title={"text": f"Keywords - {goal.objective_name}", "font": {"size": 14}},
        xaxis_title="Ocurrencias",
        height=max(180, len(kw) * 35 + 80),
        margin=dict(l=20, r=80, t=40, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def sentiment_pie_chart(satisfied: int, neutral: int, frustrated: int) -> go.Figure:
    """Pie chart for sentiment distribution."""
    labels = ["Satisfechos", "Neutrales", "Frustrados"]
    values = [satisfied, neutral, frustrated]
    colors = [GREEN, YELLOW, RED]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker={"colors": colors},
            hole=0.4,
            textinfo="label+value+percent",
            textposition="outside",
            pull=[0.02, 0, 0.02],
        )
    )
    fig.update_layout(
        title={"text": "Distribucion de sentimiento", "font": {"size": 16}},
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def sentiment_bar_chart(satisfied: int, neutral: int, frustrated: int) -> go.Figure:
    """Stacked horizontal bar showing sentiment proportions."""
    total = satisfied + neutral + frustrated
    if total == 0:
        total = 1

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["Sentimiento"],
            x=[satisfied / total * 100],
            name=f"Satisfechos ({satisfied})",
            orientation="h",
            marker_color=GREEN,
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Sentimiento"],
            x=[neutral / total * 100],
            name=f"Neutrales ({neutral})",
            orientation="h",
            marker_color=YELLOW,
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Sentimiento"],
            x=[frustrated / total * 100],
            name=f"Frustrados ({frustrated})",
            orientation="h",
            marker_color=RED,
        )
    )
    fig.update_layout(
        barmode="stack",
        height=120,
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis={"range": [0, 100], "showticklabels": False},
        yaxis={"showticklabels": False},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
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
