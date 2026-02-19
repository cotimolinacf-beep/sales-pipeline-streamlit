"""
CSV loading, preprocessing, and result aggregation utilities.
"""

import pandas as pd
import re
from collections import Counter
from models import (
    Pipeline,
    Objective,
    FunnelGoal,
    KeywordDistribution,
    PipelineResult,
    AbandonmentAnalysis,
    AbandonmentReason,
    SALES_STAGES,
    SERVICE_STAGES,
    get_stages_for_type,
)


REQUIRED_COLUMNS = ["historial", "tipificaciones"]

# Aliases for flexible column matching (lowercase)
COLUMN_ALIASES: dict[str, list[str]] = {
    "historial": [
        "historial", "historial_bot", "historial_de_mensajes_en_bot",
        "historial_mensajes", "historial_de_bot", "conversacion",
        "mensajes", "chat", "transcript", "messages",
    ],
    "tipificaciones": [
        "tipificaciones", "tipificacion", "tipificación", "tipificacón",
        "tipo", "tipos", "clasificacion", "clasificaciones",
        "tags", "etiquetas", "categorias", "categoria",
    ],
    "etapas": [
        "etapas", "etapa", "stage", "stages", "fase", "fases",
        "estado", "estados", "step", "steps", "nivel",
    ],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Match CSV columns to expected names using aliases."""
    col_lower_map = {c.lower().strip(): c for c in df.columns}

    for target, aliases in COLUMN_ALIASES.items():
        if target in df.columns:
            continue  # Already has the exact name
        matched = False
        for alias in aliases:
            if alias in col_lower_map:
                df = df.rename(columns={col_lower_map[alias]: target})
                matched = True
                break
        if not matched:
            # Try partial/substring match as last resort
            for col_lower, col_original in col_lower_map.items():
                if alias_matches_partial(col_lower, target):
                    df = df.rename(columns={col_original: target})
                    matched = True
                    break
    return df


def alias_matches_partial(col_name: str, target: str) -> bool:
    """Check if a column name partially matches a target."""
    # "historial_de_mensajes_en_bot" contains "historial"
    # "tipificacion_agente" contains "tipificacion"
    key_fragments = {
        "historial": ["historial", "mensajes_bot", "chat_bot", "transcript"],
        "tipificaciones": ["tipific", "clasific", "categori"],
        "etapas": ["etapa", "stage", "fase"],
    }
    for fragment in key_fragments.get(target, []):
        if fragment in col_name:
            return True
    return False


def load_csv(file) -> pd.DataFrame:
    """Load CSV, auto-match columns by aliases, and validate."""
    df = pd.read_csv(file, dtype=str).fillna("")
    df = _normalize_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        available = ", ".join(df.columns.tolist())
        raise ValueError(
            f"No se encontraron columnas equivalentes a: {', '.join(missing)}.\n"
            f"Columnas disponibles en el CSV: {available}\n\n"
            f"Nombres aceptados:\n"
            f"  historial: historial, historial_de_mensajes_en_bot, conversacion, mensajes, chat...\n"
            f"  tipificaciones: tipificacion, tipificaciones, clasificacion, tags, etiquetas...\n"
            f"  etapas: etapa, etapas, stage, fase, estado..."
        )
    return df


def validate_sample_diversity(df: pd.DataFrame, min_groups: int = 2) -> dict:
    """Check that a conversation sample has diverse tipificaciones/etapas."""
    # Count unique tipificaciones (split by comma, semicolon, pipe, or treat whole value)
    unique_tipificaciones = set()
    raw_unique_tipificaciones = set()
    for val in df["tipificaciones"].dropna().unique():
        val = str(val).strip()
        if not val:
            continue
        raw_unique_tipificaciones.add(val)
        # Try splitting by common separators
        for sep in [",", ";", "|"]:
            if sep in val:
                for t in val.split(sep):
                    t = t.strip()
                    if t:
                        unique_tipificaciones.add(t)
                break
        else:
            unique_tipificaciones.add(val)

    unique_etapas = set()
    if "etapas" in df.columns:
        unique_etapas = set(df["etapas"].dropna().unique()) - {""}

    # Use the larger count: split values or raw unique values
    tipif_count = max(len(unique_tipificaciones), len(raw_unique_tipificaciones))
    is_diverse = tipif_count >= min_groups or len(unique_etapas) >= min_groups

    return {
        "is_diverse": is_diverse,
        "unique_tipificaciones": sorted(unique_tipificaciones),
        "unique_etapas": sorted(unique_etapas),
        "total_conversations": len(df),
        "tipificaciones_count": tipif_count,
        "etapas_count": len(unique_etapas),
    }


def get_sample_for_auto_detect(df: pd.DataFrame, max_samples: int = 20) -> pd.DataFrame:
    """Get a diverse sample of conversations for pipeline auto-detection."""
    if len(df) <= max_samples:
        return df

    # Stratified sampling by tipificaciones (or etapas if available)
    group_col = "etapas" if "etapas" in df.columns else "tipificaciones"
    group_values = df[group_col].fillna("sin_grupo")
    groups = df.groupby(group_values)

    samples_per_group = max(1, max_samples // len(groups))
    sampled = groups.apply(
        lambda x: x.sample(n=min(len(x), samples_per_group), random_state=42)
    )
    sampled = sampled.reset_index(drop=True)

    if len(sampled) < max_samples:
        remaining = df[~df.index.isin(sampled.index)]
        extra = remaining.sample(
            n=min(len(remaining), max_samples - len(sampled)), random_state=42
        )
        sampled = pd.concat([sampled, extra]).reset_index(drop=True)

    return sampled.head(max_samples)


def conversations_to_prompt_text(df: pd.DataFrame) -> str:
    """Format conversations for LLM prompts."""
    lines = []
    for i, row in df.iterrows():
        historial = row.get("historial", "")[:1500]  # Truncate long conversations
        tipificaciones = row.get("tipificaciones", "")
        etapas = row.get("etapas", "") if "etapas" in df.columns else ""
        parts = [
            f"--- Conversacion {i + 1} ---",
            f"Tipificaciones: {tipificaciones}",
        ]
        if etapas:
            parts.append(f"Etapa: {etapas}")
        parts.append(f"Historial:\n{historial}")
        lines.append("\n".join(parts) + "\n")
    return "\n".join(lines)


def get_stage_index(stage: str, pipeline_type: str) -> int:
    """Get the position of a stage in the funnel order. Returns -1 if not found."""
    stages = get_stages_for_type(pipeline_type)
    try:
        return stages.index(stage)
    except ValueError:
        return -1


def score_conversation_for_pipeline(row: pd.Series, pipeline: Pipeline) -> int:
    """Score how well a conversation matches a pipeline. Higher = better match."""
    historial = row.get("historial", "").lower()
    tipificaciones = row.get("tipificaciones", "").lower()
    etapa_csv = row.get("etapas", "").strip() if "etapas" in row.index else ""
    score = 0

    # Stage match (strong signal)
    if etapa_csv:
        stages = pipeline.get_stages()
        if etapa_csv in stages:
            score += 5

    for obj in pipeline.objectives:
        # Success/failure criteria in tipificaciones
        if obj.success and obj.success.lower() in tipificaciones:
            score += 3
        if obj.failure and obj.failure.lower() in tipificaciones:
            score += 3
        # Keywords in historial
        for fd in obj.field_distribution:
            for kw in fd.keywords:
                if kw.lower() in historial:
                    score += 1

    return score


def classify_conversations(
    df: pd.DataFrame, pipelines: list[Pipeline]
) -> dict[str, pd.DataFrame]:
    """Classify conversations into pipelines. Returns {pipeline_id: filtered_df}."""
    if len(pipelines) <= 1:
        # Only one pipeline: all conversations go to it
        return {pipelines[0].id: df} if pipelines else {}

    # Score each conversation against each pipeline
    assignments: dict[str, list[int]] = {p.id: [] for p in pipelines}

    for idx, row in df.iterrows():
        scores = {p.id: score_conversation_for_pipeline(row, p) for p in pipelines}
        best_score = max(scores.values())

        if best_score > 0:
            # Assign to the pipeline with the highest score
            best_pipeline = max(scores, key=scores.get)
            assignments[best_pipeline].append(idx)
        else:
            # No match: assign to all pipelines
            for p_id in assignments:
                assignments[p_id].append(idx)

    result = {}
    for p in pipelines:
        indices = assignments[p.id]
        if indices:
            result[p.id] = df.loc[indices]
        else:
            result[p.id] = pd.DataFrame(columns=df.columns)

    return result


def evaluate_conversation_against_pipeline(
    row: pd.Series, pipeline: Pipeline
) -> dict:
    """
    Evaluate a single conversation against a pipeline.
    Returns dict with objective_id -> {"success": bool, "matched_keywords": list}
    """
    historial = row.get("historial", "").lower()
    etapa_csv = row.get("etapas", "").strip() if "etapas" in row.index else ""
    tipificaciones = row.get("tipificaciones", "").lower()

    stages = pipeline.get_stages()
    conv_stage_idx = get_stage_index(etapa_csv, pipeline.pipeline_type) if etapa_csv else -1

    results = {}
    for obj in pipeline.objectives:
        obj_stage_idx = get_stage_index(obj.stage, pipeline.pipeline_type)

        # Determine success based on stage progression (if etapas available)
        if conv_stage_idx >= 0 and obj_stage_idx >= 0:
            success = conv_stage_idx >= obj_stage_idx
        elif etapa_csv:
            success = etapa_csv.lower() == obj.stage.lower()
        else:
            # No etapas column: rely on keywords and tipificaciones
            success = False

        # Check for keyword matches
        matched_keywords = []
        for fd in obj.field_distribution:
            for kw in fd.keywords:
                if kw.lower() in historial:
                    matched_keywords.append(kw)

        # Also check tipificaciones for success criteria matches
        if obj.success.lower() in tipificaciones:
            success = True
        if obj.failure.lower() in tipificaciones:
            success = False

        results[obj.id] = {
            "success": success,
            "matched_keywords": matched_keywords,
        }

    return results


def aggregate_funnel_results(
    df: pd.DataFrame, pipeline: Pipeline
) -> PipelineResult:
    """
    Evaluate all conversations against a pipeline and aggregate into a PipelineResult.
    """
    total = len(df)
    objective_stats: dict[str, dict] = {}

    for obj in pipeline.objectives:
        objective_stats[obj.id] = {
            "success": 0,
            "failure": 0,
            "keywords": [],
        }

    for _, row in df.iterrows():
        eval_result = evaluate_conversation_against_pipeline(row, pipeline)
        for obj_id, result in eval_result.items():
            if obj_id in objective_stats:
                if result["success"]:
                    objective_stats[obj_id]["success"] += 1
                else:
                    objective_stats[obj_id]["failure"] += 1
                objective_stats[obj_id]["keywords"].extend(
                    result["matched_keywords"]
                )

    funnel: list[FunnelGoal] = []
    for obj in pipeline.objectives:
        stats = objective_stats[obj.id]
        s = stats["success"]
        f = stats["failure"]
        rate = (s / (s + f) * 100) if (s + f) > 0 else 0.0

        # Build keyword distribution
        kw_counter = Counter(stats["keywords"])
        kw_total = sum(kw_counter.values())
        kw_dist = [
            KeywordDistribution(
                value=kw,
                count=count,
                percentage=(count / kw_total * 100) if kw_total > 0 else 0,
            )
            for kw, count in kw_counter.most_common(10)
        ]

        funnel.append(
            FunnelGoal(
                objective_id=obj.id,
                objective_name=obj.name,
                is_conversion_indicator=obj.is_conversion_indicator,
                stage=obj.stage,
                success_count=s,
                failure_count=f,
                success_rate=round(rate, 1),
                keyword_distribution=kw_dist if kw_dist else [],
            )
        )

    # Determine abandoned conversations (didn't reach last stage)
    has_etapas = "etapas" in df.columns
    last_stage = pipeline.get_stages()[-1] if pipeline.get_stages() else ""
    last_stage_idx = get_stage_index(last_stage, pipeline.pipeline_type)
    abandoned = 0
    if has_etapas:
        for _, row in df.iterrows():
            etapa = row.get("etapas", "").strip()
            idx = get_stage_index(etapa, pipeline.pipeline_type)
            if idx < last_stage_idx:
                abandoned += 1
    completed = total - abandoned

    abandonment = AbandonmentAnalysis(
        total_conversations=total,
        abandoned_count=abandoned,
        abandonment_rate=round((abandoned / total * 100) if total > 0 else 0, 1),
        completed_count=completed,
        top_abandonment_reasons=[],  # Populated by LLM agent
    )

    return PipelineResult(
        pipeline_id=pipeline.id,
        pipeline_name=pipeline.name,
        pipeline_type=pipeline.pipeline_type,
        total_conversations=total,
        funnel=funnel,
        abandonment_analysis=abandonment,
    )


def aggregate_all_pipelines(
    df: pd.DataFrame, pipelines: list[Pipeline]
) -> tuple[list[PipelineResult], dict[str, pd.DataFrame]]:
    """Classify conversations and evaluate each pipeline against its filtered subset.

    Returns (pipeline_results, classification_map) where classification_map is
    {pipeline_id: filtered_df}.
    """
    classified = classify_conversations(df, pipelines)
    results = []
    for p in pipelines:
        filtered_df = classified.get(p.id, pd.DataFrame(columns=df.columns))
        results.append(aggregate_funnel_results(filtered_df, p))
    return results, classified
