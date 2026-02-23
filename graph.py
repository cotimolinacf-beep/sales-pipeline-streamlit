"""
LangGraph graph construction, node logic, and orchestration class.
"""

import json
import re
import pandas as pd
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents import (
    AnalysisState,
    get_llm,
    PIPELINE_IDENTIFIER_PROMPT,
    CONVERSATION_CLASSIFIER_PROMPT,
    VALUE_SUGGESTIONS_PROMPT,
)
from data_processor import (
    conversations_to_prompt_text,
)
from models import (
    Pipeline,
    EvaluationResults,
    PipelineResult,
    FunnelGoal,
    KeywordDistribution,
    AbandonmentAnalysis,
    AbandonmentReason,
    ValueSuggestion,
    ConversationDetail,
)


def _extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM response text."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Look for JSON block in markdown
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Look for any JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "No se pudo extraer JSON de la respuesta del LLM"}


# ──────────────────────────────────────────────
# Graph Nodes
# ──────────────────────────────────────────────

def pipeline_identifier_node(state: AnalysisState) -> dict:
    """Analyze sample conversations and suggest pipeline configuration."""
    llm = get_llm()
    conversations = state.get("conversations_json", "[]")

    prompt = HumanMessage(content=(
        "Analiza las siguientes conversaciones de muestra y sugiere una "
        "configuracion de pipelines con objetivos, etapas y keywords.\n\n"
        f"{conversations}"
    ))

    response = llm.invoke([PIPELINE_IDENTIFIER_PROMPT, prompt])
    result = _extract_json(response.content)

    return {
        "messages": [AIMessage(content=f"Pipeline identificado: {json.dumps(result, ensure_ascii=False)}")],
        "pipelines_config": json.dumps(result.get("pipelines", []), ensure_ascii=False),
        "current_agent": "pipeline_identifier",
    }


def _batch_conversations(conversations_text: str, batch_size: int = 20) -> list[str]:
    """Split conversations text into batches."""
    # Split by conversation separator
    parts = re.split(r"(?=--- Conversacion \d+)", conversations_text.strip())
    parts = [p.strip() for p in parts if p.strip()]

    batches = []
    for i in range(0, len(parts), batch_size):
        batch = "\n\n".join(parts[i:i + batch_size])
        batches.append(batch)
    return batches


def funnel_analyzer_node(state: AnalysisState) -> dict:
    """Classify each conversation to a pipeline and stage using LLM."""
    llm = get_llm()
    conversations = state.get("conversations_json", "[]")
    pipelines_json = state.get("pipelines_config", "[]")

    try:
        pipelines_data = json.loads(pipelines_json)
    except json.JSONDecodeError:
        pipelines_data = []

    if not pipelines_data:
        return {
            "messages": [AIMessage(content="No hay pipelines configurados.")],
            "classification_results": "[]",
            "funnel_results": "{}",
            "current_agent": "funnel_analyzer",
        }

    # Process conversations in batches
    batches = _batch_conversations(conversations)
    all_classifications = []

    pipelines_desc = json.dumps(pipelines_data, ensure_ascii=False, indent=2)

    for batch in batches:
        prompt = HumanMessage(content=(
            "Analiza CADA conversacion y clasifícala al pipeline correcto, "
            "determina la etapa alcanzada, y evalua los objetivos.\n\n"
            f"PIPELINES DISPONIBLES:\n{pipelines_desc}\n\n"
            f"CONVERSACIONES:\n{batch}\n\n"
            "Responde con JSON siguiendo la estructura indicada. "
            "Incluye TODAS las conversaciones del lote."
        ))

        response = llm.invoke([CONVERSATION_CLASSIFIER_PROMPT, prompt])
        result = _extract_json(response.content)
        batch_convs = result.get("conversations", [])
        all_classifications.extend(batch_convs)

    return {
        "messages": [AIMessage(content=f"Clasificacion completada: {len(all_classifications)} conversaciones.")],
        "classification_results": json.dumps(all_classifications, ensure_ascii=False),
        "funnel_results": "{}",
        "current_agent": "funnel_analyzer",
    }


def value_suggestions_node(state: AnalysisState) -> dict:
    """Generate value suggestions per pipeline based on classification results."""
    llm = get_llm()

    classification = state.get("classification_results", "[]")
    pipelines_json = state.get("pipelines_config", "[]")

    prompt = HumanMessage(content=(
        "Basado en los siguientes resultados de clasificacion de conversaciones, "
        "genera sugerencias de valor accionables PARA CADA PIPELINE.\n\n"
        f"CONFIGURACION DE PIPELINES:\n{pipelines_json}\n\n"
        f"CLASIFICACION DE CONVERSACIONES:\n{classification}\n\n"
        "Genera sugerencias en las 4 categorias: autogestion, conversion, "
        "cuello_botella, quick_win. Se especifico y basa tus sugerencias "
        "en los datos reales. Agrupa por pipeline_id."
    ))

    response = llm.invoke([VALUE_SUGGESTIONS_PROMPT, prompt])
    result = _extract_json(response.content)

    return {
        "messages": [AIMessage(content="Sugerencias de valor generadas.")],
        "suggestions": json.dumps(result, ensure_ascii=False),
        "current_agent": "value_suggestions",
    }


# ──────────────────────────────────────────────
# Graph Builders
# ──────────────────────────────────────────────

def build_auto_detect_graph() -> StateGraph:
    """Graph for auto-detecting pipeline configuration from sample conversations."""
    graph = StateGraph(AnalysisState)
    graph.add_node("pipeline_identifier", pipeline_identifier_node)
    graph.add_edge(START, "pipeline_identifier")
    graph.add_edge("pipeline_identifier", END)
    return graph.compile()


def build_analysis_graph() -> StateGraph:
    """
    Main analysis graph:
    START → funnel_analyzer → value_suggestions → END
    """
    graph = StateGraph(AnalysisState)

    graph.add_node("funnel_analyzer", funnel_analyzer_node)
    graph.add_node("value_suggestions", value_suggestions_node)

    graph.add_edge(START, "funnel_analyzer")
    graph.add_edge("funnel_analyzer", "value_suggestions")
    graph.add_edge("value_suggestions", END)

    return graph.compile()


# ──────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────

class SalesPipelineAnalyzer:
    """High-level orchestrator for the sales pipeline analysis system."""

    def __init__(self, provider: str = None, api_key: str = None):
        self.provider = provider
        self.api_key = api_key
        self._configure_llm()
        self.auto_detect_graph = build_auto_detect_graph()
        self.analysis_graph = build_analysis_graph()

    def _configure_llm(self):
        """Set environment variables for LLM provider."""
        if self.api_key and self.provider:
            key_map = {
                "Google Gemini": "GOOGLE_API_KEY",
                "OpenAI": "OPENAI_API_KEY",
                "Anthropic": "ANTHROPIC_API_KEY",
            }
            env_key = key_map.get(self.provider)
            if env_key:
                import os
                os.environ[env_key] = self.api_key

    def auto_detect_pipelines(self, conversations_text: str) -> list[dict]:
        """Run the pipeline identifier agent on sample conversations."""
        state: AnalysisState = {
            "messages": [],
            "pipelines_config": "[]",
            "conversations_json": conversations_text,
            "classification_results": "[]",
            "funnel_results": "{}",
            "abandonment_results": "{}",
            "suggestions": "{}",
            "current_agent": "",
        }

        result = self.auto_detect_graph.invoke(state)
        try:
            pipelines = json.loads(result.get("pipelines_config", "[]"))
            return pipelines if isinstance(pipelines, list) else []
        except json.JSONDecodeError:
            return []

    def run_analysis(
        self,
        df: pd.DataFrame,
        pipelines: list[Pipeline],
        progress_callback=None,
    ) -> EvaluationResults:
        """
        Run the full analysis pipeline:
        1. LLM classifies each conversation → pipeline + stage + objectives
        2. Aggregate LLM results into PipelineResults (funnel)
        3. LLM generates value suggestions per pipeline
        """
        import time
        from collections import Counter
        start_time = time.time()

        if progress_callback:
            progress_callback(0.1, "Preparando datos...")

        # Prepare conversations for LLM
        conversations_text = conversations_to_prompt_text(df)
        pipelines_dicts = [p.to_dict() for p in pipelines]

        # Build lookup maps
        pipeline_by_id = {p.id: p for p in pipelines}
        pipeline_id_to_name = {p.id: p.name for p in pipelines}

        # Run LLM analysis graph (classifier + suggestions)
        state: AnalysisState = {
            "messages": [],
            "pipelines_config": json.dumps(pipelines_dicts, ensure_ascii=False),
            "conversations_json": conversations_text,
            "classification_results": "[]",
            "funnel_results": "{}",
            "abandonment_results": "{}",
            "suggestions": "{}",
            "current_agent": "",
        }

        if progress_callback:
            progress_callback(0.2, "Clasificando conversaciones con IA...")

        result = self.analysis_graph.invoke(state)

        if progress_callback:
            progress_callback(0.7, "Procesando resultados...")

        # Parse LLM classification results
        classifications = []
        try:
            raw = result.get("classification_results", "[]")
            classifications = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            classifications = []

        # Build index map: conversation_index (1-based from LLM) -> classification
        conv_classifications: dict[int, dict] = {}
        for c in classifications:
            idx = c.get("conversation_index", 0)
            conv_classifications[idx] = c

        # ── Aggregate LLM results into PipelineResults ──
        # Per pipeline: count objectives success/failure, keywords, abandonment
        pipeline_stats: dict[str, dict] = {}
        for p in pipelines:
            pipeline_stats[p.id] = {
                "total": 0,
                "objectives": {obj.id: {"success": 0, "failure": 0, "keywords": []} for obj in p.objectives},
                "abandonment_reasons": [],
                "stages_reached": [],
            }

        # Per-conversation details for CSV export
        conversation_details = []

        for df_idx, (_, row) in enumerate(df.iterrows()):
            conv_idx = df_idx + 1  # LLM uses 1-based index
            c = conv_classifications.get(conv_idx, {})

            assigned_pipeline_id = c.get("pipeline_id", "")
            assigned_pipeline_name = pipeline_id_to_name.get(assigned_pipeline_id, "")
            stage_reached = c.get("stage_reached", "")
            abandonment_reason = c.get("abandonment_reason")
            llm_objectives = c.get("objectives", {})

            # If LLM didn't classify this conversation, skip stats
            if assigned_pipeline_id and assigned_pipeline_id in pipeline_stats:
                stats = pipeline_stats[assigned_pipeline_id]
                stats["total"] += 1
                stats["stages_reached"].append(stage_reached)

                if abandonment_reason:
                    stats["abandonment_reasons"].append(abandonment_reason)

                # Objective success/failure from LLM
                pipeline = pipeline_by_id.get(assigned_pipeline_id)
                if pipeline:
                    for obj in pipeline.objectives:
                        obj_result = llm_objectives.get(obj.id, {})
                        if obj_result.get("success", False):
                            stats["objectives"][obj.id]["success"] += 1
                        else:
                            stats["objectives"][obj.id]["failure"] += 1
                        kws = obj_result.get("keywords", [])
                        stats["objectives"][obj.id]["keywords"].extend(kws)

            # Build per-conversation detail
            pipe_results = {}
            if assigned_pipeline_name:
                pipeline = pipeline_by_id.get(assigned_pipeline_id)
                if pipeline:
                    obj_map = {}
                    all_kws = []
                    for obj in pipeline.objectives:
                        obj_r = llm_objectives.get(obj.id, {})
                        obj_map[obj.name] = obj_r.get("success", False)
                        all_kws.extend(obj_r.get("keywords", []))
                    pipe_results[assigned_pipeline_name] = {
                        "objectives": obj_map,
                        "keywords": all_kws,
                    }

            conversation_details.append(
                ConversationDetail(
                    index=df_idx,
                    pipeline_assigned=assigned_pipeline_name,
                    stage_assigned=stage_reached,
                    pipeline_results=pipe_results,
                )
            )

        # ── Build PipelineResult objects from aggregated stats ──
        from data_processor import get_stage_index
        pipeline_results = []
        for p in pipelines:
            stats = pipeline_stats[p.id]
            total = stats["total"]

            funnel: list[FunnelGoal] = []
            for obj in p.objectives:
                obj_stats = stats["objectives"][obj.id]
                s = obj_stats["success"]
                f = obj_stats["failure"]
                rate = round((s / (s + f) * 100) if (s + f) > 0 else 0.0, 1)

                kw_counter = Counter(obj_stats["keywords"])
                kw_total = sum(kw_counter.values())
                kw_dist = [
                    KeywordDistribution(
                        value=kw, count=count,
                        percentage=round((count / kw_total * 100) if kw_total > 0 else 0, 1),
                    )
                    for kw, count in kw_counter.most_common(10)
                ]

                funnel.append(FunnelGoal(
                    objective_id=obj.id,
                    objective_name=obj.name,
                    is_conversion_indicator=obj.is_conversion_indicator,
                    stage=obj.stage,
                    success_count=s,
                    failure_count=f,
                    success_rate=rate,
                    keyword_distribution=kw_dist,
                ))

            # Abandonment analysis
            reason_counter = Counter(stats["abandonment_reasons"])
            total_abandoned = len(stats["abandonment_reasons"])
            top_reasons = [
                AbandonmentReason(
                    reason=reason,
                    occurrences=count,
                    percentage=round((count / total_abandoned * 100) if total_abandoned > 0 else 0, 1),
                )
                for reason, count in reason_counter.most_common(5)
            ]

            abandonment = AbandonmentAnalysis(
                total_conversations=total,
                abandoned_count=total_abandoned,
                abandonment_rate=round((total_abandoned / total * 100) if total > 0 else 0, 1),
                completed_count=total - total_abandoned,
                top_abandonment_reasons=top_reasons,
            )

            pipeline_results.append(PipelineResult(
                pipeline_id=p.id,
                pipeline_name=p.name,
                pipeline_type=p.pipeline_type,
                total_conversations=total,
                funnel=funnel,
                abandonment_analysis=abandonment,
            ))

        # ── Distribute suggestions per pipeline ──
        suggestions_data = _extract_json(result.get("suggestions", "{}"))
        llm_pipelines_suggestions = suggestions_data.get("pipelines", [])
        for pr in pipeline_results:
            pr_suggestions = []
            for llm_ps in llm_pipelines_suggestions:
                if llm_ps.get("pipeline_id") == pr.pipeline_id:
                    for s in llm_ps.get("suggestions", []):
                        pr_suggestions.append(
                            ValueSuggestion(
                                category=s.get("category", "quick_win"),
                                title=s.get("title", ""),
                                description=s.get("description", ""),
                                impact=s.get("impact", "medio"),
                                metric=s.get("metric"),
                            )
                        )
            pr.suggestions = pr_suggestions

        # Fallback: if LLM didn't group by pipeline
        if not any(pr.suggestions for pr in pipeline_results):
            flat_suggestions = suggestions_data.get("suggestions", [])
            if flat_suggestions and pipeline_results:
                for s in flat_suggestions:
                    pipeline_results[0].suggestions.append(
                        ValueSuggestion(
                            category=s.get("category", "quick_win"),
                            title=s.get("title", ""),
                            description=s.get("description", ""),
                            impact=s.get("impact", "medio"),
                            metric=s.get("metric"),
                        )
                    )

        # Pass aggregated funnel to suggestions node (already done via graph)
        if progress_callback:
            progress_callback(0.95, "Finalizando...")

        elapsed = time.time() - start_time

        return EvaluationResults(
            total_conversations_analyzed=len(df),
            pipelines=pipeline_results,
            conversation_details=conversation_details,
            processing_time_seconds=round(elapsed, 2),
        )
