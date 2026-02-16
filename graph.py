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
    FUNNEL_ANALYZER_PROMPT,
    VALUE_SUGGESTIONS_PROMPT,
)
from data_processor import (
    conversations_to_prompt_text,
    aggregate_funnel_results,
    aggregate_all_pipelines,
)
from models import (
    Pipeline,
    EvaluationResults,
    ValueSuggestion,
    AbandonmentReason,
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


def funnel_analyzer_node(state: AnalysisState) -> dict:
    """Evaluate conversations against pipelines using LLM for nuanced analysis."""
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
            "funnel_results": "[]",
            "current_agent": "funnel_analyzer",
        }

    # Use LLM for conversation-level evaluation with abandonment analysis
    prompt = HumanMessage(content=(
        "Evalua las siguientes conversaciones contra los pipelines configurados.\n\n"
        f"PIPELINES:\n{json.dumps(pipelines_data, ensure_ascii=False, indent=2)}\n\n"
        f"CONVERSACIONES:\n{conversations}\n\n"
        "Para cada pipeline, analiza cada conversacion y determina:\n"
        "1. Que objetivos se cumplieron (success) y cuales no (failure)\n"
        "2. Que keywords relevantes encontraste\n"
        "3. Si la conversacion fue abandonada, identifica la razon principal\n\n"
        "Responde con JSON:\n"
        "{\n"
        '  "pipeline_results": [\n'
        "    {\n"
        '      "pipeline_id": "id",\n'
        '      "objectives": [\n'
        '        {"objective_id": "id", "success_count": N, "failure_count": N, '
        '"keywords_found": {"keyword": count}}\n'
        "      ],\n"
        '      "abandonment_reasons": [\n'
        '        {"reason": "descripcion especifica", "count": N}\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}"
    ))

    response = llm.invoke([FUNNEL_ANALYZER_PROMPT, prompt])
    result = _extract_json(response.content)

    return {
        "messages": [AIMessage(content="Analisis de funnel completado.")],
        "funnel_results": json.dumps(result, ensure_ascii=False),
        "current_agent": "funnel_analyzer",
    }


def value_suggestions_node(state: AnalysisState) -> dict:
    """Generate value suggestions per pipeline based on funnel results."""
    llm = get_llm()

    funnel = state.get("funnel_results", "{}")
    pipelines_json = state.get("pipelines_config", "[]")

    prompt = HumanMessage(content=(
        "Basado en los siguientes resultados de analisis, genera sugerencias "
        "de valor accionables PARA CADA PIPELINE.\n\n"
        f"CONFIGURACION DE PIPELINES:\n{pipelines_json}\n\n"
        f"RESULTADOS DEL FUNNEL:\n{funnel}\n\n"
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
        1. Rule-based funnel aggregation
        2. LLM-based funnel enrichment + abandonment analysis
        3. LLM-based value suggestions per pipeline
        """
        import time
        start_time = time.time()

        if progress_callback:
            progress_callback(0.1, "Preparando datos...")

        # Prepare conversations for LLM
        conversations_text = conversations_to_prompt_text(df)
        pipelines_dicts = [p.to_dict() for p in pipelines]

        # Rule-based funnel aggregation
        if progress_callback:
            progress_callback(0.2, "Analizando embudo de conversion...")
        pipeline_results = aggregate_all_pipelines(df, pipelines)

        # Run LLM analysis graph
        state: AnalysisState = {
            "messages": [],
            "pipelines_config": json.dumps(pipelines_dicts, ensure_ascii=False),
            "conversations_json": conversations_text,
            "funnel_results": json.dumps(
                [pr.to_dict() for pr in pipeline_results], ensure_ascii=False
            ),
            "abandonment_results": "{}",
            "suggestions": "{}",
            "current_agent": "",
        }

        if progress_callback:
            progress_callback(0.3, "Ejecutando agentes de analisis...")

        result = self.analysis_graph.invoke(state)

        if progress_callback:
            progress_callback(0.7, "Procesando resultados...")

        # Parse LLM results and enrich pipeline_results
        funnel_llm = _extract_json(result.get("funnel_results", "{}"))
        suggestions_data = _extract_json(result.get("suggestions", "{}"))

        # Enrich pipeline results with LLM abandonment reasons
        llm_pipeline_results = funnel_llm.get("pipeline_results", [])
        for pr in pipeline_results:
            for llm_pr in llm_pipeline_results:
                if llm_pr.get("pipeline_id") == pr.pipeline_id:
                    reasons = llm_pr.get("abandonment_reasons", [])
                    total_reasons = sum(r.get("count", 0) for r in reasons)
                    if pr.abandonment_analysis:
                        pr.abandonment_analysis.top_abandonment_reasons = [
                            AbandonmentReason(
                                reason=r["reason"],
                                occurrences=r.get("count", 0),
                                percentage=round(
                                    (r.get("count", 0) / total_reasons * 100)
                                    if total_reasons > 0
                                    else 0,
                                    2,
                                ),
                            )
                            for r in reasons[:5]
                        ]

                    # Enrich funnel with LLM objective results
                    llm_objectives = llm_pr.get("objectives", [])
                    for fg in pr.funnel:
                        for llm_obj in llm_objectives:
                            if llm_obj.get("objective_id") == fg.objective_id:
                                # Use LLM counts if they differ meaningfully
                                llm_s = llm_obj.get("success_count")
                                llm_f = llm_obj.get("failure_count")
                                if llm_s is not None and llm_f is not None:
                                    fg.success_count = llm_s
                                    fg.failure_count = llm_f
                                    total = llm_s + llm_f
                                    fg.success_rate = round(
                                        (llm_s / total * 100) if total > 0 else 0, 1
                                    )

        # Distribute suggestions per pipeline
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

        # If LLM didn't group by pipeline, try flat suggestions list as fallback
        if not any(pr.suggestions for pr in pipeline_results):
            flat_suggestions = suggestions_data.get("suggestions", [])
            if flat_suggestions:
                # Distribute evenly across pipelines
                for s in flat_suggestions:
                    suggestion = ValueSuggestion(
                        category=s.get("category", "quick_win"),
                        title=s.get("title", ""),
                        description=s.get("description", ""),
                        impact=s.get("impact", "medio"),
                        metric=s.get("metric"),
                    )
                    # Add to first pipeline as fallback
                    if pipeline_results:
                        pipeline_results[0].suggestions.append(suggestion)

        # Build per-conversation details (for CSV export)
        from data_processor import evaluate_conversation_against_pipeline
        conversation_details = []
        for idx, row in df.iterrows():
            pipe_results = {}
            for pipeline in pipelines:
                eval_r = evaluate_conversation_against_pipeline(row, pipeline)
                obj_map = {}
                all_kws = []
                for obj in pipeline.objectives:
                    r = eval_r.get(obj.id, {})
                    obj_map[obj.name] = r.get("success", False)
                    all_kws.extend(r.get("matched_keywords", []))
                pipe_results[pipeline.name] = {
                    "objectives": obj_map,
                    "keywords": all_kws,
                }

            conversation_details.append(
                ConversationDetail(
                    index=int(idx),
                    pipeline_results=pipe_results,
                )
            )

        if progress_callback:
            progress_callback(0.95, "Finalizando...")

        elapsed = time.time() - start_time

        return EvaluationResults(
            total_conversations_analyzed=len(df),
            pipelines=pipeline_results,
            conversation_details=conversation_details,
            processing_time_seconds=round(elapsed, 2),
        )
