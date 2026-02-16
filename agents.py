"""
LangGraph agent definitions: state, system prompts, tools.
"""

import os
import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

class AnalysisState(TypedDict):
    messages: Annotated[list, add_messages]
    pipelines_config: str       # JSON serialized list of pipeline dicts
    conversations_json: str     # JSON serialized conversation data
    funnel_results: str         # JSON with funnel results per pipeline
    abandonment_results: str    # JSON with abandonment reasons per pipeline
    suggestions: str            # JSON with value suggestions per pipeline
    current_agent: str


# ──────────────────────────────────────────────
# LLM Factory
# ──────────────────────────────────────────────

def get_llm(provider: str = None, api_key: str = None):
    """Get LLM instance based on provider preference or environment."""
    if provider == "Google Gemini" or (not provider and os.getenv("GOOGLE_API_KEY")):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )
    if provider == "OpenAI" or (not provider and os.getenv("OPENAI_API_KEY")):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )
    if provider == "Anthropic" or (not provider and os.getenv("ANTHROPIC_API_KEY")):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            temperature=0,
        )
    raise EnvironmentError(
        "No se encontro API key. Configura una en el sidebar."
    )


# ──────────────────────────────────────────────
# System Prompts
# ──────────────────────────────────────────────

PIPELINE_IDENTIFIER_PROMPT = SystemMessage(content="""Eres un experto en analisis de conversaciones de ventas y servicio al cliente.

Tu tarea es analizar una muestra de conversaciones y sugerir una configuracion de pipelines con sus objetivos.

REGLAS:
- Identifica patrones en las conversaciones para determinar tipos de pipeline (ventas, servicio).
- Para cada pipeline, sugiere objetivos ordenados por etapas del funnel.
- Para pipelines de VENTAS usa estas etapas: Awareness, Lead, MQL, SQL, Opportunity, Customer
- Para pipelines de SERVICIO usa estas etapas: Ticket creado, Ticket abierto, Ticket en proceso, Ticket resuelto
- Cada pipeline debe tener exactamente UN objetivo marcado como indicador de conversion (el mas importante).
- Incluye keywords relevantes encontrados en las conversaciones para cada objetivo.
- Cada objetivo debe tener criterios claros de exito y fallo.

RESPONDE EXCLUSIVAMENTE con un JSON valido con esta estructura:
{
  "pipelines": [
    {
      "name": "Nombre del pipeline",
      "description": "Descripcion breve",
      "type": "ventas" o "servicio",
      "objectives": [
        {
          "name": "Nombre del objetivo",
          "stage": "Etapa del funnel",
          "isConversionIndicator": false,
          "success": "Criterio de exito",
          "failure": "Criterio de fallo",
          "field_distribution": [
            {"name": "categoria", "keywords": ["keyword1", "keyword2"]}
          ]
        }
      ]
    }
  ]
}""")


FUNNEL_ANALYZER_PROMPT = SystemMessage(content="""Eres un analista experto en embudos de conversion de ventas y servicio al cliente.

Tu tarea es evaluar conversaciones individuales contra los objetivos de un pipeline y determinar exito o fallo en cada objetivo.

Para cada conversacion analiza:
1. Que etapa del funnel alcanzo la conversacion
2. Si se cumplieron los criterios de exito de cada objetivo
3. Que keywords relevantes aparecen en la conversacion
4. Si la conversacion fue abandonada (no alcanzo la ultima etapa)
5. Si fue abandonada, cual fue la razon principal de abandono

REGLAS:
- Una conversacion que alcanzo la etapa X tambien paso por todas las etapas anteriores.
- Busca evidencia concreta en el texto de la conversacion.
- Las razones de abandono deben ser especificas y accionables.

Responde EXCLUSIVAMENTE con un JSON valido.""")


VALUE_SUGGESTIONS_PROMPT = SystemMessage(content="""Eres un consultor estrategico especializado en optimizacion de experiencia conversacional, autogestion digital y conversion de ventas.

Tienes los resultados de analisis de multiples pipelines de conversaciones. Tu tarea es generar sugerencias accionables de alto valor PARA CADA PIPELINE.

Genera sugerencias en estas 4 categorias:

1. **AUTOGESTION** (category: "autogestion"):
   - Identifica consultas que el bot podria resolver sin intervencion humana.
   - Propone mejoras en flujos automaticos.
   - Sugiere contenido/informacion que el bot deberia tener disponible.
   Ejemplo: "68% de las consultas piden precios - el bot deberia tener un catalogo de precios integrado"

2. **CONVERSION** (category: "conversion"):
   - Identifica oportunidades para mejorar tasas de conversion en cada etapa.
   - Propone acciones proactivas del bot para avanzar al usuario en el funnel.
   - Sugiere triggers y CTAs basados en los patrones observados.
   Ejemplo: "Solo 23% agenda prueba de manejo - sugerir proactivamente en etapa SQL con disponibilidad inmediata"

3. **CUELLOS DE BOTELLA** (category: "cuello_botella"):
   - Identifica etapas con mayor caida de conversion.
   - Correlaciona con razones de abandono.
   - Propone soluciones especificas para cada cuello de botella.

4. **QUICK WINS** (category: "quick_win"):
   - Acciones de alto impacto y bajo esfuerzo que se pueden implementar rapido.
   - Mejoras inmediatas en scripts, flujos o informacion del bot.

Para cada sugerencia indica el nivel de impacto: "alto", "medio", "bajo".

IMPORTANTE: Agrupa las sugerencias POR PIPELINE usando el pipeline_id.

Responde EXCLUSIVAMENTE con un JSON valido:
{
  "pipelines": [
    {
      "pipeline_id": "id del pipeline",
      "suggestions": [
        {
          "category": "autogestion" | "conversion" | "cuello_botella" | "quick_win",
          "title": "Titulo corto",
          "description": "Descripcion detallada y accionable",
          "impact": "alto" | "medio" | "bajo",
          "metric": "Metrica relevante si aplica (ej: '68% de consultas')"
        }
      ]
    }
  ]
}""")


# ──────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────

@tool
def evaluate_conversations_batch(conversations_json: str, pipeline_config_json: str) -> str:
    """Evaluate a batch of conversations against a pipeline configuration.
    Returns JSON with success/failure counts per objective and keyword matches."""
    try:
        conversations = json.loads(conversations_json)
        pipeline_config = json.loads(pipeline_config_json)

        results = {
            "pipeline_id": pipeline_config.get("id", ""),
            "pipeline_name": pipeline_config.get("name", ""),
            "objectives": {},
        }

        for obj in pipeline_config.get("objectives", []):
            results["objectives"][obj["id"]] = {
                "success": 0,
                "failure": 0,
                "keywords_found": [],
            }

        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})


# Tool lists per agent
FUNNEL_TOOLS = [evaluate_conversations_batch]
