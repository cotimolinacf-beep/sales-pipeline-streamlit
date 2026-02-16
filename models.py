"""
Data models for the Sales Pipeline Analyzer.
Mirrors the TypeScript types from the original Next.js app.
"""

from dataclasses import dataclass, field
from typing import Optional
import uuid
from datetime import datetime


# ──────────────────────────────────────────────
# Pipeline Configuration Models
# ──────────────────────────────────────────────

SALES_STAGES = ["Awareness", "Lead", "MQL", "SQL", "Opportunity", "Customer"]
SERVICE_STAGES = [
    "Ticket creado",
    "Ticket abierto",
    "Ticket en proceso",
    "Ticket resuelto",
]

PIPELINE_TYPES = ["ventas", "servicio"]


def get_stages_for_type(pipeline_type: str) -> list[str]:
    if pipeline_type == "ventas":
        return SALES_STAGES
    return SERVICE_STAGES


@dataclass
class FieldDistribution:
    name: str
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "keywords": self.keywords}

    @classmethod
    def from_dict(cls, data: dict) -> "FieldDistribution":
        return cls(name=data["name"], keywords=data.get("keywords", []))


@dataclass
class Objective:
    name: str
    stage: str
    success: str
    failure: str
    is_conversion_indicator: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    field_distribution: list[FieldDistribution] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "isConversionIndicator": self.is_conversion_indicator,
            "stage": self.stage,
            "success": self.success,
            "failure": self.failure,
            "field_distribution": [fd.to_dict() for fd in self.field_distribution],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Objective":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            is_conversion_indicator=data.get("isConversionIndicator", False),
            stage=data["stage"],
            success=data["success"],
            failure=data["failure"],
            field_distribution=[
                FieldDistribution.from_dict(fd)
                for fd in data.get("field_distribution", [])
            ],
        )


@dataclass
class Pipeline:
    name: str
    description: str
    pipeline_type: str  # "ventas" or "servicio"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    objectives: list[Objective] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_stages(self) -> list[str]:
        return get_stages_for_type(self.pipeline_type)

    def get_conversion_indicator(self) -> Optional[Objective]:
        for obj in self.objectives:
            if obj.is_conversion_indicator:
                return obj
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.pipeline_type,
            "objectives": [o.to_dict() for o in self.objectives],
            "createdAt": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Pipeline":
        pipeline = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description", ""),
            pipeline_type=data.get("type", "ventas"),
            created_at=data.get("createdAt", datetime.now().isoformat()),
        )
        pipeline.objectives = [
            Objective.from_dict(o) for o in data.get("objectives", [])
        ]
        return pipeline


# ──────────────────────────────────────────────
# Evaluation Results Models
# ──────────────────────────────────────────────


@dataclass
class KeywordDistribution:
    value: str
    count: int
    percentage: float

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "count": self.count,
            "percentage": self.percentage,
        }


@dataclass
class FunnelGoal:
    objective_id: str
    objective_name: str
    is_conversion_indicator: bool
    stage: str
    success_count: int
    failure_count: int
    success_rate: float
    keyword_distribution: list[KeywordDistribution] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {
            "objective_id": self.objective_id,
            "objective_name": self.objective_name,
            "is_conversion_indicator": self.is_conversion_indicator,
            "stage": self.stage,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
        }
        if self.keyword_distribution:
            result["keyword_distribution"] = [
                kd.to_dict() for kd in self.keyword_distribution
            ]
        return result


@dataclass
class AbandonmentReason:
    reason: str
    occurrences: int
    percentage: float


@dataclass
class AbandonmentAnalysis:
    total_conversations: int
    abandoned_count: int
    abandonment_rate: float
    completed_count: int
    top_abandonment_reasons: list[AbandonmentReason] = field(default_factory=list)


@dataclass
class ValueSuggestion:
    category: str  # "autogestion", "conversion", "cuello_botella", "quick_win"
    title: str
    description: str
    impact: str  # "alto", "medio", "bajo"
    metric: Optional[str] = None


@dataclass
class PipelineResult:
    pipeline_id: str
    pipeline_name: str
    pipeline_type: str
    total_conversations: int
    funnel: list[FunnelGoal] = field(default_factory=list)
    abandonment_analysis: Optional[AbandonmentAnalysis] = None
    suggestions: list[ValueSuggestion] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "pipeline_type": self.pipeline_type,
            "total_conversations": self.total_conversations,
            "funnel": [g.to_dict() for g in self.funnel],
        }
        if self.abandonment_analysis:
            result["abandonment_analysis"] = {
                "total_conversations": self.abandonment_analysis.total_conversations,
                "abandoned_count": self.abandonment_analysis.abandoned_count,
                "abandonment_rate": self.abandonment_analysis.abandonment_rate,
                "completed_count": self.abandonment_analysis.completed_count,
                "top_abandonment_reasons": [
                    {
                        "reason": r.reason,
                        "occurrences": r.occurrences,
                        "percentage": r.percentage,
                    }
                    for r in self.abandonment_analysis.top_abandonment_reasons
                ],
            }
        return result


@dataclass
class ConversationDetail:
    """Per-conversation analysis detail (for CSV export)."""
    index: int
    pipeline_results: dict = field(default_factory=dict)
    # pipeline_results: {pipeline_name: {"objectives": {obj_name: bool}, "keywords": [str]}}


@dataclass
class EvaluationResults:
    total_conversations_analyzed: int
    pipelines: list[PipelineResult] = field(default_factory=list)
    conversation_details: list[ConversationDetail] = field(default_factory=list)
    processing_time_seconds: Optional[float] = None
