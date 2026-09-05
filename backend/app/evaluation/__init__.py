"""Offline/private evaluation workflow for the grounded RAG service."""

from app.evaluation.bad_cases import (
    BAD_CASE_STATUS_TRANSITIONS,
    FEEDBACK_REASON_CODES,
    apply_bad_case_transition,
    feedback_reason_to_code,
)
from app.evaluation.industry_metrics import (
    GenerationMetricReport,
    IndustryMetricReport,
    RateMetric,
    RetrievalMetricReport,
    SafetyMetricReport,
)
from app.evaluation.protocol import (
    TASK12A_PROTOCOL,
    EvaluationProtocol,
    EvaluationRunMetadata,
)
from app.evaluation.service import (
    EVALUATION_TYPES,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationGeneration,
    EvaluationReport,
    EvaluationService,
    EvaluationSet,
    calculate_report,
)

__all__ = [
    "BAD_CASE_STATUS_TRANSITIONS",
    "EVALUATION_TYPES",
    "FEEDBACK_REASON_CODES",
    "TASK12A_PROTOCOL",
    "EvaluationCase",
    "EvaluationCaseResult",
    "EvaluationGeneration",
    "EvaluationProtocol",
    "EvaluationReport",
    "EvaluationRunMetadata",
    "EvaluationService",
    "EvaluationSet",
    "GenerationMetricReport",
    "IndustryMetricReport",
    "RateMetric",
    "RetrievalMetricReport",
    "SafetyMetricReport",
    "apply_bad_case_transition",
    "calculate_report",
    "feedback_reason_to_code",
]
