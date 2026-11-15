"""Repeatable, user-reviewed comparison of vision providers on supplied images."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .vision import ImageInput, VisionAnalysisError, VisionAnalysisService, VisionProviderConfig

QUALITY_RUBRIC = (
    "visible_device_recognition",
    "differential_pair_symmetry_interpretation",
    "current_mirror_and_common_centroid_interpretation",
    "routing_observations",
    "congestion_identification",
    "recognition_of_uncertainty",
    "avoidance_of_invented_details",
    "technical_usefulness_of_recommendations",
    "quality_of_verification_suggestions",
)
EVALUATION_METRICS = ("response_latency_ms", "api_usage_and_cost")


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    images: tuple[ImageInput, ...]
    engineering_context: Mapping[str, str] = field(default_factory=dict)
    comparison_requested: bool = False
    reviewer_notes: str = ""


def new_evaluation_report(
    *,
    dataset_label: str,
    providers: Sequence[VisionProviderConfig],
    cases: Sequence[EvaluationCase],
) -> dict[str, Any]:
    """Return an empty reviewable report; no score or benchmark result is inferred."""
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_label": dataset_label,
        "evaluation_status": "user_review_required",
        "limitations": [
            "User-supplied cases are not a verified or representative analog-layout benchmark.",
            "Provider output is not engineering verification, signoff, or a measured result.",
            "Do not treat these scores as model training data or a provider ranking until reviewed.",
        ],
        "rubric": {
            "quality_scoring": "Manual score 1 (poor) through 5 (strong), or null until reviewed.",
            "quality_categories": list(QUALITY_RUBRIC),
            "observed_metrics": list(EVALUATION_METRICS),
        },
        "providers": [
            {"provider_id": config.provider, "provider": config.public_provider_name, "model": config.model}
            for config in providers
        ],
        "cases": [
            {
                "case_id": case.case_id,
                "question": case.question,
                "engineering_context": dict(case.engineering_context),
                "comparison_requested": case.comparison_requested,
                "images": [
                    {
                        "image_id": image.image_id,
                        "filename": image.filename,
                        "label": image.label or image.filename,
                        "mime_type": image.mime_type,
                        "width": image.width,
                        "height": image.height,
                    }
                    for image in case.images
                ],
                "results": {},
                "human_review": {
                    "quality_scores_1_to_5_or_null": {key: None for key in QUALITY_RUBRIC},
                    "reviewer_notes": case.reviewer_notes,
                },
            }
            for case in cases
        ],
    }


def run_case_for_providers(
    case: EvaluationCase,
    configs: Sequence[VisionProviderConfig],
) -> dict[str, dict[str, Any]]:
    """Submit identical case inputs to each explicit provider; no fallback or retries."""
    if len(configs) < 2 or {config.provider for config in configs} != {"gemini", "openai"}:
        raise VisionAnalysisError("Provider comparison requires configured Gemini and OpenAI providers.")
    results: dict[str, dict[str, Any]] = {}
    for config in configs:
        key = str(config.provider)
        try:
            service = VisionAnalysisService(config)
            analysis = service.analyze(
                case.images,
                case.question,
                comparison_requested=case.comparison_requested,
                engineering_context=case.engineering_context,
            )
            results[key] = {"status": "completed", "analysis": analysis_to_dict(analysis)}
        except VisionAnalysisError as exc:
            # Continue only because the evaluation explicitly requested both named providers.
            results[key] = {"status": "failed", "error": str(exc)}
    return results


def analysis_to_dict(analysis: Any) -> dict[str, Any]:
    """Serialize normalized provider output without serializing image bytes."""
    return {
        "visual_observations": list(analysis.visual_observations),
        "potential_issues": [asdict(item) for item in analysis.potential_issues],
        "recommended_modifications": [asdict(item) for item in analysis.recommended_modifications],
        "expected_benefits": list(analysis.expected_benefits),
        "possible_tradeoffs": list(analysis.possible_tradeoffs),
        "verification_required": list(analysis.verification_required),
        "confidence_and_limitations": list(analysis.confidence_and_limitations),
        "image_references": [asdict(item) for item in analysis.image_references],
        "provider_metadata": dict(analysis.provider_metadata),
        "comparison_summary": analysis.comparison_summary,
    }
