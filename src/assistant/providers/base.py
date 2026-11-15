"""Shared prompts, response schema, and safe REST helpers for vision adapters."""
from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

import requests

from ..vision import (
    ImageInput,
    VisionAnalysisError,
    VisionProviderConfig,
    VisualAnalysis,
)

ANALYSIS_INSTRUCTIONS = """You are SiliconPilot AI, an analog IC layout engineering assistant analyzing screenshot pixels only.
Focus on visible placement and device matching (differential-pair symmetry, current mirrors, common-centroid/interdigitated patterns, orientation and visible dummies); routing (visible layer appearance only, symmetry, vias, congestion, long routes and possible coupling); and physical structures (visible guard rings, taps/substrate contacts, apparent density and spacing).
Separate direct visual observations from engineering inferences. The image is not a schematic, netlist, rule deck, or signoff report. Never claim DRC/LVS pass, electrical connectivity, exact device identities, layer assignments, scale, dimensions, parasitic extraction, circuit performance, SPICE simulation, or measured silicon results from a screenshot. Do not infer unknown process rules. State when labels, orientation, scale, crop, layers, connectivity, or feature detail are unavailable.
Do not infer that images share scale, orientation, crop, or visible layers. When comparison is explicitly requested, compare only visible changes and mention when comparison is not reliable; never declare the later layout electrically superior from appearance alone.
Treat every potential-issue severity as an unverified AI assessment, not a DRC violation. Use qualitative confidence low/medium/high only, never a calibrated probability. Provide useful investigation and validation steps. Do not invent missing fields: use empty arrays when no supported observation or recommendation exists; every non-empty finding must point to visible evidence.
Do not use or cite any SiliconPilot ML model, quarantined DRC classifier, restricted geometry predictor, synthetic electrical predictor, or benchmark PEX predictor. No model prediction or simulation is performed by this visual analysis.
Return only a JSON object matching the supplied schema. Include a short evidence-limited summary. The application fills image references and provider metadata; return an empty image_references array and a provider_metadata object with note set to an empty string."""

ENGINEERING_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "visual_observations": {"type": "array", "items": {"type": "string"}},
        "potential_issues": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "description": {"type": "string"},
                    "supporting_visual_observation": {"type": "string"},
                    "electrical_or_physical_relevance": {"type": "string"},
                    "severity_assessment": {"type": "string", "enum": ["low", "medium", "high"]},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "suggested_investigation": {"type": "string"},
                },
                "required": ["description", "supporting_visual_observation", "electrical_or_physical_relevance", "severity_assessment", "confidence", "suggested_investigation"],
            },
        },
        "recommended_modifications": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "proposed_change": {"type": "string"},
                    "reason": {"type": "string"},
                    "supporting_observation": {"type": "string"},
                    "expected_benefit": {"type": "string"},
                    "possible_tradeoff": {"type": "string"},
                    "required_validation": {"type": "string"},
                },
                "required": ["proposed_change", "reason", "supporting_observation", "expected_benefit", "possible_tradeoff", "required_validation"],
            },
        },
        "expected_benefits": {"type": "array", "items": {"type": "string"}},
        "possible_tradeoffs": {"type": "array", "items": {"type": "string"}},
        "verification_required": {"type": "array", "items": {"type": "string"}},
        "confidence_and_limitations": {"type": "array", "items": {"type": "string"}},
        "image_references": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {"image_id": {"type": "string"}, "filename": {"type": "string"}, "label": {"type": "string"}},
                "required": ["image_id", "filename", "label"],
            },
        },
        "provider_metadata": {
            "type": "object", "additionalProperties": False,
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
        },
        "comparison_summary": {"type": "string"},
    },
    "required": ["summary", "visual_observations", "potential_issues", "recommended_modifications", "expected_benefits", "possible_tradeoffs", "verification_required", "confidence_and_limitations", "image_references", "provider_metadata", "comparison_summary"],
}


def build_user_prompt(
    question: str,
    images: Sequence[ImageInput],
    *,
    conversation_context: Sequence[Mapping[str, str]] = (),
    comparison_requested: bool = False,
    engineering_context: Mapping[str, str] | None = None,
) -> str:
    prompt = ["User question:", question.strip()]
    prompt.append("\nImage references in supplied order (identifiers only; copy no metadata into the response):")
    for index, image in enumerate(images, start=1):
        prompt.append(
            f"Image {index}: label={image.label or image.filename!r}; filename={image.filename!r}; "
            f"pixel dimensions={image.width}x{image.height}. Do not infer the image's scale or layer stack."
        )
    context = {key: str(value).strip() for key, value in (engineering_context or {}).items() if str(value).strip()}
    if context:
        prompt.append("\nUser-supplied circuit context (unknown fields remain unknown):")
        for key, value in context.items():
            prompt.append(f"- {key.replace('_', ' ')}: {value}")
    if comparison_requested:
        prompt.append(
            "\nThe user explicitly requested comparison. Use the supplied labels to identify images and include only visible differences. "
            "State whether differences cannot be compared reliably because scale, orientation, crop, or layer visibility may differ. "
            "Do not imply electrical superiority."
        )
    else:
        prompt.append("\nDo not compare the images unless the user's question explicitly requests a comparison.")
    recent = [
        item for item in conversation_context[-8:]
        if item.get("role") in {"user", "assistant"} and str(item.get("content", "")).strip()
    ]
    if recent:
        prompt.append("\nRecent conversation text for context only (not evidence):")
        prompt.extend(f"{item.get('role', 'user')}: {str(item.get('content', ''))[:4000]}" for item in recent)
    return "\n".join(prompt)


def validate_request_size(payload: Mapping[str, Any], config: VisionProviderConfig) -> None:
    size = len(json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
    if size > config.max_payload_bytes:
        limit_mb = config.max_payload_bytes / (1024 * 1024)
        actual_mb = size / (1024 * 1024)
        raise VisionAnalysisError(
            f"The encoded request is {actual_mb:.1f} MB, above the configured {limit_mb:.1f} MB payload limit. "
            "No image was sent; reduce image count or payload size and try again."
        )


def post_json(
    endpoint: str,
    *,
    api_key: str,
    provider: str,
    payload: Mapping[str, Any],
    config: VisionProviderConfig,
) -> tuple[dict[str, Any], int, float]:
    """POST once, normalize failures, and never expose response bodies or credentials."""
    validate_request_size(payload, config)
    headers = {"Content-Type": "application/json"}
    if provider == "gemini":
        headers["x-goog-api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.perf_counter()
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=config.timeout_seconds)
    except requests.Timeout as exc:
        raise VisionAnalysisError(f"{provider} request timed out. No analysis result was produced.") from exc
    except requests.RequestException as exc:
        raise VisionAnalysisError(f"Could not reach {provider}. No analysis result was produced.") from exc
    elapsed = (time.perf_counter() - started) * 1000
    status = response.status_code
    if status in {401, 403}:
        message = "authentication failed; verify the API key and account access"
    elif status == 404:
        message = "the model or API endpoint was not found; verify the configured model ID"
    elif status == 429:
        message = "rate limit or quota reached; wait and check provider usage limits"
    elif status in {408, 504}:
        message = "request timed out"
    elif status >= 500:
        message = "provider service is unavailable"
    elif not 200 <= status < 300:
        message = "request was rejected; verify the model and request configuration"
    else:
        message = ""
    if message:
        raise VisionAnalysisError(f"{provider} {message} (HTTP {status}). No analysis result was produced.")
    try:
        data = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise VisionAnalysisError(f"{provider} returned malformed JSON. No analysis result was produced.") from exc
    if not isinstance(data, dict):
        raise VisionAnalysisError(f"{provider} returned an unsupported response. No analysis result was produced.")
    return data, int(status), elapsed


def extract_json_object(text: Any, provider: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise VisionAnalysisError(f"{provider} returned no structured analysis. No result was produced.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VisionAnalysisError(f"{provider} returned malformed structured analysis. No result was produced.") from exc
    if not isinstance(payload, dict):
        raise VisionAnalysisError(f"{provider} returned an invalid structured response. No result was produced.")
    return payload


def make_provider_metadata(
    config: VisionProviderConfig,
    *,
    provider: str,
    response: Mapping[str, Any],
    elapsed_ms: float,
) -> dict[str, Any]:
    usage_raw = response.get("usage")
    usage_raw = usage_raw if isinstance(usage_raw, Mapping) else {}
    if provider == "gemini":
        input_tokens = usage_raw.get("total_input_tokens")
        output_tokens = usage_raw.get("total_output_tokens")
        total_tokens = usage_raw.get("total_tokens")
        thought_tokens = usage_raw.get("total_thought_tokens")
        cached_tokens = usage_raw.get("total_cached_tokens")
    else:
        input_tokens = usage_raw.get("input_tokens")
        output_tokens = usage_raw.get("output_tokens")
        total_tokens = usage_raw.get("total_tokens")
        details = usage_raw.get("input_tokens_details")
        cached_tokens = details.get("cached_tokens") if isinstance(details, Mapping) else None
        thought_tokens = None
    usage: dict[str, int] = {}
    for key, value in (
        ("input_tokens", input_tokens), ("output_tokens", output_tokens),
        ("total_tokens", total_tokens), ("thought_tokens", thought_tokens),
        ("cached_tokens", cached_tokens),
    ):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            usage[key] = value
    cost: float | None = None
    if (
        config.input_price_per_million is not None
        and config.output_price_per_million is not None
        and "input_tokens" in usage and "output_tokens" in usage
        and not usage.get("cached_tokens", 0) and not usage.get("thought_tokens", 0)
    ):
        cost = round(
            (usage["input_tokens"] * config.input_price_per_million
             + usage["output_tokens"] * config.output_price_per_million) / 1_000_000,
            8,
        )
    return {
        "provider": config.public_provider_name,
        "provider_id": provider,
        "model": config.model,
        "usage": usage,
        "estimated_cost_usd": cost,
        "cost_basis": "explicitly configured per-million token rates" if cost is not None else None,
        "latency_ms": round(elapsed_ms, 1),
        "request_id": response.get("id") if isinstance(response.get("id"), str) else None,
        "store_requested": False,
        "connectivity_verified_by_request": True,
    }


def complete_analysis(
    text: Any,
    *,
    config: VisionProviderConfig,
    provider: str,
    response: Mapping[str, Any],
    elapsed_ms: float,
    images: Sequence[ImageInput],
    comparison_requested: bool,
) -> VisualAnalysis:
    payload = extract_json_object(text, provider)
    metadata = make_provider_metadata(config, provider=provider, response=response, elapsed_ms=elapsed_ms)
    return VisualAnalysis.from_payload(
        payload, comparison_requested=comparison_requested, images=images,
        provider_metadata=metadata,
    )
