"""Session-scoped image validation and provider-independent layout analysis."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

import requests
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg"}
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_MAX_IMAGES = 4
DEFAULT_MAX_REQUEST_BYTES = 18 * 1024 * 1024
DEFAULT_MAX_OUTPUT_TOKENS = 3000
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_OPENAI_MODEL = "gpt-6-astra"
GEMINI_INTERACTIONS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"


class ImageValidationError(ValueError):
    """An uploaded file is not a supported, safe-to-inspect image."""


class VisionAnalysisError(RuntimeError):
    """The requested visual analysis could not be completed reliably."""


@dataclass(frozen=True)
class ImageInput:
    image_id: str
    filename: str
    mime_type: str
    data: bytes = field(repr=False)
    width: int
    height: int
    label: str = ""


@dataclass(frozen=True)
class ImageReference:
    image_id: str
    filename: str
    label: str


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    thought_tokens: int | None = None
    cached_tokens: int | None = None


@dataclass(frozen=True)
class PotentialIssue:
    description: str
    supporting_visual_observation: str
    electrical_or_physical_relevance: str
    severity_assessment: str
    confidence: str
    suggested_investigation: str

    @property
    def issue(self) -> str:  # Stage 5 API compatibility.
        return self.description

    @property
    def evidence(self) -> str:  # Stage 5 API compatibility.
        return self.supporting_visual_observation


@dataclass(frozen=True)
class Recommendation:
    proposed_change: str
    reason: str
    supporting_observation: str
    expected_benefit: str
    possible_tradeoff: str
    required_validation: str

    @property
    def recommendation(self) -> str:  # Stage 5 API compatibility.
        return self.proposed_change

    @property
    def verification_required(self) -> str:  # Stage 5 API compatibility.
        return self.required_validation


@dataclass(frozen=True)
class VisualAnalysis:
    visual_observations: tuple[str, ...]
    potential_issues: tuple[PotentialIssue, ...]
    recommended_modifications: tuple[Recommendation, ...]
    expected_benefits: tuple[str, ...]
    possible_tradeoffs: tuple[str, ...]
    verification_required: tuple[str, ...]
    confidence_and_limitations: tuple[str, ...]
    image_references: tuple[ImageReference, ...]
    provider_metadata: Mapping[str, Any]
    comparison_summary: str | None = None
    summary: str = ""

    @property
    def what_i_can_see(self) -> tuple[str, ...]:  # Stage 5 API compatibility.
        return self.visual_observations

    @property
    def what_needs_verification(self) -> tuple[str, ...]:  # Stage 5 API compatibility.
        return self.verification_required

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        comparison_requested: bool = False,
        images: Sequence[ImageInput] = (),
        provider_metadata: Mapping[str, Any] | None = None,
    ) -> "VisualAnalysis":
        """Validate provider JSON; never fill missing engineering findings with guesses."""
        if not isinstance(payload, dict):
            raise VisionAnalysisError("The vision provider returned an invalid structured response.")
        required_keys = {
            "summary", "visual_observations", "potential_issues", "recommended_modifications",
            "expected_benefits", "possible_tradeoffs", "verification_required",
            "confidence_and_limitations", "image_references", "provider_metadata",
            "comparison_summary",
        }
        if set(payload) != required_keys:
            raise VisionAnalysisError("The vision provider response does not match the required SiliconPilot schema.")
        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise VisionAnalysisError("The vision provider response is missing a valid 'summary' section.")

        def string_list(key: str) -> tuple[str, ...]:
            value = payload.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
                raise VisionAnalysisError(f"The vision provider response is missing a valid '{key}' section.")
            return tuple(item.strip() for item in value)

        issue_rows = payload.get("potential_issues")
        if not isinstance(issue_rows, list):
            raise VisionAnalysisError("The vision provider response is missing a valid 'potential_issues' section.")
        issue_keys = (
            "description", "supporting_visual_observation", "electrical_or_physical_relevance",
            "severity_assessment", "confidence", "suggested_investigation",
        )
        issues: list[PotentialIssue] = []
        for row in issue_rows:
            if not isinstance(row, dict) or set(row) != set(issue_keys) or any(
                not isinstance(row.get(key), str) or not row[key].strip() for key in issue_keys
            ):
                raise VisionAnalysisError("The vision provider returned an incomplete potential-issue item.")
            if row["severity_assessment"].strip().lower() not in {"low", "medium", "high"}:
                raise VisionAnalysisError("The vision provider returned an invalid AI-assessed severity category.")
            if row["confidence"].strip().lower() not in {"low", "medium", "high"}:
                raise VisionAnalysisError("The vision provider returned an invalid qualitative confidence label.")
            issues.append(PotentialIssue(*(row[key].strip() for key in issue_keys)))

        recommendation_rows = payload.get("recommended_modifications")
        recommendation_keys = (
            "proposed_change", "reason", "supporting_observation", "expected_benefit",
            "possible_tradeoff", "required_validation",
        )
        if not isinstance(recommendation_rows, list):
            raise VisionAnalysisError("The vision provider response is missing a valid 'recommended_modifications' section.")
        recommendations: list[Recommendation] = []
        for row in recommendation_rows:
            if not isinstance(row, dict) or set(row) != set(recommendation_keys) or any(
                not isinstance(row.get(key), str) or not row[key].strip() for key in recommendation_keys
            ):
                raise VisionAnalysisError("The vision provider returned an incomplete recommendation item.")
            recommendations.append(Recommendation(*(row[key].strip() for key in recommendation_keys)))

        comparison = payload.get("comparison_summary")
        if comparison_requested and (not isinstance(comparison, str) or not comparison.strip()):
            raise VisionAnalysisError("The provider did not return the requested image-comparison section.")
        if comparison is not None and not isinstance(comparison, str):
            raise VisionAnalysisError("The vision provider returned an invalid comparison section.")

        raw_refs = payload.get("image_references")
        if not isinstance(raw_refs, list):
            raise VisionAnalysisError("The vision provider response is missing a valid 'image_references' section.")
        if any(
            not isinstance(row, dict)
            or set(row) != {"image_id", "filename", "label"}
            or any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("image_id", "filename", "label"))
            for row in raw_refs
        ):
            raise VisionAnalysisError("The vision provider returned an invalid image reference.")
        # Filenames, IDs, and labels come from the local upload records, never model claims.
        if images:
            references = tuple(
                ImageReference(item.image_id, item.filename, item.label or item.filename) for item in images
            )
        else:
            references_list: list[ImageReference] = []
            for row in raw_refs:
                if not isinstance(row, dict) or any(
                    not isinstance(row.get(key), str) or not row[key].strip()
                    for key in ("image_id", "filename", "label")
                ):
                    raise VisionAnalysisError("The vision provider returned an invalid image reference.")
                references_list.append(ImageReference(row["image_id"], row["filename"], row["label"]))
            references = tuple(references_list)

        raw_metadata = payload.get("provider_metadata")
        if not isinstance(raw_metadata, dict) or set(raw_metadata) != {"note"} or not isinstance(raw_metadata.get("note"), str):
            raise VisionAnalysisError("The vision provider response is missing valid provider metadata.")

        return cls(
            summary=summary.strip(),
            visual_observations=string_list("visual_observations"),
            potential_issues=tuple(issues),
            recommended_modifications=tuple(recommendations),
            expected_benefits=string_list("expected_benefits"),
            possible_tradeoffs=string_list("possible_tradeoffs"),
            verification_required=string_list("verification_required"),
            confidence_and_limitations=string_list("confidence_and_limitations"),
            image_references=references,
            provider_metadata=dict(provider_metadata) if provider_metadata is not None else raw_metadata,
            comparison_summary=comparison.strip() if comparison else None,
        )

    def to_markdown(self, provider_name: str, *, comparison_requested: bool = False) -> str:
        sections = ["### 1. Summary", self.summary or "Unavailable; no summary was returned.", "", "### 2. Visual observations"]
        sections.extend(f"- **Direct visual observation:** {item}" for item in self.visual_observations)
        if not self.visual_observations:
            sections.append("- No visual observations were returned.")
        sections.extend(["", "### 3. Potential issues (AI assessment only)"])
        if self.potential_issues:
            for issue in self.potential_issues:
                sections.append(
                    f"- **Possible concern:** {issue.description}\n"
                    f"  - Supporting visual observation: {issue.supporting_visual_observation}\n"
                    f"  - Possible electrical or physical relevance: {issue.electrical_or_physical_relevance}\n"
                    f"  - AI-assessed severity category (not a verified violation): {issue.severity_assessment}\n"
                    f"  - Qualitative confidence: {issue.confidence}\n"
                    f"  - Suggested investigation: {issue.suggested_investigation}"
                )
        else:
            sections.append("- No specific potential issue was identified from the supplied images.")
        sections.append("\n### 4. Recommended modifications")
        if self.recommended_modifications:
            for item in self.recommended_modifications:
                sections.append(
                    f"- **Proposed change:** {item.proposed_change}\n"
                    f"  - Reason: {item.reason}\n"
                    f"  - Supporting observation: {item.supporting_observation}\n"
                    f"  - Expected benefit: {item.expected_benefit}\n"
                    f"  - Possible trade-off: {item.possible_tradeoff}\n"
                    f"  - Required validation: {item.required_validation}"
                )
        else:
            sections.append("- No modification is recommended from the available visual evidence.")
        sections.extend(["\n### 5. Expected benefits"])
        sections.extend(f"- {item}" for item in self.expected_benefits)
        if not self.expected_benefits:
            sections.append("- Unavailable; no benefit was returned.")
        sections.append("\n### 6. Possible trade-offs")
        sections.extend(f"- {item}" for item in self.possible_tradeoffs)
        if not self.possible_tradeoffs:
            sections.append("- Unavailable; no trade-off was returned.")
        sections.append("\n### 7. Verification required")
        sections.extend(f"- {item}" for item in self.verification_required)
        if not self.verification_required:
            sections.append("- No verification steps were returned.")
        sections.append("\n### 8. Confidence and limitations")
        sections.extend(f"- {item}" for item in self.confidence_and_limitations)
        sections.extend([
            "- Confidence labels are qualitative provider estimates, not calibrated probabilities.",
            "- A screenshot cannot establish DRC/LVS, connectivity, exact dimensions, parasitic extraction, electrical behavior, SPICE results, or measured silicon results.",
            "- No quarantined DRC classifier, restricted geometry predictor, electrical predictor, or PEX model was used to produce this visual analysis.",
        ])
        if comparison_requested:
            sections.extend(["\n### Image comparison", self.comparison_summary or "Comparison was unavailable."])
            sections.append("- Screenshots may differ in scale, orientation, crop, or visible layers; visible improvement does not establish electrical superiority.")
        sections.append("\n### 9. Image references")
        sections.extend(f"- `{item.label}` — `{item.filename}` (image ID `{item.image_id}`)" for item in self.image_references)
        sections.append("\n### 10. Provider metadata")
        sections.append(f"- Provider: `{self.provider_metadata.get('provider') or provider_name}`")
        sections.append(f"- Model: `{self.provider_metadata.get('model') or 'unavailable'}`")
        usage = self.provider_metadata.get("usage") or {}
        if usage:
            sections.append(
                "- Token usage: "
                f"input `{usage.get('input_tokens', 'unavailable')}`, "
                f"output `{usage.get('output_tokens', 'unavailable')}`, "
                f"total `{usage.get('total_tokens', 'unavailable')}`"
            )
        else:
            sections.append("- Token usage: unavailable from provider response.")
        cost = self.provider_metadata.get("estimated_cost_usd")
        if isinstance(cost, (int, float)):
            sections.append(f"- Estimated cost: `${cost:.6f}` using explicitly configured token rates; this is an estimate, not provider billing data.")
        else:
            sections.append("- Cost: not estimated; required account pricing and/or usage detail is unavailable.")
        latency = self.provider_metadata.get("latency_ms")
        if isinstance(latency, (int, float)):
            sections.append(f"- Request latency: `{latency:.0f} ms` (observed client-side).")
        sections.append("- Source: provider-generated visual analysis; no SiliconPilot ML prediction, SPICE run, DRC/LVS check, extraction, or physical measurement was performed.")
        return "\n".join(sections)


def validate_uploaded_image(upload: Any, *, max_bytes: int = MAX_IMAGE_BYTES) -> ImageInput:
    """Read and validate one Streamlit UploadedFile without writing it to disk."""
    filename = Path(str(getattr(upload, "name", ""))).name
    if not filename or Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ImageValidationError("Choose a PNG, JPG, or JPEG image.")
    try:
        data = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    except Exception as exc:
        raise ImageValidationError("The uploaded file could not be read.") from exc
    if not isinstance(data, bytes) or not data:
        raise ImageValidationError("The uploaded image is empty or unreadable.")
    if len(data) > max_bytes:
        raise ImageValidationError(f"Image exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    try:
        with Image.open(__import__("io").BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("The file is not a readable image or is corrupted.") from exc
    if image_format not in SUPPORTED_FORMATS:
        raise ImageValidationError("Only PNG, JPG, and JPEG images are supported.")
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise ImageValidationError("Image dimensions are invalid or exceed the safe pixel limit.")
    actual_mime = SUPPORTED_FORMATS[image_format]
    declared_mime = str(getattr(upload, "type", "") or "").lower()
    if declared_mime and declared_mime not in {actual_mime, "application/octet-stream"}:
        raise ImageValidationError("The uploaded file type does not match its image content.")
    suffix = Path(filename).suffix.lower()
    if (image_format == "PNG" and suffix != ".png") or (image_format == "JPEG" and suffix not in {".jpg", ".jpeg"}):
        raise ImageValidationError("The filename extension does not match the image content.")
    image_id = hashlib.sha256(filename.encode("utf-8") + b"\0" + data).hexdigest()[:20]
    return ImageInput(image_id, filename, actual_mime, data, width, height)


def remove_image_from_selection(
    image_id: str,
    selected_ids: Sequence[str],
    message_history: Sequence[Mapping[str, Any]],
) -> tuple[list[str], bool]:
    """Deselect an image and report whether its bytes are still referenced by chat."""
    updated = [item for item in selected_ids if item != image_id]
    referenced = any(image_id in message.get("image_ids", []) for message in message_history)
    return updated, referenced


def clear_image_selection(
    selected_ids: Sequence[str], message_history: Sequence[Mapping[str, Any]]
) -> tuple[list[str], set[str]]:
    referenced = {image_id for message in message_history for image_id in message.get("image_ids", [])}
    return [], referenced


def clear_all_image_data(
    image_assets: Mapping[str, Any],
    selected_ids: Sequence[str],
    image_labels: Mapping[str, str],
    message_history: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str], dict[str, str], list[dict[str, Any]]]:
    """Remove session image bytes and image-linked messages while retaining text-only chat."""
    cleaned_history = [
        dict(message)
        for message in message_history
        if not message.get("image_ids") and not message.get("source_image_ids")
    ]
    return {}, [], {}, cleaned_history


def provider_consent_granted(consents: Mapping[str, Any], provider: str | None) -> bool:
    """Consent is scoped to the chosen provider and does not follow provider switches."""
    return bool(provider and consents.get(provider) is True)


def _positive_int(raw: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, value))


def _optional_rate(raw: Any) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


@dataclass(frozen=True)
class VisionProviderConfig:
    provider: str | None
    model: str | None
    endpoint: str
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 45.0
    max_images: int = DEFAULT_MAX_IMAGES
    max_payload_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    image_detail: str = "original"

    @property
    def is_configured(self) -> bool:
        if self.provider not in {"gemini", "openai", "openai_compatible"}:
            return False
        return bool(self.model and self.api_key)

    @property
    def public_provider_name(self) -> str:
        if self.provider == "gemini":
            return "Google Gemini API"
        if self.provider == "openai":
            return "OpenAI API"
        if self.provider == "openai_compatible":
            host = urlsplit(self.endpoint).hostname or "configured endpoint"
            return f"OpenAI-compatible provider at {host}"
        return "No vision provider configured"


def _read_setting(
    env: Mapping[str, str], section: Mapping[str, Any], env_name: str, *secret_names: str
) -> str | None:
    raw = env.get(env_name)
    if raw is None:
        for name in secret_names:
            raw = section.get(name)
            if raw is not None:
                break
    return str(raw).strip() if raw is not None and str(raw).strip() else None


def resolve_provider_config(
    environ: Mapping[str, str] | None = None,
    secrets: Mapping[str, Any] | None = None,
    *,
    provider: str | None = None,
) -> VisionProviderConfig:
    """Resolve one provider, with process environment values taking precedence."""
    env = os.environ if environ is None else environ
    section: Mapping[str, Any] = {}
    if secrets and isinstance(secrets.get("vision_assistant", {}), Mapping):
        section = secrets.get("vision_assistant", {})
    configured_name = provider or _read_setting(env, section, "SILICONPILOT_VISION_PROVIDER", "provider")
    name = configured_name.lower().replace("-", "_") if configured_name else None
    if name == "google":
        name = "gemini"

    if name == "gemini":
        model = _read_setting(env, section, "SILICONPILOT_GEMINI_MODEL", "gemini_model") or DEFAULT_GEMINI_MODEL
        key = _read_setting(env, section, "GEMINI_API_KEY", "gemini_api_key")
        endpoint = GEMINI_INTERACTIONS_ENDPOINT
        input_rate = _optional_rate(_read_setting(env, section, "SILICONPILOT_GEMINI_INPUT_USD_PER_MILLION_TOKENS", "gemini_input_price_per_million"))
        output_rate = _optional_rate(_read_setting(env, section, "SILICONPILOT_GEMINI_OUTPUT_USD_PER_MILLION_TOKENS", "gemini_output_price_per_million"))
    elif name == "openai":
        model = _read_setting(env, section, "SILICONPILOT_OPENAI_MODEL", "openai_model") or DEFAULT_OPENAI_MODEL
        key = _read_setting(env, section, "OPENAI_API_KEY", "openai_api_key")
        endpoint = OPENAI_RESPONSES_ENDPOINT
        input_rate = _optional_rate(_read_setting(env, section, "SILICONPILOT_OPENAI_INPUT_USD_PER_MILLION_TOKENS", "openai_input_price_per_million"))
        output_rate = _optional_rate(_read_setting(env, section, "SILICONPILOT_OPENAI_OUTPUT_USD_PER_MILLION_TOKENS", "openai_output_price_per_million"))
    elif name == "openai_compatible":
        model = _read_setting(env, section, "SILICONPILOT_VISION_MODEL", "model")
        key = _read_setting(env, section, "OPENAI_API_KEY", "api_key")
        endpoint = _read_setting(env, section, "SILICONPILOT_VISION_API_URL", "api_url") or "https://api.openai.com/v1/chat/completions"
        input_rate = _optional_rate(_read_setting(env, section, "SILICONPILOT_VISION_INPUT_USD_PER_MILLION_TOKENS", "input_price_per_million"))
        output_rate = _optional_rate(_read_setting(env, section, "SILICONPILOT_VISION_OUTPUT_USD_PER_MILLION_TOKENS", "output_price_per_million"))
    else:
        return VisionProviderConfig(None, None, "", None)

    try:
        timeout = float(env.get("SILICONPILOT_VISION_TIMEOUT_SECONDS", section.get("timeout_seconds", 45)))
    except (TypeError, ValueError):
        timeout = 45.0
    if not math.isfinite(timeout) or timeout <= 0:
        timeout = 45.0
    max_images = _positive_int(env.get("SILICONPILOT_VISION_MAX_IMAGES", section.get("max_images")), DEFAULT_MAX_IMAGES, 1, 10)
    payload_limit_maximum = 19 if name == "gemini" else 100
    max_payload = _positive_int(
        env.get("SILICONPILOT_VISION_MAX_PAYLOAD_MB", section.get("max_payload_mb")),
        18,
        1,
        payload_limit_maximum,
    ) * 1024 * 1024
    max_output_tokens = _positive_int(
        env.get("SILICONPILOT_VISION_MAX_OUTPUT_TOKENS", section.get("max_output_tokens")),
        DEFAULT_MAX_OUTPUT_TOKENS,
        128,
        32768,
    )
    detail = str(env.get("SILICONPILOT_OPENAI_IMAGE_DETAIL", section.get("openai_image_detail", "original"))).lower()
    if detail not in {"auto", "low", "high", "original"}:
        detail = "original"
    return VisionProviderConfig(
        name, model, endpoint, key, timeout, max_images, max_payload,
        max_output_tokens, input_rate, output_rate, detail,
    )


def resolve_configured_provider_configs(
    environ: Mapping[str, str] | None = None,
    secrets: Mapping[str, Any] | None = None,
) -> dict[str, VisionProviderConfig]:
    """Return only providers with a model and credential configured."""
    env = os.environ if environ is None else environ
    names = ("gemini", "openai", "openai_compatible")
    configs = {name: resolve_provider_config(env, secrets, provider=name) for name in names}
    return {name: config for name, config in configs.items() if config.is_configured}


class VisualAnalysisProvider(Protocol):
    @property
    def display_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def analyze(
        self,
        images: Sequence[ImageInput],
        question: str,
        *,
        conversation_context: Sequence[Mapping[str, str]] = (),
        comparison_requested: bool = False,
        engineering_context: Mapping[str, str] | None = None,
    ) -> VisualAnalysis: ...


class OpenAICompatibleVisionProvider:
    """Backward-compatible adapter for the original OpenAI-compatible endpoint."""

    def __init__(self, config: VisionProviderConfig) -> None:
        if config.provider != "openai_compatible" or not config.is_configured:
            raise VisionAnalysisError("Compatible provider settings are incomplete.")
        self.config = config

    @property
    def display_name(self) -> str:
        return self.config.public_provider_name

    @property
    def model_name(self) -> str:
        return str(self.config.model)

    def analyze(
        self,
        images: Sequence[ImageInput],
        question: str,
        *,
        conversation_context: Sequence[Mapping[str, str]] = (),
        comparison_requested: bool = False,
        engineering_context: Mapping[str, str] | None = None,
    ) -> VisualAnalysis:
        import base64
        import json
        import re

        from .providers.base import (
            ANALYSIS_INSTRUCTIONS, ENGINEERING_RESPONSE_SCHEMA, build_user_prompt,
            complete_analysis, post_json,
        )

        if not images:
            raise VisionAnalysisError("Select at least one image before requesting visual analysis.")
        if not question.strip():
            raise VisionAnalysisError("Enter a question for the selected image or images.")
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": build_user_prompt(
                question, images, conversation_context=conversation_context,
                comparison_requested=comparison_requested,
                engineering_context=engineering_context,
            ),
        }]
        for index, image in enumerate(images, start=1):
            content.append({
                "type": "text",
                "text": f"Image {index} follows; label={image.label or image.filename!r}; filename={image.filename!r}.",
            })
            encoded = base64.b64encode(image.data).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image.mime_type};base64,{encoded}", "detail": self.config.image_detail},
            })
        request_data = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": ANALYSIS_INSTRUCTIONS + "\nJSON schema:\n" + json.dumps(ENGINEERING_RESPONSE_SCHEMA),
                },
                {"role": "user", "content": content},
            ],
            "max_tokens": self.config.max_output_tokens,
        }
        response, _, elapsed_ms = post_json(
            self.config.endpoint, api_key=str(self.config.api_key), provider="openai_compatible",
            payload=request_data, config=self.config,
        )
        try:
            content_text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionAnalysisError("Compatible provider returned an unsupported response. No analysis result was produced.") from exc
        if not isinstance(content_text, str):
            raise VisionAnalysisError("Compatible provider returned no structured text. No analysis result was produced.")
        content_text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content_text, flags=re.IGNORECASE)
        return complete_analysis(
            content_text, config=self.config, provider="openai_compatible", response=response,
            elapsed_ms=elapsed_ms, images=images, comparison_requested=comparison_requested,
        )


class VisionAnalysisService:
    """Provider-independent service. Provider errors never trigger fallback."""

    def __init__(self, config: VisionProviderConfig, provider: VisualAnalysisProvider | None = None) -> None:
        self.config = config
        self.provider = provider
        if provider is None and config.is_configured:
            from .providers.provider_factory import build_provider
            self.provider = build_provider(config)

    @property
    def is_available(self) -> bool:
        return self.provider is not None

    @property
    def provider_name(self) -> str:
        return self.provider.display_name if self.provider else self.config.public_provider_name

    @property
    def model_name(self) -> str:
        return self.provider.model_name if self.provider else (self.config.model or "unconfigured")

    @property
    def unavailable_reason(self) -> str:
        if self.provider is not None:
            return ""
        provider = self.config.provider
        if not provider:
            return "No vision provider configured. Configure Gemini or OpenAI API credentials and select a provider."
        missing: list[str] = []
        if not self.config.model:
            missing.append("a model ID")
        if not self.config.api_key:
            missing.append("an API key")
        if missing:
            return f"The {provider} provider is missing " + " and ".join(missing) + "."
        return f"The configured vision provider '{provider}' is unsupported."

    def analyze(
        self,
        images: Sequence[ImageInput],
        question: str,
        *,
        conversation_context: Sequence[Mapping[str, str]] = (),
        comparison_requested: bool = False,
        engineering_context: Mapping[str, str] | None = None,
    ) -> VisualAnalysis:
        if self.provider is None:
            raise VisionAnalysisError(self.unavailable_reason + " No image analysis was performed.")
        if len(images) > self.config.max_images:
            raise VisionAnalysisError(f"This provider is configured for at most {self.config.max_images} images per request.")
        if self.config.max_payload_bytes <= 0:
            raise VisionAnalysisError("Vision provider payload limit is invalid; no image was sent.")
        try:
            return self.provider.analyze(
                images, question, conversation_context=conversation_context,
                comparison_requested=comparison_requested,
                engineering_context=engineering_context or {},
            )
        except VisionAnalysisError:
            raise
        except Exception as exc:
            raise VisionAnalysisError("The vision provider failed unexpectedly. No analysis result was produced.") from exc
