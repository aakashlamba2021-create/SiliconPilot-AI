"""Page 6: session-scoped analog IC engineering assistant."""
from __future__ import annotations

from io import BytesIO
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

import streamlit as st

from dashboard.chat import SiliconPilotChatbot
from dashboard.components.header import render_header
from src.assistant.vision import (
    ImageInput,
    ImageValidationError,
    VisionAnalysisError,
    VisionAnalysisService,
    clear_all_image_data,
    clear_image_selection,
    remove_image_from_selection,
    resolve_configured_provider_configs,
    resolve_provider_config,
    validate_uploaded_image,
)

MAX_UPLOAD_MB = 10
PROVIDER_LABELS = {
    "gemini": "Google Gemini API",
    "openai": "OpenAI API",
    "openai_compatible": "OpenAI-compatible endpoint",
}
COMPARISON_REQUEST = re.compile(
    r"\b(compare|comparison|versus|vs\.?|before\s*(and|/)\s*after|original\s+(and|versus)\s+modified)\b",
    re.IGNORECASE,
)
VISUAL_ANALYSIS_INTENT = re.compile(
    r"\b(analy[sz]e|inspect|review|look\s+at|what\s+do\s+you\s+see|visual\s+comparison|compare\s+(?:these|the|both)\s+(?:images|layouts|screenshots))\b",
    re.IGNORECASE,
)
QUICK_PROMPTS = {
    "How to layout a Differential Pair?": "How do I layout a differential pair?",
    "Explain DRC vs LVS verification": "Explain DRC vs LVS",
    "How to reduce Parasitic Capacitance?": "How to reduce parasitic capacitance?",
    "Show Trained Model Registry": "What models are currently trained?",
}


def is_comparison_request(query: str) -> bool:
    return bool(COMPARISON_REQUEST.search(query))


def image_request_consent_key(
    provider: str,
    model: str,
    question: str,
    image_ids: list[str],
    image_labels: dict[str, str],
    engineering_context: dict[str, str],
    compare_mode: bool,
    nonce: int,
) -> str:
    """Bind one-time consent to the exact outgoing content and next request nonce."""
    payload = json.dumps({
        "provider": provider,
        "model": model,
        "question": question,
        "image_ids": image_ids,
        "image_labels": image_labels,
        "engineering_context": engineering_context,
        "compare_mode": compare_mode,
        "nonce": nonce,
    }, sort_keys=True, ensure_ascii=False)
    return f"assistant_consent_{provider}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def image_request_is_authorized(
    explicit_submit: bool,
    image_ids: list[str],
    provider_ids: list[str],
    consented_provider_ids: set[str],
) -> bool:
    """Require a dedicated submit action, images, and consent for every target."""
    return bool(
        explicit_submit
        and image_ids
        and provider_ids
        and set(provider_ids).issubset(consented_provider_ids)
    )


def artifact_report_key(filename: str, data: bytes, artifact_kind: str) -> str:
    """Fingerprint the exact file and parser selection that produced a report."""
    digest = hashlib.sha256(data).hexdigest()
    return f"{artifact_kind}:{filename}:{digest}"


def provider_findings_context(provider_results: dict[str, dict[str, Any]]) -> str:
    """Store concise prior vision findings so later image questions have useful context."""
    sections = ["Unverified AI visual findings from the prior image request:"]
    for provider_id, result in provider_results.items():
        analysis = result.get("analysis")
        if analysis is None:
            if result.get("error"):
                sections.append(f"{provider_id} ({result.get('status', 'unknown')}): {result['error']}")
            continue
        sections.append(f"{provider_id} ({result.get('model') or 'model unavailable'}): {analysis.summary}")
        sections.extend(f"Visible observation: {value}" for value in analysis.visual_observations[:4])
        sections.extend(
            f"Potential issue: {issue.description} (qualitative confidence {issue.confidence})"
            for issue in analysis.potential_issues[:3]
        )
        sections.extend(
            f"Recommendation: {item.proposed_change}; validate with: {item.required_validation}"
            for item in analysis.recommended_modifications[:3]
        )
    sections.append("These are image-based hypotheses, not DRC/LVS, connectivity, measurement, or signoff evidence.")
    return "\n".join(sections)


def compare_provider_findings(results: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Compare exact wording overlap only; this is not a correctness or consensus score."""
    completed = {
        name: item["analysis"] for name, item in results.items()
        if item.get("analysis") is not None
    }
    if len(completed) < 2:
        return {"agreements": [], "provider_specific": []}
    observation_sets = {
        name: {value.strip().casefold(): value for value in analysis.visual_observations}
        for name, analysis in completed.items()
    }
    common = set.intersection(*(set(values) for values in observation_sets.values()))
    agreements = [observation_sets[next(iter(observation_sets))][key] for key in sorted(common)]
    provider_specific = []
    for name, analysis in completed.items():
        for value in analysis.visual_observations:
            if value.strip().casefold() not in common:
                provider_specific.append(f"{name}: {value}")
        for item in analysis.recommended_modifications:
            provider_specific.append(f"{name} recommendation: {item.proposed_change}")
    return {"agreements": agreements, "provider_specific": provider_specific}


def build_analysis_report(
    question: str,
    images: list[ImageInput],
    provider_results: dict[str, dict[str, Any]],
    *,
    comparison: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build an exportable report without embedding image bytes or credentials."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "images": [
            {"image_id": image.image_id, "filename": image.filename, "label": image.label,
             "width": image.width, "height": image.height, "mime_type": image.mime_type}
            for image in images
        ],
        "provider_results": {
            provider: {
                "model": item.get("model"),
                "status": item.get("status"),
                "error": item.get("error"),
                "analysis": asdict(item["analysis"]) if item.get("analysis") is not None else None,
            }
            for provider, item in provider_results.items()
        },
        "provider_comparison": comparison,
        "limitations": [
            "A screenshot cannot establish DRC/LVS, connectivity, exact dimensions, parasitic extraction, electrical behavior, or measured silicon results.",
            "Provider findings are AI-generated visual assessments and are not calibrated engineering verification.",
        ],
    }


def analysis_report_markdown(report: dict[str, Any]) -> str:
    lines = ["# SiliconPilot AI image analysis", "", f"- Created (UTC): `{report['created_at_utc']}`", f"- Question: {report['question']}", "", "## Images"]
    lines.extend(f"- `{item['label'] or item['filename']}` — `{item['filename']}` ({item['width']} × {item['height']} px)" for item in report["images"])
    for provider, item in report["provider_results"].items():
        lines.extend(["", f"## {provider}: {item.get('model') or 'model unavailable'}", f"Status: {item.get('status', 'unknown')}" ])
        if item.get("error"):
            lines.append(f"Error: {item['error']}")
        analysis = item.get("analysis")
        if analysis:
            lines.append(analysis.get("summary", "Summary unavailable."))
            lines.append("### Visual observations")
            lines.extend(f"- {value}" for value in analysis.get("visual_observations", []))
            lines.append("### Potential issues")
            lines.extend(f"- {issue['description']} (qualitative confidence: {issue['confidence']}; unverified AI assessment)" for issue in analysis.get("potential_issues", []))
            lines.append("### Recommendations")
            lines.extend(f"- {value['proposed_change']} — validate: {value['required_validation']}" for value in analysis.get("recommended_modifications", []))
            lines.append("### Verification required")
            lines.extend(f"- {value}" for value in analysis.get("verification_required", []))
            lines.append("### Confidence and limitations")
            lines.extend(f"- {value}" for value in analysis.get("confidence_and_limitations", []))
            lines.append(f"Provider metadata: `{json.dumps(analysis.get('provider_metadata', {}), ensure_ascii=False)}`")
    if report.get("provider_comparison"):
        lines.extend(["", "## Provider comparison", "Exact text overlap is descriptive only; it is not a correctness score or a provider winner."])
        lines.extend(f"- Agreement: {value}" for value in report["provider_comparison"].get("agreements", []))
        lines.extend(f"- Provider-specific: {value}" for value in report["provider_comparison"].get("provider_specific", []))
    lines.extend(["", "## Limitations", *(f"- {value}" for value in report["limitations"])])
    return "\n".join(lines)


def run_vision_analysis_for_providers(
    configs: dict[str, Any],
    provider_ids: list[str],
    images: list[ImageInput],
    question: str,
    *,
    consented_provider_ids: set[str],
    conversation_context: list[dict[str, str]],
    comparison_requested: bool,
    engineering_context: dict[str, str],
    service_factory=None,
) -> dict[str, dict[str, Any]]:
    """Run explicitly selected providers with identical inputs and provider-scoped consent."""
    factory = service_factory or VisionAnalysisService
    results: dict[str, dict[str, Any]] = {}
    for provider_id in provider_ids:
        config = configs.get(provider_id)
        if provider_id not in consented_provider_ids:
            results[provider_id] = {
                "model": getattr(config, "model", None), "status": "not_sent",
                "analysis": None, "error": "No request sent because provider-specific consent was not granted.",
            }
            continue
        if config is None:
            results[provider_id] = {
                "model": None, "status": "unavailable", "analysis": None,
                "error": "No request sent because this provider is not configured.",
            }
            continue
        provider_service = factory(config)
        try:
            analysis = provider_service.analyze(
                images, question, conversation_context=conversation_context,
                comparison_requested=comparison_requested,
                engineering_context=engineering_context,
            )
            results[provider_id] = {
                "model": provider_service.model_name, "status": "completed",
                "analysis": analysis, "error": None,
            }
        except VisionAnalysisError as exc:
            results[provider_id] = {
                "model": provider_service.model_name, "status": "failed",
                "analysis": None, "error": str(exc),
            }
        except Exception:
            results[provider_id] = {
                "model": provider_service.model_name, "status": "failed",
                "analysis": None, "error": "Provider failed unexpectedly. No analysis result was produced.",
            }
    return results


def _vision_secrets() -> dict[str, Any]:
    """Read optional Streamlit secrets without surfacing secret values."""
    try:
        section = st.secrets.get("vision_assistant", {})
        return {"vision_assistant": dict(section)} if hasattr(section, "items") else {}
    except Exception:
        return {}


def _initialise_state() -> None:
    st.session_state.setdefault(
        "chat_history",
        [{
            "role": "assistant",
            "content": "Hello! Ask an analog-layout question, or upload a layout screenshot and ask what you would like me to inspect.",
            "image_ids": [],
        }],
    )
    st.session_state.setdefault("assistant_image_assets", {})
    st.session_state.setdefault("assistant_selected_image_ids", [])
    st.session_state.setdefault("assistant_image_labels", {})
    st.session_state.setdefault("assistant_upload_widget_version", 0)
    st.session_state.setdefault("assistant_image_question_version", 0)
    st.session_state.setdefault("assistant_request_nonce", 0)
    st.session_state.setdefault("assistant_selected_provider", None)
    st.session_state.setdefault("assistant_last_report", None)
    st.session_state.setdefault("assistant_artifact_report_key", None)


def _asset_record(image: ImageInput) -> dict[str, Any]:
    return {
        "image_id": image.image_id,
        "filename": image.filename,
        "mime_type": image.mime_type,
        "data": image.data,
        "width": image.width,
        "height": image.height,
    }


def _image_from_record(record: dict[str, Any], label: str = "") -> ImageInput:
    return ImageInput(
        image_id=record["image_id"],
        filename=record["filename"],
        mime_type=record["mime_type"],
        data=record["data"],
        width=record["width"],
        height=record["height"],
        label=label,
    )


def _privacy_text(provider: str) -> str:
    if provider == "gemini":
        return (
            "Google Gemini receives the selected image bytes, filenames and labels, your question, the circuit context below, "
            "and up to eight recent conversation messages, including prior assistant findings. SiliconPilot sets `store=false` on each request. The account tier is not detected: "
            "Google says content submitted on the free Gemini API tier may be used to improve its products; paid-tier content is not used for that purpose. "
            "This setting does not promise zero retention or override provider policy."
        )
    if provider == "openai":
        return (
            "OpenAI receives the selected image bytes, filenames and labels, your question, the circuit context below, "
            "and up to eight recent conversation messages, including prior assistant findings. SiliconPilot sets `store=false`. OpenAI states API inputs and outputs are not used for training by default; "
            "standard abuse-monitoring logs may retain content for up to 30 days unless different organization controls apply. "
            "Those controls and your account settings are not verified here; this does not promise zero retention."
        )
    return (
        "The configured compatible endpoint receives selected image bytes, filenames and labels, your question, the circuit context below, "
        "and up to eight recent conversation messages, including prior assistant findings. SiliconPilot cannot verify this endpoint's data use or retention policy; review it before sending proprietary layouts."
    )


def _provider_controls(secrets: dict[str, Any]) -> tuple[VisionAnalysisService | None, str | None]:
    configs = resolve_configured_provider_configs(secrets=secrets)
    preferred = resolve_provider_config(secrets=secrets).provider
    stored_choice = st.session_state.get("assistant_selected_provider")
    if stored_choice not in configs:
        stored_choice = None

    if len(configs) == 1 and not preferred and stored_choice is None:
        initial = next(iter(configs))
    elif preferred in configs and stored_choice is None:
        initial = preferred
    elif stored_choice:
        initial = stored_choice
    else:
        initial = None

    selected_provider: str | None = None
    if configs:
        options: list[str | None] = [None, *configs.keys()]
        index = options.index(initial) if initial in options else 0
        selected_provider = st.selectbox(
            "Vision provider",
            options,
            index=index,
            format_func=lambda value: "Choose a configured provider" if value is None else f"{PROVIDER_LABELS[value]} · {configs[value].model}",
            key="assistant_provider_choice_widget",
            help="Switching providers preserves chat and uploaded images. A failed request never switches providers automatically.",
        )
    st.session_state.assistant_selected_provider = selected_provider

    if selected_provider is None:
        if preferred and preferred not in configs:
            st.warning(
                f"The selected environment provider `{preferred}` is not configured with its required API key. "
                "Choose another configured provider explicitly, or provide the missing key."
            )
        elif len(configs) > 1:
            st.info("Choose which configured provider receives image submissions.")
        else:
            st.warning(
                "Live image analysis is unavailable. Configure `GEMINI_API_KEY` or `OPENAI_API_KEY`, "
                "select a model, then restart Streamlit. Upload previews and text-only chat remain available."
            )
        return None, None

    config = configs[selected_provider]
    service = VisionAnalysisService(config)
    st.caption(
        f"Active provider: **{service.provider_name}** · model `{service.model_name}`. "
        "Configuration is present; API access is not verified until a request succeeds."
    )
    st.info(_privacy_text(selected_provider))
    st.caption(
        f"Per request: up to {config.max_images} images; maximum encoded JSON payload "
        f"{config.max_payload_bytes / (1024 * 1024):.0f} MB; maximum response {config.max_output_tokens} tokens. "
        "API account tier and billing rate are not assumed."
    )
    st.caption("Consent is requested for each specific image question below and resets after submission.")
    return service, selected_provider


def _render_history() -> None:
    assets = st.session_state.assistant_image_assets
    for message in st.session_state.chat_history:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            provider_results = message.get("provider_results")
            if provider_results:
                st.markdown(message.get("content", "Independent provider outputs for the same request:"))
                columns = st.columns(min(2, len(provider_results)))
                for index, (provider, result) in enumerate(provider_results.items()):
                    with columns[index % len(columns)]:
                        st.markdown(f"#### {provider} · {result.get('model', 'model unavailable')}")
                        if result.get("error"):
                            st.error(result["error"])
                        elif result.get("analysis"):
                            st.markdown(result["analysis"].to_markdown(provider, comparison_requested=bool(message.get("comparison_requested"))))
            else:
                st.markdown(message.get("content", ""))
            labels = message.get("image_labels", {})
            for image_id in message.get("image_ids", []):
                image = assets.get(image_id)
                if image:
                    label = labels.get(image_id)
                    caption = f"{label + ' · ' if label else ''}{image['filename']} · {image['width']} × {image['height']} px"
                    st.image(
                        BytesIO(image["data"]), caption=caption, width=260,
                        alt=f"Layout screenshot {image['filename']}",
                    )
                else:
                    st.caption("A referenced image is no longer available in this session.")


def _render_selected_images(max_images: int) -> None:
    selected = st.session_state.assistant_selected_image_ids
    assets = st.session_state.assistant_image_assets
    selected[:] = [image_id for image_id in selected if image_id in assets]
    if not selected:
        return
    st.markdown(f"**Selected images ({len(selected)} of {max_images} max per request)**")
    st.caption("For a comparison, set clear labels such as `Original`, `Modified`, or `Strategy A`.")
    columns = st.columns(min(3, len(selected)))
    for index, image_id in enumerate(list(selected)):
        image = assets[image_id]
        default_label = "Original" if index == 0 else "Modified" if index == 1 else f"Alternative {index + 1}"
        with columns[index % len(columns)]:
            st.image(
                BytesIO(image["data"]),
                caption=f"{image['filename']} · {image['width']} × {image['height']} px",
                width="stretch", alt=f"Selected layout screenshot {image['filename']}",
            )
            label = st.text_input(
                "Provider label",
                value=st.session_state.assistant_image_labels.get(image_id, default_label),
                key=f"assistant_image_label_{image_id}",
                help="This label is sent with the image so a comparison can refer to it by name.",
            )
            st.session_state.assistant_image_labels[image_id] = label.strip() or default_label
            if st.button("Remove from selection", key=f"assistant_remove_{image_id}"):
                updated, referenced = remove_image_from_selection(image_id, selected, st.session_state.chat_history)
                st.session_state.assistant_selected_image_ids = updated
                if not referenced:
                    assets.pop(image_id, None)
                    st.session_state.assistant_image_labels.pop(image_id, None)
                st.session_state.assistant_upload_widget_version += 1
                st.rerun()

    if st.button("Clear current image selection", key="assistant_clear_images"):
        _, referenced = clear_image_selection(selected, st.session_state.chat_history)
        st.session_state.assistant_selected_image_ids = []
        st.session_state.assistant_image_assets = {
            image_id: image for image_id, image in assets.items() if image_id in referenced
        }
        st.session_state.assistant_upload_widget_version += 1
        st.rerun()


def _render_clear_all_images() -> None:
    if not st.session_state.assistant_image_assets:
        return
    if st.button("Clear all uploaded images and image-linked chat content", key="assistant_clear_all_images"):
        assets, selected, labels, history = clear_all_image_data(
            st.session_state.assistant_image_assets,
            st.session_state.assistant_selected_image_ids,
            st.session_state.assistant_image_labels,
            st.session_state.chat_history,
        )
        st.session_state.assistant_image_assets = assets
        st.session_state.assistant_selected_image_ids = selected
        st.session_state.assistant_image_labels = labels
        st.session_state.chat_history = history
        st.session_state.assistant_last_report = None
        st.session_state.assistant_upload_widget_version += 1
        st.rerun()


def _render_reuse_images(max_images: int) -> None:
    assets = st.session_state.assistant_image_assets
    selected = st.session_state.assistant_selected_image_ids
    if not assets:
        return
    referenced = {
        image_id for message in st.session_state.chat_history for image_id in message.get("image_ids", [])
    }
    available = [image_id for image_id in assets if image_id in referenced and image_id not in selected]
    if not available:
        return
    with st.expander("Reuse a previously uploaded image"):
        st.caption("Previously uploaded images stay in this session's chat history. Select one again only when you want it resent.")
        for image_id in available:
            image = assets[image_id]
            if st.button(f"Use {image['filename']} · {image['width']} × {image['height']} px", key=f"assistant_reuse_{image_id}"):
                if len(selected) >= max_images:
                    st.warning(f"Only {max_images} images can be selected for one request.")
                else:
                    st.session_state.assistant_selected_image_ids.append(image_id)
                    st.rerun()


def _engineering_context() -> dict[str, str]:
    with st.expander("Circuit and layout context", expanded=False):
        st.caption("Only values you enter are sent with an image request. Unknown process or device details remain unknown.")
        topology = st.text_input("Circuit topology", key="assistant_context_topology", placeholder="e.g. folded-cascode OTA")
        devices = st.text_input("Device names", key="assistant_context_devices", placeholder="e.g. input pair M1/M2")
        objective = st.text_input("Design objective", key="assistant_context_objective", placeholder="e.g. improve input offset matching")
        process = st.text_input("Process node (if known)", key="assistant_context_process")
        matching = st.text_area("Critical matching requirements", key="assistant_context_matching", height=70)
        concern = st.text_area("Specific concern or question", key="assistant_context_concern", height=70)
    return {
        "circuit_topology": topology,
        "device_names": devices,
        "design_objective": objective,
        "process_node": process,
        "critical_matching_requirements": matching,
        "specific_concern": concern,
    }


def _provider_history() -> list[dict[str, str]]:
    return [
        {"role": message.get("role", "user"), "content": message.get("content", "")}
        for message in st.session_state.chat_history[-8:]
        if message.get("content")
    ]


def _provider_comparison_controls(
    configs: dict[str, Any]
) -> bool:
    both_available = all(name in configs for name in ("gemini", "openai"))
    compare_mode = st.checkbox(
        "Compare Gemini and OpenAI on the same image request",
        value=False,
        disabled=not both_available,
        key="assistant_compare_providers",
        help="Both providers receive the same selected images, question, recent text context, and engineering context.",
    )
    if not both_available:
        st.caption("Provider comparison is unavailable until both Gemini and OpenAI have a model and credential configured.")
        return False
    if not compare_mode:
        return False

    st.warning("Comparison sends the selected images and listed context to both providers. Each provider requires its own explicit consent.")
    for provider_id in ("gemini", "openai"):
        st.info(_privacy_text(provider_id))
    return True


def _render_request_consents(
    configs: dict[str, Any],
    selected_provider: str | None,
    compare_mode: bool,
    question: str,
    image_ids: list[str],
    labels: dict[str, str],
    engineering_context: dict[str, str],
) -> dict[str, bool]:
    """Show unchecked consent controls tied to this exact request content."""
    targets = ["gemini", "openai"] if compare_mode else ([selected_provider] if selected_provider else [])
    consented: dict[str, bool] = {}
    if not targets:
        st.caption("Choose a configured provider before sending an image request.")
        return consented
    st.markdown("**Confirm this specific image request**")
    for provider_id in targets:
        if provider_id not in configs:
            consented[provider_id] = False
            continue
        config = configs[provider_id]
        key = image_request_consent_key(
            provider_id, config.model, question, image_ids, labels, engineering_context,
            compare_mode, st.session_state.assistant_request_nonce,
        )
        consented[provider_id] = st.checkbox(
            f"I consent to send this question, selected image(s), and listed context to {PROVIDER_LABELS[provider_id]} ({config.model}).",
            value=False,
            key=key,
        )
    return consented


def _render_exports() -> None:
    report = st.session_state.get("assistant_last_report")
    if not report:
        return
    with st.expander("Export latest image analysis", expanded=False):
        st.caption("Exports metadata and findings only; uploaded image bytes are not embedded.")
        json_data = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        markdown_data = analysis_report_markdown(report)
        col_json, col_markdown = st.columns(2)
        col_json.download_button(
            "Download JSON report", data=json_data, file_name="siliconpilot_image_analysis.json",
            mime="application/json", key="assistant_export_json", width="stretch",
        )
        col_markdown.download_button(
            "Download Markdown report", data=markdown_data, file_name="siliconpilot_image_analysis.md",
            mime="text/markdown", key="assistant_export_markdown", width="stretch",
        )


def render() -> None:
    render_header(
        "AI engineering assistant",
        "Analog IC design discussion, screenshot analysis, and read-only engineering artifact inspection",
    )
    _initialise_state()
    chatbot = SiliconPilotChatbot()
    secrets = _vision_secrets()
    configured_configs = resolve_configured_provider_configs(secrets=secrets)
    provider_image_limits = [config.max_images for config in configured_configs.values()]
    max_images = min([4, *provider_image_limits])
    st.markdown(
        "Ask a question, or drop one or more layout screenshots below. Visual findings are AI-generated hypotheses; "
        "they do not establish connectivity, DRC/LVS, extracted parasitics, or electrical performance."
    )

    st.markdown("#### Attach layout screenshots")
    st.caption(
        f"PNG, JPG, or JPEG · up to {MAX_UPLOAD_MB} MB each · up to {max_images} images per request · "
        "session-only storage. Images are sent only by the dedicated Analyze selected images action after per-request consent. Text chat never attaches them."
    )
    uploader_key = f"assistant_layout_upload_{st.session_state.assistant_upload_widget_version}"
    uploads = st.file_uploader(
        "Upload layout images", type=["png", "jpg", "jpeg"], accept_multiple_files=True,
        max_upload_size=MAX_UPLOAD_MB, key=uploader_key,
        help="Drag files here or browse. Add two or more labelled images when asking for a comparison.",
    )
    assets = st.session_state.assistant_image_assets
    selected = st.session_state.assistant_selected_image_ids
    seen_errors: set[str] = set()
    for upload in uploads or []:
        try:
            image = validate_uploaded_image(upload)
            if image.image_id not in selected:
                if len(selected) >= max_images:
                    st.warning(f"Only {max_images} images can be selected for one request; {image.filename} was not added.")
                    continue
                if image.image_id not in assets:
                    assets[image.image_id] = _asset_record(image)
                selected.append(image.image_id)
            elif image.image_id not in assets:
                assets[image.image_id] = _asset_record(image)
        except ImageValidationError as exc:
            message = f"{getattr(upload, 'name', 'Uploaded file')}: {exc}"
            if message not in seen_errors:
                st.error(message)
                seen_errors.add(message)

    st.caption(f"{len(st.session_state.assistant_selected_image_ids)} image(s) selected for the next request.")
    _render_selected_images(max_images)
    _render_clear_all_images()

    selected_ids = [
        image_id for image_id in st.session_state.assistant_selected_image_ids
        if image_id in st.session_state.assistant_image_assets
    ]
    image_labels = {
        image_id: st.session_state.assistant_image_labels.get(image_id, "")
        for image_id in selected_ids
    }
    image_question = st.text_input(
        "Question to send with the selected image(s)",
        key=f"assistant_image_question_{st.session_state.assistant_image_question_version}",
        placeholder="e.g. Describe the visible matching pattern and list checks for review",
    )

    with st.expander("Provider and privacy settings", expanded=True):
        service, selected_provider = _provider_controls(secrets)
        configs = resolve_configured_provider_configs(secrets=secrets)
        compare_mode = _provider_comparison_controls(configs)

    engineering_context = _engineering_context()
    request_consents = _render_request_consents(
        configs, selected_provider, compare_mode, image_question,
        selected_ids, image_labels, engineering_context,
    )
    analyze_images_clicked = st.button(
        "Analyze selected images",
        type="primary",
        disabled=not selected_ids or service is None,
        key="assistant_analyze_selected_images",
        help="Only this button can send the selected images. Consent is tied to the question, images, labels and entered context.",
    )
    if not selected_ids:
        st.caption("Upload or select an image to enable an explicit image-analysis request.")

    with st.expander("Inspect an EDA report, SPICE netlist, or native layout file", expanded=False):
        st.caption("Read-only inspection only. Imported netlists are never executed automatically; uploaded files are not stored permanently.")
        artifact = st.file_uploader(
            "Engineering artifact", type=["txt", "log", "out", "rpt", "drc", "lvs", "spef", "csv", "json", "html", "htm", "cir", "sp", "spice", "net", "gds", "gdsii", "oas", "oasis", "def", "lef"],
            accept_multiple_files=False, max_upload_size=25, key="assistant_engineering_artifact",
        )
        artifact_kind = st.selectbox(
            "Artifact type", ["EDA report", "SPICE netlist", "Native layout metadata"],
            key="assistant_artifact_kind",
        )
        artifact_bytes = artifact.getvalue() if artifact is not None else None
        current_artifact_key = (
            artifact_report_key(artifact.name, artifact_bytes, artifact_kind)
            if artifact is not None and artifact_bytes is not None else None
        )
        if st.session_state.get("assistant_artifact_report_key") != current_artifact_key:
            st.session_state.assistant_artifact_report = None
            st.session_state.assistant_artifact_report_key = None
        if st.button("Inspect artifact", key="assistant_inspect_artifact", disabled=artifact is None):
            from src.eda.inspection import ArtifactInspectionError, inspect_artifact
            try:
                st.session_state.assistant_artifact_report = inspect_artifact(
                    artifact.name, artifact_bytes, artifact_kind
                )
                st.session_state.assistant_artifact_report_key = current_artifact_key
            except ArtifactInspectionError as exc:
                st.session_state.assistant_artifact_report = {
                    "status": "unsupported", "source_filename": artifact.name,
                    "message": str(exc),
                }
                st.session_state.assistant_artifact_report_key = current_artifact_key
            except Exception as exc:
                st.session_state.assistant_artifact_report = {
                    "status": "unsupported", "source_filename": artifact.name,
                    "message": f"Artifact inspection failed safely: {type(exc).__name__}.",
                }
                st.session_state.assistant_artifact_report_key = current_artifact_key
        artifact_report = st.session_state.get("assistant_artifact_report")
        if artifact_report:
            if artifact_report.get("status") == "unsupported":
                st.warning(artifact_report.get("message", "Unsupported artifact."))
            else:
                st.json(artifact_report)
                st.download_button(
                    "Download artifact inspection JSON",
                    data=json.dumps(artifact_report, indent=2, ensure_ascii=False, default=str),
                    file_name="siliconpilot_artifact_inspection.json", mime="application/json",
                    key="assistant_artifact_export",
                )

    with st.expander("Quick topics"):
        quick_columns = st.columns(2)
        quick_query = None
        for index, (label, query_text) in enumerate(QUICK_PROMPTS.items()):
            if quick_columns[index % 2].button(label, key=f"assistant_quick_{index}"):
                quick_query = query_text

    _render_history()
    user_query = st.chat_input("Ask an analog layout, circuit, or model question?")
    query = quick_query or user_query
    if analyze_images_clicked:
        image_query = image_question.strip()
        if not image_query:
            st.warning("Enter a question for the selected image(s). No provider request was sent.")
        elif service is None or selected_provider is None:
            st.warning("No provider request was sent. Select a configured provider first.")
        else:
            comparison_requested = is_comparison_request(image_query)
            images = [
                _image_from_record(st.session_state.assistant_image_assets[image_id], image_labels[image_id])
                for image_id in selected_ids
            ]
            if comparison_requested and len(images) < 2:
                st.warning("Select at least two images and give them clear labels before requesting a visual comparison. No provider request was sent.")
            else:
                target_provider_ids = ["gemini", "openai"] if compare_mode else [selected_provider]
                consented = {
                    provider_id for provider_id in target_provider_ids
                    if request_consents.get(provider_id, False)
                }
                if not image_request_is_authorized(
                    True, selected_ids, target_provider_ids, consented
                ):
                    st.warning("No provider request was sent. Consent is required separately for every selected provider and this exact question, image set, and context.")
                    st.session_state.assistant_request_nonce += 1
                    st.rerun()
                else:
                    st.session_state.chat_history.append({
                        "role": "user", "content": image_query,
                        "image_ids": selected_ids.copy(), "image_labels": image_labels,
                    })
                    results = run_vision_analysis_for_providers(
                        configs, target_provider_ids, images, image_query,
                        consented_provider_ids=consented,
                        conversation_context=_provider_history()[:-1],
                        comparison_requested=comparison_requested,
                        engineering_context=engineering_context,
                    )
                    completed = sum(result.get("analysis") is not None for result in results.values())
                    comparison = compare_provider_findings(results) if compare_mode and completed == 2 else None
                    response = provider_findings_context(results)
                    if comparison:
                        response += (
                            "\nExact wording overlap between providers is descriptive only; "
                            "it is not an engineering consensus or correctness score."
                        )
                    st.session_state.assistant_last_report = build_analysis_report(
                        image_query, images, results, comparison=comparison
                    )
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": response, "image_ids": [],
                        "source_image_ids": selected_ids.copy(), "provider_results": results,
                        "comparison_requested": comparison_requested,
                    })
                    st.session_state.assistant_request_nonce += 1
                    st.session_state.assistant_image_question_version += 1
                    st.rerun()

    if query:
        prior_history = list(st.session_state.chat_history)
        if (
            not quick_query and selected_ids
            and VISUAL_ANALYSIS_INTENT.search(query)
        ):
            st.session_state.chat_history.append({"role": "user", "content": query, "image_ids": []})
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "This text-chat turn did not include your selected screenshots. To request image analysis, enter the question in the image-analysis box above, review the privacy details, and click **Analyze selected images**.",
                "image_ids": [],
            })
        else:
            _append_text_turn(query, chatbot, prior_history)
        st.rerun()

    _render_exports()


def _append_text_turn(
    query: str, chatbot: SiliconPilotChatbot, history: list[dict[str, Any]] | None = None
) -> None:
    st.session_state.chat_history.append({"role": "user", "content": query, "image_ids": []})
    response = chatbot.respond(query, history=history or [])
    st.session_state.chat_history.append({"role": "assistant", "content": response, "image_ids": []})
