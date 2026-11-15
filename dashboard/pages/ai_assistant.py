"""
AI Engineering Assistant Page — SiliconPilot AI
Unified Conversational Assistant with Google Gemini Multimodal Engine,
persistent chat history, in-composer image attachments, and read-only EDA artifact inspection.
"""
from __future__ import annotations

import base64
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from PIL import Image
import streamlit as st

from dashboard.chat import SiliconPilotChatbot
from dashboard.components.header import render_header
from src.assistant.vision import (
    DEFAULT_MAX_IMAGES,
    ImageInput,
    ImageValidationError,
    VisionAnalysisError,
    VisionAnalysisService,
    VisionProviderConfig,
    VisualAnalysis,
    clear_image_selection,
    remove_image_from_selection,
    resolve_configured_provider_configs,
    resolve_provider_config,
    validate_uploaded_image,
)
from src.eda.inspection import (
    ArtifactInspectionError,
    inspect_artifact,
)

MAX_UPLOAD_MB = 10
COMPARISON_REQUEST = re.compile(r"\b(compare|comparison|difference|differences|versus|vs\.?)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Backward-Compatible Helper Functions (Preserved for Unit Test Contracts)
# ---------------------------------------------------------------------------

def is_comparison_request(query: str) -> bool:
    return bool(COMPARISON_REQUEST.search(query or ""))


def artifact_report_key(filename: str, data: bytes, artifact_kind: str) -> str:
    digest = hashlib.sha256(data).hexdigest()
    return f"{artifact_kind}:{filename}:{digest}"


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
    payload = json.dumps({
        "provider": provider, "model": model, "question": question,
        "image_ids": image_ids, "image_labels": image_labels,
        "engineering_context": engineering_context, "compare_mode": compare_mode,
        "nonce": nonce,
    }, sort_keys=True, ensure_ascii=False)
    return f"assistant_consent_{provider}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def image_request_is_authorized(
    explicit_submit: bool,
    image_ids: list[str],
    provider_ids: list[str],
    consented_provider_ids: set[str],
) -> bool:
    return bool(
        explicit_submit and image_ids and provider_ids
        and set(provider_ids).issubset(consented_provider_ids)
    )


def provider_findings_context(provider_results: dict[str, dict[str, Any]]) -> str:
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
    completed = {name: item["analysis"] for name, item in results.items() if item.get("analysis") is not None}
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
                "model": item.get("model"), "status": item.get("status"), "error": item.get("error"),
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
        lines.extend(["", f"## {provider}: {item.get('model') or 'model unavailable'}", f"Status: {item.get('status', 'unknown')}"])
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
    if report.get("limitations"):
        lines.extend(["", "## Limitations", *(f"- {value}" for value in report["limitations"])])
    return "\n".join(lines)


PROVIDER_LABELS = {
    "gemini": "Google Gemini",
    "openai": "OpenAI",
    "openai_compatible": "OpenAI-compatible endpoint",
}


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
            format_func=lambda value: "Choose a configured provider" if value is None else f"{PROVIDER_LABELS.get(value, value)} · {configs[value].model}",
            key="assistant_provider_choice_widget",
            help="Switching providers preserves chat and uploaded images. A failed request never switches providers automatically.",
        )
    st.session_state.assistant_selected_provider = selected_provider

    if selected_provider is None:
        return None, None

    config = configs[selected_provider]
    service = VisionAnalysisService(config)
    return service, selected_provider


def _render_request_consents(
    configs: dict[str, Any],
    selected_provider: str | None,
    compare_mode: bool,
    question: str,
    image_ids: list[str],
    labels: dict[str, str],
    engineering_context: dict[str, str],
) -> dict[str, bool]:
    targets = ["gemini", "openai"] if compare_mode else ([selected_provider] if selected_provider else [])
    consented: dict[str, bool] = {}
    if not targets:
        st.caption("Choose a configured provider before sending an image request.")
        return consented
    st.markdown("**Confirm this specific image request**")
    nonce = getattr(st.session_state, "assistant_request_nonce", 0)
    for provider_id in targets:
        if provider_id not in configs:
            consented[provider_id] = False
            continue
        config = configs[provider_id]
        key = image_request_consent_key(
            provider_id, config.model, question, image_ids, labels, engineering_context,
            compare_mode, nonce,
        )
        consented[provider_id] = st.checkbox(
            f"I consent to send this question, selected image(s), and listed context to {PROVIDER_LABELS.get(provider_id, provider_id)} ({config.model}).",
            value=False,
            key=key,
        )
    return consented


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
                "model": getattr(config, "model", None), "status": "failed",
                "analysis": None, "error": str(exc),
            }
    return results


# ---------------------------------------------------------------------------
# State Initialization
# ---------------------------------------------------------------------------

def _initialise_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! I am SiliconPilot AI**, your Analog IC Layout Engineering Assistant.\n\n"
                    "Ask me about **transistor sizing, matching, common-centroid floorplans, differential pairs, current mirrors, DRC/LVS debugging, or SPICE simulation**.\n\n"
                    "💡 *Tip: Click **➕ Attach** beside the composer to attach layout screenshots for instant visual analysis!*"
                ),
                "images": [],
                "model": "SiliconPilot AI",
                "timestamp": datetime.now().strftime("%H:%M"),
                "status": "success",
            }
        ]
    if "staged_attachments" not in st.session_state:
        st.session_state.staged_attachments = []
    if "assistant_consent" not in st.session_state:
        st.session_state.assistant_consent = True
    if "uploader_key_nonce" not in st.session_state:
        st.session_state.uploader_key_nonce = 0
    if "last_failed_turn" not in st.session_state:
        st.session_state.last_failed_turn = None
    if "prefilled_prompt" not in st.session_state:
        st.session_state.prefilled_prompt = ""


# ---------------------------------------------------------------------------
# Main Page Renderer
# ---------------------------------------------------------------------------

def render() -> None:
    _initialise_state()
    chatbot = SiliconPilotChatbot()

    # 1. Header & Live Status Bar
    col_head, col_status = st.columns([7, 3])
    with col_head:
        render_header(
            "AI Engineering Assistant",
            "Conversational Analog IC Design Automation & Multimodal Layout Verification",
        )
    with col_status:
        if chatbot.is_gemini_active:
            st.success(f"🟢 **Gemini Active** (`{chatbot.gemini_provider.model_name}`)")
        else:
            st.info("🟠 **Offline Knowledge Base Mode**")

    # 2. Control Bar (Consent, Quick Clear, Active Diagnostics Link)
    col_c1, col_c2, col_c3 = st.columns([5, 3, 2])
    with col_c1:
        st.session_state.assistant_consent = st.checkbox(
            "🔒 Consent to transmit layout screenshots to Google Gemini API",
            value=st.session_state.assistant_consent,
            help="Required to send images to Google's multimodal API for optical analysis.",
        )
    with col_c2:
        st.caption("Visual findings are engineering hypotheses; not signoff DRC/LVS.")
    with col_c3:
        if st.button("🗑️ New Chat", key="btn_clear_chat", help="Start a clean conversation session"):
            st.session_state.chat_history = []
            st.session_state.staged_attachments = []
            st.session_state.last_failed_turn = None
            _initialise_state()
            st.rerun()

    # 3. Read-Only EDA Artifact Inspector (Collapsible Drawer)
    with st.expander("📁 Inspect EDA Artifacts (DRC/LVS Logs, SPICE Netlists, GDSII/OASIS)", expanded=False):
        st.caption("Read-only inspection only. Imported netlists are never executed automatically; uploaded files are not stored permanently.")
        col_art_upload, col_art_kind = st.columns([7, 3])
        with col_art_upload:
            artifact = st.file_uploader(
                "Upload engineering file",
                type=["txt", "log", "out", "rpt", "drc", "lvs", "spef", "csv", "json", "html", "htm", "cir", "sp", "spice", "net", "gds", "gdsii", "oas", "oasis", "def", "lef"],
                accept_multiple_files=False, max_upload_size=25, key="assistant_engineering_artifact",
            )
        with col_art_kind:
            artifact_kind = st.selectbox(
                "Artifact type", ["EDA report", "SPICE netlist", "Native layout metadata"],
                key="assistant_artifact_kind",
            )
        if artifact is not None:
            artifact_bytes = artifact.getvalue()
            current_key = artifact_report_key(artifact.name, artifact_bytes, artifact_kind)
            if st.button("Inspect Artifact", key="assistant_inspect_artifact", type="secondary"):
                try:
                    rep = inspect_artifact(artifact.name, artifact_bytes, artifact_kind)
                    st.session_state.assistant_artifact_report = rep
                    st.session_state.assistant_artifact_report_key = current_key
                except Exception as exc:
                    st.session_state.assistant_artifact_report = {
                        "status": "unsupported", "source_filename": artifact.name, "message": str(exc),
                    }
            rep = st.session_state.get("assistant_artifact_report")
            if rep:
                st.json(rep)
                col_dl, col_ask = st.columns([1, 1])
                with col_dl:
                    st.download_button(
                        "Download Inspection JSON",
                        data=json.dumps(rep, indent=2, ensure_ascii=False, default=str),
                        file_name="siliconpilot_artifact_inspection.json",
                        mime="application/json",
                    )
                with col_ask:
                    if st.button("💬 Discuss this report with Assistant", key="btn_discuss_report"):
                        summary_snippet = f"I inspected EDA artifact '{rep.get('source_filename')}'. It has {rep.get('error_count', 0)} errors and {rep.get('warning_count', 0)} warnings. Rule violations: {rep.get('rule_names', [])}. Can you explain these errors and suggest fixes?"
                        st.session_state.chat_history.append({"role": "user", "content": summary_snippet, "images": []})
                        res = chatbot.respond(summary_snippet, history=st.session_state.chat_history[:-1])
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": res.text if hasattr(res, "text") else str(res),
                            "images": [], "model": getattr(res, "model", "SiliconPilot AI"),
                            "status": getattr(res, "status", "success"),
                            "timestamp": datetime.now().strftime("%H:%M"),
                        })
                        st.rerun()

    st.markdown("---")

    # 4. Conversation History Transcript
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            # If user message had images, render them
            if msg.get("images"):
                img_cols = st.columns(min(len(msg["images"]), 4))
                for c_idx, img_rec in enumerate(msg["images"]):
                    with img_cols[c_idx]:
                        if "data" in img_rec:
                            st.image(img_rec["data"], caption=f"{img_rec.get('filename')} ({img_rec.get('width')}x{img_rec.get('height')})", use_container_width=True)
            # Message content
            if msg.get("status") == "error":
                st.error(msg["content"])
                if st.button("🔄 Retry", key=f"retry_msg_{idx}"):
                    last_user_turn = None
                    for prev_turn in reversed(st.session_state.chat_history[:idx]):
                        if prev_turn["role"] == "user":
                            last_user_turn = prev_turn
                            break
                    if last_user_turn:
                        with st.spinner("Retrying request with Gemini..."):
                            att_objs = [
                                ImageInput(
                                    image_id=im.get("image_id", "id"),
                                    filename=im.get("filename", "img"),
                                    mime_type=im.get("mime_type", "image/png"),
                                    data=im.get("data", b""),
                                    width=im.get("width", 100),
                                    height=im.get("height", 100),
                                )
                                for im in last_user_turn.get("images", [])
                            ]
                            new_res = chatbot.respond(
                                last_user_turn["content"],
                                history=st.session_state.chat_history[:idx - 1],
                                attachments=att_objs,
                            )
                            st.session_state.chat_history[idx] = {
                                "role": "assistant",
                                "content": new_res.text if hasattr(new_res, "text") else str(new_res),
                                "images": [],
                                "model": getattr(new_res, "model", "SiliconPilot AI"),
                                "latency_ms": getattr(new_res, "latency_ms", 0.0),
                                "timestamp": datetime.now().strftime("%H:%M"),
                                "status": getattr(new_res, "status", "success"),
                                "error": getattr(new_res, "error", None),
                            }
                            st.rerun()
            else:
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("model"):
                    st.caption(f"🤖 `{msg.get('model')}` · {msg.get('timestamp', '')}")

    # 5. Staged Attachments Preview Strip
    if st.session_state.staged_attachments:
        st.markdown("##### 📎 Staged Layout Attachments (Ready to send):")
        att_cols = st.columns(len(st.session_state.staged_attachments) + 2)
        for att_idx, att in enumerate(st.session_state.staged_attachments):
            with att_cols[att_idx]:
                st.image(att["data"], caption=f"{att['filename']} ({att['width']}×{att['height']})", width=120)
                if st.button("✖ Remove", key=f"btn_rem_att_{att_idx}"):
                    st.session_state.staged_attachments.pop(att_idx)
                    st.rerun()
        with att_cols[-2]:
            if st.button("🗑️ Clear All", key="btn_clear_staged", help="Discard all staged attachments"):
                st.session_state.staged_attachments.clear()
                st.session_state.uploader_key_nonce += 1
                st.rerun()
        with att_cols[-1]:
            if st.button("🚀 Send Images", key="btn_send_staged", type="primary", help="Send staged images with default analog review prompt"):
                default_prompt = "Please analyze this analog IC layout screenshot in detail, focusing on device matching, interdigitation, dummy devices, guard rings, routing symmetry, and potential layout-dependent issues."
                _submit_chat_turn(default_prompt, chatbot)

    # 6. Persistent Composer with Adjacent ➕ Attachment Button
    col_attach, col_input = st.columns([1.2, 9.8])
    with col_attach:
        with st.popover("➕ Attach", help="Attach layout screenshots (PNG, JPG, JPEG, WEBP)"):
            st.markdown("**Attach Layout Screenshots**")
            st.caption("PNG, JPG, JPEG, WEBP up to 10 MB. Max 4 images per turn.")
            uploads = st.file_uploader(
                "Upload images",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key=f"composer_uploader_{st.session_state.uploader_nonce if hasattr(st.session_state, 'uploader_nonce') else 0}",
                label_visibility="collapsed",
            )
            if uploads:
                for up in uploads:
                    try:
                        valid_img = validate_uploaded_image(up)
                        # Check duplicate
                        if not any(a["filename"] == valid_img.filename and len(a["data"]) == len(valid_img.data) for a in st.session_state.staged_attachments):
                            if len(st.session_state.staged_attachments) >= 4:
                                st.warning("Maximum of 4 images per message.")
                                break
                            st.session_state.staged_attachments.append({
                                "image_id": valid_img.image_id,
                                "filename": valid_img.filename,
                                "mime_type": valid_img.mime_type,
                                "data": valid_img.data,
                                "width": valid_img.width,
                                "height": valid_img.height,
                            })
                    except ImageValidationError as err:
                        st.error(f"{up.name}: {err}")
                if uploads:
                    st.rerun()

    with col_input:
        user_text = st.chat_input("Ask an analog IC question or describe your attached layout...")
        if user_text:
            _submit_chat_turn(user_text, chatbot)


def _submit_chat_turn(prompt: str, chatbot: SiliconPilotChatbot) -> None:
    staged = list(st.session_state.staged_attachments)
    st.session_state.staged_attachments.clear()

    # Check consent if images are present
    if staged and not st.session_state.assistant_consent:
        st.warning("⚠️ Image analysis requires consent. Please check the consent box at the top before sending layout images.")
        st.session_state.staged_attachments = staged
        return

    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt,
        "images": staged,
        "timestamp": datetime.now().strftime("%H:%M"),
    })

    # Prepare ImageInput objects
    att_inputs = [
        ImageInput(
            image_id=item["image_id"],
            filename=item["filename"],
            mime_type=item["mime_type"],
            data=item["data"],
            width=item["width"],
            height=item["height"],
        )
        for item in staged
    ]

    # Generate assistant turn
    with st.spinner("SiliconPilot AI is reasoning..."):
        res = chatbot.respond(
            prompt,
            history=st.session_state.chat_history[:-1],
            attachments=att_inputs,
        )

    # Store assistant message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": res.text if hasattr(res, "text") else str(res),
        "images": [],
        "model": getattr(res, "model", "SiliconPilot AI"),
        "latency_ms": getattr(res, "latency_ms", 0.0),
        "timestamp": datetime.now().strftime("%H:%M"),
        "status": getattr(res, "status", "success"),
        "error": getattr(res, "error", None),
    })
    st.rerun()
