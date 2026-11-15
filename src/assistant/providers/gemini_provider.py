"""Google Gemini Interactions API and Conversational REST adapter."""
from __future__ import annotations

import base64
from datetime import datetime
import json
import time
from typing import Any, Mapping, Sequence
import requests

from ..vision import ImageInput, VisionAnalysisError, VisionProviderConfig, VisualAnalysis
from .base import (
    ANALYSIS_INSTRUCTIONS,
    ENGINEERING_RESPONSE_SCHEMA,
    build_user_prompt,
    complete_analysis,
    post_json,
)

ANALOG_ENGINEERING_SYSTEM_INSTRUCTION = """You are SiliconPilot AI, a Senior Analog IC Design Automation and Layout Engineering Assistant.
You provide technical, physically accurate, and design-rule-aware guidance on Analog IC layout, circuit design, physical verification (DRC/LVS), parasitic effects (PEX), and SPICE simulation.

Your core domain expertise includes:
- MOSFET device physics: short-channel effects, subthreshold slope, velocity saturation, mobility degradation, WPE (Well Proximity Effect), LOD (Length of Diffusion / STI stress).
- Critical analog matching structures: Differential pairs, current mirrors, cascode bias networks, R-2R ladders.
- Layout symmetry & gradient cancellation: Common-centroid (2D centroid cancellation), interdigitation (1D gradient cancellation, e.g. ABBA), dummy poly/diffusion strips to mitigate optical proximity effects (OPE) and boundary etch variations, identical orientation (0° or 180°) to cancel piezoresistive stress.
- Substrate & noise isolation: Guard rings (majority/minority carrier collection), separate quiet analog vs noisy digital substrate taps, deep N-well isolation.
- Interconnect & parasitics (PEX): Lateral coupling capacitance (cross-talk), plate/fringe capacitance, sheet resistance, IR drop on thin metals (M1-M3), electromigration (EM) limits, high-speed routing on thick top metals.
- Physical Verification: DRC rule interpretation (min width/spacing, antenna ratios, density rules), LVS netlist-to-layout graph isomorphism, soft connects, floating wells.
- SPICE simulation: AC open-loop frequency response (Av, GBW, PM, gm/Id methodology), transient step response (slew rate, settling time), pole-splitting (Miller Cc compensation).

When analyzing uploaded layout or circuit screenshots:
Structure your response using these clear engineering headings:
### 1. 👁️ Visible Observations
State strictly what can be optically seen: device topologies, symmetry patterns, interdigitation order (e.g. ABBA), presence/absence of dummy devices, visible guard rings, routing orientation, layer contours.

### 2. ⚠️ Potential Engineering Issues & Risks
Explain potential risks indicated by the visible layout: systematic gradient mismatch, WPE/LOD delta, parasitic cross-coupling, asymmetric routing capacitance, gate resistance (Rg) loading, or missing substrate taps.

### 3. 🛠️ Recommended Modifications
Suggest concrete, actionable layout improvements: common-centroid floorplans, dummy device additions, cross-quad arrangements, matched routing line lengths, or shielding.

### 4. ⚖️ Expected Benefits & Trade-offs
Detail the electrical improvements (e.g., lower input offset voltage, reduced 1/f noise, higher CMRR/PSRR) alongside physical costs (e.g., increased silicon area, routing complexity, extra parasitic capacitance).

### 5. 🔍 Required Physical Verification & Simulation
Explicitly identify what must be verified externally. Note that an optical screenshot cannot verify electrical connectivity, sub-micron DRC rules, LVS correctness, or exact extracted R/C values. Mandate running foundry DRC/LVS decks and post-layout PEX SPICE simulations.

### 6. 📊 Confidence & Optical Limitations
Provide qualitative confidence (Low / Medium / High) and state any optical limitations (e.g., inability to resolve transparent via stacks, layer color ambiguity, zoom resolution).

When answering general text questions, greetings, or follow-ups:
Respond conversationally, concisely, and with technical depth. If the user refers to previously uploaded images, use that prior context intelligently. Never claim that optical inspection guarantees DRC/LVS pass or silicon validation."""


class GeminiVisionProvider:
    def __init__(self, config: VisionProviderConfig) -> None:
        if config.provider != "gemini" or not config.is_configured:
            raise VisionAnalysisError("Gemini configuration requires GEMINI_API_KEY and a configured model.")
        self.config = config

    @property
    def display_name(self) -> str:
        return self.config.public_provider_name

    @property
    def model_name(self) -> str:
        return str(self.config.model)

    def test_connection(self) -> dict[str, Any]:
        """Performs a real, minimal API connectivity probe against Google Gemini."""
        if not self.config.is_configured or not self.config.api_key:
            return {
                "status": "unconfigured",
                "model": self.model_name,
                "latency_ms": 0.0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": "GEMINI_API_KEY is not configured.",
                "message": "API key missing. Add GEMINI_API_KEY to .streamlit/secrets.toml or your environment.",
            }

        key = str(self.config.api_key).strip()
        models_to_test = [self.model_name]
        if "gemini-3-flash-preview" not in models_to_test:
            models_to_test.append("gemini-3-flash-preview")

        last_error = None
        for model in models_to_test:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            payload = {
                "contents": [{"parts": [{"text": "ping"}]}],
                "generationConfig": {"maxOutputTokens": 5},
            }
            started = time.perf_counter()
            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=15.0,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                if response.status_code == 200:
                    return {
                        "status": "verified_operational",
                        "model": model,
                        "latency_ms": round(elapsed_ms, 1),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "error": None,
                        "message": f"Successfully connected to {model} ({round(elapsed_ms, 1)} ms).",
                    }
                elif response.status_code in {401, 403}:
                    return {
                        "status": "failed",
                        "model": model,
                        "latency_ms": round(elapsed_ms, 1),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "error": f"Authentication failed (HTTP {response.status_code})",
                        "message": "Invalid API key or account permissions. Check GEMINI_API_KEY.",
                    }
                elif response.status_code == 503:
                    last_error = f"{model} is currently experiencing high demand (HTTP 503). Retrying fallback..."
                    continue
                else:
                    last_error = f"API returned HTTP {response.status_code}: {response.text[:200]}"
            except requests.Timeout:
                last_error = f"Connection to {model} timed out after 15 seconds."
            except requests.RequestException as exc:
                last_error = f"Network transport error: {exc}"

        return {
            "status": "failed",
            "model": self.model_name,
            "latency_ms": 0.0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": last_error,
            "message": f"Connection test failed: {last_error}",
        }

    def send_chat_turn(
        self,
        history: Sequence[Mapping[str, Any]],
        user_text: str,
        attachments: Sequence[ImageInput] = (),
        *,
        system_instruction: str | None = None,
        timeout: float = 35.0,
    ) -> dict[str, Any]:
        """Unified conversational method handling multi-turn text and multimodal image turns."""
        if not self.config.is_configured or not self.config.api_key:
            return {
                "text": "⚠️ Gemini API key is not configured. Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` or your environment variables to enable live AI responses.",
                "model": self.model_name,
                "status": "error",
                "latency_ms": 0.0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": "unconfigured_key",
            }

        key = str(self.config.api_key).strip()
        contents: list[dict[str, Any]] = []

        # 1. Build prior turns (compact text references for earlier images to preserve token budget)
        for msg in (history or [])[-8:]:
            role = msg.get("role")
            if role not in {"user", "assistant"}:
                continue
            gemini_role = "user" if role == "user" else "model"
            parts: list[dict[str, Any]] = []

            # Prior turn image indicator
            if msg.get("images"):
                for img_info in msg["images"]:
                    fname = img_info.get("filename", "image")
                    dims = f"{img_info.get('width', '')}x{img_info.get('height', '')}"
                    parts.append({"text": f"[User previously attached layout image: {fname} ({dims})]"})

            content_text = str(msg.get("content", "")).strip()
            if content_text:
                parts.append({"text": content_text})

            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        # 2. Build current user turn
        current_parts: list[dict[str, Any]] = []
        clean_text = user_text.strip() if user_text else ""
        if clean_text:
            current_parts.append({"text": clean_text})
        elif attachments:
            current_parts.append({
                "text": "Please analyze this analog IC layout screenshot in detail, focusing on device matching, interdigitation, dummy devices, guard rings, routing symmetry, and potential layout-dependent issues."
            })

        for img in attachments:
            b64_data = base64.b64encode(img.data).decode("ascii")
            current_parts.append({
                "inline_data": {
                    "mime_type": img.mime_type,
                    "data": b64_data,
                }
            })

        if not current_parts:
            return {
                "text": "Please enter a question or attach an image to analyze.",
                "model": self.model_name,
                "status": "error",
                "latency_ms": 0.0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": "empty_input",
            }

        contents.append({"role": "user", "parts": current_parts})

        # 3. Candidate models to try (primary + responsive preview fallback)
        candidate_models = [self.model_name]
        if "gemini-3-flash-preview" not in candidate_models:
            candidate_models.append("gemini-3-flash-preview")

        sys_prompt = system_instruction or ANALOG_ENGINEERING_SYSTEM_INSTRUCTION
        payload = {
            "system_instruction": {"parts": [{"text": sys_prompt}]},
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self.config.max_output_tokens,
                "temperature": 0.2,
            },
        }

        last_error_msg = ""
        for model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            started = time.perf_counter()
            try:
                resp = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        text_parts = [p.get("text", "") for p in parts if "text" in p]
                        output_text = "\n".join(text_parts).strip()
                        if output_text:
                            return {
                                "text": output_text,
                                "model": model,
                                "status": "success",
                                "latency_ms": round(elapsed_ms, 1),
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "error": None,
                            }
                elif resp.status_code == 503:
                    last_error_msg = f"{model} is currently experiencing high demand (HTTP 503). Retrying fallback..."
                    continue
                elif resp.status_code in {401, 403}:
                    return {
                        "text": f"❌ Gemini Authentication Failed (HTTP {resp.status_code}). Please verify that `GEMINI_API_KEY` is valid.",
                        "model": model,
                        "status": "error",
                        "latency_ms": round(elapsed_ms, 1),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "error": "auth_failure",
                    }
                else:
                    last_error_msg = f"HTTP {resp.status_code}: {resp.text[:300]}"
            except requests.Timeout:
                last_error_msg = f"Request to {model} timed out after {timeout} seconds."
            except requests.RequestException as exc:
                last_error_msg = f"Network connection error: {exc}"

        return {
            "text": f"❌ Gemini API Request Failed: {last_error_msg}\n\nClick **Retry** to resubmit your message.",
            "model": self.model_name,
            "status": "error",
            "latency_ms": 0.0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": last_error_msg,
        }

    @staticmethod
    def build_request(
        config: VisionProviderConfig,
        images: Sequence[ImageInput],
        question: str,
        *,
        conversation_context: Sequence[Mapping[str, str]] = (),
        comparison_requested: bool = False,
        engineering_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        prompt = build_user_prompt(
            question, images, conversation_context=conversation_context,
            comparison_requested=comparison_requested,
            engineering_context=engineering_context,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for index, image in enumerate(images, start=1):
            content.append({
                "type": "text",
                "text": f"Image {index} follows; label={image.label or image.filename!r}; filename={image.filename!r}.",
            })
            content.append({
                "type": "image",
                "data": base64.b64encode(image.data).decode("ascii"),
                "mime_type": image.mime_type,
            })
        return {
            "model": config.model,
            "system_instruction": ANALYSIS_INSTRUCTIONS,
            "input": content,
            "store": False,
            "generation_config": {"max_output_tokens": config.max_output_tokens},
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": ENGINEERING_RESPONSE_SCHEMA,
            },
        }

    def analyze(
        self,
        images: Sequence[ImageInput],
        question: str,
        *,
        conversation_context: Sequence[Mapping[str, str]] = (),
        comparison_requested: bool = False,
        engineering_context: Mapping[str, str] | None = None,
    ) -> VisualAnalysis:
        if not images:
            raise VisionAnalysisError("Select at least one image before requesting visual analysis.")
        if not question.strip():
            raise VisionAnalysisError("Enter a question for the selected image or images.")
        payload = self.build_request(
            self.config, images, question, conversation_context=conversation_context,
            comparison_requested=comparison_requested,
            engineering_context=engineering_context,
        )
        response, _, elapsed_ms = post_json(
            self.config.endpoint, api_key=str(self.config.api_key), provider="gemini",
            payload=payload, config=self.config,
        )
        status = response.get("status")
        if status not in (None, "completed"):
            raise VisionAnalysisError(
                f"Gemini interaction did not complete (status: {status}). No analysis result was produced."
            )
        text = _extract_interaction_text(response)
        return complete_analysis(
            text, config=self.config, provider="gemini", response=response,
            elapsed_ms=elapsed_ms, images=images,
            comparison_requested=comparison_requested,
        )


def _extract_interaction_text(response: Mapping[str, Any]) -> str | None:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    for step in response.get("steps", []) if isinstance(response.get("steps"), list) else []:
        if not isinstance(step, Mapping) or step.get("type") not in {"model_output", "text"}:
            continue
        content = step.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    return item["text"]
        if isinstance(content, Mapping) and isinstance(content.get("text"), str):
            return content["text"]
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                return item["text"]
    return None
