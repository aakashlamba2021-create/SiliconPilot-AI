"""Google Gemini Interactions API REST adapter."""
from __future__ import annotations

import base64
from typing import Any, Mapping, Sequence

from ..vision import ImageInput, VisionAnalysisError, VisionProviderConfig, VisualAnalysis
from .base import (
    ANALYSIS_INSTRUCTIONS,
    ENGINEERING_RESPONSE_SCHEMA,
    build_user_prompt,
    complete_analysis,
    post_json,
)


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
