"""OpenAI Responses API image-analysis REST adapter."""
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


class OpenAIVisionProvider:
    def __init__(self, config: VisionProviderConfig) -> None:
        if config.provider != "openai" or not config.is_configured:
            raise VisionAnalysisError("OpenAI configuration requires OPENAI_API_KEY and a configured model.")
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
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for index, image in enumerate(images, start=1):
            content.append({
                "type": "input_text",
                "text": f"Image {index} follows; label={image.label or image.filename!r}; filename={image.filename!r}.",
            })
            encoded = base64.b64encode(image.data).decode("ascii")
            content.append({
                "type": "input_image",
                "image_url": f"data:{image.mime_type};base64,{encoded}",
                "detail": config.image_detail,
            })
        return {
            "model": config.model,
            "instructions": ANALYSIS_INSTRUCTIONS,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "siliconpilot_layout_analysis",
                    "strict": True,
                    "schema": ENGINEERING_RESPONSE_SCHEMA,
                }
            },
            "max_output_tokens": config.max_output_tokens,
            "store": False,
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
            self.config.endpoint, api_key=str(self.config.api_key), provider="openai",
            payload=payload, config=self.config,
        )
        if response.get("status") == "incomplete":
            raise VisionAnalysisError("OpenAI returned an incomplete response. No analysis result was produced.")
        text = _extract_responses_text(response)
        return complete_analysis(
            text, config=self.config, provider="openai", response=response,
            elapsed_ms=elapsed_ms, images=images,
            comparison_requested=comparison_requested,
        )


def _extract_responses_text(response: Mapping[str, Any]) -> str | None:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    output = response.get("output")
    if not isinstance(output, list):
        return None
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "refusal":
                raise VisionAnalysisError("OpenAI declined the image-analysis request. No analysis result was produced.")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                return part["text"]
    return None
