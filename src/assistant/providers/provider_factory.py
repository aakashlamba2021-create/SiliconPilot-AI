"""Provider registry for configured multimodal adapters."""
from __future__ import annotations

from typing import Any

from ..vision import VisionAnalysisError, VisionProviderConfig, resolve_configured_provider_configs
from .gemini_provider import GeminiVisionProvider
from .openai_provider import OpenAIVisionProvider


def build_provider(config: VisionProviderConfig) -> Any:
    if not config.is_configured:
        raise VisionAnalysisError("Vision provider settings are incomplete.")
    if config.provider == "gemini":
        return GeminiVisionProvider(config)
    if config.provider == "openai":
        return OpenAIVisionProvider(config)
    if config.provider == "openai_compatible":
        from ..vision import OpenAICompatibleVisionProvider
        return OpenAICompatibleVisionProvider(config)
    raise VisionAnalysisError(f"Unsupported vision provider '{config.provider}'.")


def configured_providers(environ=None, secrets=None) -> dict[str, VisionProviderConfig]:
    return resolve_configured_provider_configs(environ, secrets)
