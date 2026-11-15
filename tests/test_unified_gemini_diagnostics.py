"""Unit tests for the unified Gemini conversational assistant, diagnostics telemetry, and WEBP image handling."""
from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from dashboard.chat import ChatTurnResult, SiliconPilotChatbot
from src.assistant.providers.gemini_provider import GeminiVisionProvider
from src.assistant.vision import (
    ImageInput,
    SUPPORTED_FORMATS,
    SUPPORTED_SUFFIXES,
    VisionProviderConfig,
    validate_uploaded_image,
)
from dashboard.pages.system_info import (
    _detect_gpu,
    get_system_diagnostics,
)


def _make_image_bytes(fmt: str = "PNG") -> bytes:
    stream = BytesIO()
    Image.new("RGB", (32, 32), color=(50, 100, 150)).save(stream, format=fmt)
    return stream.getvalue()


class _MockUpload:
    def __init__(self, name: str, data: bytes, mime_type: str = ""):
        self.name = name
        self._data = data
        self.type = mime_type

    def getvalue(self) -> bytes:
        return self._data


def test_webp_format_support_and_normalization():
    assert "WEBP" in SUPPORTED_FORMATS
    assert "image/webp" in SUPPORTED_FORMATS.values()
    assert ".webp" in SUPPORTED_SUFFIXES

    webp_bytes = _make_image_bytes("WEBP")
    upload = _MockUpload("layout_snapshot.webp", webp_bytes, "image/webp")
    image_input = validate_uploaded_image(upload)
    assert image_input.mime_type == "image/webp"
    assert image_input.filename == "layout_snapshot.webp"
    assert image_input.width == 32
    assert image_input.height == 32


def test_gemini_provider_test_connection_unconfigured():
    config = VisionProviderConfig(
        provider="gemini",
        model="gemini-3.8-flash",
        endpoint="https://generativelanguage.googleapis.com",
        api_key=None,
    )
    with pytest.raises(Exception) as exc_info:
        GeminiVisionProvider(config)
    assert "Gemini configuration requires GEMINI_API_KEY" in str(exc_info.value)


def test_gemini_provider_test_connection_success():
    config = VisionProviderConfig(
        provider="gemini",
        model="gemini-3.8-flash",
        endpoint="https://generativelanguage.googleapis.com",
        api_key="test-key",
    )
    provider = GeminiVisionProvider(config)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "PONG"}]
                }
            }
        ]
    }

    with patch("src.assistant.providers.gemini_provider.requests.post", return_value=fake_resp):
        res = provider.test_connection()
        assert res["status"] == "verified_operational"
        assert res["model"] == "gemini-3.8-flash"
        assert res["latency_ms"] is not None
        assert res["error"] is None
        assert "Successfully connected" in res["message"]


def test_gemini_provider_send_chat_turn_success():
    config = VisionProviderConfig(
        provider="gemini",
        model="gemini-3.8-flash",
        endpoint="https://generativelanguage.googleapis.com",
        api_key="test-key",
    )
    provider = GeminiVisionProvider(config)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Here is an analog IC layout analysis."}]
                }
            }
        ]
    }

    img_input = ImageInput("img1", "test.png", "image/png", _make_image_bytes("PNG"), 32, 32, "Test")

    with patch("src.assistant.providers.gemini_provider.requests.post", return_value=fake_resp) as mock_post:
        result = provider.send_chat_turn(
            history=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            user_text="Check the matching of this differential pair",
            attachments=[img_input],
        )

        assert result["status"] == "success"
        assert "analog IC layout" in result["text"]
        assert result["model"] == "gemini-3.8-flash"
        assert result["latency_ms"] is not None
        assert mock_post.called

        # Check payload
        called_json = mock_post.call_args.kwargs["json"]
        contents = called_json["contents"]
        # Last turn should have inline_data
        last_parts = contents[-1]["parts"]
        has_inline_data = any("inline_data" in part for part in last_parts)
        assert has_inline_data


def test_gemini_provider_send_chat_turn_503_fallback():
    config = VisionProviderConfig(
        provider="gemini",
        model="gemini-3.8-flash",
        endpoint="https://generativelanguage.googleapis.com",
        api_key="test-key",
    )
    provider = GeminiVisionProvider(config)

    fail_503 = MagicMock()
    fail_503.status_code = 503
    fail_503.text = "The model is overloaded. Please try again later."

    success_fallback = MagicMock()
    success_fallback.status_code = 200
    success_fallback.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Fallback success response"}]}}]
    }

    with patch("src.assistant.providers.gemini_provider.requests.post", side_effect=[fail_503, success_fallback]):
        result = provider.send_chat_turn(
            history=[],
            user_text="Check routing",
            attachments=[],
        )
        assert result["status"] == "success"
        assert result["text"] == "Fallback success response"
        assert result["model"] == "gemini-3-flash-preview"


def test_chat_turn_result_is_string_compatible():
    res = ChatTurnResult(
        "Response text",
        model="gemini-3-flash-preview",
        source="gemini_api",
        status="completed",
        latency_ms=120,
    )
    # Subclass of str
    assert isinstance(res, str)
    assert res == "Response text"
    assert "Response" in res
    assert res.startswith("Response")
    assert res.model == "gemini-3-flash-preview"
    assert res.source == "gemini_api"
    assert res.status == "completed"
    assert res.latency_ms == 120
    assert res.error is None


def test_chatbot_responds_with_local_kb_when_no_credentials():
    chatbot = SiliconPilotChatbot()
    # When no keys configured in env, it falls back to local knowledge base
    with patch.dict("os.environ", {}, clear=True):
        ans = chatbot.respond("What is the common-centroid layout rule?")
        assert isinstance(ans, ChatTurnResult)
        assert ans.source in {"local", "offline_knowledge_base"}
        assert len(ans) > 0


def test_system_diagnostics_passive_telemetry():
    diagnostics = get_system_diagnostics()
    assert "python_version" in diagnostics
    assert "platform" in diagnostics
    assert "cpu_logical_cores" in diagnostics
    assert "total_ram_gb" in diagnostics
    assert diagnostics["total_ram_gb"] > 0

    gpu = _detect_gpu()
    assert "name" in gpu
    assert "driver" in gpu
