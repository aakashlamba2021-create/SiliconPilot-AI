from __future__ import annotations

from io import BytesIO
import json
from unittest.mock import Mock, patch

import pytest
import requests
from PIL import Image

from dashboard.chat import SiliconPilotChatbot
from dashboard.pages.ai_assistant import is_comparison_request
from src.assistant.vision import (
    ImageValidationError,
    VisionAnalysisError,
    VisionAnalysisService,
    VisionProviderConfig,
    VisualAnalysis,
    clear_image_selection,
    remove_image_from_selection,
    resolve_provider_config,
    validate_uploaded_image,
)


class FakeUpload:
    def __init__(self, name: str, data: bytes, mime: str = "") -> None:
        self.name = name
        self.type = mime
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def image_bytes(fmt: str = "PNG") -> bytes:
    stream = BytesIO()
    Image.new("RGB", (19, 11), color=(30, 90, 140)).save(stream, format=fmt)
    return stream.getvalue()


def analysis_payload() -> dict:
    return {
        "summary": "The screenshot shows two visible device groups and a route whose symmetry needs review.",
        "visual_observations": ["Two groups of repeated shapes are visible."],
        "potential_issues": [{
            "description": "The routes appear asymmetric.",
            "supporting_visual_observation": "The visible left route is longer in this crop.",
            "electrical_or_physical_relevance": "Different route geometry could contribute to unequal parasitic loading.",
            "severity_assessment": "low",
            "confidence": "low",
            "suggested_investigation": "Inspect the native layout and compare extracted routes.",
        }],
        "recommended_modifications": [{
            "proposed_change": "Check whether the two routes can be made geometrically similar.",
            "reason": "Similar route shapes may help reduce systematic mismatch.",
            "supporting_observation": "The visible left route appears longer.",
            "expected_benefit": "Potentially more balanced parasitic loading.",
            "possible_tradeoff": "A reroute could increase congestion elsewhere.",
            "required_validation": "Inspect the native layout and compare extracted parasitics.",
        }],
        "expected_benefits": ["Potentially more balanced parasitic loading."],
        "possible_tradeoffs": ["A reroute could increase congestion elsewhere."],
        "verification_required": ["Run the appropriate DRC/LVS and extraction tools."],
        "confidence_and_limitations": ["This screenshot does not establish connectivity or exact dimensions."],
        "image_references": [],
        "provider_metadata": {"note": ""},
        "comparison_summary": "Image A and Image B cannot be quantitatively compared without a common scale.",
    }


def test_multiple_image_uploads_keep_filename_and_dimensions():
    first = validate_uploaded_image(FakeUpload("left.png", image_bytes("PNG"), "image/png"))
    second = validate_uploaded_image(FakeUpload("right.jpeg", image_bytes("JPEG"), "image/jpeg"))
    assert first.filename == "left.png"
    assert second.filename == "right.jpeg"
    assert first.width == second.width == 19
    assert first.height == second.height == 11
    assert first.mime_type == "image/png"
    assert second.mime_type == "image/jpeg"
    assert first.image_id != second.image_id


@pytest.mark.parametrize("filename,fmt,mime", [
    ("board.png", "PNG", "image/png"),
    ("board.jpg", "JPEG", "image/jpeg"),
    ("board.jpeg", "JPEG", "image/jpeg"),
])
def test_supported_formats(filename, fmt, mime):
    assert validate_uploaded_image(FakeUpload(filename, image_bytes(fmt), mime)).mime_type == mime


def test_unsupported_corrupt_oversize_and_mismatched_images_are_rejected():
    with pytest.raises(ImageValidationError, match="PNG, JPG, or JPEG"):
        validate_uploaded_image(FakeUpload("layout.bmp", b"not an image"))
    with pytest.raises(ImageValidationError, match="corrupted"):
        validate_uploaded_image(FakeUpload("layout.png", b"broken"))
    with pytest.raises(ImageValidationError, match="upload limit"):
        validate_uploaded_image(FakeUpload("layout.png", image_bytes()), max_bytes=5)
    with pytest.raises(ImageValidationError, match="does not match"):
        validate_uploaded_image(FakeUpload("layout.jpg", image_bytes("PNG"), "image/jpeg"))


def test_remove_and_clear_selection_preserve_images_referenced_by_history():
    history = [{"role": "user", "content": "inspect", "image_ids": ["used"]}]
    selected, referenced = remove_image_from_selection("used", ["used", "unused"], history)
    assert selected == ["unused"]
    assert referenced is True
    assert clear_image_selection(["unused"], history) == ([], {"used"})


def test_provider_config_requires_explicit_provider_model_and_key():
    config = resolve_provider_config(environ={})
    service = VisionAnalysisService(config)
    assert not service.is_available
    assert "No vision provider configured" in service.unavailable_reason
    image = validate_uploaded_image(FakeUpload("a.png", image_bytes()))
    with pytest.raises(VisionAnalysisError, match="No image analysis was performed"):
        service.analyze([image], "inspect this")

    configured = resolve_provider_config(
        environ={"SILICONPILOT_VISION_PROVIDER": "openai_compatible", "OPENAI_API_KEY": "secret"},
        secrets={"vision_assistant": {"model": "vision-model"}},
    )
    assert configured.is_configured
    assert configured.api_key == "secret"
    assert "secret" not in repr(configured)


def test_structured_analysis_schema_and_required_sections():
    result = VisualAnalysis.from_payload(analysis_payload(), comparison_requested=True)
    markdown = result.to_markdown("test provider", comparison_requested=True)
    for heading in (
        "Summary", "Visual observations", "Potential issues", "Recommended modifications",
        "Verification required", "Confidence and limitations", "test provider",
    ):
        assert heading in markdown
    invalid = analysis_payload()
    invalid["recommended_modifications"] = [{"proposed_change": "missing required evidence"}]
    with pytest.raises(VisionAnalysisError, match="incomplete recommendation"):
        VisualAnalysis.from_payload(invalid)
    with pytest.raises(VisionAnalysisError, match="comparison section"):
        VisualAnalysis.from_payload(analysis_payload() | {"comparison_summary": None}, comparison_requested=True)


def _fake_response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "id": "test-request",
        "choices": [{"message": {"content": json.dumps(payload)}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
    }
    return response


def test_provider_sends_multiple_images_with_filenames_and_comparison_question():
    config = VisionProviderConfig(
        provider="openai_compatible", model="configured-model", endpoint="https://vision.example/v1/chat/completions", api_key="secret"
    )
    service = VisionAnalysisService(config)
    images = [
        validate_uploaded_image(FakeUpload("before.png", image_bytes())),
        validate_uploaded_image(FakeUpload("after.jpg", image_bytes("JPEG"))),
    ]
    with patch("src.assistant.providers.base.requests.post", return_value=_fake_response(analysis_payload())) as send:
        result = service.analyze(images, "Compare before and after", comparison_requested=True)
    assert result.comparison_summary
    call = send.call_args
    body = call.kwargs["json"]
    content = body["messages"][1]["content"]
    text_parts = [item["text"] for item in content if item["type"] == "text"]
    assert any("before.png" in item for item in text_parts)
    assert any("after.jpg" in item for item in text_parts)
    assert any("Compare before and after" in item for item in text_parts)
    assert len([item for item in content if item["type"] == "image_url"]) == 2
    assert call.kwargs["headers"]["Authorization"] == "Bearer secret"
    assert '"visual_observations"' in call.kwargs["json"]["messages"][0]["content"]
    assert call.kwargs["json"]["model"] == "configured-model"
    assert result.provider_metadata["provider_id"] == "openai_compatible"
    assert [item.filename for item in result.image_references] == ["before.png", "after.jpg"]
    assert is_comparison_request("Compare the original versus modified layout")
    assert not is_comparison_request("Inspect the two uploaded screenshots separately")


@pytest.mark.parametrize(
    "side_effect,expected",
    [
        (requests.Timeout("late"), "timed out"),
        (requests.ConnectionError("offline"), "Could not reach"),
    ],
)
def test_provider_timeout_and_transport_errors_return_no_analysis(side_effect, expected):
    config = VisionProviderConfig("openai_compatible", "model", "https://vision.example/chat", "secret")
    service = VisionAnalysisService(config)
    image = validate_uploaded_image(FakeUpload("a.png", image_bytes()))
    with patch("src.assistant.providers.base.requests.post", side_effect=side_effect):
        with pytest.raises(VisionAnalysisError, match=expected):
            service.analyze([image], "inspect")


def test_provider_http_and_invalid_structured_responses_fail_closed():
    config = VisionProviderConfig("openai_compatible", "model", "https://vision.example/chat", "secret")
    service = VisionAnalysisService(config)
    image = validate_uploaded_image(FakeUpload("a.png", image_bytes()))
    bad_http = Mock()
    bad_http.status_code = 401
    with patch("src.assistant.providers.base.requests.post", return_value=bad_http):
        with pytest.raises(VisionAnalysisError, match="authentication failed.*HTTP 401"):
            service.analyze([image], "inspect")
    with patch("src.assistant.providers.base.requests.post", return_value=_fake_response({"unexpected": "shape"})):
        with pytest.raises(VisionAnalysisError, match="does not match the required SiliconPilot schema"):
            service.analyze([image], "inspect")
    malformed = Mock()
    malformed.status_code = 200
    malformed.json.side_effect = ValueError("invalid response body")
    with patch("src.assistant.providers.base.requests.post", return_value=malformed):
        with pytest.raises(VisionAnalysisError, match="malformed JSON"):
            service.analyze([image], "inspect")


def test_text_only_chat_compatibility_and_model_provenance_labels():
    chatbot = SiliconPilotChatbot()
    answer = chatbot.respond("How do I layout a differential pair?")
    assert "Common-Centroid" in answer
    registry = chatbot.respond("What models are currently trained?")
    assert "Trust status" in registry
    assert "quarantined" in registry
    assert "synthetic" in registry
    assert "excluded from recommendations and ranking" in registry
