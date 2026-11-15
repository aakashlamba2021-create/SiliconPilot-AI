from __future__ import annotations

from base64 import b64encode
from io import BytesIO
import json
from unittest.mock import Mock, patch

import pytest
from PIL import Image

import dashboard.pages.ai_assistant as assistant_page
from src.assistant.evaluation import EvaluationCase, QUALITY_RUBRIC, new_evaluation_report, run_case_for_providers
from src.assistant.providers.gemini_provider import GeminiVisionProvider
from src.assistant.providers.openai_provider import OpenAIVisionProvider
from src.assistant.providers.provider_factory import build_provider, configured_providers
from src.assistant.vision import (
    ImageInput,
    VisionAnalysisError,
    VisionAnalysisService,
    VisionProviderConfig,
    VisualAnalysis,
    provider_consent_granted,
    resolve_configured_provider_configs,
    resolve_provider_config,
)


class _State(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _FakeStreamlit:
    def __init__(self, state, *, selected=None, consent=False):
        self.session_state = state
        self.selected = selected
        self.consent = consent
        self.checkbox_defaults = []
        self.checkbox_keys = []

    def selectbox(self, _label, options, *, index=0, **_kwargs):
        return self.selected if self.selected in options else options[index]

    def caption(self, *_args, **_kwargs):
        pass

    def markdown(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def checkbox(self, _label, *, value=False, **_kwargs):
        self.checkbox_defaults.append(value)
        self.checkbox_keys.append(_kwargs.get("key"))
        return self.consent


def make_image(name: str, label: str = "") -> ImageInput:
    stream = BytesIO()
    Image.new("RGB", (21, 12), color=(30, 90, 140)).save(stream, format="PNG")
    data = stream.getvalue()
    return ImageInput("id-" + name, name, "image/png", data, 21, 12, label)


def normalized_payload() -> dict:
    return {
        "summary": "Visible routing differs; the screenshot cannot establish connectivity or electrical superiority.",
        "visual_observations": ["Two visible device groups are arranged around a central route."],
        "potential_issues": [{
            "description": "The visible routing appears asymmetric.",
            "supporting_visual_observation": "One route bends around the left group while the other appears more direct.",
            "electrical_or_physical_relevance": "Different route shapes may contribute to unequal parasitic loading.",
            "severity_assessment": "low",
            "confidence": "low",
            "suggested_investigation": "Inspect native routing and compare extracted parasitics.",
        }],
        "recommended_modifications": [{
            "proposed_change": "Review whether the two routes can use similar geometry.",
            "reason": "A closer geometric match may reduce systematic imbalance.",
            "supporting_observation": "The two visible routes take different paths.",
            "expected_benefit": "Potentially more balanced routing parasitics.",
            "possible_tradeoff": "The route change may increase local congestion.",
            "required_validation": "Run DRC/LVS and compare extracted net parasitics.",
        }],
        "expected_benefits": ["Potentially more balanced routing parasitics."],
        "possible_tradeoffs": ["The route change may increase local congestion."],
        "verification_required": ["Run native DRC/LVS and extraction; screenshot review is not signoff."],
        "confidence_and_limitations": ["Connectivity and layout scale are not established from this screenshot."],
        "image_references": [],
        "provider_metadata": {"note": ""},
        "comparison_summary": "The images have visibly different routing, but scale alignment is unknown.",
    }


def mocked_response(payload: dict, *, provider: str) -> Mock:
    response = Mock()
    response.status_code = 200
    if provider == "gemini":
        response.json.return_value = {
            "id": "gemini-request-1",
            "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": json.dumps(payload)}]}],
            "usage": {
                "total_input_tokens": 100,
                "total_output_tokens": 60,
                "total_tokens": 180,
                "total_thought_tokens": 20,
                "total_cached_tokens": 0,
            },
        }
    else:
        response.json.return_value = {
            "id": "openai-request-1",
            "status": "completed",
            "output_text": json.dumps(payload),
            "usage": {"input_tokens": 100, "output_tokens": 60, "total_tokens": 160, "input_tokens_details": {"cached_tokens": 0}},
        }
    return response


def test_provider_configs_only_expose_configured_providers_and_factory_types():
    configs = resolve_configured_provider_configs(environ={
        "GEMINI_API_KEY": "gemini-secret",
        "OPENAI_API_KEY": "openai-secret",
        "SILICONPILOT_OPENAI_MODEL": "gpt-vision-alternative",
    })
    assert set(configs) == {"gemini", "openai"}
    assert configs["gemini"].model == "gemini-3.8-flash"
    assert configs["openai"].model == "gpt-vision-alternative"
    assert isinstance(build_provider(configs["gemini"]), GeminiVisionProvider)
    assert isinstance(build_provider(configs["openai"]), OpenAIVisionProvider)
    assert not configured_providers(environ={})
    assert not resolve_provider_config(environ={"SILICONPILOT_VISION_PROVIDER": "gemini"}).is_configured


def test_gemini_request_and_response_are_normalized_and_keep_local_image_associations():
    config = resolve_provider_config(environ={"GEMINI_API_KEY": "secret"}, provider="gemini")
    images = [make_image("before.png", "Original"), make_image("after.png", "Modified")]
    body = GeminiVisionProvider.build_request(
        config, images, "Compare these layouts", comparison_requested=True,
        engineering_context={"circuit_topology": "differential pair"},
    )
    assert body["model"] == "gemini-3.8-flash"
    assert body["store"] is False
    assert body["generation_config"]["max_output_tokens"] == config.max_output_tokens
    assert body["response_format"]["schema"]["required"]
    image_parts = [part for part in body["input"] if part.get("type") == "image"]
    assert len(image_parts) == 2
    assert image_parts[0]["mime_type"] == "image/png"
    assert image_parts[0]["data"] == b64encode(images[0].data).decode("ascii")
    assert any("Original" in part.get("text", "") for part in body["input"])
    assert any("Modified" in part.get("text", "") for part in body["input"])
    response = mocked_response(normalized_payload(), provider="gemini")
    with patch("src.assistant.providers.base.requests.post", return_value=response) as send:
        result = VisionAnalysisService(config).analyze(
            images, "Compare these layouts", comparison_requested=True,
            engineering_context={"circuit_topology": "differential pair"},
        )
    assert send.call_args.kwargs["headers"]["x-goog-api-key"] == "secret"
    assert "Authorization" not in send.call_args.kwargs["headers"]
    assert result.comparison_summary
    assert [(item.filename, item.label) for item in result.image_references] == [
        ("before.png", "Original"), ("after.png", "Modified")
    ]
    assert result.provider_metadata["usage"]["input_tokens"] == 100
    assert result.provider_metadata["store_requested"] is False
    assert "calibrated probabilities" in result.to_markdown("Gemini")
    assert "not a verified violation" in result.to_markdown("Gemini")


def test_openai_responses_request_uses_images_schema_limits_and_stateless_mode():
    config = resolve_provider_config(
        environ={"OPENAI_API_KEY": "secret", "SILICONPILOT_OPENAI_MODEL": "gpt-6-astra"},
        provider="openai",
    )
    images = [make_image("layout.png", "Placement A"), make_image("routing.png", "Routing view")]
    body = OpenAIVisionProvider.build_request(config, images, "Inspect matching")
    assert body["model"] == "gpt-6-astra"
    assert body["store"] is False
    assert body["max_output_tokens"] == config.max_output_tokens
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    user_content = body["input"][0]["content"]
    image_parts = [part for part in user_content if part.get("type") == "input_image"]
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"].startswith("data:image/png;base64,")
    assert image_parts[0]["detail"] == "original"
    response = mocked_response(normalized_payload(), provider="openai")
    with patch("src.assistant.providers.base.requests.post", return_value=response) as send:
        result = VisionAnalysisService(config).analyze(images, "Inspect matching")
    assert send.call_args.args[0] == "https://api.openai.com/v1/responses"
    assert send.call_args.kwargs["headers"]["Authorization"] == "Bearer secret"
    assert result.provider_metadata["provider_id"] == "openai"
    assert result.provider_metadata["usage"]["total_tokens"] == 160
    assert [reference.label for reference in result.image_references] == ["Placement A", "Routing view"]


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, "authentication failed"),
        (404, "model or API endpoint was not found"),
        (429, "rate limit or quota reached"),
        (503, "provider service is unavailable"),
    ],
)
def test_openai_http_failures_are_normalized_without_raw_response(status, expected):
    config = resolve_provider_config(environ={"OPENAI_API_KEY": "secret"}, provider="openai")
    response = Mock()
    response.status_code = status
    response.text = "secret provider diagnostic payload"
    with patch("src.assistant.providers.base.requests.post", return_value=response):
        with pytest.raises(VisionAnalysisError, match=expected) as error:
            VisionAnalysisService(config).analyze([make_image("layout.png")], "Inspect")
    assert "secret provider diagnostic payload" not in str(error.value)
    assert "secret" not in str(error.value)


def test_malformed_incomplete_and_oversize_responses_fail_without_fabricated_analysis():
    config = resolve_provider_config(environ={"GEMINI_API_KEY": "secret"}, provider="gemini")
    malformed = mocked_response({"visual_observations": ["partial"]}, provider="gemini")
    with patch("src.assistant.providers.base.requests.post", return_value=malformed):
        with pytest.raises(VisionAnalysisError, match="does not match the required SiliconPilot schema"):
            VisionAnalysisService(config).analyze([make_image("layout.png")], "Inspect")
    incomplete = Mock()
    incomplete.status_code = 200
    incomplete.json.return_value = {"status": "incomplete", "steps": [], "usage": {}}
    with patch("src.assistant.providers.base.requests.post", return_value=incomplete):
        with pytest.raises(VisionAnalysisError, match="did not complete"):
            VisionAnalysisService(config).analyze([make_image("layout.png")], "Inspect")

    restricted = VisionProviderConfig(
        "gemini", "model", "https://example.invalid", "secret", max_payload_bytes=20,
    )
    with patch("src.assistant.providers.base.requests.post") as send:
        with pytest.raises(VisionAnalysisError, match="above the configured"):
            VisionAnalysisService(restricted).analyze([make_image("layout.png")], "Inspect")
    send.assert_not_called()


def test_configured_image_count_limit_and_missing_provider_do_not_make_requests():
    config = VisionProviderConfig("openai", "model", "https://example.invalid", "secret", max_images=1)
    with patch("src.assistant.providers.base.requests.post") as send:
        with pytest.raises(VisionAnalysisError, match="at most 1 images"):
            VisionAnalysisService(config).analyze([make_image("a.png"), make_image("b.png")], "Inspect")
    send.assert_not_called()

    absent = VisionAnalysisService(resolve_provider_config(environ={}))
    with patch("src.assistant.providers.base.requests.post") as send:
        with pytest.raises(VisionAnalysisError, match="No image analysis was performed"):
            absent.analyze([make_image("a.png")], "Inspect")
    send.assert_not_called()


def test_no_silent_provider_fallback_and_consent_is_provider_scoped():
    gemini = resolve_provider_config(environ={"GEMINI_API_KEY": "g-key"}, provider="gemini")
    # An OpenAI credential elsewhere does not make a failed Gemini request switch providers.
    response = Mock()
    response.status_code = 429
    with patch("src.assistant.providers.base.requests.post", return_value=response) as send:
        with pytest.raises(VisionAnalysisError, match="rate limit"):
            VisionAnalysisService(gemini).analyze([make_image("a.png")], "Inspect")
    assert send.call_count == 1
    assert send.call_args.args[0] == gemini.endpoint
    consents = {"gemini": True}
    assert provider_consent_granted(consents, "gemini")
    assert not provider_consent_granted(consents, "openai")
    assert not provider_consent_granted(consents, None)


def test_dashboard_provider_switch_preserves_chat_and_images_but_does_not_reuse_consent():
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.delenv("SILICONPILOT_VISION_PROVIDER", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        monkeypatch.setenv("OPENAI_API_KEY", "o-key")
        history = [{"role": "user", "content": "prior turn", "image_ids": ["kept"]}]
        assets = {"kept": {"filename": "kept.png", "data": b"private-bytes"}}
        state = _State(
            assistant_request_nonce=0,
            assistant_selected_provider=None,
            chat_history=history,
            assistant_image_assets=assets,
        )
        configs = resolve_configured_provider_configs()
        fake = _FakeStreamlit(state, selected="openai", consent=True)
        with patch.object(assistant_page, "st", fake):
            service, provider = assistant_page._provider_controls({})
        assert service.config.provider == "openai"
        assert provider == "openai"
        with patch.object(assistant_page, "st", fake):
            consented = assistant_page._render_request_consents(
                configs, provider, False, "Inspect this image", ["kept"],
                {"kept": "Original"}, {"topology": "OTA"},
            )
        assert consented == {"openai": True}
        openai_key = fake.checkbox_keys[-1]
        assert fake.checkbox_defaults[-1] is False

        fake.selected = "gemini"
        fake.consent = False
        with patch.object(assistant_page, "st", fake):
            service, provider = assistant_page._provider_controls({})
        assert service.config.provider == "gemini"
        assert provider == "gemini"
        with patch.object(assistant_page, "st", fake):
            consented = assistant_page._render_request_consents(
                configs, provider, False, "Inspect this image", ["kept"],
                {"kept": "Original"}, {"topology": "OTA"},
            )
        assert consented == {"gemini": False}
        assert fake.checkbox_defaults[-1] is False
        assert fake.checkbox_keys[-1] != openai_key
        assert state.chat_history is history
        assert state.assistant_image_assets is assets
    finally:
        monkeypatch.undo()


def test_consent_key_changes_after_a_request_nonce_even_when_content_is_identical():
    common = ("openai", "gpt-6-astra", "Inspect", ["image-1"], {"image-1": "Original"}, {}, False)
    assert assistant_page.image_request_consent_key(*common, 0) != assistant_page.image_request_consent_key(*common, 1)


def test_usage_cost_is_only_estimated_when_rates_and_complete_usage_are_configured():
    config = VisionProviderConfig(
        "openai", "model", "https://example.invalid", "secret",
        input_price_per_million=10, output_price_per_million=20,
    )
    data = {
        "id": "estimate-test",
        "usage": {"input_tokens": 2000, "output_tokens": 1000, "total_tokens": 3000, "input_tokens_details": {"cached_tokens": 0}},
    }
    analysis = VisualAnalysis.from_payload(
        normalized_payload(), images=[make_image("layout.png")],
        provider_metadata=__import__("src.assistant.providers.base", fromlist=["make_provider_metadata"]).make_provider_metadata(
            config, provider="openai", response=data, elapsed_ms=30,
        ),
    )
    assert analysis.provider_metadata["estimated_cost_usd"] == 0.04
    unknown_rates = VisionProviderConfig("openai", "model", "https://example.invalid", "secret")
    no_estimate = __import__("src.assistant.providers.base", fromlist=["make_provider_metadata"]).make_provider_metadata(
        unknown_rates, provider="openai", response=data, elapsed_ms=30,
    )
    assert no_estimate["estimated_cost_usd"] is None


def test_evaluation_framework_submits_identical_user_case_to_both_providers_with_empty_scores():
    configs = [
        resolve_provider_config(environ={"GEMINI_API_KEY": "g-key"}, provider="gemini"),
        resolve_provider_config(environ={"OPENAI_API_KEY": "o-key"}, provider="openai"),
    ]
    image = make_image("reviewed.png", "Reviewed layout")
    case = EvaluationCase(
        case_id="case-1", question="Inspect matching", images=(image,),
        engineering_context={"circuit_topology": "OTA"},
    )
    report = new_evaluation_report(dataset_label="user supplied", providers=configs, cases=[case])
    assert report["cases"][0]["human_review"]["quality_scores_1_to_5_or_null"] == {
        rubric: None for rubric in QUALITY_RUBRIC
    }
    responses = [mocked_response(normalized_payload(), provider="gemini"), mocked_response(normalized_payload(), provider="openai")]
    with patch("src.assistant.providers.base.requests.post", side_effect=responses) as send:
        results = run_case_for_providers(case, configs)
    assert set(results) == {"gemini", "openai"}
    assert all(item["status"] == "completed" for item in results.values())
    calls = send.call_args_list
    gemini_image = next(item for item in calls[0].kwargs["json"]["input"] if item.get("type") == "image")
    openai_image = next(item for item in calls[1].kwargs["json"]["input"][0]["content"] if item.get("type") == "input_image")
    assert gemini_image["data"] == b64encode(image.data).decode("ascii")
    assert openai_image["image_url"] == "data:image/png;base64," + b64encode(image.data).decode("ascii")
    assert all("OTA" in str(call.kwargs["json"]) for call in calls)
