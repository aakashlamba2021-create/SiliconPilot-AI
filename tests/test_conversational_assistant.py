from __future__ import annotations

from dashboard.chat import SiliconPilotChatbot
from dashboard.pages.ai_assistant import (
    artifact_report_key,
    analysis_report_markdown,
    build_analysis_report,
    compare_provider_findings,
    image_request_consent_key,
    image_request_is_authorized,
    provider_findings_context,
    run_vision_analysis_for_providers,
)
from src.assistant.vision import (
    ImageInput,
    VisionProviderConfig,
    VisualAnalysis,
    clear_all_image_data,
)


def _analysis(provider: str, observation: str) -> VisualAnalysis:
    payload = {
        "summary": f"Summary from {provider}.",
        "visual_observations": [observation],
        "potential_issues": [],
        "recommended_modifications": [],
        "expected_benefits": [],
        "possible_tradeoffs": [],
        "verification_required": [],
        "confidence_and_limitations": ["Screenshot evidence is limited."],
        "image_references": [],
        "provider_metadata": {"note": ""},
        "comparison_summary": None,
    }
    return VisualAnalysis.from_payload(payload, provider_metadata={"provider": provider, "model": "mock-model"})


def test_common_conversation_intents_are_friendly_and_local():
    chatbot = SiliconPilotChatbot()
    assert chatbot.respond("hi!").startswith("Hey!")
    assert "Doing well" in chatbot.respond("How are you?")
    assert "I can explain" in chatbot.respond("What can you do?")
    assert "Upload a PNG" in chatbot.respond("Please analyze this layout")


def test_short_engineering_followup_uses_recent_user_context():
    chatbot = SiliconPilotChatbot()
    history = [{"role": "user", "content": "I'm working on a differential pair."}]
    answer = chatbot.respond("matching", history=history)
    assert "Differential Pair" in answer
    assert "Common-Centroid" in answer


def test_provider_comparison_uses_exact_overlap_without_a_winner_score():
    results = {
        "gemini": {"analysis": _analysis("Gemini", "Two device groups appear symmetrically arranged.")},
        "openai": {"analysis": _analysis("OpenAI", "Two device groups appear symmetrically arranged.")},
    }
    comparison = compare_provider_findings(results)
    assert comparison["agreements"] == ["Two device groups appear symmetrically arranged."]
    assert "score" not in str(comparison).lower()


def test_image_request_requires_explicit_action_and_consent_for_each_provider():
    assert not image_request_is_authorized(False, ["image"], ["gemini"], {"gemini"})
    assert not image_request_is_authorized(True, ["image"], ["gemini", "openai"], {"gemini"})
    assert not image_request_is_authorized(True, [], ["gemini"], {"gemini"})
    assert image_request_is_authorized(True, ["image"], ["gemini"], {"gemini"})


def test_image_consent_is_bound_to_provider_model_prompt_images_context_and_nonce():
    base = image_request_consent_key(
        "gemini", "gemini-3.8-flash", "Inspect matching", ["image-a"],
        {"image-a": "Original"}, {"topology": "OTA"}, False, 0,
    )
    assert base == image_request_consent_key(
        "gemini", "gemini-3.8-flash", "Inspect matching", ["image-a"],
        {"image-a": "Original"}, {"topology": "OTA"}, False, 0,
    )
    assert base != image_request_consent_key(
        "gemini", "gemini-3.8-flash", "Inspect routing", ["image-a"],
        {"image-a": "Original"}, {"topology": "OTA"}, False, 0,
    )
    assert base != image_request_consent_key(
        "gemini", "gemini-3.8-flash", "Inspect matching", ["image-a"],
        {"image-a": "Original"}, {"topology": "OTA"}, False, 1,
    )


def test_provider_findings_are_available_as_unverified_followup_context():
    result = {"gemini": {"model": "mock-model", "status": "completed", "analysis": _analysis("Gemini", "A symmetric group is visible."), "error": None}}
    context = provider_findings_context(result)
    assert "Summary from Gemini" in context
    assert "A symmetric group is visible." in context
    assert "not DRC/LVS" in context


def test_provider_comparison_sends_identical_inputs_only_with_separate_consents():
    image = ImageInput("id", "a.png", "image/png", b"bytes", 2, 2, "Original")
    configs = {
        provider: VisionProviderConfig(provider, "model-" + provider, "https://example.invalid", "secret")
        for provider in ("gemini", "openai")
    }
    calls = []

    class FakeService:
        def __init__(self, config):
            self.config = config
            self.model_name = config.model

        def analyze(self, images, question, **kwargs):
            calls.append((self.config.provider, images, question, kwargs))
            return _analysis(self.config.provider, "A repeated device group is visible.")

    history = [{"role": "user", "content": "Differential pair"}]
    context = {"circuit_topology": "OTA"}
    results = run_vision_analysis_for_providers(
        configs, ["gemini", "openai"], [image], "Compare routes",
        consented_provider_ids={"gemini", "openai"},
        conversation_context=history, comparison_requested=True,
        engineering_context=context, service_factory=FakeService,
    )
    assert all(result["status"] == "completed" for result in results.values())
    assert len(calls) == 2
    assert calls[0][1:] == calls[1][1:]

    calls.clear()
    blocked = run_vision_analysis_for_providers(
        configs, ["gemini", "openai"], [image], "Compare routes",
        consented_provider_ids={"gemini"}, conversation_context=history,
        comparison_requested=True, engineering_context=context, service_factory=FakeService,
    )
    assert blocked["gemini"]["status"] == "completed"
    assert blocked["openai"]["status"] == "not_sent"
    assert len(calls) == 1


def test_analysis_export_excludes_image_bytes_and_keeps_provider_provenance():
    image = ImageInput("image-1", "layout.png", "image/png", b"private image bytes", 19, 11, "Original")
    results = {"gemini": {"model": "mock-model", "status": "completed", "analysis": _analysis("Gemini", "A route is visible."), "error": None}}
    report = build_analysis_report("Inspect matching", [image], results)
    encoded = str(report)
    assert "layout.png" in encoded
    assert "private image bytes" not in encoded
    assert "mock-model" in encoded
    markdown = analysis_report_markdown(report)
    assert "Limitations" in markdown
    assert "Provider metadata" in markdown


def test_artifact_report_key_tracks_content_and_parser_selection():
    original = artifact_report_key("review.rpt", b"DRC errors: 0", "EDA report")
    assert original == artifact_report_key("review.rpt", b"DRC errors: 0", "EDA report")
    assert original != artifact_report_key("review.rpt", b"DRC errors: 1", "EDA report")
    assert original != artifact_report_key("review.rpt", b"DRC errors: 0", "SPICE netlist")


def test_clear_all_removes_session_images_and_linked_messages_but_keeps_text_chat():
    history = [
        {"role": "user", "content": "text question", "image_ids": []},
        {"role": "user", "content": "inspect", "image_ids": ["image-1"]},
        {"role": "assistant", "content": "analysis", "image_ids": [], "source_image_ids": ["image-1"]},
    ]
    assets, selected, labels, new_history = clear_all_image_data(
        {"image-1": {"data": b"private"}}, ["image-1"], {"image-1": "Original"}, history
    )
    assert assets == {}
    assert selected == []
    assert labels == {}
    assert new_history == [history[0]]
