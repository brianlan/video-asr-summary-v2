from __future__ import annotations

from importlib import import_module
from unittest.mock import MagicMock

import pytest


def _setup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_ACCESS_TOKEN", "token")


def test_summarize_returns_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    summarizer_module = import_module("video_asr_summary.summarizer")
    ChataiSummarizer = summarizer_module.ChataiSummarizer

    _setup_env(monkeypatch)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "# Title\n\n## Section\n- point A\n- point B"}}
        ]
    }
    mock_response.raise_for_status.return_value = None

    captured_payload = {}

    def fake_post(
        url: str,
        headers: dict[str, object],
        json: dict[str, object],
        timeout: int,
    ) -> MagicMock:  # type: ignore[override]
        captured_payload["url"] = url
        captured_payload["headers"] = headers
        captured_payload["json"] = json
        captured_payload["timeout"] = timeout
        return mock_response

    monkeypatch.setattr("video_asr_summary.summarizer.requests.post", fake_post)

    summarizer = ChataiSummarizer()
    result = summarizer.summarize("text to summarize")

    assert result == "# Title\n\n## Section\n- point A\n- point B"
    assert captured_payload["url"].endswith("/chat/completions")
    assert captured_payload["headers"]["Authorization"] == "Bearer token"
    system_prompt = captured_payload["json"]["messages"][0]["content"]
    assert "Role: You are an expert editor and content analyst." in system_prompt
    assert "Begin with a single H1 heading" in system_prompt


def test_summarize_raises_when_content_not_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summarizer_module = import_module("video_asr_summary.summarizer")
    ChataiSummarizer = summarizer_module.ChataiSummarizer

    _setup_env(monkeypatch)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": {"summary": "hello"}}}]
    }
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(
        "video_asr_summary.summarizer.requests.post",
        lambda *args, **kwargs: mock_response,
    )

    summarizer = ChataiSummarizer()
    with pytest.raises(RuntimeError, match="content is not text"):
        summarizer.summarize("another text")


def test_structured_summarizer_single_pass_prompt_includes_timestamped_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    audio_module = import_module("video_asr_summary.audio")
    summarizer_module = import_module("video_asr_summary.summarizer")
    FrameContext = audio_module.FrameContext
    StructuredSummarizer = summarizer_module.StructuredSummarizer

    _setup_env(monkeypatch)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "# Structured output"}}]
    }
    mock_response.raise_for_status.return_value = None

    captured_payload = {}

    def fake_post(
        url: str,
        headers: dict[str, object],
        json: dict[str, object],
        timeout: int,
    ) -> MagicMock:  # type: ignore[override]
        captured_payload["url"] = url
        captured_payload["headers"] = headers
        captured_payload["json"] = json
        captured_payload["timeout"] = timeout
        return mock_response

    monkeypatch.setattr("video_asr_summary.summarizer.requests.post", fake_post)

    frame_contexts = [
        FrameContext(
            timestamp_start=0.0,
            timestamp_end=5.0,
            frame_path=Path("frame_00001.jpg"),
            ocr_text="Slide title appears",
        ),
        FrameContext(
            timestamp_start=5.0,
            timestamp_end=10.0,
            frame_path=Path("frame_00002.jpg"),
            ocr_text="   ",
        ),
    ]

    summarizer = StructuredSummarizer()
    result = summarizer.summarize("Raw transcript text", frame_contexts, language="en")

    assert result == "# Structured output"
    assert captured_payload["url"].endswith("/chat/completions")
    assert captured_payload["headers"]["Authorization"] == "Bearer token"

    payload = captured_payload["json"]
    system_prompt = payload["messages"][0]["content"]
    assert "[MM:SS]" in system_prompt

    user_content = payload["messages"][1]["content"]
    assert "=== TRANSCRIPT ===" in user_content
    assert "Raw transcript text" in user_content
    assert "=== VISUAL CONTEXT (Time-Ordered) ===" in user_content
    assert "[00:00-00:05] Slide title appears" in user_content
    assert "[00:05-00:10]" not in user_content


def test_transcript_corrector_posts_transcript_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summarizer_module = import_module("video_asr_summary.summarizer")
    TranscriptCorrector = summarizer_module.TranscriptCorrector

    _setup_env(monkeypatch)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"corrected_transcript": "Clean text"}'}}]
    }
    mock_response.raise_for_status.return_value = None

    captured_payload = {}

    def fake_post(
        url: str,
        headers: dict[str, object],
        json: dict[str, object],
        timeout: int,
    ) -> MagicMock:  # type: ignore[override]
        captured_payload["url"] = url
        captured_payload["headers"] = headers
        captured_payload["json"] = json
        captured_payload["timeout"] = timeout
        return mock_response

    monkeypatch.setattr("video_asr_summary.summarizer.requests.post", fake_post)

    corrector = TranscriptCorrector()
    result = corrector.correct(
        "Raw tx",
        image_context=["Frame 1 shows Launch Day"],
        language="en",
    )

    assert result == "Clean text"
    payload = captured_payload["json"]
    assert payload["model"] == "gpt-5-mini"
    user_content = payload["messages"][1]["content"]
    assert "Raw tx" in user_content
    assert "Launch Day" in user_content


def test_transcript_corrector_accepts_markdown_wrapped_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summarizer_module = import_module("video_asr_summary.summarizer")
    TranscriptCorrector = summarizer_module.TranscriptCorrector

    _setup_env(monkeypatch)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '```json\n{\n  "corrected_transcript": "Texto limpio"\n}\n```'
                }
            }
        ]
    }
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(
        "video_asr_summary.summarizer.requests.post",
        lambda *args, **kwargs: mock_response,
    )

    corrector = TranscriptCorrector()
    result = corrector.correct(
        "Entrada", image_context=["Frame shows texto"], language="es"
    )

    assert result == "Texto limpio"


def test_transcript_corrector_falls_back_to_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summarizer_module = import_module("video_asr_summary.summarizer")
    TranscriptCorrector = summarizer_module.TranscriptCorrector

    _setup_env(monkeypatch)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Clean text with fixes"}}]
    }
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(
        "video_asr_summary.summarizer.requests.post",
        lambda *args, **kwargs: mock_response,
    )

    corrector = TranscriptCorrector()
    assert (
        corrector.correct("noisy", image_context=[], language="en")
        == "Clean text with fixes"
    )


def test_transcript_corrector_repairs_trailing_comma_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("json_repair")
    summarizer_module = import_module("video_asr_summary.summarizer")
    TranscriptCorrector = summarizer_module.TranscriptCorrector

    _setup_env(monkeypatch)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"corrected_transcript": "Ok", }'}}]
    }
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(
        "video_asr_summary.summarizer.requests.post",
        lambda *args, **kwargs: mock_response,
    )

    corrector = TranscriptCorrector()
    assert corrector.correct("src", image_context=["ctx"], language="en") == "Ok"
