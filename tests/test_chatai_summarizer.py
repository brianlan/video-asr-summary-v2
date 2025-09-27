from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _setup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_ACCESS_TOKEN", "token")


def test_summarize_parses_plain_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary.summarizer import ChataiSummarizer

    _setup_env(monkeypatch)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"summary": "short", "highlights": ["a", "b"]}'
                }
            }
        ]
    }
    mock_response.raise_for_status.return_value = None

    captured_payload = {}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> MagicMock:  # type: ignore[override]
        captured_payload["url"] = url
        captured_payload["headers"] = headers
        captured_payload["json"] = json
        captured_payload["timeout"] = timeout
        return mock_response

    monkeypatch.setattr("video_asr_summary.summarizer.requests.post", fake_post)

    summarizer = ChataiSummarizer()
    result = summarizer.summarize("text to summarize")

    assert result == {"summary": "short", "highlights": ["a", "b"]}
    assert captured_payload["url"].endswith("/chat/completions")
    assert captured_payload["headers"]["Authorization"] == "Bearer token"


def test_summarize_parses_markdown_wrapped_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary.summarizer import ChataiSummarizer

    _setup_env(monkeypatch)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "```json\n{\\n  \"summary\": \"hello\"\n}\n```"
                }
            }
        ]
    }
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(
        "video_asr_summary.summarizer.requests.post", lambda *args, **kwargs: mock_response
    )

    summarizer = ChataiSummarizer()
    result = summarizer.summarize("another text")

    assert result == {"summary": "hello"}
