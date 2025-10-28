from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _setup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_ACCESS_TOKEN", "token")


def test_summarize_returns_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary.summarizer import ChataiSummarizer

    _setup_env(monkeypatch)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "# Title\n\n## Section\n- point A\n- point B"
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

    assert result == "# Title\n\n## Section\n- point A\n- point B"
    assert captured_payload["url"].endswith("/chat/completions")
    assert captured_payload["headers"]["Authorization"] == "Bearer token"
    system_prompt = captured_payload["json"]["messages"][0]["content"]
    assert "Reorganize the provided transcript" in system_prompt
    assert "Begin with a single H1 heading" in system_prompt


def test_summarize_raises_when_content_not_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary.summarizer import ChataiSummarizer

    _setup_env(monkeypatch)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": {"summary": "hello"}
                }
            }
        ]
    }
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(
        "video_asr_summary.summarizer.requests.post", lambda *args, **kwargs: mock_response
    )

    summarizer = ChataiSummarizer()
    with pytest.raises(RuntimeError, match="content is not text"):
        summarizer.summarize("another text")
