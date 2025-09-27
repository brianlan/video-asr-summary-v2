from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake-audio")
    return audio


def test_transcribe_posts_audio_with_language(audio_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary.asr_client import BailianASRClient

    monkeypatch.setenv("BAILIAN_API_KEY", "secret")

    mock_response = MagicMock()
    mock_response.json.return_value = {"output": {"text": "hello world"}}
    mock_response.raise_for_status.return_value = None

    captured_payload = {}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> MagicMock:  # type: ignore[override]
        captured_payload["url"] = url
        captured_payload["headers"] = headers
        captured_payload["json"] = json
        captured_payload["timeout"] = timeout
        return mock_response

    monkeypatch.setattr("video_asr_summary.asr_client.requests.post", fake_post)

    client = BailianASRClient()
    transcript = client.transcribe(audio_file, language="zh-CN")

    assert transcript == "hello world"
    assert "Authorization" in captured_payload["headers"]
    assert captured_payload["headers"]["Authorization"] == "Bearer secret"
    assert captured_payload["json"]["parameters"]["language"] == "zh-CN"
    encoded_audio = captured_payload["json"]["input"]["audio"]["content"]
    assert base64.b64decode(encoded_audio) == b"fake-audio"


def test_transcribe_raises_on_error(audio_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary.asr_client import BailianASRClient

    monkeypatch.setenv("BAILIAN_API_KEY", "secret")

    def fake_post(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("boom")

    monkeypatch.setattr("video_asr_summary.asr_client.requests.post", fake_post)

    client = BailianASRClient()
    with pytest.raises(RuntimeError):
        client.transcribe(audio_file)
