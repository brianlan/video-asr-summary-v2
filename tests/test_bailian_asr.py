from __future__ import annotations
from pathlib import Path

import pytest


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake-audio")
    return audio


def test_transcribe_posts_audio_with_language(audio_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary.asr_client import BailianASRClient

    monkeypatch.setenv("BAILIAN_API_KEY", "secret")

    captured_call = {}

    class FakeResponse:
        def __init__(self) -> None:
            self.output = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"text": "hello world"},
                            ],
                        }
                    }
                ]
            }

    def fake_call(*, model: str, messages: list, api_key: str, result_format: str, asr_options: dict, **kwargs) -> FakeResponse:  # type: ignore[override]
        captured_call["model"] = model
        captured_call["messages"] = messages
        captured_call["api_key"] = api_key
        captured_call["result_format"] = result_format
        captured_call["asr_options"] = asr_options
        captured_call["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr(
        "video_asr_summary.asr_client.MultiModalConversation.call",
        staticmethod(fake_call),
    )

    client = BailianASRClient()
    transcript = client.transcribe(audio_file, language="zh-CN")

    assert transcript == "hello world"
    assert captured_call["model"] == "qwen3-asr-flash"
    assert captured_call["api_key"] == "secret"
    assert captured_call["result_format"] == "message"
    assert captured_call["asr_options"]["language"] == "zh-CN"

    messages = captured_call["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == [{"text": ""}]

    audio_message = messages[1]
    assert audio_message["role"] == "user"
    assert audio_message["content"] == [{"audio": str(audio_file)}]


def test_transcribe_raises_on_error(audio_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary.asr_client import BailianASRClient

    monkeypatch.setenv("BAILIAN_API_KEY", "secret")

    def fake_call(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "video_asr_summary.asr_client.MultiModalConversation.call",
        staticmethod(fake_call),
    )

    client = BailianASRClient()
    with pytest.raises(RuntimeError):
        client.transcribe(audio_file)
