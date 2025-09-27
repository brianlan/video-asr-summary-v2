from __future__ import annotations

from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import pytest


def test_pipeline_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary import pipeline

    audio_path = tmp_path / "audio.wav"

    mock_extract = MagicMock(return_value=audio_path)
    monkeypatch.setattr("video_asr_summary.pipeline.extract_audio", mock_extract)

    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = "transcribed text"

    fake_summarizer = MagicMock()
    fake_summarizer.summarize.return_value = {"summary": "short"}

    result = pipeline.process_video(
        video_path=tmp_path / "input.mp4",
        language="es-ES",
        bailian_client=fake_asr,
        summarizer=fake_summarizer,
    )

    assert result == {"transcript": "transcribed text", "summary": {"summary": "short"}}
    mock_extract.assert_called_once_with(tmp_path / "input.mp4")
    fake_asr.transcribe.assert_called_once_with(audio_path, language="es-ES")
    fake_summarizer.summarize.assert_called_once()


def test_pipeline_uses_default_clients(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from video_asr_summary import pipeline

    monkeypatch.setattr("video_asr_summary.pipeline.extract_audio", lambda path: tmp_path / "audio.wav")

    created_clients: Dict[str, object] = {}

    class StubASR:
        def __init__(self, *args, **kwargs):
            created_clients["asr"] = True

        def transcribe(self, *_args, **_kwargs):
            return "text"

    class StubSummarizer:
        def __init__(self, *args, **kwargs):
            created_clients["summarizer"] = True

        def summarize(self, *_args, **_kwargs):
            return {"summary": "value"}

    monkeypatch.setattr("video_asr_summary.pipeline.BailianASRClient", StubASR)
    monkeypatch.setattr("video_asr_summary.pipeline.ChataiSummarizer", StubSummarizer)

    result = pipeline.process_video(video_path=tmp_path / "input.mp4")

    assert created_clients == {"asr": True, "summarizer": True}
    assert result == {"transcript": "text", "summary": {"summary": "value"}}
