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
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )

    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = "transcribed text"

    fake_summarizer = MagicMock()
    fake_summarizer.summarize.return_value = "# Summary\n\n- short"

    result = pipeline.process_video(
        video_path=tmp_path / "input.mp4",
        language="es-ES",
        bailian_client=fake_asr,
        summarizer=fake_summarizer,
    )

    assert result == {"transcript": "transcribed text", "summary": "# Summary\n\n- short"}
    mock_extract.assert_called_once_with(
        tmp_path / "input.mp4",
        sample_rate=16000,
        audio_format="mp3",
        audio_bitrate="64k",
    )
    fake_asr.transcribe.assert_called_once_with(audio_path, language="es-ES")
    fake_summarizer.summarize.assert_called_once()


def test_pipeline_uses_default_clients(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from video_asr_summary import pipeline

    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_audio",
        lambda *args, **kwargs: tmp_path / "audio.wav",
    )
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [tmp_path / "audio.wav"],
    )

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
            return "# Value"

    monkeypatch.setattr("video_asr_summary.pipeline.BailianASRClient", StubASR)
    monkeypatch.setattr("video_asr_summary.pipeline.ChataiSummarizer", StubSummarizer)

    result = pipeline.process_video(video_path=tmp_path / "input.mp4")

    assert created_clients == {"asr": True, "summarizer": True}
    assert result == {"transcript": "text", "summary": "# Value"}


def test_pipeline_transcribes_multiple_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary import pipeline

    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.mp3"
    chunk_paths = [tmp_path / "chunk_0.mp3", tmp_path / "chunk_1.mp3"]
    for path in chunk_paths:
        path.write_bytes(b"chunk")

    monkeypatch.setattr("video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: chunk_paths,
    )

    fake_asr = MagicMock()
    fake_asr.transcribe.side_effect = ["first", "second"]

    fake_summarizer = MagicMock()
    fake_summarizer.summarize.return_value = "# ok"

    result = pipeline.process_video(
        video_path=video_path,
        language="zh",
        bailian_client=fake_asr,
        summarizer=fake_summarizer,
    )

    assert fake_asr.transcribe.call_count == 2
    fake_asr.transcribe.assert_any_call(chunk_paths[0], language="zh")
    fake_asr.transcribe.assert_any_call(chunk_paths[1], language="zh")

    assert result["transcript"] == "first\n\nsecond"
    fake_summarizer.summarize.assert_called_once_with("first\n\nsecond", language="zh")


def test_pipeline_uses_local_backend_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary import pipeline

    audio_path = tmp_path / "audio.wav"
    monkeypatch.setattr("video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )

    created: dict[str, object] = {}

    class StubLocalClient:
        def __init__(self, **kwargs) -> None:
            created["kwargs"] = kwargs

        def transcribe(self, path: Path, *, language: str) -> str:
            created["transcribe"] = (path, language)
            return "local transcript"

    monkeypatch.setattr("video_asr_summary.pipeline.LocalQwenASRClient", StubLocalClient)

    fake_summarizer = MagicMock()
    fake_summarizer.summarize.return_value = "# done"

    result = pipeline.process_video(
        video_path=tmp_path / "input.mp4",
        language="fr",
        asr_backend="local",
        local_asr_options={"model_path": "/models/qwen"},
        summarizer=fake_summarizer,
    )

    assert result == {"transcript": "local transcript", "summary": "# done"}
    assert created["kwargs"] == {"model_path": "/models/qwen"}
    assert created["transcribe"] == (audio_path, "fr")
    fake_summarizer.summarize.assert_called_once_with("local transcript", language="fr")


def test_pipeline_raises_for_unknown_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary import pipeline

    audio_path = tmp_path / "audio.wav"
    monkeypatch.setattr("video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )

    with pytest.raises(ValueError):
        pipeline.process_video(tmp_path / "video.mp4", asr_backend="unknown")
