from __future__ import annotations

import argparse
import importlib.util
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


def test_pipeline_allows_overriding_summarizer_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from video_asr_summary import pipeline

    audio_path = tmp_path / "audio.wav"
    monkeypatch.setattr("video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )

    class StubASR:
        def transcribe(self, path: Path, *, language: str) -> str:
            return "full transcript"

    captured: dict[str, str] = {}

    class StubSummarizer:
        def __init__(self, *, model: str = "gpt-5-mini") -> None:
            captured["model"] = model

        def summarize(self, *_args, **_kwargs) -> str:
            return "# stub summary"

    monkeypatch.setattr("video_asr_summary.pipeline.BailianASRClient", StubASR)
    monkeypatch.setattr("video_asr_summary.pipeline.ChataiSummarizer", StubSummarizer)

    result = pipeline.process_video(
        video_path=tmp_path / "clip.mp4",
        summarizer_model="gpt-custom",
    )

    assert captured["model"] == "gpt-custom"
    assert result == {"transcript": "full transcript", "summary": "# stub summary"}


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


def test_pipeline_returns_transcript_when_summarizer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_asr_summary import pipeline

    audio_path = tmp_path / "audio.wav"
    monkeypatch.setattr("video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )

    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = "fully transcribed"

    class ExplodingSummarizer:
        def summarize(self, *_args, **_kwargs):
            raise RuntimeError("network unavailable")

    result = pipeline.process_video(
        video_path=tmp_path / "broken.mp4",
        bailian_client=fake_asr,
        summarizer=ExplodingSummarizer(),
    )

    assert result["transcript"] == "fully transcribed"
    assert result["summary"] is None
    assert result["summarizer_error"] == "network unavailable"


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


def test_pipeline_corrects_transcript_with_image_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary import pipeline

    audio_path = tmp_path / "audio.wav"
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_path = frame_dir / "frame_00001.jpg"
    frame_path.write_bytes(b"image")

    monkeypatch.setattr("video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )

    captured: dict[str, object] = {}

    class StubLocalASR:
        def transcribe(self, *_args, **_kwargs) -> str:
            return "raw transcript"

    class StubVisionClient:
        def __init__(self) -> None:
            self.calls: list[Path] = []

        def describe_image(self, path: Path, *, language: str) -> str:
            self.calls.append(path)
            return f"Text on {path.name}: Launch Day"

    class StubCorrector:
        def correct(self, transcript: str, *, image_context: list[str], language: str) -> str:
            captured["transcript"] = transcript
            captured["image_context"] = image_context
            captured["language"] = language
            return "corrected transcript"

    class StubSummarizer:
        def summarize(self, transcript: str, *, language: str) -> str:
            captured["summarizer_input"] = transcript
            captured["summarizer_language"] = language
            return "# summary"

    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_video_frames",
        lambda *args, **kwargs: [frame_path],
    )

    result = pipeline.process_video(
        video_path=tmp_path / "input.mp4",
        language="en",
        asr_backend="local",
        local_client=StubLocalASR(),
        enable_image_context=True,
        frame_interval_seconds=5,
        frame_output_dir=frame_dir,
        vision_client=StubVisionClient(),
        transcript_corrector=StubCorrector(),
        summarizer=StubSummarizer(),
    )

    assert result == {"transcript": "corrected transcript", "summary": "# summary"}
    assert captured["transcript"] == "raw transcript"
    assert captured["summarizer_input"] == "corrected transcript"
    assert frame_dir.joinpath("frame_00001.txt").read_text() == "Text on frame_00001.jpg: Launch Day"
    assert captured["image_context"] == ["Text on frame_00001.jpg: Launch Day"]


def test_pipeline_skips_image_context_for_remote_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary import pipeline

    audio_path = tmp_path / "audio.wav"
    monkeypatch.setattr("video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )

    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = "remote transcript"

    fake_summarizer = MagicMock()
    fake_summarizer.summarize.return_value = "# sum"

    def fail_extract(*_args, **_kwargs):
        raise AssertionError("frame extraction should not run")

    monkeypatch.setattr("video_asr_summary.pipeline.extract_video_frames", fail_extract)

    result = pipeline.process_video(
        video_path=tmp_path / "clip.mp4",
        enable_image_context=True,
        bailian_client=fake_asr,
        summarizer=fake_summarizer,
    )

    assert result["transcript"] == "remote transcript"
    fake_summarizer.summarize.assert_called_once_with("remote transcript", language="en")


def test_pipeline_reuses_local_asr_runtime_for_vision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary import pipeline
    from video_asr_summary.asr_client import LocalQwenASRClient

    audio_path = tmp_path / "audio.wav"
    frame_path = tmp_path / "frame_00001.jpg"
    frame_path.write_bytes(b"image")

    monkeypatch.setattr("video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )
    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_video_frames",
        lambda *args, **kwargs: [frame_path],
    )

    captured_vision_kwargs: dict[str, object] = {}

    class RecordingVisionClient:
        def __init__(self, **kwargs) -> None:
            captured_vision_kwargs.update(kwargs)

        def describe_image(self, *_args, **_kwargs) -> str:
            return "desc"

    monkeypatch.setattr("video_asr_summary.pipeline.LocalQwenVisionClient", RecordingVisionClient)

    class StubLocalQwen(LocalQwenASRClient):
        def __init__(self) -> None:
            self.model_path = "stub-model"

        def transcribe(self, *_args, **_kwargs) -> str:
            return "raw"

        def export_runtime_components(self) -> dict[str, object]:
            return {
                "llm": "llm",
                "processor": "processor",
                "sampling_params": "sampling",
                "process_mm_info": lambda *args, **kwargs: (None, None, None),
            }

        @property
        def model_path_str(self) -> str:  # type: ignore[override]
            return self.model_path

    class StubCorrector:
        def correct(self, *_args, **_kwargs) -> str:
            return "corrected"

    class StubSummarizer:
        def summarize(self, *_args, **_kwargs) -> str:
            return "# summary"

    result = pipeline.process_video(
        video_path=tmp_path / "clip.mp4",
        asr_backend="local",
        local_client=StubLocalQwen(),
        enable_image_context=True,
        transcript_corrector=StubCorrector(),
        summarizer=StubSummarizer(),
    )

    assert result == {"transcript": "corrected", "summary": "# summary"}
    assert captured_vision_kwargs["llm"] == "llm"
    assert captured_vision_kwargs["processor"] == "processor"
    assert captured_vision_kwargs["sampling_params"] == "sampling"


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


def test_process_video_cli_exposes_message_receiver_id(monkeypatch: pytest.MonkeyPatch) -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "process_video.py"
    spec = importlib.util.spec_from_file_location("process_video_cli", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)  # type: ignore[assignment]

    def run_with_args(argv: list[str]) -> argparse.Namespace:
        monkeypatch.setattr(cli.sys, "argv", argv)
        return cli.parse_args()

    args_default = run_with_args(["process_video", "sample.mp4"])
    assert args_default.message_receiver_id == "1gc832ed"

    args_custom = run_with_args(
        ["process_video", "sample.mp4", "--message-receiver-id", "custom-user"]
    )
    assert args_custom.message_receiver_id == "custom-user"


def test_process_video_cli_prints_transcript_when_lark_publish_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "process_video.py"
    spec = importlib.util.spec_from_file_location("process_video_cli", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)  # type: ignore[assignment]

    video_file = tmp_path / "clip.mp4"
    video_file.write_text("video")

    monkeypatch.setattr(cli, "process_video", lambda *args, **kwargs: {
        "transcript": "ASR transcript",
        "summary": "# summary",
    })

    def always_fail(*_args, **_kwargs):
        raise cli.LarkDocError("lark network rejected request")

    monkeypatch.setattr(cli, "create_summary_document", always_fail)

    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "process_video",
            str(video_file),
            "--publish-to-lark",
        ],
    )

    cli.main()

    captured = capsys.readouterr()
    assert "ASR transcript" in captured.out
    assert "lark network rejected request" in captured.out
