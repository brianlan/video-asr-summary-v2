from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
from unittest.mock import MagicMock, call

import pytest


def _load_pipeline_module():
    return importlib.import_module("video_asr_summary.pipeline")


def test_pipeline_runs_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _load_pipeline_module()

    audio_path = tmp_path / "audio.wav"

    mock_extract = MagicMock(return_value=audio_path)
    monkeypatch.setattr("video_asr_summary.pipeline.extract_audio", mock_extract)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )
    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_video_frames",
        lambda *args, **kwargs: [],
    )

    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = "transcribed text"

    fake_summarizer = MagicMock()
    fake_summarizer.summarize.return_value = "# Summary\n\n- short"

    result = pipeline.process_video(
        video_path=tmp_path / "input.mp4",
        language="es-ES",
        asr_client=fake_asr,
        summarizer=fake_summarizer,
    )

    assert result == {
        "transcript": "transcribed text",
        "corrected_transcript": None,
        "summary": "# Summary\n\n- short",
    }
    mock_extract.assert_called_once_with(
        tmp_path / "input.mp4",
        sample_rate=16000,
        audio_format="mp3",
        audio_bitrate="64k",
    )
    fake_asr.transcribe.assert_called_once_with(audio_path, language="es-ES")
    fake_summarizer.summarize.assert_called_once_with(
        "transcribed text", [], language="es-ES"
    )


def test_pipeline_uses_default_clients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline = _load_pipeline_module()

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

    monkeypatch.setattr("video_asr_summary.pipeline.LocalASRClient", StubASR)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.StructuredSummarizer", StubSummarizer
    )

    result = pipeline.process_video(video_path=tmp_path / "input.mp4")

    assert created_clients == {"asr": True, "summarizer": True}
    assert result == {
        "transcript": "text",
        "corrected_transcript": None,
        "summary": "# Value",
    }


def test_pipeline_allows_overriding_structured_summarizer_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline = _load_pipeline_module()

    audio_path = tmp_path / "audio.wav"
    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path
    )
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

    monkeypatch.setattr("video_asr_summary.pipeline.LocalASRClient", StubASR)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.StructuredSummarizer", StubSummarizer
    )

    result = pipeline.process_video(
        video_path=tmp_path / "clip.mp4",
        summarizer_model="gpt-custom",
    )

    assert captured["model"] == "gpt-custom"
    assert result == {
        "transcript": "full transcript",
        "corrected_transcript": None,
        "summary": "# stub summary",
    }


def test_pipeline_transcribes_multiple_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _load_pipeline_module()

    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.mp3"
    chunk_paths = [tmp_path / "chunk_0.mp3", tmp_path / "chunk_1.mp3"]
    for path in chunk_paths:
        path.write_bytes(b"chunk")

    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path
    )
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
        asr_client=fake_asr,
        summarizer=fake_summarizer,
    )

    assert fake_asr.transcribe.call_count == 2
    fake_asr.transcribe.assert_any_call(chunk_paths[0], language="zh")
    fake_asr.transcribe.assert_any_call(chunk_paths[1], language="zh")

    assert result["transcript"] == "first\n\nsecond"
    fake_summarizer.summarize.assert_called_once_with(
        "first\n\nsecond", [], language="zh"
    )


def test_pipeline_integration_asr_ocr_structured_summarizer_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _load_pipeline_module()

    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.mp3"
    chunks = [tmp_path / "chunk_0.mp3", tmp_path / "chunk_1.mp3"]
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_paths = [frame_dir / "frame_00001.jpg", frame_dir / "frame_00002.jpg"]
    for frame in frame_paths:
        frame.write_bytes(b"img")
    frames = [
        SimpleNamespace(
            timestamp_start=0.0,
            timestamp_end=5.0,
            frame_path=frame_paths[0],
            ocr_text="",
        ),
        SimpleNamespace(
            timestamp_start=5.0,
            timestamp_end=10.0,
            frame_path=frame_paths[1],
            ocr_text="",
        ),
    ]

    split_mock = MagicMock(return_value=chunks)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path
    )
    monkeypatch.setattr("video_asr_summary.pipeline.split_audio_on_silence", split_mock)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_video_frames",
        lambda *args, **kwargs: frames,
    )

    captured: dict[str, object] = {}
    asr_calls: list[tuple[Path, str]] = []
    ocr_calls: list[tuple[Path, str]] = []

    class StubASR:
        def transcribe(self, path: Path, *, language: str) -> str:
            asr_calls.append((path, language))
            if path == chunks[0]:
                return "first part"
            return "second part"

    class StubOCR:
        def describe_image(self, path: Path, *, language: str) -> str:
            ocr_calls.append((path, language))
            return f"ocr:{path.name}"

    class StubSummarizer:
        def summarize(
            self, transcript: str, frame_contexts: list[object], *, language: str
        ) -> str:
            captured["summarizer_transcript"] = transcript
            captured["summarizer_frame_contexts"] = frame_contexts
            captured["summarizer_language"] = language
            return "# summary"

    result = pipeline.process_video(
        video_path=video_path,
        language="ja",
        max_segment_duration=12.5,
        asr_client=StubASR(),
        enable_image_context=True,
        frame_output_dir=frame_dir,
        ocr_client=StubOCR(),
        summarizer=StubSummarizer(),
    )

    assert result == {
        "transcript": "first part\n\nsecond part",
        "corrected_transcript": None,
        "summary": "# summary",
    }
    assert asr_calls == [(chunks[0], "ja"), (chunks[1], "ja")]
    assert ocr_calls == [(frame_paths[0], "ja"), (frame_paths[1], "ja")]
    assert frames[0].ocr_text == "ocr:frame_00001.jpg"
    assert frames[1].ocr_text == "ocr:frame_00002.jpg"
    assert captured["summarizer_transcript"] == "first part\n\nsecond part"
    assert captured["summarizer_frame_contexts"] == frames
    assert captured["summarizer_language"] == "ja"
    split_mock.assert_called_once()
    assert split_mock.call_args.kwargs["max_duration"] == 12.5


def test_pipeline_integration_ocr_failure_skips_failed_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _load_pipeline_module()

    audio_path = tmp_path / "audio.mp3"
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_path = frame_dir / "frame_00001.jpg"
    frame_path.write_bytes(b"img")
    frame = SimpleNamespace(
        timestamp_start=0.0,
        timestamp_end=5.0,
        frame_path=frame_path,
        ocr_text="",
    )

    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path
    )
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )
    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_video_frames",
        lambda *args, **kwargs: [frame],
    )

    class StubASR:
        def transcribe(self, *_args, **_kwargs) -> str:
            return "raw transcript"

    class FlakyOCR:
        def describe_image(self, path: Path, *, language: str) -> str:
            if path == frame_path:
                raise RuntimeError("ocr server unavailable")
            return "unreachable"

    fake_summarizer = MagicMock()
    fake_summarizer.summarize.return_value = "# summary"

    result = pipeline.process_video(
        video_path=tmp_path / "video.mp4",
        asr_client=StubASR(),
        enable_image_context=True,
        frame_output_dir=frame_dir,
        ocr_client=FlakyOCR(),
        summarizer=fake_summarizer,
    )

    assert result == {
        "transcript": "raw transcript",
        "corrected_transcript": None,
        "summary": "# summary",
    }
    assert frame.ocr_text == ""
    fake_summarizer.summarize.assert_called_once_with(
        "raw transcript", [], language="en"
    )


def test_pipeline_long_audio_chunking_joins_transcript_and_preserves_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _load_pipeline_module()

    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.mp3"
    chunk_paths = [
        tmp_path / "chunk_000.mp3",
        tmp_path / "chunk_001.mp3",
        tmp_path / "chunk_002.mp3",
    ]

    split_mock = MagicMock(return_value=chunk_paths)
    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path
    )
    monkeypatch.setattr("video_asr_summary.pipeline.split_audio_on_silence", split_mock)

    fake_asr = MagicMock()
    fake_asr.transcribe.side_effect = ["part one", " part two ", ""]
    fake_summarizer = MagicMock()
    fake_summarizer.summarize.return_value = "# done"

    result = pipeline.process_video(
        video_path=video_path,
        language="fr-FR",
        max_segment_duration=30.0,
        asr_client=fake_asr,
        summarizer=fake_summarizer,
    )

    assert result["transcript"] == "part one\n\npart two"
    split_mock.assert_called_once()
    assert split_mock.call_args.kwargs["max_duration"] == 30.0
    fake_asr.transcribe.assert_has_calls(
        [
            call(chunk_paths[0], language="fr-FR"),
            call(chunk_paths[1], language="fr-FR"),
            call(chunk_paths[2], language="fr-FR"),
        ]
    )
    fake_summarizer.summarize.assert_called_once_with(
        "part one\n\npart two", [], language="fr-FR"
    )


def test_pipeline_returns_transcript_when_summarizer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _load_pipeline_module()

    audio_path = tmp_path / "audio.wav"
    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path
    )
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
        asr_client=fake_asr,
        summarizer=ExplodingSummarizer(),
    )

    assert result["transcript"] == "fully transcribed"
    assert result["summary"] is None
    assert result["summarizer_error"] == "network unavailable"


def test_pipeline_uses_asr_client_with_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test pipeline uses asr_client when injected with asr_options."""
    pipeline = _load_pipeline_module()

    audio_path = tmp_path / "audio.wav"
    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path
    )
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )

    created: dict[str, object] = {}

    class StubASRClient:
        def __init__(self, **kwargs) -> None:
            created["kwargs"] = kwargs

        def transcribe(self, path: Path, *, language: str) -> str:
            created["transcribe"] = (path, language)
            return "local transcript"

    monkeypatch.setattr("video_asr_summary.pipeline.LocalASRClient", StubASRClient)

    fake_summarizer = MagicMock()
    fake_summarizer.summarize.return_value = "# done"

    result = pipeline.process_video(
        video_path=tmp_path / "input.mp4",
        language="fr",
        asr_options={"model": "asr-model"},
        summarizer=fake_summarizer,
    )

    assert result == {
        "transcript": "local transcript",
        "corrected_transcript": None,
        "summary": "# done",
    }
    assert created["kwargs"] == {"model": "asr-model"}
    assert created["transcribe"] == (audio_path, "fr")
    fake_summarizer.summarize.assert_called_once_with(
        "local transcript", [], language="fr"
    )


def test_pipeline_populates_frame_ocr_text_for_structured_summarizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test pipeline uses LocalOCRClient for image context correction."""
    pipeline = _load_pipeline_module()

    audio_path = tmp_path / "audio.wav"
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_path = frame_dir / "frame_00001.jpg"
    frame_path.write_bytes(b"image")
    frame = SimpleNamespace(
        timestamp_start=0.0,
        timestamp_end=5.0,
        frame_path=frame_path,
        ocr_text="",
    )

    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path
    )
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )

    captured: dict[str, object] = {}

    class StubASR:
        def transcribe(self, *_args, **_kwargs) -> str:
            return "raw transcript"

    class StubOCRClient:
        def __init__(self) -> None:
            self.calls: list[Path] = []

        def describe_image(self, path: Path, *, language: str) -> str:
            self.calls.append(path)
            return f"Text on {path.name}: Launch Day"

    class StubSummarizer:
        def summarize(
            self, transcript: str, frame_contexts: list[object], *, language: str
        ) -> str:
            captured["transcript"] = transcript
            captured["frame_contexts"] = frame_contexts
            captured["summarizer_input"] = transcript
            captured["summarizer_language"] = language
            return "# summary"

    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_video_frames",
        lambda *args, **kwargs: [frame],
    )

    result = pipeline.process_video(
        video_path=tmp_path / "input.mp4",
        language="en",
        asr_client=StubASR(),
        enable_image_context=True,
        frame_interval_seconds=5,
        frame_output_dir=frame_dir,
        ocr_client=StubOCRClient(),
        summarizer=StubSummarizer(),
    )

    assert result == {
        "transcript": "raw transcript",
        "corrected_transcript": None,
        "summary": "# summary",
    }
    assert captured["transcript"] == "raw transcript"
    assert captured["summarizer_input"] == "raw transcript"
    assert captured["frame_contexts"] == [frame]
    assert frame.ocr_text == "Text on frame_00001.jpg: Launch Day"


def test_pipeline_uses_ocr_client_options_when_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test pipeline uses ocr_options when creating LocalOCRClient."""
    pipeline = _load_pipeline_module()

    audio_path = tmp_path / "audio.wav"
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_path = frame_dir / "frame_00001.jpg"
    frame_path.write_bytes(b"image")
    frame = SimpleNamespace(
        timestamp_start=0.0,
        timestamp_end=5.0,
        frame_path=frame_path,
        ocr_text="",
    )

    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path
    )
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )

    captured_ocr_kwargs: dict[str, object] = {}

    class StubASR:
        def transcribe(self, *_args, **_kwargs) -> str:
            return "raw transcript"

    class StubOCRClient:
        def __init__(self, **kwargs) -> None:
            captured_ocr_kwargs.update(kwargs)

        def describe_image(self, path: Path, *, language: str) -> str:
            return "OCR text"

    class StubSummarizer:
        def summarize(
            self, transcript: str, frame_contexts: list[object], *, language: str
        ) -> str:
            return "# summary"

    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_video_frames",
        lambda *args, **kwargs: [frame],
    )
    monkeypatch.setattr("video_asr_summary.pipeline.LocalOCRClient", StubOCRClient)

    result = pipeline.process_video(
        video_path=tmp_path / "input.mp4",
        language="en",
        asr_client=StubASR(),
        enable_image_context=True,
        frame_interval_seconds=5,
        frame_output_dir=frame_dir,
        ocr_options={"model": "custom-ocr-model"},
        summarizer=StubSummarizer(),
    )

    assert result == {
        "transcript": "raw transcript",
        "corrected_transcript": None,
        "summary": "# summary",
    }
    assert captured_ocr_kwargs.get("model") == "custom-ocr-model"


def test_pipeline_transcript_correction_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _load_pipeline_module()

    audio_path = tmp_path / "audio.wav"
    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path
    )
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )

    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = "raw transcript"
    fake_summarizer = MagicMock()
    fake_summarizer.summarize.return_value = "# summary"
    fake_corrector = MagicMock()
    fake_corrector.correct.return_value = "corrected transcript"

    result = pipeline.process_video(
        video_path=tmp_path / "input.mp4",
        asr_client=fake_asr,
        summarizer=fake_summarizer,
        transcript_corrector=fake_corrector,
    )

    assert result == {
        "transcript": "raw transcript",
        "corrected_transcript": None,
        "summary": "# summary",
    }
    fake_corrector.correct.assert_not_called()
    fake_summarizer.summarize.assert_called_once_with(
        "raw transcript", [], language="en"
    )


def test_pipeline_transcript_correction_uses_time_ordered_ocr_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _load_pipeline_module()

    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.mp3"
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_paths = [
        frame_dir / "frame_00003.jpg",
        frame_dir / "frame_00001.jpg",
        frame_dir / "frame_00002.jpg",
    ]
    for frame_path in frame_paths:
        frame_path.write_bytes(b"img")

    frames = [
        SimpleNamespace(
            timestamp_start=10.0,
            timestamp_end=15.0,
            frame_path=frame_paths[0],
            ocr_text="",
        ),
        SimpleNamespace(
            timestamp_start=0.0,
            timestamp_end=5.0,
            frame_path=frame_paths[1],
            ocr_text="",
        ),
        SimpleNamespace(
            timestamp_start=5.0,
            timestamp_end=10.0,
            frame_path=frame_paths[2],
            ocr_text="",
        ),
    ]

    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path
    )
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )
    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_video_frames",
        lambda *args, **kwargs: frames,
    )

    class StubASR:
        def transcribe(self, *_args, **_kwargs) -> str:
            return "raw transcript"

    class StubOCR:
        def describe_image(self, path: Path, *, language: str) -> str:
            mapping = {
                frame_paths[0]: "  third line  ",
                frame_paths[1]: "first line",
                frame_paths[2]: "   ",
            }
            return mapping[path]

    captured: dict[str, object] = {}

    class StubCorrector:
        def correct(
            self,
            transcript: str,
            *,
            image_context: list[str] | None = None,
            language: str | None = None,
        ) -> str:
            captured["corrector_transcript"] = transcript
            captured["corrector_image_context"] = image_context
            captured["corrector_language"] = language
            return "corrected transcript"

    class StubSummarizer:
        def summarize(
            self, transcript: str, frame_contexts: list[object], *, language: str
        ) -> str:
            captured["summarizer_transcript"] = transcript
            captured["summarizer_frame_contexts"] = frame_contexts
            captured["summarizer_language"] = language
            return "# summary"

    result = pipeline.process_video(
        video_path=video_path,
        language="ja",
        asr_client=StubASR(),
        enable_image_context=True,
        frame_output_dir=frame_dir,
        ocr_client=StubOCR(),
        transcript_corrector=StubCorrector(),
        enable_transcript_correction=True,
        summarizer=StubSummarizer(),
    )

    assert result == {
        "transcript": "raw transcript",
        "corrected_transcript": "corrected transcript",
        "summary": "# summary",
    }
    assert captured["corrector_transcript"] == "raw transcript"
    assert captured["corrector_language"] == "ja"
    assert captured["corrector_image_context"] == [
        "[00:00-00:05] first line",
        "[00:10-00:15] third line",
    ]
    assert captured["summarizer_transcript"] == "raw transcript"
    assert captured["summarizer_frame_contexts"] == frames
    assert captured["summarizer_language"] == "ja"


def test_pipeline_returns_summary_when_transcript_correction_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _load_pipeline_module()

    audio_path = tmp_path / "audio.wav"
    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_audio", lambda *args, **kwargs: audio_path
    )
    monkeypatch.setattr(
        "video_asr_summary.pipeline.split_audio_on_silence",
        lambda *args, **kwargs: [audio_path],
    )
    monkeypatch.setattr(
        "video_asr_summary.pipeline.extract_video_frames",
        lambda *args, **kwargs: [],
    )

    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = "raw transcript"
    fake_summarizer = MagicMock()
    fake_summarizer.summarize.return_value = "# summary"

    class ExplodingCorrector:
        def correct(self, *_args, **_kwargs):
            raise RuntimeError("correction backend unavailable")

    result = pipeline.process_video(
        video_path=tmp_path / "input.mp4",
        asr_client=fake_asr,
        transcript_corrector=ExplodingCorrector(),
        enable_transcript_correction=True,
        summarizer=fake_summarizer,
    )

    assert result == {
        "transcript": "raw transcript",
        "corrected_transcript": None,
        "summary": "# summary",
        "transcript_correction_error": "correction backend unavailable",
    }
    fake_summarizer.summarize.assert_called_once_with(
        "raw transcript", [], language="en"
    )


def test_process_video_cli_exposes_message_receiver_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "process_video.py"
    )
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


def test_process_video_cli_exposes_transcript_correction_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "process_video.py"
    )
    spec = importlib.util.spec_from_file_location("process_video_cli", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)  # type: ignore[assignment]

    def run_with_args(argv: list[str]) -> argparse.Namespace:
        monkeypatch.setattr(cli.sys, "argv", argv)
        return cli.parse_args()

    args_default = run_with_args(["process_video", "sample.mp4"])
    assert args_default.enable_transcript_correction is False

    args_enabled = run_with_args(
        ["process_video", "sample.mp4", "--enable-transcript-correction"]
    )
    assert args_enabled.enable_transcript_correction is True

    args_disabled = run_with_args(
        ["process_video", "sample.mp4", "--no-enable-transcript-correction"]
    )
    assert args_disabled.enable_transcript_correction is False


def test_process_video_cli_load_env_file_sets_missing_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "process_video.py"
    )
    spec = importlib.util.spec_from_file_location("process_video_cli", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_ACCESS_TOKEN=from-dotenv\nLARK_MESSAGE_RECEIVER_ID=dotenv-user\n"
    )

    monkeypatch.delenv("OPENAI_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LARK_MESSAGE_RECEIVER_ID", raising=False)

    cli.load_env_file(env_file)

    assert cli.os.environ["OPENAI_ACCESS_TOKEN"] == "from-dotenv"
    assert cli.os.environ["LARK_MESSAGE_RECEIVER_ID"] == "dotenv-user"


def test_process_video_cli_load_env_file_does_not_override_existing_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "process_video.py"
    )
    spec = importlib.util.spec_from_file_location("process_video_cli", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_ACCESS_TOKEN=from-dotenv\n")

    monkeypatch.setenv("OPENAI_ACCESS_TOKEN", "already-set")

    cli.load_env_file(env_file)

    assert cli.os.environ["OPENAI_ACCESS_TOKEN"] == "already-set"


def test_process_video_cli_main_loads_dotenv_before_parsing_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "process_video.py"
    )
    spec = importlib.util.spec_from_file_location("process_video_cli", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    video_file = tmp_path / "clip.mp4"
    video_file.write_text("video")
    (tmp_path / ".env").write_text("LARK_MESSAGE_RECEIVER_ID=dotenv-user\n")

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        cli,
        "process_video",
        lambda *args, **kwargs: {
            "transcript": "ASR transcript",
            "summary": "# summary",
        },
    )

    captured: dict[str, str] = {}

    def fake_create_summary_document(*args, **kwargs):
        captured["message_receiver_id"] = kwargs["message_receiver_id"]
        return {"url": "https://example.com/doc"}

    monkeypatch.setattr(cli, "create_summary_document", fake_create_summary_document)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["process_video", str(video_file), "--publish-to-lark"],
    )

    cli.main()
    _ = capsys.readouterr()

    assert captured["message_receiver_id"] == "dotenv-user"


def test_process_video_cli_env_defaults_for_asr_and_ocr_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "process_video.py"
    )
    spec = importlib.util.spec_from_file_location("process_video_cli", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    monkeypatch.setenv("ASR_URL", "http://127.0.0.1:9002")
    monkeypatch.setenv("OCR_URL", "http://127.0.0.1:9001")
    monkeypatch.setenv("ASR_MODEL", "my-asr")
    monkeypatch.setenv("OCR_MODEL", "my-ocr")

    monkeypatch.setattr(cli.sys, "argv", ["process_video", "sample.mp4"])
    args = cli.parse_args()

    assert args.asr_url == "http://127.0.0.1:9002"
    assert args.ocr_url == "http://127.0.0.1:9001"
    assert args.asr_model == "my-asr"
    assert args.ocr_model == "my-ocr"


def test_process_video_cli_main_loads_dotenv_for_asr_ocr_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "process_video.py"
    )
    spec = importlib.util.spec_from_file_location("process_video_cli", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    video_file = tmp_path / "clip.mp4"
    video_file.write_text("video")
    (tmp_path / ".env").write_text(
        "ASR_URL=http://127.0.0.1:9002\n"
        "OCR_URL=http://127.0.0.1:9001\n"
        "ASR_MODEL=dotenv-asr\n"
        "OCR_MODEL=dotenv-ocr\n"
    )

    monkeypatch.chdir(tmp_path)

    captured: dict[str, object] = {}

    def fake_process_video(*args, **kwargs):
        captured["asr_options"] = kwargs["asr_options"]
        captured["ocr_options"] = kwargs["ocr_options"]
        captured["enable_transcript_correction"] = kwargs[
            "enable_transcript_correction"
        ]
        return {"transcript": "ASR transcript", "summary": "# summary"}

    monkeypatch.setattr(cli, "process_video", fake_process_video)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "process_video",
            str(video_file),
            "--enable-image-context",
            "--enable-transcript-correction",
        ],
    )

    cli.main()
    _ = capsys.readouterr()

    assert captured["asr_options"] == {
        "base_url": "http://127.0.0.1:9002",
        "model": "dotenv-asr",
    }
    assert captured["ocr_options"] == {
        "base_url": "http://127.0.0.1:9001",
        "model": "dotenv-ocr",
    }
    assert captured["enable_transcript_correction"] is True


def test_process_video_cli_prints_transcript_when_lark_publish_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "process_video.py"
    )
    spec = importlib.util.spec_from_file_location("process_video_cli", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    video_file = tmp_path / "clip.mp4"
    video_file.write_text("video")

    monkeypatch.setattr(
        cli,
        "process_video",
        lambda *args, **kwargs: {
            "transcript": "ASR transcript",
            "summary": "# summary",
        },
    )

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


def test_process_video_cli_publishes_correction_failure_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "process_video.py"
    )
    spec = importlib.util.spec_from_file_location("process_video_cli", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    video_file = tmp_path / "clip.mp4"
    video_file.write_text("video")

    monkeypatch.setattr(
        cli,
        "process_video",
        lambda *args, **kwargs: {
            "transcript": "ASR transcript",
            "corrected_transcript": None,
            "transcript_correction_error": "backend down",
            "summary": "# summary",
        },
    )

    captured: dict[str, str] = {}

    def fake_create_summary_document(*args, **kwargs):
        captured["corrected_transcript"] = kwargs["corrected_transcript"]
        return {"url": "https://example.com/doc"}

    monkeypatch.setattr(cli, "create_summary_document", fake_create_summary_document)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "process_video",
            str(video_file),
            "--publish-to-lark",
            "--enable-transcript-correction",
        ],
    )

    cli.main()
    _ = capsys.readouterr()

    assert captured["corrected_transcript"] == "Correction failed: backend down"
