from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

audio_module = importlib.import_module("video_asr_summary.audio")
FrameContext = audio_module.FrameContext
extract_audio = audio_module.extract_audio
extract_video_frames = audio_module.extract_video_frames


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake-video-bytes")
    return video


def test_extract_audio_invokes_ffmpeg(sample_video: Path, tmp_path: Path) -> None:
    with patch("video_asr_summary.audio.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        output_path = extract_audio(
            sample_video, output_path=tmp_path / "out.wav", sample_rate=16000
        )

    assert output_path == tmp_path / "out.wav"
    mock_run.assert_called_once()
    called_args = mock_run.call_args[0][0]
    assert called_args[0] == "ffmpeg"
    assert "-ar" in called_args
    assert "-ac" in called_args  # ensure mono conversion
    assert "pcm_s16le" in called_args


def test_extract_audio_supports_mp3(sample_video: Path, tmp_path: Path) -> None:
    with patch("video_asr_summary.audio.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        output = extract_audio(
            sample_video,
            output_path=tmp_path / "compressed.mp3",
            sample_rate=8000,
            audio_format="mp3",
            audio_bitrate="48k",
        )

    assert output == tmp_path / "compressed.mp3"
    called_args = mock_run.call_args[0][0]
    assert "libmp3lame" in called_args
    assert "-b:a" in called_args
    assert "48k" in called_args
    assert "-ar" in called_args and "8000" in called_args


def test_extract_audio_raises_when_ffmpeg_fails(sample_video: Path) -> None:
    with patch(
        "video_asr_summary.audio.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
    ):
        with pytest.raises(RuntimeError):
            extract_audio(sample_video)


def test_extract_video_frames_invokes_ffmpeg(
    sample_video: Path, tmp_path: Path
) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_2 = frame_dir / "frame_00002.jpg"
    frame_1 = frame_dir / "frame_00001.jpg"
    frame_2.write_bytes(b"image-bytes")
    frame_1.write_bytes(b"image-bytes")

    with patch("video_asr_summary.audio.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        frames = extract_video_frames(
            sample_video,
            output_dir=frame_dir,
            interval_seconds=2.5,
        )

    assert frames == [
        FrameContext(
            timestamp_start=0.0,
            timestamp_end=2.5,
            frame_path=frame_1,
        ),
        FrameContext(
            timestamp_start=2.5,
            timestamp_end=5.0,
            frame_path=frame_2,
        ),
    ]
    called_args = mock_run.call_args[0][0]
    assert called_args[0] == "ffmpeg"
    assert "fps=1/2.5" in called_args
    assert called_args[-1].startswith(str(frame_dir))


def test_extract_video_frames_validates_interval(
    sample_video: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        extract_video_frames(sample_video, output_dir=tmp_path, interval_seconds=0)


def test_split_audio_on_silence_returns_original_when_short(tmp_path: Path) -> None:
    clip = Sine(440).to_audio_segment(duration=1500).apply_gain(-6)
    audio_path = tmp_path / "short.wav"
    clip.export(audio_path, format="wav")

    segments = audio_module.split_audio_on_silence(audio_path, max_duration=5.0)

    assert segments == [audio_path]


def test_split_audio_on_silence_aligns_segments_with_silence(tmp_path: Path) -> None:
    speech = Sine(440).to_audio_segment(duration=2200).apply_gain(-6)
    pause = AudioSegment.silent(duration=900)
    composite = speech + pause + speech + pause + speech

    source_path = tmp_path / "input.wav"
    composite.export(source_path, format="wav")

    segments = audio_module.split_audio_on_silence(
        source_path,
        max_duration=3.0,
        silence_threshold_db=-35,
        min_silence_len_ms=600,
        output_dir=tmp_path,
    )

    assert len(segments) == 3
    durations = [len(AudioSegment.from_file(path)) for path in segments]
    total_duration = sum(durations)

    assert all(duration <= 3500 for duration in durations)
    assert pytest.approx(total_duration, rel=0.05) == len(composite)

    # Ensure we created new chunk files rather than overwriting the original
    assert all(path != source_path for path in segments)


def test_format_timestamp_zeros() -> None:
    assert audio_module.format_timestamp(0) == "[00:00]"


def test_format_timestamp_minutes_seconds() -> None:
    assert audio_module.format_timestamp(65) == "[01:05]"
    assert audio_module.format_timestamp(0) == "[00:00]"
    assert audio_module.format_timestamp(59) == "[00:59]"
    assert audio_module.format_timestamp(60) == "[01:00]"
    assert audio_module.format_timestamp(120) == "[02:00]"
    assert audio_module.format_timestamp(3665) == "[61:05]"  # 61 minutes, 5 seconds


def test_frame_context_dataclass() -> None:
    fc = FrameContext(
        timestamp_start=0.0,
        timestamp_end=5.0,
        frame_path=Path("frame.jpg"),
    )
    assert fc.timestamp_start == 0.0
    assert fc.timestamp_end == 5.0
    assert fc.frame_path == Path("frame.jpg")
    assert fc.ocr_text == ""

    fc_with_ocr = FrameContext(
        timestamp_start=10.0,
        timestamp_end=15.0,
        frame_path=Path("frame2.jpg"),
        ocr_text="test text",
    )
    assert fc_with_ocr.ocr_text == "test text"
