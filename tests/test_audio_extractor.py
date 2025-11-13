from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from video_asr_summary.audio import extract_audio, extract_video_frames


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake-video-bytes")
    return video


def test_extract_audio_invokes_ffmpeg(sample_video: Path, tmp_path: Path) -> None:
    with patch("video_asr_summary.audio.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        output_path = extract_audio(sample_video, output_path=tmp_path / "out.wav", sample_rate=16000)

    assert output_path == tmp_path / "out.wav"
    mock_run.assert_called_once()
    called_args = mock_run.call_args[0][0]
    assert called_args[0] == "ffmpeg"
    assert "-ar" in called_args
    assert "-ac" in called_args  # ensure mono conversion
    assert "pcm_s16le" in called_args


def test_extract_audio_supports_mp3(sample_video: Path, tmp_path: Path) -> None:
    from video_asr_summary.audio import extract_audio

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
    from video_asr_summary.audio import extract_audio

    with patch("video_asr_summary.audio.subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
        with pytest.raises(RuntimeError):
            extract_audio(sample_video)


def test_extract_video_frames_invokes_ffmpeg(sample_video: Path, tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    expected_frame = frame_dir / "frame_00001.jpg"
    expected_frame.write_bytes(b"image-bytes")

    with patch("video_asr_summary.audio.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        frames = extract_video_frames(
            sample_video,
            output_dir=frame_dir,
            interval_seconds=2.5,
        )

    assert frames == [expected_frame]
    called_args = mock_run.call_args[0][0]
    assert called_args[0] == "ffmpeg"
    assert "fps=1/2.5" in called_args
    assert called_args[-1].startswith(str(frame_dir))


def test_extract_video_frames_validates_interval(sample_video: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        extract_video_frames(sample_video, output_dir=tmp_path, interval_seconds=0)


def test_split_audio_on_silence_returns_original_when_short(tmp_path: Path) -> None:
    from video_asr_summary.audio import split_audio_on_silence

    clip = Sine(440).to_audio_segment(duration=1500).apply_gain(-6)
    audio_path = tmp_path / "short.wav"
    clip.export(audio_path, format="wav")

    segments = split_audio_on_silence(audio_path, max_duration=5.0)

    assert segments == [audio_path]


def test_split_audio_on_silence_aligns_segments_with_silence(tmp_path: Path) -> None:
    from video_asr_summary.audio import split_audio_on_silence

    speech = Sine(440).to_audio_segment(duration=2200).apply_gain(-6)
    pause = AudioSegment.silent(duration=900)
    composite = speech + pause + speech + pause + speech

    source_path = tmp_path / "input.wav"
    composite.export(source_path, format="wav")

    segments = split_audio_on_silence(
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
