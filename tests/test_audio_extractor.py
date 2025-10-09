from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake-video-bytes")
    return video


def test_extract_audio_invokes_ffmpeg(sample_video: Path, tmp_path: Path) -> None:
    from video_asr_summary.audio import extract_audio

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
