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


def test_extract_audio_raises_when_ffmpeg_fails(sample_video: Path) -> None:
    from video_asr_summary.audio import extract_audio

    with patch("video_asr_summary.audio.subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
        with pytest.raises(RuntimeError):
            extract_audio(sample_video)
