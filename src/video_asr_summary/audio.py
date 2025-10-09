from __future__ import annotations

import subprocess
from pathlib import Path


def extract_audio(
    input_video: Path | str,
    output_path: Path | str | None = None,
    *,
    sample_rate: int = 16000,
    audio_format: str = "wav",
    audio_bitrate: str | None = None,
) -> Path:
    """Extract a mono WAV track from ``input_video``.

    Args:
        input_video: Path to the source video file.
        output_path: Optional target path for the extracted audio. Defaults to the
            same stem as ``input_video`` with a ``.wav`` suffix.
        sample_rate: Audio sampling rate in Hz.

    Returns:
        Path to the extracted audio file.
    """

    input_path = Path(input_video)
    normalized_format = audio_format.lower()

    if output_path is None:
        target = input_path.with_suffix(f".{normalized_format}")
    else:
        target = Path(output_path)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
    ]

    if normalized_format == "wav":
        command.extend(["-acodec", "pcm_s16le"])
    elif normalized_format in {"mp3", "mpeg"}:
        command.extend(["-acodec", "libmp3lame"])
        if audio_bitrate:
            command.extend(["-b:a", audio_bitrate])
    elif normalized_format in {"flac"}:
        command.extend(["-acodec", "flac"])
    else:
        command.extend(["-acodec", normalized_format])
        if audio_bitrate:
            command.extend(["-b:a", audio_bitrate])

    command.append(str(target))

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - handled in tests
        raise RuntimeError("Failed to extract audio with ffmpeg") from exc

    return target
