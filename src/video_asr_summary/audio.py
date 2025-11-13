from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple

from pydub import AudioSegment
from pydub.silence import detect_silence


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


def extract_video_frames(
    input_video: Path | str,
    *,
    interval_seconds: float = 5.0,
    output_dir: Path | str | None = None,
    image_format: str = "jpg",
    frame_prefix: str = "frame",
) -> list[Path]:
    """Extract JPEG frames every ``interval_seconds`` seconds using ffmpeg.

    Returns a list of extracted frame paths sorted in ascending order.
    """

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    source = Path(input_video)
    target_dir = Path(output_dir) if output_dir is not None else source.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    pattern = target_dir / f"{frame_prefix}_%05d.{image_format}"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"fps=1/{interval_seconds}",
        "-qscale:v",
        "2",
        str(pattern),
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - handled via tests
        raise RuntimeError("Failed to extract video frames with ffmpeg") from exc

    frame_paths = sorted(target_dir.glob(f"{frame_prefix}_*.{image_format}"))
    return frame_paths


def split_audio_on_silence(
    audio_path: Path | str,
    *,
    max_duration: float = 60.0,
    min_silence_len_ms: int = 600,
    silence_threshold_db: int = -40,
    keep_silence_ms: int = 0,
    seek_step_ms: int = 1,
    output_dir: Path | None = None,
    min_export_duration_ms: int = 300,
    silent_rms_threshold: int = 20,
) -> list[Path]:
    """Split ``audio_path`` into segments no longer than ``max_duration`` seconds.

    Segments prefer to end at detected silence regions to minimize cutting speech.
    Falls back to uniform slicing when no silence is available. Returns a list of
    paths pointing to the chunk files. If splitting is unnecessary the original
    file path is returned.
    """

    source = Path(audio_path)
    audio = AudioSegment.from_file(source)

    max_duration_ms = int(max(max_duration, 0) * 1000)
    if max_duration_ms <= 0:
        raise ValueError("max_duration must be positive")

    total_ms = len(audio)
    if total_ms <= max_duration_ms:
        return [source]

    silence_ranges: List[Tuple[int, int]] = detect_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_threshold_db,
        seek_step=seek_step_ms,
    )
    silence_ranges.sort(key=lambda pair: pair[0])

    target_dir = Path(output_dir) if output_dir is not None else source.parent
    if output_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)

    segments: list[Path] = []
    segment_start = 0
    segment_index = 1
    tolerance_ms = max(min_silence_len_ms, int(max_duration_ms * 0.2))
    min_chunk_ms = min(max_duration_ms, max(500, int(max_duration_ms * 0.25)))

    def export_chunk(start_ms: int, end_ms: int, index: int) -> Path | None:
        chunk = audio[start_ms:end_ms]
        duration_ms = len(chunk)

        if duration_ms < max(min_export_duration_ms, 1) and chunk.rms <= silent_rms_threshold:
            return None

        if chunk.rms <= silent_rms_threshold and duration_ms < min_silence_len_ms:
            return None

        suffix = source.suffix.lstrip(".") or "wav"
        chunk_name = f"{source.stem}_part{index:03d}.{suffix}"
        chunk_path = target_dir / chunk_name
        chunk.export(chunk_path, format=suffix)
        return chunk_path

    silence_idx = 0
    total_silences = len(silence_ranges)

    while segment_start < total_ms:
        target_end = min(segment_start + max_duration_ms, total_ms)
        best_idx = None
        best_split = None

        for idx in range(silence_idx, total_silences):
            silence_start, silence_end = silence_ranges[idx]
            if silence_end <= segment_start:
                silence_idx = idx + 1
                continue
            if silence_start < segment_start + min_chunk_ms:
                continue

            if silence_start <= target_end:
                best_idx = idx
                best_split = (silence_start + silence_end) // 2
                continue

            if silence_start <= target_end + tolerance_ms and best_split is None:
                best_idx = idx
                best_split = (silence_start + silence_end) // 2
            break

        if best_idx is not None and best_split is not None:
            silence_idx = best_idx + 1
            segment_end = max(segment_start + min_chunk_ms, best_split)
        else:
            segment_end = target_end

        if segment_end <= segment_start:
            segment_end = min(total_ms, segment_start + max(100, max_duration_ms // 4))

        if keep_silence_ms and segment_end < total_ms:
            segment_end = min(total_ms, segment_end + keep_silence_ms)

        chunk_path = export_chunk(segment_start, segment_end, segment_index)
        if chunk_path is not None:
            segments.append(chunk_path)
            segment_index += 1

        segment_start = segment_end

    return segments
