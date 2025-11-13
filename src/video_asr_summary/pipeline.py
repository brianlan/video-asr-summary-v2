from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

from .asr_client import BailianASRClient, LocalQwenASRClient
from .audio import extract_audio, split_audio_on_silence
from .summarizer import ChataiSummarizer


def process_video(
    video_path: Path | str,
    *,
    language: str = "en",
    audio_format: str = "mp3",
    audio_sample_rate: int = 16000,
    audio_bitrate: str | None = "64k",
    max_segment_duration: float = 60.0,
    bailian_client: Optional[BailianASRClient] = None,
    bailian_options: Optional[dict[str, Any]] = None,
    asr_backend: str = "bailian",
    local_client: Optional[LocalQwenASRClient] = None,
    local_asr_options: Optional[dict[str, Any]] = None,
    summarizer: Optional[ChataiSummarizer] = None,
    summarizer_model: str | None = None,
    cleanup: bool = False,
) -> dict[str, Any]:
    """Run the end-to-end pipeline from video to summary."""

    video = Path(video_path)
    audio_path = extract_audio(
        video,
        sample_rate=audio_sample_rate,
        audio_format=audio_format,
        audio_bitrate=audio_bitrate,
    )

    backend = asr_backend.lower()
    if backend == "bailian":
        if bailian_client is not None:
            asr = bailian_client
        else:
            options = bailian_options or {}
            asr = BailianASRClient(**options)
    elif backend == "local":
        if local_client is not None:
            asr = local_client
        else:
            options = local_asr_options or {}
            asr = LocalQwenASRClient(**options)
    else:
        raise ValueError(f"Unsupported ASR backend: {asr_backend}")
    transcripts: list[str] = []

    with TemporaryDirectory() as tmpdir:
        chunk_paths = split_audio_on_silence(
            audio_path,
            max_duration=max_segment_duration,
            output_dir=Path(tmpdir),
        )
        for chunk in chunk_paths:
            transcripts.append(asr.transcribe(Path(chunk), language=language))

    transcript = "\n\n".join(part.strip() for part in transcripts if part.strip())

    if not transcript:
        transcript = ""

    if summarizer is not None:
        llm = summarizer
    else:
        summarizer_kwargs: dict[str, Any] = {}
        if summarizer_model is not None:
            summarizer_kwargs["model"] = summarizer_model
        llm = ChataiSummarizer(**summarizer_kwargs)
    summary: str | None
    summarizer_error: str | None = None
    try:
        summary = llm.summarize(transcript, language=language)
    except Exception as exc:  # noqa: BLE001 - we must preserve the transcript on any failure
        summary = None
        summarizer_error = str(exc) or exc.__class__.__name__

    if cleanup:
        audio_path.unlink(missing_ok=True)

    result: dict[str, Any] = {
        "transcript": transcript,
        "summary": summary,
    }
    if summarizer_error is not None:
        result["summarizer_error"] = summarizer_error

    return result
