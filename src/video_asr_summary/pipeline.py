from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

from .asr_client import LocalASRClient, LocalOCRClient
from .audio import extract_audio, extract_video_frames, split_audio_on_silence
from .summarizer import StructuredSummarizer, TranscriptCorrector


def process_video(
    video_path: Path | str,
    *,
    language: str = "en",
    audio_format: str = "mp3",
    audio_sample_rate: int = 16000,
    audio_bitrate: str | None = "64k",
    max_segment_duration: float = 60.0,
    asr_client: Optional[LocalASRClient] = None,
    asr_options: Optional[dict[str, Any]] = None,
    summarizer: Optional[StructuredSummarizer] = None,
    summarizer_model: str | None = None,
    enable_image_context: bool = False,
    frame_interval_seconds: float = 5.0,
    frame_output_dir: Path | str | None = None,
    ocr_client: Optional[LocalOCRClient] = None,
    ocr_options: Optional[dict[str, Any]] = None,
    transcript_corrector: Optional[TranscriptCorrector] = None,
    transcript_corrector_model: str | None = None,
    cleanup: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """Run the end-to-end pipeline from video to summary."""

    video = Path(video_path)
    audio_path = extract_audio(
        video,
        sample_rate=audio_sample_rate,
        audio_format=audio_format,
        audio_bitrate=audio_bitrate,
    )

    if asr_client is not None:
        asr = asr_client
    else:
        options = asr_options or {}
        asr = LocalASRClient(**options)

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

    frame_contexts = []

    if enable_image_context:
        frame_contexts = _collect_frame_contexts(
            video=video,
            language=language,
            frame_interval_seconds=frame_interval_seconds,
            frame_output_dir=frame_output_dir,
            ocr_client=ocr_client,
            ocr_options=ocr_options,
            debug=debug,
        )

    if summarizer is not None:
        llm = summarizer
    else:
        summarizer_kwargs: dict[str, Any] = {}
        if summarizer_model is not None:
            summarizer_kwargs["model"] = summarizer_model
        llm = StructuredSummarizer(**summarizer_kwargs)
    summary: str | None
    summarizer_error: str | None = None
    try:
        summary = llm.summarize(transcript, frame_contexts, language=language)
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


def _collect_frame_contexts(
    *,
    video: Path,
    language: str,
    frame_interval_seconds: float,
    frame_output_dir: Path | str | None,
    ocr_client: Optional[LocalOCRClient],
    ocr_options: Optional[dict[str, Any]],
    debug: bool,
) -> list[Any]:
    frame_tempdir: TemporaryDirectory[str] | None = None
    if frame_output_dir is None:
        frame_tempdir = TemporaryDirectory()
        frames_dir = Path(frame_tempdir.name)
    else:
        frames_dir = Path(frame_output_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        frames = extract_video_frames(
            video,
            interval_seconds=frame_interval_seconds,
            output_dir=frames_dir,
        )
        if not frames:
            return []

        ocr = ocr_client
        if ocr is None:
            options = ocr_options or {}
            ocr = LocalOCRClient(**options)

        enriched_frames: list[Any] = []
        for frame in frames:
            frame_path = frame.frame_path
            try:
                frame.ocr_text = ocr.describe_image(
                    frame_path, language=language
                ).strip()
            except Exception as exc:  # noqa: BLE001 - one frame failure should not fail whole run
                if debug:
                    print(f"[debug] skipped frame OCR failure: {frame_path}: {exc}")
                continue
            enriched_frames.append(frame)

        return enriched_frames
    finally:
        if frame_tempdir is not None:
            frame_tempdir.cleanup()
