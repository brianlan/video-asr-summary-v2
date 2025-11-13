from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

from .asr_client import BailianASRClient, LocalQwenASRClient, LocalQwenVisionClient
from .audio import extract_audio, extract_video_frames, split_audio_on_silence
from .summarizer import ChataiSummarizer, TranscriptCorrector


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
    enable_image_context: bool = False,
    frame_interval_seconds: float = 5.0,
    frame_output_dir: Path | str | None = None,
    vision_client: Optional[LocalQwenVisionClient] = None,
    vision_options: Optional[dict[str, Any]] = None,
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

    corrected_transcript = transcript
    local_asr_client = asr if isinstance(asr, LocalQwenASRClient) else None

    if debug:
        print("[debug] transcript-before-correction:\n", transcript)

    if enable_image_context and backend == "local" and transcript.strip():
        corrected_transcript = _apply_image_context_corrections(
            video=video,
            language=language,
            transcript=transcript,
            frame_interval_seconds=frame_interval_seconds,
            frame_output_dir=frame_output_dir,
            vision_client=vision_client,
            vision_options=vision_options,
            local_asr_client=local_asr_client,
            transcript_corrector=transcript_corrector,
            transcript_corrector_model=transcript_corrector_model,
            debug=debug,
        )

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
        summary = llm.summarize(corrected_transcript, language=language)
    except Exception as exc:  # noqa: BLE001 - we must preserve the transcript on any failure
        summary = None
        summarizer_error = str(exc) or exc.__class__.__name__

    if cleanup:
        audio_path.unlink(missing_ok=True)

    result: dict[str, Any] = {
        "transcript": corrected_transcript,
        "summary": summary,
    }
    if summarizer_error is not None:
        result["summarizer_error"] = summarizer_error

    return result


def _apply_image_context_corrections(
    *,
    video: Path,
    language: str,
    transcript: str,
    frame_interval_seconds: float,
    frame_output_dir: Path | str | None,
    vision_client: Optional[LocalQwenVisionClient],
    vision_options: Optional[dict[str, Any]],
    local_asr_client: Optional[LocalQwenASRClient],
    transcript_corrector: Optional[TranscriptCorrector],
    transcript_corrector_model: str | None,
    debug: bool,
) -> str:
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
            return transcript

        vision = vision_client
        if vision is None:
            options = dict(vision_options or {})
            if local_asr_client is not None:
                shared = local_asr_client.export_runtime_components()
                shared.setdefault("model_path", local_asr_client.model_path_str)
                for key, value in shared.items():
                    options.setdefault(key, value)
            vision = LocalQwenVisionClient(**options)

        description_files: list[Path] = []
        for frame in frames:
            description = vision.describe_image(frame, language=language)
            text_path = frame.with_suffix(".txt")
            text_path.write_text(description)
            description_files.append(text_path)

        image_context: list[str] = []
        for text_file in description_files:
            try:
                saved = text_file.read_text().strip()
            except OSError:
                continue
            if saved:
                image_context.append(saved)

        if not image_context:
            return transcript

        corrector = transcript_corrector
        if corrector is None:
            corrector_kwargs: dict[str, Any] = {}
            if transcript_corrector_model is not None:
                corrector_kwargs["model"] = transcript_corrector_model
            corrector = TranscriptCorrector(**corrector_kwargs)

        corrected = corrector.correct(transcript, image_context=image_context, language=language)
        if debug:
            print("[debug] transcript-after-correction:\n", corrected)
        return corrected
    finally:
        if frame_tempdir is not None:
            frame_tempdir.cleanup()
