from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .asr_client import BailianASRClient
from .audio import extract_audio
from .summarizer import ChataiSummarizer


def process_video(
    video_path: Path | str,
    *,
    language: str = "en",
    audio_format: str = "mp3",
    audio_sample_rate: int = 16000,
    audio_bitrate: str | None = "64k",
    bailian_client: Optional[BailianASRClient] = None,
    summarizer: Optional[ChataiSummarizer] = None,
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

    asr = bailian_client or BailianASRClient()
    transcript = asr.transcribe(audio_path, language=language)

    llm = summarizer or ChataiSummarizer()
    summary = llm.summarize(transcript, language=language)

    if cleanup:
        audio_path.unlink(missing_ok=True)

    return {
        "transcript": transcript,
        "summary": summary,
    }
