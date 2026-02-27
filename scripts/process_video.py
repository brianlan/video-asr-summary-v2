import argparse
import os
import sys


from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from urllib.parse import unquote, urlparse

import yt_dlp
from video_asr_summary import process_video  # pyright: ignore[reportMissingImports]
from video_asr_summary.lark_docs import (  # pyright: ignore[reportMissingImports]
    LarkDocError,
    create_summary_document,
    derive_lark_title,
)


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the video ASR and summarization pipeline."
    )
    parser.add_argument("video", help="Path to the input video file or a video URL")
    parser.add_argument(
        "--language", default="zh", help="Language code to pass to ASR and summarizer"
    )
    parser.add_argument(
        "--audio-format",
        dest="audio_format",
        default="mp3",
        help="Audio format produced by ffmpeg",
    )
    parser.add_argument(
        "--audio-sample-rate",
        dest="audio_sample_rate",
        type=int,
        default=16000,
        help="Sample rate for extracted audio",
    )
    parser.add_argument(
        "--audio-bitrate",
        dest="audio_bitrate",
        default="48k",
        help="Bitrate for extracted audio",
    )
    parser.add_argument(
        "--max-segment-duration",
        dest="max_segment_duration",
        type=float,
        default=60.0,
        help="Maximum duration in seconds for each ASR chunk",
    )
    parser.add_argument(
        "--asr-url",
        dest="asr_url",
        default=os.getenv("ASR_URL", "http://127.0.0.1:8002"),
        help="ASR service base URL",
    )
    parser.add_argument(
        "--ocr-url",
        dest="ocr_url",
        default=os.getenv("OCR_URL", "http://127.0.0.1:8001"),
        help="OCR service base URL",
    )
    parser.add_argument(
        "--asr-model",
        dest="asr_model",
        default=os.getenv("ASR_MODEL", "whisper"),
        help="ASR model name",
    )
    parser.add_argument(
        "--ocr-model",
        dest="ocr_model",
        default=os.getenv("OCR_MODEL", "qwen2-vl"),
        help="OCR model name",
    )
    parser.add_argument(
        "--enable-image-context",
        action=argparse.BooleanOptionalAction,
        dest="enable_image_context",
        default=False,
        help="Enable using image context to correct the transcript",
    )
    parser.add_argument(
        "--enable-transcript-correction",
        action=argparse.BooleanOptionalAction,
        dest="enable_transcript_correction",
        default=False,
        help="Enable LLM transcript correction using OCR frame context",
    )
    parser.add_argument(
        "--image-context-frame-interval-seconds",
        dest="image_context_frame_interval_seconds",
        type=float,
        default=5.0,
        help="Interval in seconds between extracted video frames for image context",
    )
    parser.add_argument(
        "--summary-only",
        dest="summary_only",
        action="store_true",
        help="Print only the summary text instead of the full result payload",
    )
    parser.add_argument(
        "--publish-to-lark",
        dest="publish_to_lark",
        action="store_true",
        help="Create a Lark doc with the generated summary",
    )
    parser.add_argument(
        "--lark-folder-token",
        dest="lark_folder_token",
        help="Optional Lark folder token where the document should be created",
    )
    parser.add_argument(
        "--lark-app-id",
        dest="lark_app_id",
        help="Override the Lark app ID (defaults to LARK_APP_ID env var)",
    )
    parser.add_argument(
        "--lark-app-secret",
        dest="lark_app_secret",
        help="Override the Lark app secret (defaults to LARK_APP_SECRET env var)",
    )
    parser.add_argument(
        "--lark-user-access-token",
        dest="lark_user_access_token",
        help="Override the Lark user access token (defaults to LARK_USER_ACCESS_TOKEN env var)",
    )
    parser.add_argument(
        "--lark-user-subdomain",
        dest="lark_user_subdomain",
        help="Override the Lark user subdomain (defaults to LARK_USER_SUBDOMAIN env var)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose pipeline debugging output",
    )
    parser.add_argument(
        "--lark-api-domain",
        dest="lark_api_domain",
        help="Override the Lark OpenAPI domain (defaults to LARK_API_DOMAIN env var or SDK default)",
    )
    parser.add_argument(
        "--message-receiver-id",
        dest="message_receiver_id",
        default=os.getenv("LARK_MESSAGE_RECEIVER_ID", "1gc832ed"),
        help="User ID that receives Lark notification messages",
    )
    parser.add_argument(
        "--summarizer-model",
        dest="summarizer_model",
        default="gpt-5-mini",
        help="Override the model used for summarization (defaults to SUMMARIZER_MODEL env var or SDK default)",
    )
    return parser.parse_args()


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def default_title_from_video(video_input: str) -> str:
    if is_url(video_input):
        parsed = urlparse(video_input)
        name = Path(unquote(parsed.path)).stem or parsed.netloc
    else:
        name = Path(video_input).stem
    return name or "Video Summary"


def download_video(video_url: str, output_dir: Path) -> Path:
    options: dict[str, Any] = {
        "outtmpl": str(output_dir / "%(title).200s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "retries": 5,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        },
    }
    with yt_dlp.YoutubeDL(cast(Any, options)) as downloader:
        info = downloader.extract_info(video_url, download=True)
        filepath = downloader.prepare_filename(info)
    return Path(filepath)


def main() -> None:
    load_env_file(Path(".env"))
    args = parse_args()

    asr_options = {
        "base_url": args.asr_url,
        "model": args.asr_model,
    }
    ocr_options = {
        "base_url": args.ocr_url,
        "model": args.ocr_model,
    }
    use_ocr = args.enable_image_context or args.enable_transcript_correction

    video_input = args.video
    if is_url(video_input):
        with TemporaryDirectory() as tmpdir:
            downloaded_video = download_video(video_input, Path(tmpdir))
            result = process_video(
                video_path=downloaded_video,
                language=args.language,
                audio_format=args.audio_format,
                audio_sample_rate=args.audio_sample_rate,
                audio_bitrate=args.audio_bitrate,
                max_segment_duration=args.max_segment_duration,
                frame_interval_seconds=args.image_context_frame_interval_seconds,
                asr_options=asr_options,
                ocr_options=ocr_options if use_ocr else None,
                summarizer_model=args.summarizer_model,
                enable_image_context=args.enable_image_context,
                enable_transcript_correction=args.enable_transcript_correction,
                debug=args.debug,
            )
    else:
        video_path = Path(video_input)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        result = process_video(
            video_path=video_path,
            language=args.language,
            audio_format=args.audio_format,
            audio_sample_rate=args.audio_sample_rate,
            audio_bitrate=args.audio_bitrate,
            max_segment_duration=args.max_segment_duration,
            frame_interval_seconds=args.image_context_frame_interval_seconds,
            asr_options=asr_options,
            ocr_options=ocr_options if use_ocr else None,
            summarizer_model=args.summarizer_model,
            enable_image_context=args.enable_image_context,
            enable_transcript_correction=args.enable_transcript_correction,
            debug=args.debug,
        )

    summary_text = result.get("summary")
    summarizer_error = result.get("summarizer_error")
    if summarizer_error:
        print(f"Summarizer error encountered: {summarizer_error}")
    lark_doc = None

    if args.publish_to_lark:
        if summary_text is None or not str(summary_text).strip():
            print("No summary content generated; skipping Lark publishing.")
        else:
            doc_title = derive_lark_title(
                summary_text, default_title_from_video(video_input)
            )
            try:
                lark_doc = create_summary_document(
                    summary_text,
                    title=doc_title,
                    folder_token=args.lark_folder_token,
                    app_id=args.lark_app_id,
                    app_secret=args.lark_app_secret,
                    user_access_token=args.lark_user_access_token,
                    user_subdomain=args.lark_user_subdomain,
                    api_domain=args.lark_api_domain,
                    message_receiver_id=args.message_receiver_id,
                )
                result["lark_document"] = lark_doc
                print(f"Lark document created: {lark_doc['url']}")
            except LarkDocError as exc:
                result["lark_error"] = str(exc)
                print(f"Lark publishing failed: {exc}")

    if args.summary_only:
        if summary_text is not None and str(summary_text).strip():
            print(summary_text)
        else:
            transcript_text = result.get("transcript")
            if transcript_text:
                print(transcript_text)
            else:
                print(result)
    else:
        print(result)


if __name__ == "__main__":
    main()
