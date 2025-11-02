import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlparse

import yt_dlp
from video_asr_summary import process_video
from video_asr_summary.lark_docs import (
	LarkDocError,
	create_summary_document,
	derive_lark_title,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the video ASR and summarization pipeline.")
	parser.add_argument("video", help="Path to the input video file or a video URL")
	parser.add_argument("--language", default="zh", help="Language code to pass to ASR and summarizer")
	parser.add_argument("--audio-format", dest="audio_format", default="mp3", help="Audio format produced by ffmpeg")
	parser.add_argument(
		"--asr-backend",
		choices=("bailian", "local"),
		default="bailian",
		help="Choose the ASR backend to use (bailian cloud or local).",
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
		"--local-model-path",
		dest="local_model_path",
		help="Path to the local Qwen Omni model weights",
	)
	parser.add_argument(
		"--local-prompt-template",
		dest="local_prompt_template",
		help="Custom prompt template for the local ASR backend",
	)
	parser.add_argument(
		"--local-temperature",
		dest="local_temperature",
		type=float,
		default=0.6,
		help="Sampling temperature for the local ASR backend",
	)
	parser.add_argument(
		"--local-top-p",
		dest="local_top_p",
		type=float,
		default=0.95,
		help="Top-p nucleus sampling value for the local ASR backend",
	)
	parser.add_argument(
		"--local-top-k",
		dest="local_top_k",
		type=int,
		default=20,
		help="Top-k sampling value for the local ASR backend",
	)
	parser.add_argument(
		"--local-max-tokens",
		dest="local_max_tokens",
		type=int,
		default=16384,
		help="Maximum tokens to generate for local ASR",
	)
	parser.add_argument(
		"--local-max-model-len",
		dest="local_max_model_len",
		type=int,
		default=32768,
		help="Maximum model length when running the local ASR backend",
	)
	parser.add_argument(
		"--local-max-num-seqs",
		dest="local_max_num_seqs",
		type=int,
		default=8,
		help="Maximum parallel sequences for the local ASR backend",
	)
	parser.add_argument(
		"--local-gpu-memory-utilization",
		dest="local_gpu_memory_utilization",
		type=float,
		default=0.95,
		help="GPU memory utilization fraction for the local ASR backend",
	)
	parser.add_argument(
		"--local-tensor-parallel-size",
		dest="local_tensor_parallel_size",
		type=int,
		help="Override tensor parallel size for the local ASR backend",
	)
	parser.add_argument(
		"--local-seed",
		dest="local_seed",
		type=int,
		default=1234,
		help="Random seed for the local ASR backend",
	)
	parser.add_argument(
		"--local-use-audio-in-video",
		action=argparse.BooleanOptionalAction,
		dest="local_use_audio_in_video",
		default=True,
		help="Enable using the audio track when processing video chunks in the local backend",
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
		"--lark-tenant-token",
		dest="lark_tenant_token",
		help="Override the Lark tenant access token (defaults to LARK_TENANT_ACCESS_TOKEN env var)",
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
		"--lark-api-domain",
		dest="lark_api_domain",
		help="Override the Lark OpenAPI domain (defaults to LARK_API_DOMAIN env var or SDK default)",
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
	options = {
		"outtmpl": str(output_dir / "%(title).200s.%(ext)s"),
		"quiet": True,
		"no_warnings": True,
		"nocheckcertificate": True,
		"retries": 5,
		"http_headers": {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
		},
	}
	with yt_dlp.YoutubeDL(options) as downloader:
		info = downloader.extract_info(video_url, download=True)
		filepath = downloader.prepare_filename(info)
	return Path(filepath)


def main() -> None:
	args = parse_args()

	local_asr_options = {
		"model_path": args.local_model_path,
		"prompt_template": args.local_prompt_template,
		"temperature": args.local_temperature,
		"top_p": args.local_top_p,
		"top_k": args.local_top_k,
		"max_tokens": args.local_max_tokens,
		"max_model_len": args.local_max_model_len,
		"max_num_seqs": args.local_max_num_seqs,
		"gpu_memory_utilization": args.local_gpu_memory_utilization,
		"tensor_parallel_size": args.local_tensor_parallel_size,
		"seed": args.local_seed,
		"use_audio_in_video": args.local_use_audio_in_video,
	}
	local_asr_options = {key: value for key, value in local_asr_options.items() if value is not None}

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
				asr_backend=args.asr_backend,
				local_asr_options=local_asr_options,
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
			asr_backend=args.asr_backend,
			local_asr_options=local_asr_options,
		)

	summary_text = result.get("summary")
	lark_doc = None

	if args.publish_to_lark:
		if summary_text is None or not str(summary_text).strip():
			raise RuntimeError("No summary content generated; cannot publish to Lark.")
		doc_title = derive_lark_title(summary_text, default_title_from_video(video_input))
		try:
			lark_doc = create_summary_document(
				summary_text,
				title=doc_title,
				folder_token=args.lark_folder_token,
				app_id=args.lark_app_id,
				app_secret=args.lark_app_secret,
				user_access_token=args.lark_user_access_token,
				tenant_access_token=args.lark_tenant_token,
				user_subdomain=args.lark_user_subdomain,
				api_domain=args.lark_api_domain,
			)
			result["lark_document"] = lark_doc
			print(f"Lark document created: {lark_doc['url']}")
		except LarkDocError as exc:
			raise RuntimeError(str(exc)) from exc

	if args.summary_only:
		print(summary_text if summary_text is not None else result)
	else:
		print(result)


if __name__ == "__main__":
	main()
