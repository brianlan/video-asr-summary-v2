import argparse
from pathlib import Path

from video_asr_summary import process_video


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the video ASR and summarization pipeline.")
	parser.add_argument("video", type=Path, help="Path to the input video file")
	parser.add_argument("--language", default="zh", help="Language code to pass to ASR and summarizer")
	parser.add_argument("--audio-format", dest="audio_format", default="mp3", help="Audio format produced by ffmpeg")
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
		"--summary-only",
		dest="summary_only",
		action="store_true",
		help="Print only the summary text instead of the full result payload",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	result = process_video(
		video_path=args.video,
		language=args.language,
		audio_format=args.audio_format,
		audio_sample_rate=args.audio_sample_rate,
		audio_bitrate=args.audio_bitrate,
		max_segment_duration=args.max_segment_duration,
	)

	if args.summary_only:
		summary_text = result.get("summary", {}).get("summary")
		if summary_text is None:
			print(result)
		else:
			print(summary_text)
	else:
		print(result)


if __name__ == "__main__":
	main()
