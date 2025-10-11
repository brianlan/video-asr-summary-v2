import argparse
from pathlib import Path

from video_asr_summary import process_video


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the video ASR and summarization pipeline.")
	parser.add_argument("video", type=Path, help="Path to the input video file")
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
	return parser.parse_args()


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

	result = process_video(
		video_path=args.video,
		language=args.language,
		audio_format=args.audio_format,
		audio_sample_rate=args.audio_sample_rate,
		audio_bitrate=args.audio_bitrate,
		max_segment_duration=args.max_segment_duration,
		asr_backend=args.asr_backend,
		local_asr_options=local_asr_options,
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
