# video-asr-summary-v2

Pipeline for extracting audio from video, transcribing it with Alibaba Bailian ASR, and summarizing via chatai's OpenAI-compatible API.

## Quick start

1. Ensure the environment variables `BAILIAN_API_KEY` and `OPENAI_ACCESS_TOKEN` are exported.
2. Activate the provided Conda environment `video_asr_summary_py311` and install dependencies:

	```bash
	/Users/rlan/miniforge3/envs/video_asr_summary_py311/bin/python -m pip install -e '.[dev]'
	```

3. Run the pipeline:

	```python
	from pathlib import Path

	from video_asr_summary import process_video

	result = process_video(Path("/path/to/video.mp4"), language="en-US")
	print(result["summary"]["summary"])
	```

4. Execute tests with:

	```bash
	/Users/rlan/miniforge3/envs/video_asr_summary_py311/bin/python -m pytest
	```

## Notes

- External API calls are isolated inside `BailianASRClient` and `ChataiSummarizer`; both accept custom endpoints and credentials for testing or overrides.
- `process_video` compresses extracted audio to MP3 (configurable) before upload so large source videos stay within Bailian's size limits.
- `BailianASRClient` now routes through DashScope's `MultiModalConversation` API, uploading local audio files automatically and supporting optional per-call language overrides.
- `process_video` accepts optional injected client instances and supports automatic cleanup of the intermediate audio file via the `cleanup` flag.