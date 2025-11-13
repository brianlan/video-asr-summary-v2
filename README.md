# video-asr-summary-v2

Pipeline for extracting audio from video, transcribing it with Alibaba Bailian ASR, and summarizing via chatai's OpenAI-compatible API.

## Prepare the environment to run the program
- conda create -p /ssd4/envs/vllm_torch271_py310_cu128 python=3.10
- pip install git+https://github.com/huggingface/transformers@0419ff881d7bb503f4fc0f0a7a5aac3d012c9b91
- pip install --pre torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
- pip install uv
- git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git
- cd vllm
- python use_existing_torch.py
- pip install -r requirements/build.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple
- MAX_JOBS=16 uv pip install --no-build-isolation -e .
- pip install accelerate-1.11.0
- pip install yt_dlp
- pip install pydub
- pip install lark-oapi
- pip install qwen-omni-utils -U
- pip install -U --no-build-isolation https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu128torch2.7-cp310-cp310-linux_x86_64.whl


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
