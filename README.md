# video-asr-summary-v2

This tool extracts audio from videos, transcribes it using a local ASR service, and generates a summary through an OpenAI-compatible API. It also supports optional visual context via a local OCR service to improve transcription accuracy.

## Architecture

The pipeline relies on two local services:
- **ASR Server**: Located at `http://127.0.0.1:8002/v1/audio/transcriptions`. It handles audio transcription.
- **OCR Server**: Located at `http://127.0.0.1:8001/v1/chat/completions`. It provides visual descriptions of video frames to correct transcription errors.

## Environment Setup

This project assumes an existing Python environment at `/ssd4/envs/vllm_torch271_py310_cu128`. We do not modify this environment.

To install the project in your current environment:

```bash
PYTHONPATH=src pip install -e '.[dev]'
```

## Quick Start

### Python API

```python
from pathlib import Path
from video_asr_summary import process_video

# Run the pipeline
result = process_video(Path("path/to/video.mp4"), language="zh")

print("Transcript:", result["transcript"])
print("Summary:", result["summary"])
```

### CLI

Run the pipeline on a local file:

```bash
PYTHONPATH=src python scripts/process_video.py path/to/video.mp4 --language zh
```

Run the pipeline on a video URL (requires `yt-dlp`):

```bash
PYTHONPATH=src python scripts/process_video.py "https://www.youtube.com/watch?v=example" --language en
```

### CLI Options

- `video`: Path to video file or URL.
- `--asr-url`: ASR service base URL (default: `http://127.0.0.1:8002`).
- `--asr-model`: ASR model name (default: `whisper`).
- `--ocr-url`: OCR service base URL (default: `http://127.0.0.1:8001`).
- `--ocr-model`: OCR model name (default: `qwen2-vl`).
- `--enable-image-context`: Enable visual context to help correct transcriptions.
- `--language`: Language code for ASR and summarizer (default: `zh`).
- `--summary-only`: Print only the summary text.
- `--publish-to-lark`: Create a Lark doc with the generated summary.

## Development

Run tests:

```bash
PYTHONPATH=src pytest
```

## Notes

- The pipeline uses `LocalASRClient` and `LocalOCRClient` to communicate with local servers.
- Localhost calls bypass environment proxies.
- Audio is compressed to MP3 before being sent to the ASR service.
- Transcript correction uses visual descriptions obtained from the OCR service.

