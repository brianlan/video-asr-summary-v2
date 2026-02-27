from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests


def test_transcribe_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful transcription returns the text."""
    from video_asr_summary.asr_client import LocalASRClient

    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "Hello world transcription"}
    mock_response.raise_for_status.return_value = None

    captured_payload: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> MagicMock:  # type: ignore[override]
        captured_payload["url"] = url
        captured_payload["kwargs"] = kwargs
        return mock_response

    monkeypatch.setattr(requests, "post", fake_post)

    client = LocalASRClient(base_url="http://127.0.0.1:8002")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    result = client.transcribe(audio_file, language="en")

    assert result == "Hello world transcription"
    assert captured_payload["url"] == "http://127.0.0.1:8002/v1/audio/transcriptions"


def test_transcribe_posts_form_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that transcribe sends multipart form data."""
    from video_asr_summary.asr_client import LocalASRClient

    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "test"}
    mock_response.raise_for_status.return_value = None

    captured_files: dict[str, Any] = {}

    def fake_post(url: str, files: dict | None = None, **kwargs: Any) -> MagicMock:  # type: ignore[override]
        captured_files["files"] = files
        return mock_response

    monkeypatch.setattr(requests, "post", fake_post)

    client = LocalASRClient(base_url="http://127.0.0.1:8002", model="whisper-base")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    client.transcribe(audio_file)

    assert captured_files["files"] is not None
    assert "file" in captured_files["files"]


def test_transcribe_raises_on_http_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that HTTP errors raise RuntimeError."""
    from video_asr_summary.asr_client import LocalASRClient

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

    client = LocalASRClient(base_url="http://127.0.0.1:8002")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    with pytest.raises(RuntimeError, match="ASR request failed"):
        client.transcribe(audio_file)


def test_transcribe_raises_on_connection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that connection errors raise RuntimeError."""
    from video_asr_summary.asr_client import LocalASRClient

    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError()),
    )

    client = LocalASRClient(base_url="http://127.0.0.1:8002")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    with pytest.raises(RuntimeError, match="connection"):
        client.transcribe(audio_file)


def test_transcribe_raises_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that timeout raises RuntimeError."""
    from video_asr_summary.asr_client import LocalASRClient

    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout()),
    )

    client = LocalASRClient(base_url="http://127.0.0.1:8002")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    with pytest.raises(RuntimeError, match="timeout"):
        client.transcribe(audio_file)


def test_transcribe_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test retry logic: fails twice then succeeds."""
    from video_asr_summary.asr_client import LocalASRClient

    call_count = 0
    sleep_count = 0

    def fake_post_with_retry(url: str, **kwargs: Any) -> MagicMock:  # type: ignore[override]
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            mock_err = MagicMock()
            mock_err.raise_for_status.side_effect = requests.HTTPError(
                "503 Service Unavailable"
            )
            return mock_err
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "Success after retry"}
        mock_response.raise_for_status.return_value = None
        return mock_response

    def fake_sleep(_delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1

    monkeypatch.setattr(requests, "post", fake_post_with_retry)
    monkeypatch.setattr("video_asr_summary.asr_client.time.sleep", fake_sleep)

    client = LocalASRClient(base_url="http://127.0.0.1:8002", max_retries=3)
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    result = client.transcribe(audio_file)

    assert result == "Success after retry"
    assert call_count == 3  # 2 failures + 1 success
    assert sleep_count == 2  # exponential backoff on attempts 0 and 1

def test_transcribe_raises_after_max_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that it raises after exhausting retries."""
    from video_asr_summary.asr_client import LocalASRClient

    call_count = 0
    sleep_count = 0

    def fake_post_always_fails(url: str, **kwargs: Any) -> MagicMock:  # type: ignore[override]
        nonlocal call_count
        call_count += 1
        mock_err = MagicMock()
        mock_err.raise_for_status.side_effect = requests.HTTPError(
            "503 Service Unavailable"
        )
        return mock_err

    def fake_sleep(_delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1

    monkeypatch.setattr(requests, "post", fake_post_always_fails)
    monkeypatch.setattr("video_asr_summary.asr_client.time.sleep", fake_sleep)

    client = LocalASRClient(base_url="http://127.0.0.1:8002", max_retries=3)
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    with pytest.raises(RuntimeError, match="ASR request failed"):
        client.transcribe(audio_file)

    assert call_count == 3  # max_retries attempts
    assert sleep_count == 2  # exponential backoff on attempts 0 and 1

def test_transcribe_with_custom_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that language parameter is passed correctly."""
    from video_asr_summary.asr_client import LocalASRClient

    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "transcription"}
    mock_response.raise_for_status.return_value = None

    captured_form: dict[str, Any] = {}

    def fake_post(
        url: str, files: dict | None = None, data: dict | None = None, **kwargs: Any
    ) -> MagicMock:  # type: ignore[override]
        captured_form["files"] = files
        captured_form["data"] = data
        return mock_response

    monkeypatch.setattr(requests, "post", fake_post)

    client = LocalASRClient(base_url="http://127.0.0.1:8002")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    client.transcribe(audio_file, language="zh")

    # Verify language is passed in form data
    assert captured_form["data"] is not None
    assert captured_form["data"].get("language") == "zh"


# === LocalOCRClient tests ===


def test_ocr_describe_image_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test successful OCR call with valid image path."""
    from video_asr_summary.asr_client import LocalOCRClient

    image_path = tmp_path / "test_image.jpg"
    image_path.write_bytes(b"fake-image-data")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Extracted text from image"}}]
    }
    mock_response.raise_for_status.return_value = None

    captured_payload = {}

    def fake_post(url: str, **kwargs: Any) -> MagicMock:  # type: ignore[override]
        captured_payload["url"] = url
        captured_payload["kwargs"] = kwargs
        return mock_response

    monkeypatch.setattr(requests, "post", fake_post)

    client = LocalOCRClient(base_url="http://127.0.0.1:8001")
    result = client.describe_image(image_path)

    assert result == "Extracted text from image"
    assert captured_payload["url"] == "http://127.0.0.1:8001/v1/chat/completions"

    request_json = captured_payload["kwargs"].get("json")
    assert request_json is not None
    assert request_json["model"] == "qwen2-vl"
    assert "messages" in request_json
    # Verify the message contains image_url with file:// prefix and prompt text
    content = request_json["messages"][0]["content"]
    assert any("file://" in str(item) for item in content)
    assert any("text" in item and isinstance(item["text"], str) for item in content)


def test_ocr_describe_image_invalid_path_raises_file_not_found(tmp_path: Path) -> None:
    """Test that FileNotFoundError is raised for missing image path."""
    from video_asr_summary.asr_client import LocalOCRClient

    nonexistent_path = tmp_path / "nonexistent.jpg"

    client = LocalOCRClient()
    with pytest.raises(FileNotFoundError):
        client.describe_image(nonexistent_path)


def test_ocr_describe_image_http_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test HTTP error handling in OCR client."""
    from video_asr_summary.asr_client import LocalOCRClient

    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake-image-data")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = RuntimeError(
        "HTTP 500: Internal Server Error"
    )

    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: mock_response,
    )

    client = LocalOCRClient()
    with pytest.raises(RuntimeError, match="HTTP 500"):
        client.describe_image(image_path)


def test_ocr_describe_image_connection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test connection error handling in OCR client."""
    from video_asr_summary.asr_client import LocalOCRClient

    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake-image-data")

    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ConnectionError("Connection refused")
        ),
    )

    client = LocalOCRClient()
    with pytest.raises(ConnectionError):
        client.describe_image(image_path)


def test_ocr_describe_image_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test timeout handling in OCR client."""
    from video_asr_summary.asr_client import LocalOCRClient

    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake-image-data")

    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            TimeoutError("Request timed out")
        ),
    )

    client = LocalOCRClient()
    with pytest.raises(TimeoutError):
        client.describe_image(image_path)


def test_ocr_describe_image_with_custom_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test OCR with custom language parameter."""
    from video_asr_summary.asr_client import LocalOCRClient

    image_path = tmp_path / "test_image.jpg"
    image_path.write_bytes(b"fake-image-data")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Chinese text: 你好世界"}}]
    }
    mock_response.raise_for_status.return_value = None

    captured_payload = {}

    def fake_post(url: str, **kwargs: Any) -> MagicMock:  # type: ignore[override]
        captured_payload["url"] = url
        captured_payload["kwargs"] = kwargs
        return mock_response

    monkeypatch.setattr(requests, "post", fake_post)

    client = LocalOCRClient()
    result = client.describe_image(image_path, language="zh-CN")

    assert "你好世界" in result
    request_json = captured_payload["kwargs"].get("json")
    assert request_json is not None
    # Language should be passed in the prompt text
    assert "zh-CN" in str(request_json["messages"][0]["content"])


# === Streaming tests for LocalASRClient ===


def test_transcribe_streaming_sse_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test streaming response parsing with SSE format (data: prefix)."""
    from video_asr_summary.asr_client import LocalASRClient

    # Simulate SSE streaming response lines
    sse_lines = [
        b'data: {"text": "Hello "}\n',
        b'data: {"text": "world "}\n',
        b'data: {"text": "transcription"}\n',
        b"data: [DONE]\n",
    ]

    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/event-stream"}
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

    client = LocalASRClient(base_url="http://127.0.0.1:8002")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    result = client.transcribe(audio_file, language="en")

    assert result == "Hello world transcription"


def test_transcribe_streaming_without_data_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test streaming response parsing without data: prefix."""
    from video_asr_summary.asr_client import LocalASRClient

    # Simulate SSE streaming without data: prefix
    sse_lines = [
        b'{"text": "First chunk"}\n',
        b'{"text": " second chunk"}\n',
        b"[DONE]\n",
    ]

    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/event-stream"}
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

    client = LocalASRClient(base_url="http://127.0.0.1:8002")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    result = client.transcribe(audio_file)

    assert result == "First chunk second chunk"


def test_transcribe_streaming_openai_delta_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test streaming response with OpenAI-style delta format."""
    from video_asr_summary.asr_client import LocalASRClient

    # OpenAI streaming format with choices[0].delta.content
    sse_lines = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
        b'data: {"choices": [{"delta": {"content": " from"}}]}\n',
        b'data: {"choices": [{"delta": {"content": " OpenAI"}}]}\n',
        b"data: [DONE]\n",
    ]

    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/event-stream"}
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

    client = LocalASRClient(base_url="http://127.0.0.1:8002")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    result = client.transcribe(audio_file)

    assert result == "Hello from OpenAI"


def test_transcribe_streaming_openai_message_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test streaming response with OpenAI-style message format."""
    from video_asr_summary.asr_client import LocalASRClient

    # OpenAI non-streaming format with choices[0].message.content
    sse_lines = [
        b'data: {"choices": [{"message": {"content": "Complete message"}}]}\n',
        b"data: [DONE]\n",
    ]

    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/event-stream"}
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

    client = LocalASRClient(base_url="http://127.0.0.1:8002")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    result = client.transcribe(audio_file)

    assert result == "Complete message"


def test_transcribe_streaming_empty_lines_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that empty lines in SSE stream are ignored."""
    from video_asr_summary.asr_client import LocalASRClient

    # SSE with empty lines and comments (should be ignored)
    sse_lines = [
        b"",
        b'data: {"text": "Hello"}\n',
        b"",
        b": this is a comment\n",  # SSE comments start with :
        b'data: {"text": " World"}\n',
        b"",
        b"data: [DONE]\n",
    ]

    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/event-stream"}
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

    client = LocalASRClient(base_url="http://127.0.0.1:8002")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    result = client.transcribe(audio_file)

    assert result == "Hello World"


def test_transcribe_streaming_malformed_json_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that malformed JSON lines are skipped during streaming."""
    from video_asr_summary.asr_client import LocalASRClient

    # SSE with some malformed lines
    sse_lines = [
        b'data: {"text": "Valid"}\n',
        b"data: {invalid json}\n",
        b'data: {"text": " text"}\n',
        b"data: [DONE]\n",
    ]

    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/event-stream"}
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

    client = LocalASRClient(base_url="http://127.0.0.1:8002")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    result = client.transcribe(audio_file)

    assert result == "Valid text"


def test_transcribe_non_streaming_json_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test non-streaming JSON response (fallback when not SSE)."""
    from video_asr_summary.asr_client import LocalASRClient

    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {"text": "Non-streaming result"}
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

    client = LocalASRClient(base_url="http://127.0.0.1:8002")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    result = client.transcribe(audio_file)

    assert result == "Non-streaming result"


def test_transcribe_streaming_no_done_terminator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test streaming without [DONE] terminator (exhausts iterator)."""
    from video_asr_summary.asr_client import LocalASRClient

    # SSE without [DONE] - should process all lines
    sse_lines = [
        b'data: {"text": "No"}\n',
        b'data: {"text": " terminator"}\n',
    ]

    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/event-stream"}
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

    client = LocalASRClient(base_url="http://127.0.0.1:8002")
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake-audio-data")
    result = client.transcribe(audio_file)

    assert result == "No terminator"
