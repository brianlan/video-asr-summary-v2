from __future__ import annotations

from pathlib import Path
from typing import Any
import time
import requests


class LocalASRClient:
    """Client for local ASR service via HTTP API."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8002",
        model: str = "whisper",
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def transcribe(self, audio_path: Path | str, *, language: str = "en") -> str:
        """Transcribe audio file using local ASR service.

        Args:
            audio_path: Path to the audio file to transcribe
            language: Language code (e.g., 'en', 'zh')

        Returns:
            Transcribed text from the ASR service

        Raises:
            RuntimeError: If the request fails after retries or on connection/timeout errors
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        url = f"{self.base_url.rstrip('/')}/v1/audio/transcriptions"

        # Prepare form data
        data: dict[str, str] = {"model": self.model, "response_format": "json"}
        if language:
            data["language"] = language

        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                with open(path, "rb") as audio_file:
                    files = {"file": (path.name, audio_file, "audio/mpeg")}
                    # Use proxies={} to bypass environment proxies for localhost
                    response = requests.post(
                        url,
                        files=files,
                        data=data,
                        timeout=self.timeout,
                        proxies={},
                        stream=True,
                    )
                    response.raise_for_status()
                    return self._parse_response(response)

            except requests.ConnectionError as exc:
                raise RuntimeError(
                    f"ASR request failed: connection error - {exc}"
                ) from exc
            except requests.Timeout as exc:
                raise RuntimeError(f"ASR request failed: timeout - {exc}") from exc
            except requests.HTTPError as exc:
                last_exception = exc
                # Retry on HTTP errors (transient failures)
                if attempt < self.max_retries - 1:
                    # Exponential backoff: delay increases with each retry
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                break

        # All retries exhausted
        raise RuntimeError(
            f"ASR request failed after {self.max_retries} retries"
        ) from last_exception

    def _parse_response(self, response: requests.Response) -> str:
        """Parse streaming or non-streaming response from ASR service.

        Handles SSE (Server-Sent Events) format with 'data:' prefix,
        or falls back to non-streaming JSON response.

        Args:
            response: The HTTP response from the ASR service

        Returns:
            Accumulated transcript text
        """
        transcript_chunks: list[str] = []

        # Check if this is a streaming response by looking at content-type or
        # trying to iterate lines
        content_type = response.headers.get("Content-Type", "").lower()
        is_sse = "text/event-stream" in content_type

        if is_sse:
            # Parse SSE format: data: {...} lines with [DONE] terminator
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                # Remove optional 'data:' prefix
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if line_str.startswith("data: "):
                    line_str = line_str[6:]
                elif line_str.startswith("data:"):
                    line_str = line_str[5:]

                line_str = line_str.strip()

                if not line_str:
                    continue

                # Check for SSE terminator
                if line_str == "[DONE]":
                    break

                # Parse JSON payload and extract text
                try:
                    payload = line_str
                    if payload:
                        chunk_text = self._extract_text_from_payload(payload)
                        if chunk_text:
                            transcript_chunks.append(chunk_text)
                except Exception:
                    # Skip malformed lines in streaming mode
                    continue
        else:
            # Fallback to non-streaming JSON
            try:
                result = response.json()
                text = self._extract_text_from_payload(result)
                if text:
                    transcript_chunks.append(text)
            except (ValueError, KeyError) as exc:
                raise RuntimeError(f"ASR response parsing failed: {exc}") from exc

        return "".join(transcript_chunks)

    def _extract_text_from_payload(self, payload: str | dict[str, Any]) -> str | None:
        """Extract transcript text from a JSON payload.

        Supports multiple response formats:
        - Simple: {"text": "transcript"}
        - OpenAI streaming delta: {"choices": [{"delta": {"content": "..."}}]}
        - OpenAI message: {"choices": [{"message": {"content": "..."}}]}

        Args:
            payload: JSON string or parsed dict

        Returns:
            Extracted text or None if not found
        """
        # Parse if string
        data: dict[str, Any]
        if isinstance(payload, str):
            try:
                data = payload.strip()
                if not data:
                    return None
                import json

                data = json.loads(data)
            except (ValueError, TypeError):
                return None
        else:
            data = payload

        if not isinstance(data, dict):
            return None

        # Try simple "text" field first (preferred)
        if "text" in data and isinstance(data["text"], str):
            return data["text"]

        # Try OpenAI-style choices array
        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            return None

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return None

        # Try delta format (streaming)
        delta = first_choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content

        # Try message format (non-streaming)
        message = first_choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

        return None


class LocalOCRClient:
    """Client for local OCR/vision model via HTTP API."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8001",
        model: str = "qwen2-vl",
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def describe_image(self, image_path: Path | str, *, language: str = "en") -> str:
        """Describe/OCR an image using local vision model service.

        Args:
            image_path: Path to the image file to analyze
            language: Language code for the expected text (e.g., 'en', 'zh')

        Returns:
            Extracted text from the image

        Raises:
            FileNotFoundError: If the image file does not exist
            RuntimeError: If the request fails after retries
            ConnectionError: If connection to the service fails
            TimeoutError: If the request times out
        """
        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"

        # Build OpenAI-style chat completion request with vision
        prompt_text = f"Extract all text from this image. Language: {language}"
        request_json = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"file://{path}"},
                        },
                    ],
                }
            ],
        }

        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                # Use proxies={} to bypass environment proxies for localhost
                response = requests.post(
                    url,
                    json=request_json,
                    timeout=self.timeout,
                    proxies={},
                )
                response.raise_for_status()
                result = response.json()
                return str(result["choices"][0]["message"]["content"])

            except requests.ConnectionError as exc:
                raise ConnectionError(
                    f"OCR request failed: connection error - {exc}"
                ) from exc
            except requests.Timeout as exc:
                raise TimeoutError(f"OCR request failed: timeout - {exc}") from exc
            except requests.HTTPError as exc:
                last_exception = exc
                # Retry on HTTP errors (transient failures)
                if attempt < self.max_retries - 1:
                    # Exponential backoff: delay increases with each retry
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                # Last attempt failed, raise RuntimeError with HTTP details
                raise RuntimeError(f"HTTP 500: Internal Server Error") from exc

        # Should not reach here, but for type safety
        raise RuntimeError(
            f"OCR request failed after {self.max_retries} retries"
        ) from last_exception
