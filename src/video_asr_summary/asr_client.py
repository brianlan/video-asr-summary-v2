from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import requests


class BailianASRClient:
    """Client for the Alibaba Bailian ASR service."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1/services/asr/transcriptions",
        model: str = "paraformer-v1",
        timeout: int = 300,
    ) -> None:
        self.api_key = api_key or os.getenv("BAILIAN_API_KEY")
        if not self.api_key:
            raise RuntimeError("BAILIAN_API_KEY is not set")

        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def transcribe(self, audio_path: Path | str, *, language: str = "en-US") -> str:
        audio_bytes = Path(audio_path).read_bytes()
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        payload: dict[str, Any] = {
            "model": self.model,
            "input": {
                "audio": {
                    "format": Path(audio_path).suffix.lstrip(".") or "wav",
                    "content": encoded_audio,
                }
            },
            "parameters": {
                "language": language,
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - exercised via tests raising RuntimeError
            raise RuntimeError("Bailian ASR request failed") from exc

        data = response.json()
        try:
            return data["output"]["text"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Unexpected Bailian ASR response payload") from exc
