from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from dashscope import MultiModalConversation


class BailianASRClient:
    """Client for the Alibaba Bailian ASR service using MultiModalConversation."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "qwen3-asr-flash",
        timeout: int = 300,
        system_prompt: str = "",
        default_asr_options: dict[str, Any] | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("BAILIAN_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("BAILIAN_API_KEY is not set")
        self.api_key: str = str(resolved_api_key)

        self.model = model
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.default_asr_options = default_asr_options or {
            "enable_lid": True,
            "enable_itn": False,
        }

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        language: str = "en",
        context: str | None = None,
    ) -> str:
        path = Path(audio_path).resolve()

        system_text = context if context is not None else self.system_prompt

        messages = [
            {
                "role": "system",
                "content": [{"text": system_text}],
            },
            {
                "role": "user",
                "content": [{"audio": str(path)}],
            },
        ]

        asr_options: dict[str, Any] = {**self.default_asr_options}
        if language:
            asr_options["language"] = language

        try:
            response = MultiModalConversation.call(
                model=self.model,
                api_key=self.api_key,
                messages=messages,
                result_format="message",
                asr_options=asr_options,
                timeout=self.timeout,
            )
        except Exception as exc:  # pragma: no cover - raised via tests
            raise RuntimeError("Bailian ASR request failed") from exc

        transcript = self._extract_transcript(response)
        if transcript is None:
            raise RuntimeError("Unexpected Bailian ASR response payload")

        return transcript

    @staticmethod
    def _extract_transcript(response: Any) -> str | None:
        output = getattr(response, "output", None)
        if output is None:
            return None

        choices = BailianASRClient._get_attr_or_key(output, "choices")
        if not choices:
            return None

        first_choice = choices[0]
        message = BailianASRClient._get_attr_or_key(first_choice, "message")
        if not message:
            return None

        content = BailianASRClient._get_attr_or_key(message, "content")
        if isinstance(content, str):
            return content
        if isinstance(content, Iterable):
            for element in content:
                text_value = BailianASRClient._get_attr_or_key(element, "text")
                if isinstance(text_value, str):
                    return text_value
        return None

    @staticmethod
    def _get_attr_or_key(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)
