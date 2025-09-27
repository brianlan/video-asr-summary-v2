from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


class ChataiSummarizer:
    """Client for the chatai OpenAI-compatible summarization endpoint."""

    def __init__(
        self,
        *,
        api_token: str | None = None,
        base_url: str = "https://www.chataiapi.com/v1",
        model: str = "gpt-4o-mini",
        timeout: int = 120,
    ) -> None:
        self.api_token = api_token or os.getenv("OPENAI_ACCESS_TOKEN")
        if not self.api_token:
            raise RuntimeError("OPENAI_ACCESS_TOKEN is not set")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def summarize(
        self,
        text: str,
        *,
        language: str | None = None,
        instructions: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        prompt_instructions = instructions or (
            "You are a summarization assistant. Return a concise JSON object with keys "
            "'summary' and 'highlights' (an array of strings)."
        )
        if language:
            prompt_instructions += f" Respond in {language}."

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt_instructions},
                {"role": "user", "content": text},
            ],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError("Summarization request failed") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected summarization response payload") from exc

        return _parse_json_content(content)


def _parse_json_content(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_PATTERN.search(content)
        if not match:
            raise RuntimeError("Summarization response is not valid JSON") from None
        block = match.group(1).strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            block = block.encode("utf-8").decode("unicode_escape")
            return json.loads(block)
