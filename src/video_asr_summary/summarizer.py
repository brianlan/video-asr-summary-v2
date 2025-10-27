from __future__ import annotations

import os
from typing import Any

import requests


class ChataiSummarizer:
    """Client for the chatai OpenAI-compatible summarization endpoint."""

    def __init__(
        self,
        *,
        api_token: str | None = None,
        base_url: str = "https://api.chataiapi.com/v1",
        model: str = "gpt-4o-mini", # other available models: ["gpt-4o-mini"]
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
    ) -> str:
        prompt_instructions = instructions or (
            "You are a writing assistant. Reorganize the provided transcript into a clear, well-structured "
            "markdown document. Remove filler words, rewrite informal phrases into formal prose, and group "
            "related ideas under concise headings and bullet lists when appropriate. Do not return JSON; "
            "respond using markdown only."
        )
        if language:
            prompt_instructions += f" Produce the markdown in {language}."

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

        if not isinstance(content, str):
            raise RuntimeError("Summarization response content is not text")

        return content
