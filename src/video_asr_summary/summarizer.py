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
            "Role: You are an expert editor and content analyst.\n\n"
            "Task: I will provide you with a raw, verbatim transcript from a video. This text is highly conversational, likely containing filler words, redundancies, and a loose logical structure. Your task is to reconstruct this text into a well-structured, logically coherent, and readable article, not a brief summary.\n\n"
            "Key Requirements:\n"
            "  - Cleanse and Refine: Remove all meaningless filler words (e.g., 'uh,' 'um,' 'like,' 'you know'), verbal tics, and obvious repetitions. Convert overly conversational phrasing into more formal, fluid written language.\n"
            "  - Build Logical Structure: Reorganize the content. Use clear headings, subheadings (if necessary), and paragraphs to separate distinct topics or arguments. Ensure the article has a clear introduction, body, and conclusion.\n"
            "  - Preserve Core Arguments & Reasoning (Most Important): Clearly identify all main points or conclusions in the text. For each point, you must retain the key evidence, reasoning, data, or examples the speaker used to support it. Do not just state 'The speaker believes X.' You must elaborate: 'The speaker believes X because of Y and Z.'\n"
            "  - Retain Significant Details & Anecdotes: While ensuring flow, preserve specific details, anecdotes, or personal stories that add depth, context, or interest. If the speaker told a short story to illustrate a point, retain the essence of that story.\n"
            "  - Maintain Neutrality and Fidelity: Preserve the speaker's original intent and stance. Do not add your own opinions or interpretations.\n\n"
            "Output Format: \n"
            "  - Use Markdown for formatting.\n"
            "  - Begin with a single H1 heading that serves as a concise title capturing the overall theme.\n"
            "  - You must not include any conversational introduction, preamble, or concluding remarks (e.g., do not say 'Okay, here is...' or 'I have processed...')..\n\n"
        )
        if language:
            prompt_instructions += f"Produce the markdown in {language}."

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
