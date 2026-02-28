from __future__ import annotations

import json
import os
import re
from typing import Any, Sequence

import requests

from .audio import FrameContext, format_timestamp

try:  # pragma: no cover - dependency declared but guard for optional envs
    import json_repair
except ImportError:  # pragma: no cover
    json_repair = None


class ChataiSummarizer:
    """Client for the chatai OpenAI-compatible summarization endpoint."""

    def __init__(
        self,
        *,
        api_token: str | None = None,
        base_url: str = "https://api.chataiapi.com/v1",
        model: str = "gpt-5-mini",  # other available models: ["gpt-5-mini"]
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
        except (
            requests.RequestException
        ) as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError("Summarization request failed") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected summarization response payload") from exc

        if not isinstance(content, str):
            raise RuntimeError("Summarization response content is not text")

        return content


class StructuredSummarizer:
    def __init__(
        self,
        *,
        api_token: str | None = None,
        base_url: str = "https://api.chataiapi.com/v1",
        model: str = "gpt-5-mini",
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
        transcript: str,
        frame_contexts: Sequence[FrameContext],
        *,
        language: str | None = None,
        instructions: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        prompt_instructions = instructions or (
            "Role: You are an expert editor and content analyst.\n\n"
            "Task: I will provide two synchronized sources from one video: a raw transcript and "
            "time-ordered visual context extracted from frames. Use both sources in one pass to "
            "produce a coherent, faithful markdown article.\n\n"
            "Key Requirements:\n"
            "  - Improve clarity while preserving the speaker's intent, reasoning, examples, and important details.\n"
            "  - Use visual context only to resolve ambiguity or strengthen factual accuracy; do not invent facts.\n"
            "  - Keep structure clear with a title, sections, and concise paragraphs or bullets where helpful.\n"
            "  - Include [MM:SS] timestamps in the final markdown for key statements so readers can trace claims to the video timeline.\n\n"
            "Output Format:\n"
            "  - Use Markdown.\n"
            "  - Begin with a single H1 heading as the article title.\n"
            "  - Do not include conversational preamble or closing remarks.\n"
        )
        if language:
            prompt_instructions += f"Produce the markdown in {language}."

        visual_context = self._render_visual_context(frame_contexts)
        user_content = (
            "=== TRANSCRIPT ===\n"
            f"{transcript.strip()}\n\n"
            "=== VISUAL CONTEXT (Time-Ordered) ===\n"
            f"{visual_context}"
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt_instructions},
                {"role": "user", "content": user_content},
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
        except (
            requests.RequestException
        ) as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError("Structured summarization request failed") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected structured summarization response payload"
            ) from exc

        if not isinstance(content, str):
            raise RuntimeError("Structured summarization response content is not text")

        return content

    @staticmethod
    def _render_visual_context(frame_contexts: Sequence[FrameContext]) -> str:
        lines: list[str] = []
        ordered_contexts = sorted(
            frame_contexts,
            key=lambda frame: (
                frame.timestamp_start,
                frame.timestamp_end,
                str(frame.frame_path),
            ),
        )

        for frame in ordered_contexts:
            ocr_text = frame.ocr_text.strip()
            if not ocr_text:
                continue
            start = format_timestamp(frame.timestamp_start).strip("[]")
            end = format_timestamp(frame.timestamp_end).strip("[]")
            lines.append(f"[{start}-{end}] {ocr_text}")

        if not lines:
            return "(no non-empty visual context)"
        return "\n".join(lines)


class TranscriptCorrector:
    """LLM client that refines transcripts using visual context."""

    def __init__(
        self,
        *,
        api_token: str | None = None,
        base_url: str = "https://api.chataiapi.com/v1",
        model: str = "gpt-5-mini",
        timeout: int = 120,
    ) -> None:
        self.api_token = api_token or os.getenv("OPENAI_ACCESS_TOKEN")
        if not self.api_token:
            raise RuntimeError("OPENAI_ACCESS_TOKEN is not set")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def correct(
        self,
        transcript: str,
        *,
        image_context: Sequence[str] | None = None,
        language: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        details = [
            context.strip() for context in (image_context or []) if context.strip()
        ]

        instructions = (
            "You refine noisy ASR transcripts using additional observations captured from video frames. "
            "Focus on correcting factual errors: fix spelling of names, places, technical terms, and abbreviations. "
            "Use the image context to resolve ambiguous references. "
            "Do NOT rewrite the speaker's voice, conversational style, or personality. "
            "Preserve all demonstrations, real-time reactions, specific examples, and personal anecdotes. "
            "No need to make the corrected transcript too formal."
        )
        # if language:
        #     instructions += f" Respond in {language}."

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "transcript": transcript,
                            "image_context": details,
                            "language": language,
                        },
                        ensure_ascii=False,
                    ),
                },
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
        except (
            requests.RequestException
        ) as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError("Transcript correction request failed") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected transcript correction response payload"
            ) from exc

        if not isinstance(content, str):
            raise RuntimeError("Transcript correction response content is not text")

        return self._extract_corrected_transcript(content)

    @staticmethod
    def _extract_corrected_transcript(content: str) -> str:
        text = content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        candidates = [text]
        if not fenced:
            structured = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if structured:
                snippet = structured.group(0).strip()
                if snippet and snippet not in candidates:
                    candidates.append(snippet)

        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            payload = TranscriptCorrector._parse_candidate(candidate)
            if payload is None:
                continue
            corrected = payload.get("corrected_transcript")
            if isinstance(corrected, str) and corrected.strip():
                return corrected.strip()

        if text:
            return text
        raise RuntimeError("Transcript correction response payload is invalid JSON")

    @staticmethod
    def _parse_candidate(candidate: str) -> dict[str, Any] | None:
        try:
            loaded = json.loads(candidate)
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            if json_repair is None:
                return None
            try:
                repaired = json_repair.loads(candidate)
            except Exception:
                return None
            return repaired if isinstance(repaired, dict) else None
