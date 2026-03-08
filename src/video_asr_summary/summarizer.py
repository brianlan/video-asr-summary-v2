from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any, Sequence

import requests

from .audio import FrameContext, format_timestamp

try:  # pragma: no cover - dependency declared but guard for optional envs
    import json_repair
except ImportError:  # pragma: no cover
    json_repair = None


DEFAULT_OPENAI_COMPAT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_OPENAI_COMPAT_MODEL = "deepseek-reasoner"
STRUCTURED_SUMMARIZER_DEFAULT_MAX_TOKENS = 4096


def _post_with_proxy_bypass(url: str, **kwargs: Any) -> requests.Response:
    try:
        return requests.post(url, proxies={}, **kwargs)
    except TypeError as exc:
        if "proxies" not in str(exc):
            raise
        return requests.post(url, **kwargs)


class OpenAICompatibleSummarizer:
    def __init__(
        self,
        *,
        api_token: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.api_token = api_token or os.getenv("OPENAI_ACCESS_TOKEN")
        if not self.api_token:
            raise RuntimeError("OPENAI_ACCESS_TOKEN is not set")

        resolved_base_url = base_url or os.getenv(
            "OPENAI_COMPAT_BASE_URL", DEFAULT_OPENAI_COMPAT_BASE_URL
        )
        resolved_model = model or os.getenv(
            "OPENAI_COMPAT_MODEL", DEFAULT_OPENAI_COMPAT_MODEL
        )
        self.base_url = resolved_base_url.rstrip("/")
        self.model = resolved_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

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

        response: requests.Response | None = None
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = _post_with_proxy_bypass(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exception = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                raise RuntimeError(
                    f"Summarization request failed after {self.max_retries} retries"
                ) from exc
            except requests.HTTPError as exc:
                last_exception = exc
                status_code = (
                    exc.response.status_code if exc.response is not None else None
                )
                if status_code is not None and 500 <= status_code < 600:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2**attempt))
                        continue
                    raise RuntimeError(
                        f"Summarization request failed after {self.max_retries} retries"
                    ) from exc
                raise RuntimeError("Summarization request failed") from exc
            except (
                requests.RequestException
            ) as exc:  # pragma: no cover - exercised via tests
                raise RuntimeError("Summarization request failed") from exc

        if response is None:
            raise RuntimeError(
                f"Summarization request failed after {self.max_retries} retries"
            ) from last_exception

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
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.api_token = api_token or os.getenv("OPENAI_ACCESS_TOKEN")
        if not self.api_token:
            raise RuntimeError("OPENAI_ACCESS_TOKEN is not set")

        resolved_base_url = base_url or os.getenv(
            "OPENAI_COMPAT_BASE_URL", DEFAULT_OPENAI_COMPAT_BASE_URL
        )
        resolved_model = model or os.getenv(
            "OPENAI_COMPAT_MODEL", DEFAULT_OPENAI_COMPAT_MODEL
        )
        self.base_url = resolved_base_url.rstrip("/")
        self.model = resolved_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

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

        return self._request_content(
            prompt_instructions,
            user_content,
            max_tokens=max_tokens,
        )

    def summarize_with_correction(
        self,
        transcript: str,
        frame_contexts: Sequence[FrameContext],
        *,
        language: str | None = None,
        instructions: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, str]:
        transcript_paragraphs = self._split_transcript_paragraphs(transcript)
        paragraph_count = len(transcript_paragraphs)
        paragraph_contract = (
            "Paragraph Contract for corrected_transcript:\n"
            f"  - The input transcript has {paragraph_count} paragraphs identified by markers [[P1]] through [[P{paragraph_count}]].\n"
            f"  - corrected_transcript MUST contain every marker [[P1]] through [[P{paragraph_count}]] exactly once, in order.\n"
            "  - The text for each paragraph must appear immediately after its marker.\n"
            f"  - After removing markers, corrected_transcript must still map to exactly {paragraph_count} paragraphs separated by blank lines.\n"
            '  - Example (2 paragraphs): "[[P1]]\\nFirst paragraph.\\n\\n[[P2]]\\nSecond paragraph."\n'
        )
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
            "  - Return only a JSON object with keys: summary, corrected_transcript.\n"
            "  - summary must be markdown and begin with a single H1 heading as the article title.\n"
            "  - corrected_transcript must correct factual errors only and preserve paragraph boundaries exactly.\n"
            "  - Do not include conversational preamble or closing remarks.\n"
        )
        prompt_instructions += f"\n{paragraph_contract}"
        if language:
            prompt_instructions += f"Produce both fields in {language}."

        visual_context = self._render_visual_context(frame_contexts)
        marked_transcript = self._render_marked_transcript(transcript_paragraphs)
        user_content = (
            "=== TRANSCRIPT ===\n"
            f"{marked_transcript}\n\n"
            "=== VISUAL CONTEXT (Time-Ordered) ===\n"
            f"{visual_context}"
        )

        content = self._request_content(
            prompt_instructions,
            user_content,
            max_tokens=max_tokens,
        )
        if not content.strip():
            fallback_summary = self.summarize(
                transcript,
                frame_contexts,
                language=language,
                max_tokens=max_tokens,
            ).strip()
            if not fallback_summary:
                raise RuntimeError(
                    "Structured summarize-with-correction returned empty content"
                )
            return {
                "summary": fallback_summary,
                "corrected_transcript": transcript.strip(),
                "error": "model response was empty; used summary-only fallback",
            }

        parsed = self._extract_json_payload(content)
        if parsed is None:
            fallback_summary = content.strip()
            if not fallback_summary:
                raise RuntimeError(
                    "Structured summarize-with-correction response payload is invalid JSON"
                )
            return {
                "summary": fallback_summary,
                "corrected_transcript": transcript.strip(),
                "error": "model response was not valid JSON; using raw response as summary",
            }

        summary = parsed.get("summary")
        corrected_transcript = parsed.get("corrected_transcript")
        result: dict[str, str] = {
            "summary": summary.strip() if isinstance(summary, str) else "",
            "corrected_transcript": transcript.strip(),
        }

        if isinstance(corrected_transcript, str) and corrected_transcript.strip():
            corrected_value = corrected_transcript.strip()
            marker_parsed = self._parse_marked_corrected_transcript(
                corrected_value,
                expected_paragraph_count=paragraph_count,
            )
            if marker_parsed is not None:
                corrected_value = marker_parsed
            self._validate_correction_alignment(
                transcript,
                corrected_value,
                summary=result["summary"],
                corrected_transcript=corrected_value,
            )
            result["corrected_transcript"] = corrected_value
            return result

        result["error"] = "missing corrected_transcript in model response"
        return result

    def _request_content(
        self,
        system_prompt: str,
        user_content: str,
        *,
        max_tokens: int | None,
    ) -> str:
        effective_max_tokens = (
            max_tokens
            if max_tokens is not None
            else STRUCTURED_SUMMARIZER_DEFAULT_MAX_TOKENS
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": effective_max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        response: requests.Response | None = None
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = _post_with_proxy_bypass(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exception = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                raise RuntimeError(
                    f"Structured summarization request failed after {self.max_retries} retries"
                ) from exc
            except requests.HTTPError as exc:
                last_exception = exc
                status_code = (
                    exc.response.status_code if exc.response is not None else None
                )
                if status_code is not None and (
                    status_code == 429 or 500 <= status_code < 600
                ):
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2**attempt))
                        continue
                    raise RuntimeError(
                        f"Structured summarization request failed after {self.max_retries} retries"
                    ) from exc
                raise RuntimeError("Structured summarization request failed") from exc
            except (
                requests.RequestException
            ) as exc:  # pragma: no cover - exercised via tests
                raise RuntimeError("Structured summarization request failed") from exc

        if response is None:
            raise RuntimeError(
                f"Structured summarization request failed after {self.max_retries} retries"
            ) from last_exception

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
    def _extract_json_payload(content: str) -> dict[str, Any] | None:
        text = content.strip()
        fenced = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)```", text, flags=re.DOTALL)
        candidates = [fenced.group(1).strip() if fenced else text]

        structured = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if structured:
            snippet = structured.group(0).strip()
            if snippet and snippet not in candidates:
                candidates.append(snippet)

        for candidate in candidates:
            payload = TranscriptCorrector._parse_candidate(candidate)
            if payload is not None:
                return payload
        return None

    @staticmethod
    def _validate_correction_alignment(
        original: str,
        corrected: str,
        *,
        summary: str,
        corrected_transcript: str,
    ) -> None:
        original_parts = StructuredSummarizer._split_transcript_paragraphs(original)
        corrected_parts = StructuredSummarizer._split_transcript_paragraphs(corrected)

        if len(original_parts) != len(corrected_parts):
            raise TranscriptAlignmentError(
                "Corrected transcript paragraph count does not match original transcript",
                summary=summary,
                corrected_transcript=corrected_transcript,
            )

        for index, (source_para, corrected_para) in enumerate(
            zip(original_parts, corrected_parts),
            start=1,
        ):
            ratio = SequenceMatcher(None, source_para, corrected_para).ratio()
            if ratio < 0.6:
                raise TranscriptAlignmentError(
                    f"Corrected transcript paragraph {index} similarity ratio {ratio:.3f} is below 0.6",
                    summary=summary,
                    corrected_transcript=corrected_transcript,
                )

    @staticmethod
    def _split_transcript_paragraphs(transcript: str) -> list[str]:
        return [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", transcript.strip())
            if paragraph.strip()
        ]

    @staticmethod
    def _render_marked_transcript(paragraphs: Sequence[str]) -> str:
        marked_paragraphs: list[str] = []
        for index, paragraph in enumerate(paragraphs, start=1):
            marked_paragraphs.append(f"[[P{index}]]\n{paragraph.strip()}")
        return "\n\n".join(marked_paragraphs)

    @staticmethod
    def _parse_marked_corrected_transcript(
        corrected_transcript: str,
        *,
        expected_paragraph_count: int,
    ) -> str | None:
        if expected_paragraph_count <= 0:
            return None

        marker_pattern = r"\[\[P(\d+)]]"
        marker_matches = list(re.finditer(marker_pattern, corrected_transcript))

        if len(marker_matches) != expected_paragraph_count:
            return None
        if corrected_transcript[: marker_matches[0].start()].strip():
            return None

        paragraphs: list[str] = []
        for expected_index, marker in enumerate(marker_matches, start=1):
            if int(marker.group(1)) != expected_index:
                return None
            body_start = marker.end()
            body_end = (
                marker_matches[expected_index].start()
                if expected_index < len(marker_matches)
                else len(corrected_transcript)
            )
            paragraph_body = corrected_transcript[body_start:body_end].strip()
            if not paragraph_body:
                return None
            paragraphs.append(paragraph_body)

        if len(paragraphs) != expected_paragraph_count:
            return None
        return "\n\n".join(paragraphs)

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


class TranscriptAlignmentError(RuntimeError):
    def __init__(
        self, message: str, *, summary: str, corrected_transcript: str | None = None
    ) -> None:
        super().__init__(message)
        self.summary = summary
        self.corrected_transcript = corrected_transcript


class TranscriptCorrector:
    """LLM client that refines transcripts using visual context."""

    def __init__(
        self,
        *,
        api_token: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.api_token = api_token or os.getenv("OPENAI_ACCESS_TOKEN")
        if not self.api_token:
            raise RuntimeError("OPENAI_ACCESS_TOKEN is not set")

        resolved_base_url = base_url or os.getenv(
            "OPENAI_COMPAT_BASE_URL", DEFAULT_OPENAI_COMPAT_BASE_URL
        )
        resolved_model = model or os.getenv(
            "OPENAI_COMPAT_MODEL", DEFAULT_OPENAI_COMPAT_MODEL
        )
        self.base_url = resolved_base_url.rstrip("/")
        self.model = resolved_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

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

        response: requests.Response | None = None
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = _post_with_proxy_bypass(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exception = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                raise RuntimeError(
                    f"Transcript correction request failed after {self.max_retries} retries"
                ) from exc
            except requests.HTTPError as exc:
                last_exception = exc
                status_code = (
                    exc.response.status_code if exc.response is not None else None
                )
                if status_code is not None and 500 <= status_code < 600:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2**attempt))
                        continue
                    raise RuntimeError(
                        f"Transcript correction request failed after {self.max_retries} retries"
                    ) from exc
                raise RuntimeError("Transcript correction request failed") from exc
            except (
                requests.RequestException
            ) as exc:  # pragma: no cover - exercised via tests
                raise RuntimeError("Transcript correction request failed") from exc

        if response is None:
            raise RuntimeError(
                f"Transcript correction request failed after {self.max_retries} retries"
            ) from last_exception

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


class ChataiSummarizer(OpenAICompatibleSummarizer):
    pass
