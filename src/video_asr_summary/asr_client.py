from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

try:  # pragma: no cover - allow tests to run when dashscope is unavailable
    from dashscope import MultiModalConversation
except ImportError:  # pragma: no cover - tests provide a stub via monkeypatch
    class MultiModalConversation:  # type: ignore[override]
        @staticmethod
        def call(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("dashscope.MultiModalConversation is required for BailianASRClient")


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

        error_code = self._get_attr_or_key(response, "code")
        if isinstance(error_code, str) and error_code.strip():
            error_message = self._get_attr_or_key(response, "message") or ""
            raise RuntimeError(
                f"Bailian ASR error: {error_code}: {error_message}"
            )

        transcript = self._extract_transcript(response)
        if transcript is None:
            raise RuntimeError("Unexpected Bailian ASR response payload")

        return transcript

    @staticmethod
    def _get_attr_or_key(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    @staticmethod
    def _extract_transcript(response: Any) -> str | None:
        output = BailianASRClient._get_attr_or_key(response, "output")
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


class LocalQwenASRClient:
    """Client that runs Qwen Omni ASR locally via vLLM."""

    DEFAULT_MODEL_PATH = "/ssd4/models/cpatonn-mirror/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        prompt_template: str = "Please output the ASR result of the audio.",
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 16384,
        max_model_len: int = 32768,
        max_num_seqs: int = 8,
        gpu_memory_utilization: float = 0.95,
        tensor_parallel_size: Optional[int] = None,
        limit_mm_per_prompt: Optional[dict[str, int]] = None,
        seed: int = 1234,
        trust_remote_code: bool = True,
        use_audio_in_video: bool = True,
        llm: Any | None = None,
        processor: Any | None = None,
        sampling_params: Any | None = None,
        process_mm_info: Callable[[list, Any], tuple[Any, Any, Any]] | None = None,
        extra_llm_kwargs: Optional[dict[str, Any]] = None,
        extra_sampling_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model_path = str(model_path or self.DEFAULT_MODEL_PATH)
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self._tensor_parallel_size = tensor_parallel_size
        self.limit_mm_per_prompt = limit_mm_per_prompt or {"image": 3, "video": 3, "audio": 3}
        self.seed = seed
        self.trust_remote_code = trust_remote_code
        self.use_audio_in_video = use_audio_in_video
        self._llm = llm
        self._processor = processor
        self._sampling_params = sampling_params
        self._process_mm_info = process_mm_info
        self.extra_llm_kwargs = extra_llm_kwargs or {}
        self.extra_sampling_kwargs = extra_sampling_kwargs or {}

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        language: str = "en",
        media_type: str = "audio",
    ) -> str:
        path = Path(audio_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        os.environ.setdefault("VLLM_USE_V1", "0")

        prompt_text = self.prompt_template.format(media_type=media_type, language=language)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": media_type, media_type: str(path)},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        processor = self._ensure_processor()
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        mm_messages = self._process_mm_messages()
        audios, images, videos = mm_messages(messages, use_audio_in_video=self.use_audio_in_video)

        multi_modal_data: dict[str, Any] = {}
        if audios is not None:
            multi_modal_data["audio"] = audios
        if images is not None:
            multi_modal_data["image"] = images
        if videos is not None:
            multi_modal_data["video"] = videos

        inputs = {
            "prompt": prompt,
            "multi_modal_data": multi_modal_data,
            "mm_processor_kwargs": {
                "use_audio_in_video": self.use_audio_in_video,
            },
        }

        llm = self._ensure_llm()
        sampling_params = self._ensure_sampling_params()
        outputs = llm.generate([inputs], sampling_params=sampling_params)

        try:
            text = outputs[0].outputs[0].text
        except (IndexError, AttributeError, TypeError) as exc:
            raise RuntimeError("Local ASR response payload is malformed") from exc

        return str(text).strip()

    def _ensure_llm(self) -> Any:
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _ensure_processor(self) -> Any:
        if self._processor is None:
            self._processor = self._create_processor()
        return self._processor

    def _ensure_sampling_params(self) -> Any:
        if self._sampling_params is None:
            self._sampling_params = self._create_sampling_params()
        return self._sampling_params

    def _process_mm_messages(self) -> Callable[[list, Any], tuple[Any, Any, Any]]:
        if self._process_mm_info is None:
            self._process_mm_info = self._load_process_mm_info()
        return self._process_mm_info

    def _create_llm(self) -> Any:
        try:
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover - handled in tests via injection
            raise RuntimeError("vLLM is required for LocalQwenASRClient") from exc

        tensor_parallel_size = self._tensor_parallel_size
        if tensor_parallel_size is None:
            tensor_parallel_size = self._infer_tensor_parallel_size()

        return LLM(
            model=self.model_path,
            trust_remote_code=self.trust_remote_code,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            max_num_seqs=self.max_num_seqs,
            max_model_len=self.max_model_len,
            seed=self.seed,
            **self.extra_llm_kwargs,
        )

    def _create_processor(self) -> Any:
        try:
            from transformers import Qwen3OmniMoeProcessor
        except ImportError as exc:  # pragma: no cover - handled in tests via injection
            raise RuntimeError("transformers is required for LocalQwenASRClient") from exc

        return Qwen3OmniMoeProcessor.from_pretrained(self.model_path)

    def _create_sampling_params(self) -> Any:
        try:
            from vllm import SamplingParams
        except ImportError as exc:  # pragma: no cover - handled in tests via injection
            raise RuntimeError("vLLM SamplingParams is required for LocalQwenASRClient") from exc

        params: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
        }
        params.update(self.extra_sampling_kwargs)
        return SamplingParams(**params)

    @staticmethod
    def _load_process_mm_info() -> Callable[[list, Any], tuple[Any, Any, Any]]:
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError as exc:  # pragma: no cover - handled in tests via injection
            raise RuntimeError("qwen_omni_utils.process_mm_info is required for LocalQwenASRClient") from exc
        return process_mm_info

    @staticmethod
    def _infer_tensor_parallel_size() -> int:
        try:
            import torch
        except ImportError:  # pragma: no cover - torch may be unavailable during tests
            return 1

        count = getattr(torch.cuda, "device_count", lambda: 0)()
        return count if count else 1
