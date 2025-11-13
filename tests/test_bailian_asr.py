from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake-audio")
    return audio


def test_transcribe_posts_audio_with_language(audio_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary.asr_client import BailianASRClient

    monkeypatch.setenv("BAILIAN_API_KEY", "secret")

    captured_call = {}

    class FakeResponse:
        def __init__(self) -> None:
            self.output = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"text": "hello world"},
                            ],
                        }
                    }
                ]
            }

    def fake_call(*, model: str, messages: list, api_key: str, result_format: str, asr_options: dict, **kwargs) -> FakeResponse:  # type: ignore[override]
        captured_call["model"] = model
        captured_call["messages"] = messages
        captured_call["api_key"] = api_key
        captured_call["result_format"] = result_format
        captured_call["asr_options"] = asr_options
        captured_call["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr(
        "video_asr_summary.asr_client.MultiModalConversation.call",
        staticmethod(fake_call),
    )

    client = BailianASRClient()
    transcript = client.transcribe(audio_file, language="zh-CN")

    assert transcript == "hello world"
    assert captured_call["model"] == "qwen3-asr-flash"
    assert captured_call["api_key"] == "secret"
    assert captured_call["result_format"] == "message"
    assert captured_call["asr_options"]["language"] == "zh-CN"

    messages = captured_call["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == [{"text": ""}]

    audio_message = messages[1]
    assert audio_message["role"] == "user"
    assert audio_message["content"] == [{"audio": str(audio_file)}]


def test_transcribe_raises_on_error(audio_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary.asr_client import BailianASRClient

    monkeypatch.setenv("BAILIAN_API_KEY", "secret")

    def fake_call(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "video_asr_summary.asr_client.MultiModalConversation.call",
        staticmethod(fake_call),
    )

    client = BailianASRClient()
    with pytest.raises(RuntimeError):
        client.transcribe(audio_file)


def test_local_qwen_asr_transcribe_uses_injected_dependencies(tmp_path: Path) -> None:
    from video_asr_summary.asr_client import LocalQwenASRClient

    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"fake")

    captured: dict[str, object] = {}

    class FakeProcessor:
        def apply_chat_template(self, messages: list, *, tokenize: bool, add_generation_prompt: bool) -> str:
            captured["messages"] = messages
            captured["tokenize"] = tokenize
            captured["add_generation_prompt"] = add_generation_prompt
            return "prompt-text"

    class FakeLLM:
        def generate(self, prompts: list, *, sampling_params: object) -> list:
            captured["prompts"] = prompts
            captured["sampling_params"] = sampling_params

            class GeneratedOutput:
                def __init__(self, text: str) -> None:
                    self.outputs = [MagicMock(text=text)]

            return [GeneratedOutput("  final text  ")]

    def fake_process_mm_info(messages: list, *, use_audio_in_video: bool) -> tuple[list[str], None, None]:
        captured["mm_messages"] = messages
        captured["use_audio_in_video"] = use_audio_in_video
        return (["audio-bytes"], None, None)

    client = LocalQwenASRClient(
        model_path="dummy",
        llm=FakeLLM(),
        processor=FakeProcessor(),
        sampling_params=object(),
        process_mm_info=fake_process_mm_info,
        prompt_template="Please output the ASR result of the above {media_type} in {language}.",
    )

    transcript = client.transcribe(audio_path, language="ja")

    assert transcript == "final text"

    messages = captured["messages"]
    assert messages[0]["role"] == "user"
    audio_content = messages[0]["content"][0]
    assert audio_content["type"] == "audio"
    assert audio_content["audio"] == str(audio_path.resolve())

    prompt_content = messages[0]["content"][1]
    assert prompt_content["text"].lower().startswith("please output")
    assert "ja" in prompt_content["text"]

    prompts = captured["prompts"]
    assert prompts[0]["prompt"] == "prompt-text"
    assert prompts[0]["multi_modal_data"]["audio"] == ["audio-bytes"]
    assert captured["sampling_params"] is not None
    assert captured["use_audio_in_video"] is True


def test_local_qwen_vision_describe_image_uses_injected_dependencies(tmp_path: Path) -> None:
    from video_asr_summary.asr_client import LocalQwenVisionClient

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")

    captured: dict[str, object] = {}

    class FakeProcessor:
        def apply_chat_template(self, messages: list, *, tokenize: bool, add_generation_prompt: bool) -> str:
            captured["messages"] = messages
            captured["tokenize"] = tokenize
            captured["add_generation_prompt"] = add_generation_prompt
            return "prompt-text"

    class FakeLLM:
        def generate(self, prompts: list, *, sampling_params: object) -> list:
            captured["prompts"] = prompts
            captured["sampling_params"] = sampling_params

            class GeneratedOutput:
                def __init__(self, text: str) -> None:
                    self.outputs = [MagicMock(text=text)]

            return [GeneratedOutput("View shows text: Launch Day")] 

    def fake_process_mm_info(messages: list, *, use_audio_in_video: bool) -> tuple[None, list[str], None]:
        captured["mm_messages"] = messages
        captured["use_audio_in_video"] = use_audio_in_video
        return (None, ["image-bytes"], None)

    client = LocalQwenVisionClient(
        model_path="dummy",
        llm=FakeLLM(),
        processor=FakeProcessor(),
        sampling_params=object(),
        process_mm_info=fake_process_mm_info,
        prompt_template="Describe the following scene in {language}.",
    )

    description = client.describe_image(image_path, language="en")

    assert "Launch Day" in description

    messages = captured["messages"]
    assert messages[0]["role"] == "user"
    image_content = messages[0]["content"][0]
    assert image_content["type"] == "image"
    assert image_content["image"] == str(image_path.resolve())

    prompt_content = messages[0]["content"][1]
    assert "Describe" in prompt_content["text"]
    assert "en" in prompt_content["text"]

    prompts = captured["prompts"]
    assert prompts[0]["prompt"] == "prompt-text"
    assert prompts[0]["multi_modal_data"]["image"] == ["image-bytes"]
    assert captured["sampling_params"] is not None


def test_local_qwen_asr_exports_runtime_components(tmp_path: Path) -> None:
    from video_asr_summary.asr_client import LocalQwenASRClient

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"0")

    fake_llm = object()
    fake_processor = object()
    fake_sampling = object()

    def fake_process(messages: list, *, use_audio_in_video: bool):
        return (None, None, None)

    client = LocalQwenASRClient(
        model_path="dummy",
        llm=fake_llm,
        processor=fake_processor,
        sampling_params=fake_sampling,
        process_mm_info=fake_process,
    )

    shared = client.export_runtime_components()

    assert shared["llm"] is fake_llm
    assert shared["processor"] is fake_processor
    assert shared["sampling_params"] is fake_sampling
    assert shared["process_mm_info"] is fake_process
