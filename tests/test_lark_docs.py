from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import pytest

from video_asr_summary.lark_docs import (  # pyright: ignore[reportMissingImports]
    LarkDocError,
    create_summary_document,
    derive_lark_title,
)


def test_derive_lark_title_prefers_heading() -> None:
    summary = "# Heading Title\n\nContent line."
    assert derive_lark_title(summary, "Fallback") == "Heading Title"


def test_derive_lark_title_uses_fallback_when_needed() -> None:
    assert derive_lark_title("   \n", "Fallback Title") == "Fallback Title"


def test_ensure_client_builder_sets_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    recorded: dict[str, Any] = {}

    class FakeBuilder:
        def app_id(self, value):
            recorded["app_id"] = value
            return self

        def app_secret(self, value):
            recorded["app_secret"] = value
            return self

        def log_level(self, value):
            recorded["log_level"] = value
            return self

        def timeout(self, value):
            recorded["timeout"] = value
            return self

        def domain(self, value):
            recorded["domain"] = value
            return self

        def enable_set_token(self, value):
            recorded["enable_set_token"] = value
            return self

    class FakeClientEntry:
        @staticmethod
        def builder():
            return FakeBuilder()

    monkeypatch.setattr(lark_docs, "Client", FakeClientEntry)

    builder, request_option = lark_docs._ensure_client_builder(
        "app-id",
        "app-secret",
        tenant_access_token=None,
        user_access_token=None,
        domain=None,
    )

    assert isinstance(builder, FakeBuilder)
    assert request_option is None
    assert recorded["timeout"] == 30


def test_request_tenant_access_token_bypasses_proxy_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("LARK_DISABLE_PROXY", "1")

    recorded: dict[str, Any] = {}

    def fake_post(url, headers=None, json=None, timeout=10, proxies=None):
        recorded["url"] = url
        recorded["headers"] = headers
        recorded["json"] = json
        recorded["timeout"] = timeout
        recorded["proxies"] = proxies
        return SimpleNamespace(
            status_code=200,
            ok=True,
            json=lambda: {
                "code": 0,
                "msg": "ok",
                "tenant_access_token": "token-xyz",
                "expire": 7200,
            },
        )

    monkeypatch.setattr(lark_docs.requests, "post", fake_post)

    token, _, _ = lark_docs._request_tenant_access_token("app-id", "app-secret")

    assert token == "token-xyz"
    assert recorded["url"].endswith("/tenant_access_token/internal")
    assert recorded["proxies"] == {}


def test_create_summary_document_builds_blocks_and_uses_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    recorded: dict[str, Any] = {}

    monkeypatch.setattr(
        lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: None
    )

    class FakeDocumentResource:
        def create(self, request, option=None):
            recorded["title"] = request.request_body.title
            recorded["folder"] = request.request_body.folder_token
            recorded["document_option"] = option
            recorded["create_document_called"] = True
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    document=SimpleNamespace(
                        document_id="Doc123",
                        revision_id=1,
                        title=request.request_body.title,
                    )
                ),
            )

    class FakeBlockResource:
        def create(self, request, option=None):
            call = {
                "block_id": request.block_id,
                "index": request.request_body.index,
                "children": request.request_body.children,
                "option": option,
            }
            recorded.setdefault("block_calls", []).append(call)

            created_children = []
            for idx, block in enumerate(request.request_body.children):
                block.block_id = f"blk_{len(recorded['block_calls'])}_{idx}"
                created_children.append(block)

            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    children=created_children,
                    document_revision_id=len(recorded["block_calls"]),
                    client_token="token",
                ),
            )

    class FakeDocxV1:
        def __init__(self):
            self.document = FakeDocumentResource()
            self.document_block_children = FakeBlockResource()

    class FakeDocxService:
        def __init__(self):
            self.v1 = FakeDocxV1()

    class FakeClient:
        def __init__(self):
            self.docx = FakeDocxService()

    class FakeBuilder:
        def build(self):
            return FakeClient()

    request_option = SimpleNamespace(option="user_access")

    def fake_builder(
        app_id: str,
        app_secret: str,
        *,
        tenant_access_token=None,
        user_access_token=None,
        domain=None,
    ):
        recorded["app_id"] = app_id
        recorded["app_secret"] = app_secret
        recorded["tenant"] = tenant_access_token
        recorded["user_access_token"] = user_access_token
        recorded["domain"] = domain
        return FakeBuilder(), request_option

    monkeypatch.setattr(lark_docs, "_ensure_client_builder", fake_builder)

    result = create_summary_document(
        "# Top Heading\n\nIntro paragraph with **bold** text.\n\n- First bullet\n- Second bullet\n\n1. First item\n2. Second item\n",
        title="Provided Title",
        folder_token="fld_token",
        app_id="app-id",
        app_secret="app-secret",
        user_access_token="user-access-token",
    )

    assert result["document_id"] == "Doc123"
    assert result["title"] == "Provided Title"
    assert result["url"].endswith("/Doc123")

    assert recorded["app_id"] == "app-id"
    assert recorded["app_secret"] == "app-secret"
    assert recorded["tenant"] is None
    assert recorded["user_access_token"] == "user-access-token"
    assert recorded["domain"] is None
    assert recorded["title"] == "Provided Title"
    assert recorded["folder"] == "fld_token"
    assert recorded["document_option"] is request_option
    assert recorded["create_document_called"] is True

    block_calls = recorded["block_calls"]
    assert len(block_calls) == 1

    batched_call = block_calls[0]
    assert batched_call["block_id"] == "Doc123"
    assert batched_call["index"] == -1
    assert batched_call["option"] is request_option
    assert len(batched_call["children"]) == 6

    children = batched_call["children"]

    heading_block = children[0]
    assert heading_block.block_type == 3
    assert heading_block.heading1.elements[0].text_run.content == "Top Heading"

    paragraph_block = children[1]
    assert paragraph_block.block_type == 2
    sentence_elements = paragraph_block.text.elements
    assert [elem.text_run.content for elem in sentence_elements] == [
        "Intro paragraph with ",
        "bold",
        " text.",
    ]
    bold_styles = [
        elem.text_run.text_element_style.bold
        if elem.text_run.text_element_style
        else False
        for elem in sentence_elements
    ]
    assert bold_styles == [False, True, False]

    bullet_blocks = children[2:4]
    assert [block.block_type for block in bullet_blocks] == [2, 2]
    bullet_texts = [
        [elem.text_run.content for elem in block.text.elements]
        for block in bullet_blocks
    ]
    assert bullet_texts == [["• ", "First bullet"], ["• ", "Second bullet"]]

    ordered_blocks = children[4:]
    assert [block.block_type for block in ordered_blocks] == [2, 2]
    ordered_texts = [
        [elem.text_run.content for elem in block.text.elements]
        for block in ordered_blocks
    ]
    assert ordered_texts == [["1. ", "First item"], ["2. ", "Second item"]]


def test_create_summary_document_batches_large_block_insert_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(
        lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: "token-1"
    )

    recorded: dict[str, Any] = {"batch_sizes": [], "batch_indexes": []}

    class FakeDocumentResource:
        def create(self, request, option=None):
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    document=SimpleNamespace(
                        document_id="DocBatch",
                        revision_id=1,
                        title=request.request_body.title,
                    )
                ),
            )

    class FakeBlockResource:
        def create(self, request, option=None):
            recorded["batch_sizes"].append(len(request.request_body.children))
            recorded["batch_indexes"].append(request.request_body.index)
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    children=request.request_body.children,
                    document_revision_id=1,
                    client_token="token",
                ),
            )

    class FakeBuilder:
        def build(self):
            v1 = SimpleNamespace(
                document=FakeDocumentResource(),
                document_block_children=FakeBlockResource(),
            )
            return SimpleNamespace(docx=SimpleNamespace(v1=v1))

    monkeypatch.setattr(
        lark_docs,
        "_ensure_client_builder",
        lambda *args, **kwargs: (FakeBuilder(), None),
    )

    summary = "\n".join(f"- bullet {idx}" for idx in range(1, 121))

    result = create_summary_document(
        summary, title="Title", app_id="app-id", app_secret="app-secret"
    )

    assert result["document_id"] == "DocBatch"
    assert recorded["batch_sizes"] == [50, 50, 20]
    assert recorded["batch_indexes"] == [-1, -1, -1]


def test_create_summary_document_preserves_block_order_across_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(
        lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: "token-1"
    )

    recorded_texts: list[str] = []

    class FakeDocumentResource:
        def create(self, request, option=None):
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    document=SimpleNamespace(
                        document_id="DocOrder",
                        revision_id=1,
                        title=request.request_body.title,
                    )
                ),
            )

    class FakeBlockResource:
        def create(self, request, option=None):
            for block in request.request_body.children:
                text = "".join(
                    element.text_run.content for element in block.text.elements
                )
                recorded_texts.append(text)
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    children=request.request_body.children,
                    document_revision_id=1,
                    client_token="token",
                ),
            )

    class FakeBuilder:
        def build(self):
            v1 = SimpleNamespace(
                document=FakeDocumentResource(),
                document_block_children=FakeBlockResource(),
            )
            return SimpleNamespace(docx=SimpleNamespace(v1=v1))

    monkeypatch.setattr(
        lark_docs,
        "_ensure_client_builder",
        lambda *args, **kwargs: (FakeBuilder(), None),
    )

    summary = "\n".join(f"{idx}. item {idx:03d}" for idx in range(1, 121))

    result = create_summary_document(
        summary, title="Title", app_id="app-id", app_secret="app-secret"
    )

    assert result["document_id"] == "DocOrder"
    expected_texts = [f"{idx}. item {idx:03d}" for idx in range(1, 121)]
    assert recorded_texts == expected_texts


def test_create_summary_document_appends_corrected_transcript_after_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(
        lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: None
    )

    recorded: dict[str, Any] = {}

    class FakeDocumentResource:
        def create(self, request, option=None):
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    document=SimpleNamespace(
                        document_id="DocCorrected",
                        revision_id=1,
                        title=request.request_body.title,
                    )
                ),
            )

    class FakeBlockResource:
        def create(self, request, option=None):
            recorded["children"] = request.request_body.children
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    children=request.request_body.children,
                    document_revision_id=1,
                    client_token="token",
                ),
            )

    class FakeBuilder:
        def build(self):
            v1 = SimpleNamespace(
                document=FakeDocumentResource(),
                document_block_children=FakeBlockResource(),
            )
            return SimpleNamespace(docx=SimpleNamespace(v1=v1))

    monkeypatch.setattr(
        lark_docs,
        "_ensure_client_builder",
        lambda *args, **kwargs: (FakeBuilder(), None),
    )

    create_summary_document(
        "# Summary",
        title="Title",
        corrected_transcript="a\n\nb",
        app_id="app-id",
        app_secret="app-secret",
    )

    children = recorded["children"]
    assert children[0].heading1.elements[0].text_run.content == "Summary"
    assert children[1].heading2.elements[0].text_run.content == "Corrected Transcript"
    assert children[2].text.elements[0].text_run.content == "a"
    assert children[3].text.elements[0].text_run.content == "b"


def test_create_summary_document_sends_notification_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    recorded: dict[str, Any] = {}

    monkeypatch.setattr(
        lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: "token-123"
    )

    class FakeDocumentResource:
        def create(self, request, option=None):
            recorded["title"] = request.request_body.title
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    document=SimpleNamespace(
                        document_id="DocSuccess",
                        revision_id=1,
                        title=request.request_body.title,
                    )
                ),
            )

    class FakeBlockResource:
        def create(self, request, option=None):
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    children=[], document_revision_id=1, client_token="token"
                ),
            )

    class FakeDocxV1:
        def __init__(self):
            self.document = FakeDocumentResource()
            self.document_block_children = FakeBlockResource()

    class FakeDocxService:
        def __init__(self):
            self.v1 = FakeDocxV1()

    class FakeClient:
        def __init__(self):
            self.docx = FakeDocxService()

    class FakeBuilder:
        def build(self):
            return FakeClient()

    monkeypatch.setattr(
        lark_docs,
        "_ensure_client_builder",
        lambda *args, **kwargs: (FakeBuilder(), None),
    )

    messages: list[tuple[str, str, str]] = []

    def fake_send(access_token: str, receiver_id: str, message: str) -> None:
        messages.append((access_token, receiver_id, message))

    monkeypatch.setattr(lark_docs, "_send_personal_message", fake_send)

    result = create_summary_document(
        "Summary paragraph.",
        title="Notification Title",
        app_id="app-id",
        app_secret="app-secret",
        message_receiver_id="user-999",
    )

    assert result["document_id"] == "DocSuccess"
    assert messages == [
        (
            "token-123",
            "user-999",
            "Lark document created successfully.\nDocument ID: DocSuccess\nTitle: Notification Title\nURL: https://open.feishu.cn/docx/DocSuccess",
        )
    ]


def test_create_summary_document_reports_error_via_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(
        lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: "token-456"
    )

    class FakeDocumentResource:
        def create(self, request, option=None):
            return SimpleNamespace(code=1, msg="permission denied", data=None)

    class FakeDocxV1:
        def __init__(self):
            self.document = FakeDocumentResource()
            self.document_block_children = None

    class FakeDocxService:
        def __init__(self):
            self.v1 = FakeDocxV1()

    class FakeClient:
        def __init__(self):
            self.docx = FakeDocxService()

    class FakeBuilder:
        def build(self):
            return FakeClient()

    monkeypatch.setattr(
        lark_docs,
        "_ensure_client_builder",
        lambda *args, **kwargs: (FakeBuilder(), None),
    )

    messages: list[tuple[str, str, str]] = []

    def fake_send(access_token: str, receiver_id: str, message: str) -> None:
        messages.append((access_token, receiver_id, message))

    monkeypatch.setattr(lark_docs, "_send_personal_message", fake_send)

    with pytest.raises(LarkDocError) as excinfo:
        create_summary_document(
            "Summary paragraph.",
            title="Error Title",
            app_id="app-id",
            app_secret="app-secret",
            message_receiver_id="user-abc",
        )

    assert "Failed to create Lark document" in str(excinfo.value)
    assert messages[0][0] == "token-456"
    assert messages[0][1] == "user-abc"
    assert (
        "Failed to create Lark document.\nError: Failed to create Lark document: permission denied"
        in messages[0][2]
    )
    assert "operation=docx.v1.document.create" in messages[0][2]


def test_create_summary_document_requires_non_empty_summary() -> None:
    with pytest.raises(LarkDocError):
        create_summary_document("   ", title="Title", app_id="app", app_secret="secret")


def test_create_summary_document_uses_cached_tenant_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    cache_dir = tmp_path / "cache-dir"
    cache_dir.mkdir()
    cache_file = cache_dir / "tenant_token.json"
    cache_file.write_text(
        json.dumps(
            {
                "token": "cached-token",
                "updated_at": time.time() - 7200 - 10,
                "expires_at": time.time() + 3600,
            }
        )
    )
    monkeypatch.setenv("LARK_TENANT_TOKEN_CACHE_PATH", str(cache_dir))

    monkeypatch.setattr(
        lark_docs.requests,
        "post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected refresh")
        ),
    )

    recorded: dict[str, Any] = {}

    class FakeDocumentResource:
        def create(self, request, option=None):
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    document=SimpleNamespace(
                        document_id="DocCache",
                        revision_id=1,
                        title=request.request_body.title,
                    )
                ),
            )

    class FakeBlockResource:
        def create(self, request, option=None):
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    children=[], document_revision_id=1, client_token="token"
                ),
            )

    class FakeDocxV1:
        def __init__(self):
            self.document = FakeDocumentResource()
            self.document_block_children = FakeBlockResource()

    class FakeDocxService:
        def __init__(self):
            self.v1 = FakeDocxV1()

    class FakeClient:
        def __init__(self):
            self.docx = FakeDocxService()

    class FakeBuilder:
        def build(self):
            return FakeClient()

    def fake_builder(
        app_id: str,
        app_secret: str,
        *,
        tenant_access_token=None,
        user_access_token=None,
        domain=None,
    ):
        recorded["tenant"] = tenant_access_token
        return FakeBuilder(), None

    monkeypatch.setattr(lark_docs, "_ensure_client_builder", fake_builder)

    result = create_summary_document(
        "Content.", title="Token Title", app_id="app-id", app_secret="app-secret"
    )

    assert result["document_id"] == "DocCache"
    assert recorded["tenant"] == "cached-token"


def test_create_summary_document_refreshes_expired_tenant_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    cache_dir = tmp_path / "cache-dir"
    cache_dir.mkdir()
    cache_file = cache_dir / "tenant_token.json"
    cache_file.write_text(
        json.dumps({"token": "old-token", "updated_at": time.time() - 7200 - 10})
    )
    monkeypatch.setenv("LARK_TENANT_TOKEN_CACHE_PATH", str(cache_dir))

    post_calls: dict[str, Any] = {}

    def fake_post(url, headers=None, json=None, timeout=10):
        post_calls["url"] = url
        post_calls["payload"] = json
        post_calls["called_at"] = time.time()
        return SimpleNamespace(
            status_code=200,
            ok=True,
            json=lambda: {
                "code": 0,
                "msg": "ok",
                "tenant_access_token": "new-token",
                "expire": 7200,
            },
        )

    monkeypatch.setattr(lark_docs.requests, "post", fake_post)

    recorded: dict[str, Any] = {}

    class FakeDocumentResource:
        def create(self, request, option=None):
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    document=SimpleNamespace(
                        document_id="DocRefreshed",
                        revision_id=1,
                        title=request.request_body.title,
                    )
                ),
            )

    class FakeBlockResource:
        def create(self, request, option=None):
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    children=[], document_revision_id=1, client_token="token"
                ),
            )

    class FakeDocxV1:
        def __init__(self):
            self.document = FakeDocumentResource()
            self.document_block_children = FakeBlockResource()

    class FakeDocxService:
        def __init__(self):
            self.v1 = FakeDocxV1()

    class FakeClient:
        def __init__(self):
            self.docx = FakeDocxService()

    class FakeBuilder:
        def build(self):
            return FakeClient()

    def fake_builder(
        app_id: str,
        app_secret: str,
        *,
        tenant_access_token=None,
        user_access_token=None,
        domain=None,
    ):
        recorded["tenant"] = tenant_access_token
        return FakeBuilder(), None

    monkeypatch.setattr(lark_docs, "_ensure_client_builder", fake_builder)

    result = create_summary_document(
        "Content.", title="Refresh Title", app_id="app-id", app_secret="app-secret"
    )

    assert result["document_id"] == "DocRefreshed"
    assert recorded["tenant"] == "new-token"
    assert post_calls["url"].endswith("/tenant_access_token/internal")
    assert post_calls["payload"] == {"app_id": "app-id", "app_secret": "app-secret"}
    cache_payload = json.loads(cache_file.read_text())
    assert cache_payload["token"] == "new-token"
    assert cache_payload["updated_at"] >= post_calls["called_at"] - 1
    assert cache_payload["expires_at"] >= cache_payload["updated_at"] + 7199


def test_create_summary_document_uses_legacy_cached_tenant_token_with_refresh_interval(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    cache_file = tmp_path / "tenant_token.json"
    cache_file.write_text(
        json.dumps({"token": "legacy-cached-token", "updated_at": time.time() - 60})
    )
    monkeypatch.setenv("LARK_TENANT_TOKEN_CACHE_PATH", str(cache_file))

    monkeypatch.setattr(
        lark_docs.requests,
        "post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected refresh")
        ),
    )

    recorded: dict[str, Any] = {}

    class FakeDocumentResource:
        def create(self, request, option=None):
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    document=SimpleNamespace(
                        document_id="DocLegacyCache",
                        revision_id=1,
                        title=request.request_body.title,
                    )
                ),
            )

    class FakeBlockResource:
        def create(self, request, option=None):
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    children=[], document_revision_id=1, client_token="token"
                ),
            )

    class FakeDocxV1:
        def __init__(self):
            self.document = FakeDocumentResource()
            self.document_block_children = FakeBlockResource()

    class FakeDocxService:
        def __init__(self):
            self.v1 = FakeDocxV1()

    class FakeClient:
        def __init__(self):
            self.docx = FakeDocxService()

    class FakeBuilder:
        def build(self):
            return FakeClient()

    def fake_builder(
        app_id: str,
        app_secret: str,
        *,
        tenant_access_token=None,
        user_access_token=None,
        domain=None,
    ):
        recorded["tenant"] = tenant_access_token
        return FakeBuilder(), None

    monkeypatch.setattr(lark_docs, "_ensure_client_builder", fake_builder)

    result = create_summary_document(
        "Content.",
        title="Legacy Refresh Interval Title",
        app_id="app-id",
        app_secret="app-secret",
    )

    assert result["document_id"] == "DocLegacyCache"
    assert recorded["tenant"] == "legacy-cached-token"


def _fake_builder_for_response(response: object):
    class FakeBuilder:
        def build(self):
            doc = SimpleNamespace(create=lambda request, option=None: response)
            v1 = SimpleNamespace(document=doc, document_block_children=None)
            return SimpleNamespace(docx=SimpleNamespace(v1=v1))

    return FakeBuilder()


def test_create_summary_document_error_includes_log_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(
        lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: "token-1"
    )
    response = SimpleNamespace(
        code=999, msg="internal error", get_log_id=lambda: "req-abc123logid", data=None
    )
    monkeypatch.setattr(
        lark_docs,
        "_ensure_client_builder",
        lambda *args, **kwargs: (_fake_builder_for_response(response), None),
    )

    with pytest.raises(LarkDocError) as excinfo:
        create_summary_document(
            "Summary", title="Title", app_id="app-id", app_secret="app-secret"
        )

    error_msg = str(excinfo.value)
    assert "operation=docx.v1.document.create" in error_msg
    assert "code=999" in error_msg
    assert "log_id=req-abc123logid" in error_msg


def test_create_summary_document_error_includes_truncated_redacted_raw_excerpt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(
        lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: "token-1"
    )
    raw = (
        b'{"tenant_access_token":"tenant-secret","access_token":"user-secret","app_secret":"super-secret","payload":"'
        + (b"x" * 12000)
        + b'"}'
    )
    response = SimpleNamespace(
        code=888, msg="server error", raw=SimpleNamespace(content=raw), data=None
    )
    monkeypatch.setattr(
        lark_docs,
        "_ensure_client_builder",
        lambda *args, **kwargs: (_fake_builder_for_response(response), None),
    )

    with pytest.raises(LarkDocError) as excinfo:
        create_summary_document(
            "Summary", title="Title", app_id="app-id", app_secret="app-secret"
        )

    error_msg = str(excinfo.value)
    assert "raw_excerpt=" in error_msg
    assert "...[truncated]" in error_msg
    assert "tenant-secret" not in error_msg
    assert "user-secret" not in error_msg
    assert "super-secret" not in error_msg


def test_create_summary_document_block_insert_error_includes_block_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(
        lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: "token-1"
    )

    class FakeDocumentResource:
        def create(self, request, option=None):
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    document=SimpleNamespace(
                        document_id="DocBlockErr",
                        revision_id=1,
                        title=request.request_body.title,
                    )
                ),
            )

    class FakeBlockResource:
        def create(self, request, option=None):
            return SimpleNamespace(code=1, msg="insert failed", data=None)

    class FakeBuilder:
        def build(self):
            v1 = SimpleNamespace(
                document=FakeDocumentResource(),
                document_block_children=FakeBlockResource(),
            )
            return SimpleNamespace(docx=SimpleNamespace(v1=v1))

    monkeypatch.setattr(
        lark_docs,
        "_ensure_client_builder",
        lambda *args, **kwargs: (FakeBuilder(), None),
    )

    with pytest.raises(LarkDocError) as excinfo:
        create_summary_document(
            "# Heading\n\nParagraph",
            title="Title",
            app_id="app-id",
            app_secret="app-secret",
        )

    assert "operation=docx.v1.document_block_children.create" in str(excinfo.value)


def test_block_insert_retries_on_transient_frequency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(
        lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: "token-1"
    )

    recorded: dict[str, Any] = {"block_create_calls": 0}
    sleeps: list[float] = []

    def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(lark_docs.time, "sleep", fake_sleep)
    monkeypatch.setattr(lark_docs.random, "uniform", lambda a, b: a)

    class FakeDocumentResource:
        def create(self, request, option=None):
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    document=SimpleNamespace(
                        document_id="DocRetry",
                        revision_id=1,
                        title=request.request_body.title,
                    )
                ),
            )

    class FakeBlockResource:
        def create(self, request, option=None):
            recorded["block_create_calls"] += 1
            if recorded["block_create_calls"] < 3:
                return SimpleNamespace(code=99991400, msg="frequency limit", data=None)
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    children=[], document_revision_id=1, client_token="token"
                ),
            )

    class FakeBuilder:
        def build(self):
            v1 = SimpleNamespace(
                document=FakeDocumentResource(),
                document_block_children=FakeBlockResource(),
            )
            return SimpleNamespace(docx=SimpleNamespace(v1=v1))

    monkeypatch.setattr(
        lark_docs,
        "_ensure_client_builder",
        lambda *args, **kwargs: (FakeBuilder(), None),
    )

    result = create_summary_document(
        "Summary paragraph.", title="Title", app_id="app-id", app_secret="app-secret"
    )

    assert result["document_id"] == "DocRetry"
    assert recorded["block_create_calls"] == 3
    assert len(sleeps) == 2


def test_create_document_refreshes_token_once_on_invalid_tenant_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_asr_summary import lark_docs  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(
        lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: "stale-token"
    )
    monkeypatch.setattr(
        lark_docs, "_store_cached_tenant_token", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(lark_docs.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lark_docs.random, "uniform", lambda a, b: a)

    recorded: dict[str, Any] = {
        "create_calls": 0,
        "token_refresh_calls": 0,
        "builder_tokens": [],
    }

    def fake_request_token(app_id: str, app_secret: str) -> tuple[str, float, float]:
        recorded["token_refresh_calls"] += 1
        updated_at = time.time()
        return "fresh-token", updated_at, updated_at + 7200

    monkeypatch.setattr(lark_docs, "_request_tenant_access_token", fake_request_token)

    class FakeDocumentResource:
        def create(self, request, option=None):
            recorded["create_calls"] += 1
            if recorded["create_calls"] == 1:
                return SimpleNamespace(
                    code=99991663, msg="invalid tenant token", data=None
                )
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    document=SimpleNamespace(
                        document_id="DocFresh",
                        revision_id=1,
                        title=request.request_body.title,
                    )
                ),
            )

    class FakeBlockResource:
        def create(self, request, option=None):
            return SimpleNamespace(
                code=0,
                msg="success",
                data=SimpleNamespace(
                    children=[], document_revision_id=1, client_token="token"
                ),
            )

    class FakeBuilder:
        def __init__(self, tenant_access_token: str | None):
            self._tenant_access_token = tenant_access_token

        def build(self):
            v1 = SimpleNamespace(
                document=FakeDocumentResource(),
                document_block_children=FakeBlockResource(),
            )
            return SimpleNamespace(
                docx=SimpleNamespace(v1=v1),
                tenant_access_token=self._tenant_access_token,
            )

    def fake_builder(
        app_id: str,
        app_secret: str,
        *,
        tenant_access_token=None,
        user_access_token=None,
        domain=None,
    ):
        recorded["builder_tokens"].append(tenant_access_token)
        return FakeBuilder(tenant_access_token), None

    monkeypatch.setattr(lark_docs, "_ensure_client_builder", fake_builder)

    result = create_summary_document(
        "Summary paragraph.", title="Title", app_id="app-id", app_secret="app-secret"
    )

    assert result["document_id"] == "DocFresh"
    assert recorded["create_calls"] == 2
    assert recorded["token_refresh_calls"] == 1
    assert recorded["builder_tokens"] == ["stale-token", "fresh-token"]
