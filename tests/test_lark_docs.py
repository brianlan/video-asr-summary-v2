from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from video_asr_summary.lark_docs import (
	LarkDocError,
	create_summary_document,
	derive_lark_title,
)


def test_derive_lark_title_prefers_heading() -> None:
	summary = "# Heading Title\n\nContent line."
	assert derive_lark_title(summary, "Fallback") == "Heading Title"


def test_derive_lark_title_uses_fallback_when_needed() -> None:
	assert derive_lark_title("   \n", "Fallback Title") == "Fallback Title"


def test_create_summary_document_builds_blocks_and_uses_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
	from video_asr_summary import lark_docs

	recorded: dict[str, object] = {}

	monkeypatch.setattr(lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: None)

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
				data=SimpleNamespace(children=created_children, document_revision_id=len(recorded["block_calls"]), client_token="token"),
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

	def fake_builder(app_id: str, app_secret: str, *, tenant_access_token=None, user_access_token=None, domain=None):
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
	assert batched_call["index"] == 0
	assert batched_call["option"] is request_option
	assert len(batched_call["children"]) == 6

	children = batched_call["children"]

	heading_block = children[0]
	assert heading_block.block_type == 3
	assert heading_block.heading1.elements[0].text_run.content == "Top Heading"

	paragraph_block = children[1]
	assert paragraph_block.block_type == 2
	sentence_elements = paragraph_block.text.elements
	assert [elem.text_run.content for elem in sentence_elements] == ["Intro paragraph with ", "bold", " text."]
	bold_styles = [elem.text_run.text_element_style.bold if elem.text_run.text_element_style else False for elem in sentence_elements]
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


def test_create_summary_document_sends_notification_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
	from video_asr_summary import lark_docs

	recorded: dict[str, object] = {}

	monkeypatch.setattr(lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: "token-123")

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
				data=SimpleNamespace(children=[], document_revision_id=1, client_token="token"),
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

	monkeypatch.setattr(lark_docs, "_ensure_client_builder", lambda *args, **kwargs: (FakeBuilder(), None))

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


def test_create_summary_document_reports_error_via_notification(monkeypatch: pytest.MonkeyPatch) -> None:
	from video_asr_summary import lark_docs

	monkeypatch.setattr(lark_docs, "_resolve_tenant_access_token", lambda *args, **kwargs: "token-456")

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

	monkeypatch.setattr(lark_docs, "_ensure_client_builder", lambda *args, **kwargs: (FakeBuilder(), None))

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
	assert messages == [
		(
			"token-456",
			"user-abc",
			"Failed to create Lark document.\nError: Failed to create Lark document: permission denied",
		)
	]


def test_create_summary_document_requires_non_empty_summary() -> None:
	with pytest.raises(LarkDocError):
		create_summary_document("   ", title="Title", app_id="app", app_secret="secret")


def test_create_summary_document_uses_cached_tenant_token(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
	from video_asr_summary import lark_docs

	cache_file = tmp_path / "tenant_token.json"
	cache_file.write_text(json.dumps({"token": "cached-token", "updated_at": time.time()}))
	monkeypatch.setenv("LARK_TENANT_TOKEN_CACHE_PATH", str(cache_file))

	recorded: dict[str, object] = {}

	class FakeDocumentResource:
		def create(self, request, option=None):
			return SimpleNamespace(
				code=0,
				msg="success",
				data=SimpleNamespace(
					document=SimpleNamespace(document_id="DocCache", revision_id=1, title=request.request_body.title)
				),
			)

	class FakeBlockResource:
		def create(self, request, option=None):
			return SimpleNamespace(
				code=0,
				msg="success",
				data=SimpleNamespace(children=[], document_revision_id=1, client_token="token"),
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

	def fake_builder(app_id: str, app_secret: str, *, tenant_access_token=None, user_access_token=None, domain=None):
		recorded["tenant"] = tenant_access_token
		return FakeBuilder(), None

	monkeypatch.setattr(lark_docs, "_ensure_client_builder", fake_builder)

	result = create_summary_document("Content.", title="Token Title", app_id="app-id", app_secret="app-secret")

	assert result["document_id"] == "DocCache"
	assert recorded["tenant"] == "cached-token"


def test_create_summary_document_refreshes_expired_tenant_token(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
	from video_asr_summary import lark_docs

	cache_file = tmp_path / "tenant_token.json"
	cache_file.write_text(json.dumps({"token": "old-token", "updated_at": time.time() - 7200 - 10}))
	monkeypatch.setenv("LARK_TENANT_TOKEN_CACHE_PATH", str(cache_file))

	post_calls: dict[str, object] = {}

	def fake_post(url, headers=None, json=None, timeout=10):
		post_calls["url"] = url
		post_calls["payload"] = json
		post_calls["called_at"] = time.time()
		return SimpleNamespace(
			status_code=200,
			ok=True,
			json=lambda: {"code": 0, "msg": "ok", "tenant_access_token": "new-token", "expire": 7200},
		)

	monkeypatch.setattr(lark_docs.requests, "post", fake_post)

	recorded: dict[str, object] = {}

	class FakeDocumentResource:
		def create(self, request, option=None):
			return SimpleNamespace(
				code=0,
				msg="success",
				data=SimpleNamespace(
					document=SimpleNamespace(document_id="DocRefreshed", revision_id=1, title=request.request_body.title)
				),
			)

	class FakeBlockResource:
		def create(self, request, option=None):
			return SimpleNamespace(
				code=0,
				msg="success",
				data=SimpleNamespace(children=[], document_revision_id=1, client_token="token"),
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

	def fake_builder(app_id: str, app_secret: str, *, tenant_access_token=None, user_access_token=None, domain=None):
		recorded["tenant"] = tenant_access_token
		return FakeBuilder(), None

	monkeypatch.setattr(lark_docs, "_ensure_client_builder", fake_builder)

	result = create_summary_document("Content.", title="Refresh Title", app_id="app-id", app_secret="app-secret")

	assert result["document_id"] == "DocRefreshed"
	assert recorded["tenant"] == "new-token"
	assert post_calls["url"].endswith("/tenant_access_token/internal")
	assert post_calls["payload"] == {"app_id": "app-id", "app_secret": "app-secret"}
	cache_payload = json.loads(cache_file.read_text())
	assert cache_payload["token"] == "new-token"
	assert cache_payload["updated_at"] >= post_calls["called_at"] - 1
