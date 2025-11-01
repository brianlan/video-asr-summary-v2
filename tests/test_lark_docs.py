from __future__ import annotations

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

	class FakeDocumentResource:
		def create(self, request, option=None):
			recorded["title"] = request.request_body.title
			recorded["folder"] = request.request_body.folder_token
			recorded["document_option"] = option
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
			recorded["blocks"] = request.request_body.children
			recorded["index"] = request.request_body.index
			recorded["block_option"] = option
			return SimpleNamespace(code=0, msg="success", data=None)

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

	def fake_builder(app_id: str, app_secret: str, *, tenant_access_token=None, user_access_token=None):
		recorded["app_id"] = app_id
		recorded["app_secret"] = app_secret
		recorded["tenant"] = tenant_access_token
		recorded["user_access_token"] = user_access_token
		return FakeBuilder(), request_option

	monkeypatch.setattr(lark_docs, "_ensure_client_builder", fake_builder)

	result = create_summary_document(
		"First paragraph.\n\nSecond paragraph.",
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
	assert recorded["title"] == "Provided Title"
	assert recorded["folder"] == "fld_token"
	assert recorded["index"] == 0
	assert recorded["document_option"] is request_option
	assert recorded["block_option"] is request_option

	blocks = recorded["blocks"]
	assert len(blocks) == 2
	contents = [block.text.elements[0].text_run.content for block in blocks]
	assert contents == ["First paragraph.", "Second paragraph."]


def test_create_summary_document_requires_non_empty_summary() -> None:
	with pytest.raises(LarkDocError):
		create_summary_document("   ", title="Title", app_id="app", app_secret="secret")
