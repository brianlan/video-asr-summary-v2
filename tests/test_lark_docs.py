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


def test_create_summary_document_requires_non_empty_summary() -> None:
	with pytest.raises(LarkDocError):
		create_summary_document("   ", title="Title", app_id="app", app_secret="secret")
