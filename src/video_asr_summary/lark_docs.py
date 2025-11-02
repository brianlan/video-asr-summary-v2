from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from lark_oapi.client import Client, ClientBuilder, LogLevel, RequestOption
from lark_oapi.api.docx.v1 import (
	Block,
	BlockBuilder,
	CreateDocumentBlockChildrenRequest,
	CreateDocumentBlockChildrenRequestBody,
	CreateDocumentRequest,
	CreateDocumentRequestBody,
	Text,
	TextBuilder,
	TextElementBuilder,
	TextElementStyleBuilder,
	TextRunBuilder,
)


class LarkDocError(RuntimeError):
	"""Raised when a request to the Lark OpenAPI fails."""


def _ensure_client_builder(
	app_id: str,
	app_secret: str,
	*,
	tenant_access_token: str | None,
	user_access_token: str | None,
	domain: str | None,
) -> tuple[ClientBuilder, Optional[RequestOption]]:
	builder = Client.builder().app_id(app_id).app_secret(app_secret).log_level(LogLevel.ERROR)
	if domain:
		builder.domain(domain)
	request_option: Optional[RequestOption] = None
	if user_access_token:
		builder.enable_set_token(True)
		request_option = RequestOption.builder().user_access_token(user_access_token).build()
	elif tenant_access_token:
		builder.enable_set_token(True)
		request_option = RequestOption.builder().tenant_access_token(tenant_access_token).build()
	return builder, request_option


@dataclass(frozen=True)
class _InlineSpan:
	text: str
	bold: bool = False


@dataclass(frozen=True)
class _DocumentElement:
	kind: str  # heading, paragraph, unordered_list, ordered_list
	spans: Sequence[_InlineSpan] | None = None
	items: Sequence[Sequence[_InlineSpan]] | None = None
	level: int | None = None


_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_PATTERN = re.compile(r"^\s*[-*+]\s+(.*)$")
_ORDERED_PATTERN = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")

_PARAGRAPH_BLOCK_TYPE = 2
# Lark docx block types; see https://github.com/larksuite/oapi-sdk-python/issues/57 for enum mapping.
_HEADING_BLOCK_TYPES = {level: level + 2 for level in range(1, 10)}


def _normalize_whitespace(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def _parse_inline_spans(text: str) -> List[_InlineSpan]:
	normalized = _normalize_whitespace(text)
	if not normalized:
		return []

	spans: List[_InlineSpan] = []
	cursor = 0
	for match in _BOLD_PATTERN.finditer(normalized):
		start, end = match.span()
		if start > cursor:
			prefix = normalized[cursor:start]
			if prefix:
				spans.append(_InlineSpan(prefix))
		bold_text = match.group(1)
		if bold_text:
			spans.append(_InlineSpan(bold_text, bold=True))
		cursor = end

	if cursor < len(normalized):
		tail = normalized[cursor:]
		if tail:
			spans.append(_InlineSpan(tail))

	return spans or [_InlineSpan(normalized)]


def _parse_summary_elements(summary: str) -> List[_DocumentElement]:
	elements: List[_DocumentElement] = []
	paragraph_lines: List[str] = []
	current_list_type: str | None = None
	current_list_items: List[str] = []

	def flush_paragraph() -> None:
		nonlocal paragraph_lines
		content = _normalize_whitespace(" ".join(paragraph_lines))
		if content:
			elements.append(_DocumentElement(kind="paragraph", spans=_parse_inline_spans(content)))
		paragraph_lines = []

	def flush_list() -> None:
		nonlocal current_list_type, current_list_items
		if current_list_type and current_list_items:
			list_spans = [_parse_inline_spans(item) for item in current_list_items if _normalize_whitespace(item)]
			if list_spans:
				kind = "ordered_list" if current_list_type == "ol" else "unordered_list"
				elements.append(_DocumentElement(kind=kind, items=list_spans))
		current_list_type = None
		current_list_items = []

	for raw_line in summary.splitlines():
		line = raw_line.rstrip()
		if not line.strip():
			if paragraph_lines:
				flush_paragraph()
			if current_list_type:
				flush_list()
			continue

		heading_match = _HEADING_PATTERN.match(line)
		if heading_match:
			if paragraph_lines:
				flush_paragraph()
			if current_list_type:
				flush_list()
			hashes, content = heading_match.groups()
			level = min(len(hashes), 3)
			content_spans = _parse_inline_spans(content)
			if content_spans:
				elements.append(_DocumentElement(kind="heading", level=level, spans=content_spans))
			continue

		bullet_match = _BULLET_PATTERN.match(line)
		if bullet_match:
			if paragraph_lines:
				flush_paragraph()
			if current_list_type and current_list_type != "ul":
				flush_list()
			current_list_type = "ul"
			current_list_items.append(bullet_match.group(1))
			continue

		ordered_match = _ORDERED_PATTERN.match(line)
		if ordered_match:
			if paragraph_lines:
				flush_paragraph()
			if current_list_type and current_list_type != "ol":
				flush_list()
			current_list_type = "ol"
			current_list_items.append(ordered_match.group(1))
			continue

		if current_list_type:
			flush_list()

		paragraph_lines.append(line.strip())

	if paragraph_lines:
		flush_paragraph()
	if current_list_type:
		flush_list()

	return elements


def _build_text(spans: Sequence[_InlineSpan]) -> Text:
	text_elements = []
	for span in spans:
		if not span.text:
			continue
		run = TextRunBuilder().content(span.text)
		if span.bold:
			run.text_element_style(TextElementStyleBuilder().bold(True).build())
		text_elements.append(TextElementBuilder().text_run(run.build()).build())
	if not text_elements:
		text_elements.append(TextElementBuilder().text_run(TextRunBuilder().content("").build()).build())
	return TextBuilder().elements(text_elements).build()


def _blocks_for_element(element: _DocumentElement) -> List[Block]:
	if element.kind == "heading" and element.spans:
		level = element.level or 1
		text = _build_text(element.spans)
		block_type = _HEADING_BLOCK_TYPES.get(level, _PARAGRAPH_BLOCK_TYPE)
		builder = BlockBuilder().block_type(block_type)
		if level == 1:
			builder.heading1(text)
		elif level == 2:
			builder.heading2(text)
		else:
			builder.heading3(text)
		return [builder.build()]

	if element.kind == "paragraph" and element.spans:
		text = _build_text(element.spans)
		return [BlockBuilder().block_type(_PARAGRAPH_BLOCK_TYPE).text(text).build()]

 
	if element.kind in {"unordered_list", "ordered_list"}:
		return []

	return []



def _append_elements_to_document(
	block_resource,
	document_id: str,
	elements: Iterable[_DocumentElement],
	request_option: Optional[RequestOption],
) -> None:
	index = 0

	def create_blocks(parent_id: str, insert_index: int, blocks: List[Block], *, ctx: str) -> List[Block]:
		body = CreateDocumentBlockChildrenRequestBody.builder().index(insert_index).children(blocks).build()
		request = (
			CreateDocumentBlockChildrenRequest.builder()
			.document_id(document_id)
			.block_id(parent_id)
			.request_body(body)
			.build()
		)
		response = block_resource.create(request, request_option)
		if response.code != 0:
			raise LarkDocError(
				f"Failed to insert summary into Lark document: {response.msg} (context: {ctx})"
			)
		children = getattr(getattr(response, "data", None), "children", None) or []
		return children

	for element in elements:
		from loguru import logger
		logger.debug(f"Processing element: {element}")
		if element.kind == "unordered_list" and element.items:
			for spans in element.items:
				if not spans:
					continue
				prefixed_spans = [_InlineSpan("• ")] + list(spans)
				text = _build_text(prefixed_spans)
				block = BlockBuilder().block_type(_PARAGRAPH_BLOCK_TYPE).text(text).build()
				create_blocks(document_id, index, [block], ctx="unordered-list:paragraph")
				index += 1
			continue

		if element.kind == "ordered_list" and element.items:
			for order_index, spans in enumerate(element.items, start=1):
				if not spans:
					continue
				prefix = _InlineSpan(f"{order_index}. ")
				prefixed_spans = [prefix] + list(spans)
				text = _build_text(prefixed_spans)
				block = BlockBuilder().block_type(_PARAGRAPH_BLOCK_TYPE).text(text).build()
				create_blocks(document_id, index, [block], ctx="ordered-list:paragraph")
				index += 1
			continue

		blocks = _blocks_for_element(element)
		if not blocks:
			continue
		create_blocks(document_id, index, blocks, ctx=f"{element.kind}")
		index += len(blocks)


def derive_lark_title(summary: str, fallback: str) -> str:
	for line in summary.splitlines():
		candidate = line.strip()
		if not candidate:
			continue
		if candidate.startswith("#"):
			candidate = candidate.lstrip("#").strip()
		candidate = candidate.strip()
		if candidate:
			return candidate[:200]
	return fallback


def create_summary_document(
	summary: str,
	*,
	title: str,
	folder_token: str | None = None,
	app_id: str | None = None,
	app_secret: str | None = None,
	tenant_access_token: str | None = None,
	user_access_token: str | None = None,
	user_subdomain: str | None = None,
	api_domain: str | None = None,
) -> dict[str, str]:
	content = summary.strip()
	if not content:
		raise LarkDocError("Summary content is empty; cannot create a Lark document.")

	app_id = app_id or os.getenv("LARK_APP_ID")
	app_secret = app_secret or os.getenv("LARK_APP_SECRET")
	folder_token = folder_token or os.getenv("LARK_FOLDER_TOKEN")
	tenant_access_token = tenant_access_token or os.getenv("LARK_TENANT_ACCESS_TOKEN")
	user_access_token = user_access_token or os.getenv("LARK_USER_ACCESS_TOKEN")
	user_subdomain = user_subdomain or os.getenv("LARK_USER_SUBDOMAIN")
	api_domain = api_domain or os.getenv("LARK_API_DOMAIN")

	if not app_id or not app_secret:
		raise LarkDocError("Lark app credentials are missing. Provide app_id and app_secret or set LARK_APP_ID/LARK_APP_SECRET.")

	builder, request_option = _ensure_client_builder(
		app_id,
		app_secret,
		tenant_access_token=tenant_access_token,
		user_access_token=user_access_token,
		domain=api_domain,
	)
	client = builder.build()

	body_builder = CreateDocumentRequestBody.builder().title(title)
	if folder_token:
		body_builder.folder_token(folder_token)

	create_req = CreateDocumentRequest.builder().request_body(body_builder.build()).build()
	create_resp = client.docx.v1.document.create(create_req, request_option)
	if create_resp.code != 0 or not create_resp.data or not create_resp.data.document:
		raise LarkDocError(f"Failed to create Lark document: {create_resp.msg}")

	document = create_resp.data.document
	document_id = document.document_id

	elements = _parse_summary_elements(content)
	if elements:
		_append_elements_to_document(
			client.docx.v1.document_block_children,
			document_id,
			elements,
			request_option,
		)

	base_host = f"{user_subdomain}.feishu.cn" if user_subdomain else "open.feishu.cn"

	return {
		"document_id": document_id,
		"title": title,
		"url": f"https://{base_host}/docx/{document_id}",
	}


__all__ = ["LarkDocError", "create_summary_document", "derive_lark_title"]
