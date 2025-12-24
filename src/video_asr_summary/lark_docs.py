from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
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

import requests


class LarkDocError(RuntimeError):
	"""Raised when a request to the Lark OpenAPI fails."""


_TOKEN_REQUEST_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
_TOKEN_REFRESH_INTERVAL = 2 * 60 * 60  # seconds
_TOKEN_CACHE_ENV = "LARK_TENANT_TOKEN_CACHE_PATH"


def _tenant_token_cache_path() -> Path:
	override = os.getenv(_TOKEN_CACHE_ENV)
	if override:
		return Path(override)

	base_dir = os.getenv("XDG_CACHE_HOME")
	if base_dir:
		base = Path(base_dir)
	else:
		base = Path.home() / ".cache"
	return base / "video_asr_summary" / "tenant_token.json"


def _load_cached_tenant_token(cache_path: Path) -> tuple[str, float] | None:
	try:
		data = json.loads(cache_path.read_text())
	except FileNotFoundError:
		return None
	except (OSError, json.JSONDecodeError):
		return None

	token = data.get("token")
	updated_at = data.get("updated_at")
	if not isinstance(token, str) or not token:
		return None
	if not isinstance(updated_at, (int, float)):
		return None
	return token, float(updated_at)


def _store_cached_tenant_token(cache_path: Path, token: str, updated_at: float) -> None:
	try:
		cache_path.parent.mkdir(parents=True, exist_ok=True)
		cache_path.write_text(json.dumps({"token": token, "updated_at": updated_at}))
	except OSError:
		pass


def _request_tenant_access_token(app_id: str, app_secret: str) -> tuple[str, float]:
	headers = {"Content-Type": "application/json"}
	payload = {"app_id": app_id, "app_secret": app_secret}

	try:
		response = requests.post(_TOKEN_REQUEST_URL, headers=headers, json=payload, timeout=10)
	except requests.RequestException as exc:  # pragma: no cover - network issues handled at runtime
		raise LarkDocError(f"Failed to obtain tenant access token: {exc}") from exc

	try:
		data = response.json()
	except ValueError as exc:
		raise LarkDocError("Failed to parse tenant access token response as JSON.") from exc

	if data.get("code") != 0 or not data.get("tenant_access_token"):
		message = data.get("msg") or "unknown error"
		raise LarkDocError(f"Failed to obtain tenant access token (code={data.get('code')}): {message}")

	return data["tenant_access_token"], time.time()


def _resolve_tenant_access_token(
	app_id: str,
	app_secret: str,
	tenant_access_token: str | None,
) -> str | None:
	cache_path = _tenant_token_cache_path()
	now = time.time()

	# If a token was explicitly provided, check if it's still valid
	if tenant_access_token:
		# Try to load cached token info to check expiration
		cached = _load_cached_tenant_token(cache_path)
		if cached:
			cached_token, updated_at = cached
			# If the cached token matches the provided token and is still valid, use it
			if cached_token == tenant_access_token and (now - updated_at < _TOKEN_REFRESH_INTERVAL):
				return tenant_access_token
		# Either token not in cache, mismatched, or expired - get a fresh one
		token, updated_at = _request_tenant_access_token(app_id, app_secret)
		_store_cached_tenant_token(cache_path, token, updated_at)
		return token

	# No token provided, use cache or request new one
	cached = _load_cached_tenant_token(cache_path)
	if cached:
		token, updated_at = cached
		if now - updated_at < _TOKEN_REFRESH_INTERVAL:
			return token

	token, updated_at = _request_tenant_access_token(app_id, app_secret)
	_store_cached_tenant_token(cache_path, token, updated_at)
	return token


def _send_personal_message(access_token: str, user_id: str, message: str) -> None:
	url = "https://open.feishu.cn/open-apis/im/v1/messages"
	headers = {
		"Content-Type": "application/json",
		"Authorization": f"Bearer {access_token}",
	}
	params = {"receive_id_type": "user_id"}
	body = {
		"receive_id": user_id,
		"msg_type": "text",
		"content": json.dumps({"text": message}),
	}
	requests.post(url, headers=headers, params=params, json=body, timeout=10)


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

	if element.kind == "unordered_list" and element.items:
		blocks: List[Block] = []
		for spans in element.items:
			if not spans:
				continue
			prefixed_spans = [_InlineSpan("• ")] + list(spans)
			text = _build_text(prefixed_spans)
			blocks.append(BlockBuilder().block_type(_PARAGRAPH_BLOCK_TYPE).text(text).build())
		return blocks

	if element.kind == "ordered_list" and element.items:
		blocks = []
		for order_index, spans in enumerate(element.items, start=1):
			if not spans:
				continue
			prefix = _InlineSpan(f"{order_index}. ")
			prefixed_spans = [prefix] + list(spans)
			text = _build_text(prefixed_spans)
			blocks.append(BlockBuilder().block_type(_PARAGRAPH_BLOCK_TYPE).text(text).build())
		return blocks

	return []



def _append_elements_to_document(
	block_resource,
	document_id: str,
	elements: Iterable[_DocumentElement],
	request_option: Optional[RequestOption],
) -> None:
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

	blocks_to_insert: List[Block] = []

	for element in elements:
		blocks = _blocks_for_element(element)
		if not blocks:
			continue
		blocks_to_insert.extend(blocks)

	if not blocks_to_insert:
		return

	create_blocks(document_id, 0, blocks_to_insert, ctx="batched-insert")


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
	message_receiver_id: str | None = None,
) -> dict[str, str]:
	content = summary.strip()
	if not content:
		raise LarkDocError("Summary content is empty; cannot create a Lark document.")

	app_id = app_id or os.getenv("LARK_APP_ID")
	app_secret = app_secret or os.getenv("LARK_APP_SECRET")
	folder_token = folder_token or os.getenv("LARK_FOLDER_TOKEN")
	provided_tenant_access_token = tenant_access_token or os.getenv("LARK_TENANT_ACCESS_TOKEN")
	user_access_token = user_access_token or os.getenv("LARK_USER_ACCESS_TOKEN")
	user_subdomain = user_subdomain or os.getenv("LARK_USER_SUBDOMAIN")
	api_domain = api_domain or os.getenv("LARK_API_DOMAIN")
	message_receiver_id = message_receiver_id or os.getenv("LARK_MESSAGE_RECEIVER_ID")

	if not app_id or not app_secret:
		raise LarkDocError("Lark app credentials are missing. Provide app_id and app_secret or set LARK_APP_ID/LARK_APP_SECRET.")

	tenant_access_token = _resolve_tenant_access_token(app_id, app_secret, provided_tenant_access_token)

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

	result: dict[str, str] | None = None
	error: LarkDocError | None = None
	attempted = False

	try:
		attempted = True
		create_resp = client.docx.v1.document.create(create_req, request_option)
		if create_resp.code != 0 or not create_resp.data or not create_resp.data.document:
			# If user access token failed and we have app credentials, try with tenant token as fallback
			if user_access_token and not provided_tenant_access_token and tenant_access_token:
				print(f"[DEBUG] User token failed (code={create_resp.code}, msg={create_resp.msg}), retrying with tenant token", file=sys.stderr)
				builder_fallback, request_option_fallback = _ensure_client_builder(
					app_id,
					app_secret,
					tenant_access_token=tenant_access_token,
					user_access_token=None,  # Don't use user token this time
					domain=api_domain,
				)
				client_fallback = builder_fallback.build()
				create_resp = client_fallback.docx.v1.document.create(create_req, request_option_fallback)
				if create_resp.code != 0 or not create_resp.data or not create_resp.data.document:
					raise LarkDocError(f"Failed to create Lark document (tenant token fallback also failed): {create_resp.msg}")
				else:
					print(f"[DEBUG] Tenant token fallback succeeded", file=sys.stderr)
					# Update client and request_option for subsequent operations
					client = client_fallback
					request_option = request_option_fallback
			else:
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
		result = {
			"document_id": document_id,
			"title": title,
			"url": f"https://{base_host}/docx/{document_id}",
		}
	except LarkDocError as exc:
		error = exc
	except Exception as exc:  # pragma: no cover - unexpected errors still need conversion
		error = LarkDocError(str(exc))

	if tenant_access_token and message_receiver_id and attempted:
		if result is not None:
			message = (
				"Lark document created successfully.\n"
				f"Document ID: {result['document_id']}\n"
				f"Title: {result['title']}\n"
				f"URL: {result['url']}"
			)
		else:
			error_text = str(error) if error is not None else "Unknown error"
			message = f"Failed to create Lark document.\nError: {error_text}"
		try:
			_send_personal_message(tenant_access_token, message_receiver_id, message)
		except Exception:
			pass

	if error is not None:
		raise error

	if result is None:
		raise LarkDocError("Unknown error occurred while creating Lark document.")

	return result


__all__ = ["LarkDocError", "create_summary_document", "derive_lark_title"]
