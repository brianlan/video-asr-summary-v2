from __future__ import annotations

import os
import re
from typing import List, Optional

from lark_oapi.client import Client, ClientBuilder, LogLevel, RequestOption
from lark_oapi.api.docx.v1 import (
	Block,
	BlockBuilder,
	CreateDocumentBlockChildrenRequest,
	CreateDocumentBlockChildrenRequestBody,
	CreateDocumentRequest,
	CreateDocumentRequestBody,
	TextBuilder,
	TextElementBuilder,
	TextRunBuilder,
)


class LarkDocError(RuntimeError):
	"""Raised when a request to the Lark OpenAPI fails."""


def _ensure_client_builder(app_id: str, app_secret: str, *, tenant_access_token: str | None, user_access_token: str | None) -> tuple[ClientBuilder, Optional[RequestOption]]:
	builder = Client.builder().app_id(app_id).app_secret(app_secret).log_level(LogLevel.ERROR)
	request_option: Optional[RequestOption] = None
	if user_access_token:
		builder.enable_set_token(True)
		request_option = RequestOption.builder().user_access_token(user_access_token).build()
	elif tenant_access_token:
		builder.enable_set_token(True)
		request_option = RequestOption.builder().tenant_access_token(tenant_access_token).build()
	return builder, request_option


def _summary_to_blocks(summary: str) -> List[Block]:
	paragraphs = [
		chunk.strip()
		for chunk in re.split(r"\n\s*\n", summary.strip())
		if chunk.strip()
	]
	if not paragraphs:
		return []

	blocks = []
	for paragraph in paragraphs:
		text = TextBuilder().elements(
			[
				TextElementBuilder()
				.text_run(TextRunBuilder().content(paragraph).build())
				.build()
			]
		).build()
		block = BlockBuilder().block_type(2).text(text).build()
		blocks.append(block)
	return blocks


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

	if not app_id or not app_secret:
		raise LarkDocError("Lark app credentials are missing. Provide app_id and app_secret or set LARK_APP_ID/LARK_APP_SECRET.")

	builder, request_option = _ensure_client_builder(app_id, app_secret, tenant_access_token=tenant_access_token, user_access_token=user_access_token)
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

	blocks = _summary_to_blocks(content)
	if blocks:
		block_body = CreateDocumentBlockChildrenRequestBody.builder().index(0).children(blocks).build()
		block_req = CreateDocumentBlockChildrenRequest.builder().document_id(document_id).block_id(document_id).request_body(block_body).build()
		block_resp = client.docx.v1.document_block_children.create(block_req, request_option)
		if block_resp.code != 0:
			raise LarkDocError(f"Failed to insert summary into Lark document: {block_resp.msg}")
	
	return {
		"document_id": document_id,
		"title": title,
		"url": f"https://{user_subdomain}.feishu.cn/docx/{document_id}",
	}


__all__ = ["LarkDocError", "create_summary_document", "derive_lark_title"]
