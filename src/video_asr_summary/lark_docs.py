from __future__ import annotations

import json
import os
import random
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, TypeVar

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


_TOKEN_REQUEST_URL = (
    "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
)
_TOKEN_REFRESH_INTERVAL = 2 * 60 * 60  # seconds
_TOKEN_CACHE_ENV = "LARK_TENANT_TOKEN_CACHE_PATH"
_MAX_RAW_EXCERPT_LEN = 2000
_RETRY_MAX_ATTEMPTS = 5
_RETRY_BASE_DELAY_SECONDS = 1.0
_RETRY_MAX_DELAY_SECONDS = 20.0
_RETRY_JITTER_RATIO = 0.2
MAX_BLOCKS_PER_REQUEST = 50
_CORRECTED_TRANSCRIPT_MAX_BLOCK_CHARS = 1500
_TRANSIENT_SDK_CODES = {99991400, 99991401}
_INVALID_TENANT_TOKEN_CODE = 99991663
_DISABLE_PROXY_ENV = "LARK_DISABLE_PROXY"
_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
_T = TypeVar("_T")


def _is_proxy_disabled() -> bool:
    return os.getenv(_DISABLE_PROXY_ENV, "0") == "1"


def _requests_post_with_optional_no_proxy(url: str, **kwargs: Any):
    if _is_proxy_disabled():
        kwargs["proxies"] = {}
    return requests.post(url, **kwargs)


@contextmanager
def _clear_proxy_env_for_sdk() -> Iterator[None]:
    if not _is_proxy_disabled():
        yield
        return

    original_values = {key: os.environ.get(key) for key in _PROXY_ENV_KEYS}
    for key in _PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_sdk_call(func: Callable[[], _T]) -> _T:
    with _clear_proxy_env_for_sdk():
        return func()


def _redact_token_like_fields(text: str) -> str:
    for key in ("tenant_access_token", "access_token", "app_secret"):
        text = re.sub(rf'("{re.escape(key)}"\s*:\s*")[^"]*(")', r"\1[REDACTED]\2", text)
        text = re.sub(rf"('{re.escape(key)}'\s*:\s*')[^']*(')", r"\1[REDACTED]\2", text)
        text = re.sub(rf"({re.escape(key)}=)[^&\s\"]+", r"\1[REDACTED]", text)
    return text


def _format_lark_error(response: object, operation: str) -> str:
    parts = [f"operation={operation}"]
    code = getattr(response, "code", None)
    msg = getattr(response, "msg", None)
    if code is not None:
        parts.append(f"code={code}")
    if msg is not None:
        parts.append(f"msg={msg}")
    get_log_id = getattr(response, "get_log_id", None)
    if callable(get_log_id):
        try:
            log_id = get_log_id()
        except Exception:
            log_id = None
        if log_id:
            parts.append(f"log_id={log_id}")
    raw = getattr(response, "raw", None)
    content = getattr(raw, "content", None) if raw is not None else None
    if content is not None:
        raw_text = (
            content.decode("utf-8", errors="replace")
            if isinstance(content, bytes)
            else str(content)
        )
        raw_text = _redact_token_like_fields(raw_text)
        if len(raw_text) > _MAX_RAW_EXCERPT_LEN:
            raw_text = raw_text[:_MAX_RAW_EXCERPT_LEN] + "...[truncated]"
        parts.append(f"raw_excerpt={raw_text}")
    return " ".join(parts)


def _is_transient_lark_code_or_msg(code: object, msg: object) -> bool:
    if code in _TRANSIENT_SDK_CODES:
        return True
    message = str(msg or "").lower()
    return "frequency" in message or "too many" in message


def _is_transient_network_exception(exc: Exception) -> bool:
    return isinstance(
        exc, (requests.RequestException, ConnectionError, TimeoutError, OSError)
    )


def _retry_with_backoff(
    operation: str,
    func,
    *,
    is_transient_result,
    is_transient_exception,
    on_invalid_tenant_token=None,
    max_attempts: int = _RETRY_MAX_ATTEMPTS,
    base_delay_seconds: float = _RETRY_BASE_DELAY_SECONDS,
    max_delay_seconds: float = _RETRY_MAX_DELAY_SECONDS,
    jitter_ratio: float = _RETRY_JITTER_RATIO,
    sleep_fn=None,
    jitter_fn=None,
):
    sleep_fn = sleep_fn or time.sleep
    jitter_fn = jitter_fn or random.uniform
    refreshed_tenant_token = False

    for attempt_idx in range(max_attempts):
        try:
            result = func()
        except Exception as exc:
            if attempt_idx >= max_attempts - 1 or not is_transient_exception(exc):
                raise
            delay = min(max_delay_seconds, base_delay_seconds * (2**attempt_idx))
            jitter = delay * jitter_ratio
            sleep_fn(jitter_fn(max(0.0, delay - jitter), delay + jitter))
            continue

        code = getattr(result, "code", None)
        if (
            code == _INVALID_TENANT_TOKEN_CODE
            and on_invalid_tenant_token
            and not refreshed_tenant_token
        ):
            on_invalid_tenant_token()
            refreshed_tenant_token = True
            continue

        if is_transient_result(result):
            if attempt_idx >= max_attempts - 1:
                return result
            delay = min(max_delay_seconds, base_delay_seconds * (2**attempt_idx))
            jitter = delay * jitter_ratio
            sleep_fn(jitter_fn(max(0.0, delay - jitter), delay + jitter))
            continue

        return result

    raise LarkDocError(f"Retry loop exhausted unexpectedly for operation: {operation}")


def _tenant_token_cache_path() -> Path:
    override = os.getenv(_TOKEN_CACHE_ENV)
    if override:
        override_path = Path(override)
        if override.endswith(".json") or override_path.suffix:
            return override_path
        return override_path / "tenant_token.json"

    base_dir = os.getenv("XDG_CACHE_HOME")
    if base_dir:
        base = Path(base_dir)
    else:
        base = Path.home() / ".cache"
    return base / "video_asr_summary" / "tenant_token.json"


def _load_cached_tenant_token(
    cache_path: Path,
) -> tuple[str, float, float | None] | None:
    try:
        data = json.loads(cache_path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None

    token = data.get("token")
    updated_at = data.get("updated_at")
    expires_at = data.get("expires_at")
    if not isinstance(token, str) or not token:
        return None
    if not isinstance(updated_at, (int, float)):
        return None
    if expires_at is not None and not isinstance(expires_at, (int, float)):
        return None
    return (
        token,
        float(updated_at),
        (float(expires_at) if expires_at is not None else None),
    )


def _store_cached_tenant_token(
    cache_path: Path, token: str, updated_at: float, expires_at: float
) -> None:
    temp_path: Path | None = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            json.dump(
                {
                    "token": token,
                    "updated_at": updated_at,
                    "expires_at": expires_at,
                },
                temp_file,
            )
            temp_path = Path(temp_file.name)

        os.replace(temp_path, cache_path)
    except OSError:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        pass


def _request_tenant_access_token(
    app_id: str, app_secret: str
) -> tuple[str, float, float]:
    headers = {"Content-Type": "application/json"}
    payload = {"app_id": app_id, "app_secret": app_secret}

    def request_once():
        return _requests_post_with_optional_no_proxy(
            _TOKEN_REQUEST_URL,
            headers=headers,
            json=payload,
            timeout=10,
        )

    def is_transient_response(response) -> bool:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and 500 <= status_code <= 599:
            return True
        try:
            data = response.json()
        except ValueError:
            return False
        return _is_transient_lark_code_or_msg(data.get("code"), data.get("msg"))

    try:
        response = _retry_with_backoff(
            "requests.post tenant_access_token",
            request_once,
            is_transient_result=is_transient_response,
            is_transient_exception=_is_transient_network_exception,
        )
    except (
        requests.RequestException
    ) as exc:  # pragma: no cover - network issues handled at runtime
        raise LarkDocError(f"Failed to obtain tenant access token: {exc}") from exc

    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        raise LarkDocError(f"Failed to obtain tenant access token: HTTP {status_code}")

    try:
        data = response.json()
    except ValueError as exc:
        raise LarkDocError(
            "Failed to parse tenant access token response as JSON."
        ) from exc

    if data.get("code") != 0 or not data.get("tenant_access_token"):
        message = data.get("msg") or "unknown error"
        raise LarkDocError(
            f"Failed to obtain tenant access token (code={data.get('code')}): {message}"
        )

    updated_at = time.time()
    expire = data.get("expire")
    if isinstance(expire, (int, float)):
        expires_at = updated_at + float(expire)
    else:
        expires_at = updated_at + _TOKEN_REFRESH_INTERVAL
    return data["tenant_access_token"], updated_at, expires_at


def _is_cached_token_valid(
    now: float, updated_at: float, expires_at: float | None
) -> bool:
    if expires_at is not None:
        return now < (expires_at - 300)
    return now - updated_at < _TOKEN_REFRESH_INTERVAL


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
            cached_token, updated_at, expires_at = cached
            # If the cached token matches the provided token and is still valid, use it
            if cached_token == tenant_access_token and _is_cached_token_valid(
                now, updated_at, expires_at
            ):
                return tenant_access_token
        # Either token not in cache, mismatched, or expired - get a fresh one
        token, updated_at, expires_at = _request_tenant_access_token(app_id, app_secret)
        _store_cached_tenant_token(cache_path, token, updated_at, expires_at)
        return token

    # No token provided, use cache or request new one
    cached = _load_cached_tenant_token(cache_path)
    if cached:
        token, updated_at, expires_at = cached
        if _is_cached_token_valid(now, updated_at, expires_at):
            return token

    token, updated_at, expires_at = _request_tenant_access_token(app_id, app_secret)
    _store_cached_tenant_token(cache_path, token, updated_at, expires_at)
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
    _requests_post_with_optional_no_proxy(
        url,
        headers=headers,
        params=params,
        json=body,
        timeout=10,
    )


def _ensure_client_builder(
    app_id: str,
    app_secret: str,
    *,
    tenant_access_token: str | None,
    user_access_token: str | None,
    domain: str | None,
) -> tuple[ClientBuilder, Optional[RequestOption]]:
    builder = (
        Client.builder()
        .app_id(app_id)
        .app_secret(app_secret)
        .log_level(LogLevel.ERROR)
        .timeout(30)
    )
    if domain:
        builder.domain(domain)
    request_option: Optional[RequestOption] = None
    if user_access_token:
        builder.enable_set_token(True)
        request_option = (
            RequestOption.builder().user_access_token(user_access_token).build()
        )
    elif tenant_access_token:
        builder.enable_set_token(True)
        request_option = (
            RequestOption.builder().tenant_access_token(tenant_access_token).build()
        )
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
            elements.append(
                _DocumentElement(kind="paragraph", spans=_parse_inline_spans(content))
            )
        paragraph_lines = []

    def flush_list() -> None:
        nonlocal current_list_type, current_list_items
        if current_list_type and current_list_items:
            list_spans = [
                _parse_inline_spans(item)
                for item in current_list_items
                if _normalize_whitespace(item)
            ]
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
                elements.append(
                    _DocumentElement(kind="heading", level=level, spans=content_spans)
                )
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


def _parse_corrected_transcript_elements(
    corrected_transcript: str,
) -> List[_DocumentElement]:
    elements: List[_DocumentElement] = [
        _DocumentElement(
            kind="heading",
            level=2,
            spans=[_InlineSpan("Corrected Transcript")],
        )
    ]
    paragraphs = re.split(r"\n\s*\n", corrected_transcript)
    for paragraph in paragraphs:
        if paragraph == "":
            continue
        for start in range(0, len(paragraph), _CORRECTED_TRANSCRIPT_MAX_BLOCK_CHARS):
            chunk = paragraph[start : start + _CORRECTED_TRANSCRIPT_MAX_BLOCK_CHARS]
            if chunk == "":
                continue
            elements.append(
                _DocumentElement(kind="paragraph", spans=[_InlineSpan(chunk)])
            )
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
        text_elements.append(
            TextElementBuilder().text_run(TextRunBuilder().content("").build()).build()
        )
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
            blocks.append(
                BlockBuilder().block_type(_PARAGRAPH_BLOCK_TYPE).text(text).build()
            )
        return blocks

    if element.kind == "ordered_list" and element.items:
        blocks = []
        for order_index, spans in enumerate(element.items, start=1):
            if not spans:
                continue
            prefix = _InlineSpan(f"{order_index}. ")
            prefixed_spans = [prefix] + list(spans)
            text = _build_text(prefixed_spans)
            blocks.append(
                BlockBuilder().block_type(_PARAGRAPH_BLOCK_TYPE).text(text).build()
            )
        return blocks

    return []


def _append_elements_to_document(
    block_resource,
    document_id: str,
    elements: Iterable[_DocumentElement],
    request_option: Optional[RequestOption],
) -> None:
    def create_blocks(
        parent_id: str, insert_index: int, blocks: List[Block], *, ctx: str
    ) -> List[Block]:
        body = (
            CreateDocumentBlockChildrenRequestBody.builder()
            .index(insert_index)
            .children(blocks)
            .build()
        )
        request = (
            CreateDocumentBlockChildrenRequest.builder()
            .document_id(document_id)
            .block_id(parent_id)
            .request_body(body)
            .build()
        )
        response = _retry_with_backoff(
            "docx.v1.document_block_children.create",
            lambda: _run_sdk_call(
                lambda: block_resource.create(request, request_option)
            ),
            is_transient_result=lambda resp: _is_transient_lark_code_or_msg(
                getattr(resp, "code", None), getattr(resp, "msg", None)
            ),
            is_transient_exception=_is_transient_network_exception,
        )
        if response.code != 0:
            raise LarkDocError(
                f"Failed to insert summary into Lark document: {response.msg} ({_format_lark_error(response, 'docx.v1.document_block_children.create')})"
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

    for batch_start in range(0, len(blocks_to_insert), MAX_BLOCKS_PER_REQUEST):
        batch_blocks = blocks_to_insert[
            batch_start : batch_start + MAX_BLOCKS_PER_REQUEST
        ]
        create_blocks(document_id, -1, batch_blocks, ctx="batched-insert")


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
    corrected_transcript: str | None = None,
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
    provided_tenant_access_token = tenant_access_token or os.getenv(
        "LARK_TENANT_ACCESS_TOKEN"
    )
    user_access_token = user_access_token or os.getenv("LARK_USER_ACCESS_TOKEN")
    user_subdomain = user_subdomain or os.getenv("LARK_USER_SUBDOMAIN")
    api_domain = api_domain or os.getenv("LARK_API_DOMAIN")
    message_receiver_id = message_receiver_id or os.getenv("LARK_MESSAGE_RECEIVER_ID")

    if not app_id or not app_secret:
        raise LarkDocError(
            "Lark app credentials are missing. Provide app_id and app_secret or set LARK_APP_ID/LARK_APP_SECRET."
        )

    tenant_access_token = _resolve_tenant_access_token(
        app_id, app_secret, provided_tenant_access_token
    )

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

    create_req = (
        CreateDocumentRequest.builder().request_body(body_builder.build()).build()
    )

    result: dict[str, str] | None = None
    error: LarkDocError | None = None
    attempted = False
    cache_path = _tenant_token_cache_path()

    def _refresh_tenant_token_state(state: dict[str, Any]) -> None:
        nonlocal tenant_access_token
        token, updated_at, expires_at = _request_tenant_access_token(app_id, app_secret)
        _store_cached_tenant_token(cache_path, token, updated_at, expires_at)
        tenant_access_token = token
        builder_refreshed, request_option_refreshed = _ensure_client_builder(
            app_id,
            app_secret,
            tenant_access_token=tenant_access_token,
            user_access_token=None,
            domain=api_domain,
        )
        state["client"] = builder_refreshed.build()
        state["request_option"] = request_option_refreshed

    def _create_document_with_retry(
        state: dict[str, Any], *, allow_token_refresh: bool
    ):
        on_invalid_tenant_token = (
            (lambda: _refresh_tenant_token_state(state))
            if allow_token_refresh
            else None
        )
        return _retry_with_backoff(
            "docx.v1.document.create",
            lambda: _run_sdk_call(
                lambda: state["client"].docx.v1.document.create(
                    create_req, state["request_option"]
                )
            ),  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue]
            is_transient_result=lambda resp: _is_transient_lark_code_or_msg(
                getattr(resp, "code", None), getattr(resp, "msg", None)
            ),
            is_transient_exception=_is_transient_network_exception,
            on_invalid_tenant_token=on_invalid_tenant_token,
        )

    try:
        attempted = True
        primary_state: dict[str, Any] = {
            "client": client,
            "request_option": request_option,
        }
        create_resp = _create_document_with_retry(
            primary_state,
            allow_token_refresh=bool(tenant_access_token and not user_access_token),
        )
        client = primary_state["client"]
        request_option = primary_state["request_option"]
        if (
            create_resp.code != 0
            or not create_resp.data
            or not create_resp.data.document
        ):
            # If user access token failed and we have app credentials, try with tenant token as fallback
            if (
                user_access_token
                and not provided_tenant_access_token
                and tenant_access_token
            ):
                print(
                    f"[DEBUG] User token failed (code={create_resp.code}, msg={create_resp.msg}), retrying with tenant token",
                    file=sys.stderr,
                )
                builder_fallback, request_option_fallback = _ensure_client_builder(
                    app_id,
                    app_secret,
                    tenant_access_token=tenant_access_token,
                    user_access_token=None,  # Don't use user token this time
                    domain=api_domain,
                )
                client_fallback = builder_fallback.build()
                fallback_state: dict[str, Any] = {
                    "client": client_fallback,
                    "request_option": request_option_fallback,
                }
                create_resp = _create_document_with_retry(
                    fallback_state,
                    allow_token_refresh=bool(tenant_access_token),
                )
                client_fallback = fallback_state["client"]
                request_option_fallback = fallback_state["request_option"]
                if (
                    create_resp.code != 0
                    or not create_resp.data
                    or not create_resp.data.document
                ):
                    raise LarkDocError(
                        f"Failed to create Lark document (tenant token fallback also failed): {create_resp.msg} ({_format_lark_error(create_resp, 'docx.v1.document.create')})"
                    )
                else:
                    print(f"[DEBUG] Tenant token fallback succeeded", file=sys.stderr)
                    # Update client and request_option for subsequent operations
                    client = client_fallback
                    request_option = request_option_fallback
            else:
                raise LarkDocError(
                    f"Failed to create Lark document: {create_resp.msg} ({_format_lark_error(create_resp, 'docx.v1.document.create')})"
                )

        document = create_resp.data.document
        document_id = document.document_id

        elements = _parse_summary_elements(content)
        if corrected_transcript is not None:
            elements.extend(_parse_corrected_transcript_elements(corrected_transcript))
        if elements:
            _append_elements_to_document(
                client.docx.v1.document_block_children,  # pyright: ignore[reportOptionalMemberAccess]
                document_id,  # pyright: ignore[reportArgumentType]
                elements,
                request_option,
            )

        base_host = (
            f"{user_subdomain}.feishu.cn" if user_subdomain else "open.feishu.cn"
        )
        result = {  # pyright: ignore[reportAssignmentType]
            "document_id": document_id,
            "title": title,
            "url": f"https://{base_host}/docx/{document_id}",
        }
    except LarkDocError as exc:
        error = exc
    except (
        Exception
    ) as exc:  # pragma: no cover - unexpected errors still need conversion
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
