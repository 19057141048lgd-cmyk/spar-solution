"""Bohrium 论文检索 Provider。

适配器只负责 Bohrium API 的请求、响应校验和 ``PaperDoc`` 映射。网络层可
通过 ``transport`` 注入，单元测试不需要真实 Key 或外部网络。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from ..paperdoc import validate_paper_doc
from .base import ProviderError, ProviderResult


DEFAULT_BASE_URL = "https://open.bohrium.com"
SEARCH_PATH = "/openapi/v2/paper/rag/pass/keyword"
CONTENT_PATH = "/openapi/v1/lkm/papers/content/batch"

# transport(method, url, headers, body, timeout) -> (status_code, JSON-like body)
Transport = Callable[[str, str, Mapping[str, str], bytes, float], Any]


@dataclass(frozen=True)
class BohriumConfig:
    """Bohrium 连接配置；Key 只保存在实例内，不提供打印方法。"""

    base_url: str = DEFAULT_BASE_URL
    access_key: str = ""
    timeout: float = 30.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "BohriumConfig":
        values = os.environ if environ is None else environ
        return cls(
            base_url=values.get("BOHRIUM_BASE_URL", DEFAULT_BASE_URL),
            access_key=values.get("BOHR_ACCESS_KEY", ""),
            timeout=_read_timeout(values.get("BOHRIUM_TIMEOUT")),
        )


def _read_timeout(value: str | None) -> float:
    if not value:
        return 30.0
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return 30.0
    return timeout if timeout > 0 else 30.0


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "value", "name", "title", "content"):
            if key in value:
                text = _as_text(value[key])
                if text:
                    return text
        return ""
    return str(value).strip()


def _pick(item: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in item and item[name] not in (None, ""):
            return item[name]
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _authors(value: Any) -> list[str]:
    result: list[str] = []
    for author in _as_list(value):
        if isinstance(author, Mapping):
            name = _pick(author, "name", "fullName", "full_name", "authorName", "author_name")
        else:
            name = author
        text = _as_text(name)
        if text:
            result.append(text)
    return result


def _year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, int):
            return value if 0 < value < 3000 else None
        match = re.search(r"\b(\d{4})\b", _as_text(value))
        return int(match.group(1)) if match else None
    except (TypeError, ValueError):
        return None


def _normalise_doi(value: Any) -> str | None:
    doi = _as_text(value)
    if not doi:
        return None
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I).strip()
    return doi or None


def _identifier(item: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    doi = _normalise_doi(_pick(item, "doi", "DOI"))
    arxiv = _as_text(_pick(item, "arxiv_id", "arxivId", "arxivID")) or None
    unique = _as_text(_pick(item, "paper_id", "paperId", "id", "unique_id", "uniqueId")) or None
    return doi, arxiv, unique


def _relation_items(value: Any) -> list[Any]:
    result: list[Any] = []
    for relation in _as_list(value):
        if isinstance(relation, Mapping):
            item = dict(relation)
            item.setdefault("relation_source", "bohrium")
            result.append(item)
        else:
            relation_id = _as_text(relation)
            if relation_id:
                result.append({"id": relation_id, "relation_source": "bohrium"})
    return result


def _nested_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """兼容 data 中包裹 ``paper``/``metadata`` 的返回项。"""

    merged: dict[str, Any] = {}
    for key in ("paper", "paperInfo", "metadata"):
        value = record.get(key)
        if isinstance(value, Mapping):
            merged.update(value)
    merged.update(record)
    return merged


def _content_text(item: Mapping[str, Any]) -> str:
    value = _pick(item, "content", "text", "full_text", "fullText", "body", "markdown")
    return _as_text(value)


class BohriumProvider:
    """Bohrium keyword-RAG 与批量全文接口适配器。"""

    name = "bohrium"

    def __init__(
        self,
        access_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        self.access_key = access_key.strip() if isinstance(access_key, str) else ""
        self.base_url = base_url.rstrip("/") or DEFAULT_BASE_URL
        self.timeout = timeout if timeout > 0 else 30.0
        self._transport = transport

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BohriumProvider":
        """从 ``load_config`` 或 ``ProviderSettings`` 兼容读取配置。"""

        if hasattr(config, "access_key"):
            return cls(
                getattr(config, "access_key", ""),
                base_url=getattr(config, "base_url", DEFAULT_BASE_URL),
            )
        return cls(
            str(config.get("BOHR_ACCESS_KEY", "")),
            base_url=str(config.get("BOHRIUM_BASE_URL", DEFAULT_BASE_URL)),
        )

    def search(
        self,
        query: str,
        *,
        page_size: int = 10,
        cursor: str | None = None,
        words: Sequence[str] | None = None,
        question: str | None = None,
        request_type: int = 0,
        **_: Any,
    ) -> ProviderResult:
        if not isinstance(query, str) or not query.strip():
            raise ProviderError(self.name, "config", "query must be a non-empty string")
        page = _parse_page(cursor)
        query_words = list(words) if words is not None else [query.strip()]
        payload: dict[str, Any] = {
            "words": _validate_words(query_words),
            "question": (question if question is not None else query).strip(),
            "type": request_type,
            "pageSize": _validate_page_size(page_size),
        }
        if page is not None:
            payload["page"] = page
        response = self._post_json(SEARCH_PATH, payload)
        data = _extract_data(response, self.name)
        records = [self._to_paper_doc(item, page=page or 1, endpoint=SEARCH_PATH) for item in data]
        next_cursor = _next_cursor(response, page)
        total = _total(response, len(records))
        return ProviderResult(
            self.name,
            "search",
            records,
            next_cursor=next_cursor,
            total=total,
            provenance={"endpoint": self._url(SEARCH_PATH), "page": page or 1},
        )

    def read(
        self,
        paper_id: str,
        *,
        cursor: str | None = None,
        **_: Any,
    ) -> ProviderResult:
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ProviderError(self.name, "config", "paper_id must be a non-empty string")
        return self.read_many([paper_id], cursor=cursor)

    def read_many(self, paper_ids: Sequence[str], *, cursor: str | None = None) -> ProviderResult:
        ids = [str(item).strip() for item in paper_ids if str(item).strip()]
        if not ids:
            raise ProviderError(self.name, "config", "paper_ids must contain at least one id")
        payload: dict[str, Any] = {"paperIds": ids}
        page = _parse_page(cursor)
        if page is not None:
            payload["page"] = page
        response = self._post_json(CONTENT_PATH, payload)
        data = _extract_data(response, self.name)
        records = [
            self._to_paper_doc(item, page=page or 1, endpoint=CONTENT_PATH, forced_id=ids[index] if index < len(ids) else None, content_mode=True)
            for index, item in enumerate(data)
        ]
        return ProviderResult(
            self.name,
            "read",
            records,
            next_cursor=_next_cursor(response, page),
            total=_total(response, len(records)),
            provenance={"endpoint": self._url(CONTENT_PATH), "page": page or 1},
        )

    # Explicit alias used by callers that do not use the BaseProvider name.
    fetch_content = read_many
    fetch_full_text = read_many

    def relations(self, paper_id: str, *, relation: str = "all", cursor: str | None = None, **_: Any) -> ProviderResult:
        """Bohrium 当前给定接口没有关系图端点，显式报告不支持。"""

        raise ProviderError(self.name, "unsupported", f"relations endpoint is not configured (requested {relation})")

    def _url(self, path: str) -> str:
        return urljoin(f"{self.base_url}/", path.lstrip("/"))

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self.access_key:
            raise ProviderError(self.name, "config", "BOHR_ACCESS_KEY is not configured")
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.access_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            if self._transport is not None:
                raw_response = self._transport("POST", self._url(path), headers, body, self.timeout)
                status, raw_body = _transport_response(raw_response)
            else:
                request = Request(self._url(path), data=body, headers=headers, method="POST")
                with urlopen(request, timeout=self.timeout) as response:  # nosec B310: configured HTTPS API
                    status, raw_body = response.status, response.read()
        except HTTPError as exc:
            code = "auth" if exc.code in {401, 403} else "rate" if exc.code == 429 else "http"
            raise ProviderError(self.name, code, f"HTTP status {exc.code}", retryable=exc.code >= 500 or exc.code == 429, status_code=exc.code) from exc
        except TimeoutError as exc:
            raise ProviderError(self.name, "timeout", "request timed out", retryable=True) from exc
        except URLError as exc:
            raise ProviderError(self.name, "network", "request failed", retryable=True) from exc
        except OSError as exc:
            raise ProviderError(self.name, "network", "request failed", retryable=True) from exc

        if not isinstance(status, int) or status < 200 or status >= 300:
            code = "auth" if status in {401, 403} else "rate" if status == 429 else "http"
            raise ProviderError(self.name, code, f"HTTP status {status}", retryable=status >= 500 or status == 429, status_code=status)
        try:
            if isinstance(raw_body, Mapping):
                response = dict(raw_body)
            else:
                if isinstance(raw_body, bytes):
                    raw_body = raw_body.decode("utf-8")
                response = json.loads(raw_body)
        except (UnicodeDecodeError, TypeError, ValueError) as exc:
            raise ProviderError(self.name, "parse", "response is not valid JSON", status_code=status) from exc
        if not isinstance(response, dict):
            raise ProviderError(self.name, "parse", "response root must be an object", status_code=status)
        code = response.get("code")
        if code not in (0, "0", None):
            message = _as_text(response.get("message")) or "Bohrium returned a business error"
            raise ProviderError(self.name, "business", message, status_code=status, details={"provider_code": code})
        if "code" not in response:
            raise ProviderError(self.name, "parse", "response missing code", status_code=status)
        return response

    def _to_paper_doc(
        self,
        raw: Any,
        *,
        page: int,
        endpoint: str,
        forced_id: str | None = None,
        content_mode: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ProviderError(self.name, "parse", "data item must be an object")
        item = _nested_record(raw)
        doi, arxiv_id, unique_id = _identifier(item)
        title = _as_text(_pick(item, "title", "paperTitle", "paper_title", "name"))
        abstract = _as_text(_pick(item, "abstract", "summary", "description", "paperAbstract"))
        full_text = _content_text(item) if content_mode or _pick(item, "content", "full_text", "fullText", "body", "markdown") else ""
        source_id = unique_id or forced_id
        if not doi and not source_id and not title:
            raise ProviderError(self.name, "parse", "paper item has no identifier or title")
        identity = doi or source_id or hashlib.sha1(f"{title}|{_year(_pick(item, 'year', 'publicationYear', 'date'))}".encode("utf-8")).hexdigest()[:20]
        paper_id = f"doi:{doi}" if doi else f"bohrium:{identity}"
        status = "fulltext" if full_text else "abstract" if abstract else "metadata"
        content_ref = f"bohrium://papers/{identity}/content" if full_text else None
        chunks = []
        if full_text:
            chunks = [{"chunk_id": f"{paper_id}:content:0", "content_ref": content_ref, "offset": 0, "section": None, "page": None}]
        doc = {
            "schema_version": "paperdoc.v1",
            "paper_id": paper_id,
            "identifiers": {
                "doi": doi,
                "arxiv_id": arxiv_id,
                "s2_id": None,
                "openalex_id": None,
                "pmid": _as_text(_pick(item, "pmid", "pmId")) or None,
                "pmcid": _as_text(_pick(item, "pmcid", "pmcId")) or None,
                "sciverse_doc_id": None,
                "unique_id": unique_id or forced_id,
            },
            "bibliography": {
                "title": title,
                "authors": _authors(_pick(item, "authors", "author", "creators")),
                "year": _year(_pick(item, "year", "publicationYear", "date")),
                "venue": _as_text(_pick(item, "venue", "journal", "containerTitle")) or None,
                "abstract": abstract or None,
                "fields": [_as_text(value) for value in _as_list(_pick(item, "fields", "categories", "keywords")) if _as_text(value)],
            },
            "access": {
                "landing_url": _as_text(_pick(item, "landing_url", "landingUrl", "url", "link", "paperUrl")) or None,
                "pdf_url": _as_text(_pick(item, "pdf_url", "pdfUrl", "pdf")) or None,
                "oa_url": _as_text(_pick(item, "oa_url", "oaUrl", "openAccessUrl")) or None,
                "full_text_status": status,
                "content_type": "text/plain" if full_text else "abstract" if abstract else None,
            },
            "content": {"content_ref": content_ref, "chunks": chunks, "sections": [], "char_count": len(full_text)},
            "relations": {
                "references": _relation_items(_pick(item, "references", "reference")),
                "citations": _relation_items(_pick(item, "citations", "citation")),
                "related_works": _relation_items(_pick(item, "related_works", "relatedWorks")),
            },
            "scores": {"retrieval": None, "relevance": None, "constraint": None, "quality": None, "evidence": None, "citation": None, "novelty": None, "final": None, "confidence": None},
            "provenance": {
                "sources": [self.name],
                "query_id": None,
                "subquery_id": None,
                "iteration": 0,
                "parent_node_id": None,
                "endpoints": [self._url(endpoint)],
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "pages": [page],
                "reconciliation": {"complete": True},
                "warnings": [],
            },
            "evidence_refs": [],
            "status": {"hard_constraints_pass": None, "evidence_status": status, "provider_errors": []},
        }
        try:
            validate_paper_doc(doc)
        except Exception as exc:
            raise ProviderError(self.name, "parse", "mapped PaperDoc failed validation") from exc
        return doc


def _validate_words(words: Sequence[str]) -> list[str]:
    if isinstance(words, (str, bytes)) or not isinstance(words, Sequence):
        raise ProviderError("bohrium", "config", "words must be a non-empty array")
    result = [_as_text(item) for item in words]
    result = [item for item in result if item]
    if not result:
        raise ProviderError("bohrium", "config", "words must be a non-empty array")
    return result


def _validate_page_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ProviderError("bohrium", "config", "page_size must be an integer between 1 and 100")
    return value


def _parse_page(cursor: str | None) -> int | None:
    if cursor in (None, ""):
        return None
    try:
        page = int(cursor)
    except (TypeError, ValueError) as exc:
        raise ProviderError("bohrium", "config", "cursor must be a numeric page") from exc
    if page < 1:
        raise ProviderError("bohrium", "config", "cursor page must be >= 1")
    return page


def _transport_response(raw_response: Any) -> tuple[int, Any]:
    if isinstance(raw_response, tuple) and len(raw_response) == 2:
        status, body = raw_response
        return int(status), body
    # A mock may return a JSON-like object directly; that is a successful 200.
    return 200, raw_response


def _extract_data(response: Mapping[str, Any], source: str) -> list[Mapping[str, Any]]:
    if "data" not in response:
        raise ProviderError(source, "parse", "response missing data")
    data = response["data"]
    if isinstance(data, list):
        records = data
    elif isinstance(data, Mapping):
        nested = next((data[key] for key in ("papers", "results", "items", "records", "list") if isinstance(data.get(key), list)), None)
        records = nested if nested is not None else [data] if any(key in data for key in ("title", "paperTitle", "paperId", "doi", "id")) else None
        if records is None:
            raise ProviderError(source, "parse", "response data object has no paper list")
    else:
        raise ProviderError(source, "parse", "response data must be an array or paper object")
    if any(not isinstance(item, Mapping) for item in records):
        raise ProviderError(source, "parse", "response data contains a non-object paper")
    return list(records)


def _next_cursor(response: Mapping[str, Any], page: int | None) -> str | None:
    value = _pick(response, "next_cursor", "nextCursor", "nextPage", "next_page")
    if value not in (None, ""):
        return str(value)
    data = response.get("data")
    if isinstance(data, Mapping):
        value = _pick(data, "next_cursor", "nextCursor", "nextPage", "next_page")
        if value not in (None, ""):
            return str(value)
    return None


def _total(response: Mapping[str, Any], fallback: int) -> int:
    value = _pick(response, "total", "totalCount", "count")
    if value is None and isinstance(response.get("data"), Mapping):
        value = _pick(response["data"], "total", "totalCount", "count")
    try:
        return int(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


__all__ = ["BohriumConfig", "BohriumProvider", "CONTENT_PATH", "DEFAULT_BASE_URL", "SEARCH_PATH", "Transport"]
