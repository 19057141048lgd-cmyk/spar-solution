"""arXiv Atom API provider used by the P1 comparison protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from ..config import redact_url
from ..paperdoc import validate_paper_doc
from .base import ProviderError, ProviderResult


DEFAULT_BASE_URL = "https://export.arxiv.org/api/query"
_ATOM = "http://www.w3.org/2005/Atom"
_ARXIV = "http://arxiv.org/schemas/atom"


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: bytes | str
    headers: Mapping[str, str] = field(default_factory=dict)


Transport = Callable[[str, str, Mapping[str, str], float], Any]


def _normalise_response(value: Any) -> TransportResponse:
    if isinstance(value, TransportResponse):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return TransportResponse(int(value[0]), value[1])
    raise ProviderError("arxiv", "network", "transport returned an unsupported response")


def _default_transport(method: str, url: str, headers: Mapping[str, str], timeout: float) -> TransportResponse:
    request = Request(url, method=method, headers=dict(headers))
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310: configured public API URL
            return TransportResponse(
                int(getattr(response, "status", response.getcode())),
                response.read(),
                dict(response.headers.items()),
            )
    except HTTPError as exc:
        return TransportResponse(exc.code, exc.read(), dict(exc.headers.items()) if exc.headers else {})
    except (TimeoutError, URLError, OSError) as exc:
        raise ProviderError("arxiv", "network", f"request failed: {type(exc).__name__}") from exc


def _text(node: ElementTree.Element | None) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def _year(value: str) -> int | None:
    match = re.match(r"(\d{4})", value.strip())
    return int(match.group(1)) if match else None


def _arxiv_id(value: str) -> str | None:
    value = value.strip().rstrip("/")
    if not value:
        return None
    value = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", value, flags=re.I)
    value = re.sub(r"\.pdf$", "", value, flags=re.I)
    value = re.sub(r"^arxiv:", "", value, flags=re.I)
    return re.sub(r"v\d+$", "", value, flags=re.I).casefold() or None


def _find(entry: ElementTree.Element, name: str) -> ElementTree.Element | None:
    return entry.find(f"{{{_ATOM}}}{name}")


def _search_expression(query: str) -> str:
    """把自然语言查询转成宽召回的 arXiv 表达式。

    Provider 层不应把用户问题强制收紧为全词 ``AND``。复杂问题由上层
    QueryPlanner 拆成多个变体；单个变体这里使用短语和 ``OR``，再由后续
    去重、重排和约束判断恢复精度。只允许安全的词和短语进入表达式，避免
    把用户输入当作 arXiv 查询语法执行。
    """

    phrases = [item.strip() for item in re.findall(r'"([^"\r\n]+)"', query) if item.strip()]
    remainder = re.sub(r'"[^"\r\n]+"', " ", query)
    terms = re.findall(r"[\w]+(?:[-'][\w]+)?", remainder, flags=re.UNICODE)
    clauses: list[str] = []
    for phrase in phrases[:4]:
        words = re.findall(r"[\w]+(?:[-'][\w]+)?", phrase, flags=re.UNICODE)
        if words:
            clauses.append(f'all:"{" ".join(words[:8])}"')
    clauses.extend(f"all:{term}" for term in terms[:12] if term.casefold() not in {"and", "or"})
    if not clauses:
        raise ProviderError("arxiv", "config", "query must contain searchable terms")
    return "(" + " OR ".join(clauses) + ")"


def _validate_year(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1900 <= value <= 2200:
        raise ProviderError("arxiv", "config", f"{name} must be a four-digit year")
    return value


class ArxivProvider:
    """Search arXiv's public Atom endpoint without third-party dependencies."""

    name = "arxiv"
    source = "arxiv"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport: Transport | None = None,
        timeout: float = 20.0,
    ) -> None:
        parts = urlsplit(str(base_url).strip())
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ProviderError(self.source, "config", "base_url must be an http(s) URL")
        self.base_url = str(base_url).strip().rstrip("?")
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ProviderError(self.source, "config", "timeout must be positive")
        self.transport = transport or _default_transport

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None, **kwargs: Any) -> "ArxivProvider":
        values = config or {}
        return cls(base_url=str(values.get("ARXIV_BASE_URL", DEFAULT_BASE_URL)), **kwargs)

    def search(
        self,
        query: str,
        *,
        page_size: int = 10,
        cursor: str | None = None,
        per_page: int | None = None,
        page: int | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        cutoff_year: int | None = None,
        **_: Any,
    ) -> ProviderResult:
        if not isinstance(query, str) or not query.strip():
            raise ProviderError(self.source, "config", "query must be a non-empty string")
        size = page_size if per_page is None else per_page
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 200:
            raise ProviderError(self.source, "config", "page_size must be an integer between 1 and 200")
        if cursor not in (None, ""):
            try:
                page = int(cursor)
            except (TypeError, ValueError) as exc:
                raise ProviderError(self.source, "config", "cursor must be a numeric page") from exc
        if page is None:
            page = 1
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ProviderError(self.source, "config", "page must be a positive integer")
        start = _validate_year(start_year, "start_year")
        end = _validate_year(end_year, "end_year")
        cutoff = _validate_year(cutoff_year, "cutoff_year")
        if cutoff is not None:
            end = min(end, cutoff) if end is not None else cutoff
        if start is not None and end is not None and start > end:
            raise ProviderError(self.source, "config", "start_year must not exceed end_year")
        expression = _search_expression(query)
        date_filter = None
        if start is not None or end is not None:
            lower = f"{start or 1900}01010000"
            upper = f"{end or 2200}12312359"
            date_filter = f"submittedDate:[{lower} TO {upper}]"
            expression = f"{expression} AND {date_filter}"
        params = {"search_query": expression, "start": (page - 1) * size, "max_results": size}
        url = f"{self.base_url}&{urlencode(params)}" if "?" in self.base_url else f"{self.base_url}?{urlencode(params)}"
        try:
            response = _normalise_response(self.transport("GET", url, {"Accept": "application/atom+xml", "User-Agent": "spar-p1/arxiv"}, self.timeout))
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.source, "network", f"transport failed: {type(exc).__name__}") from exc
        if not 200 <= response.status < 300:
            code = "rate" if response.status == 429 else "network"
            raise ProviderError(self.source, code, f"HTTP status {response.status}", status_code=response.status, retryable=response.status >= 500 or response.status == 429)
        try:
            raw = response.body.decode("utf-8") if isinstance(response.body, bytes) else str(response.body)
            root = ElementTree.fromstring(raw)
        except (UnicodeDecodeError, ElementTree.ParseError, TypeError) as exc:
            raise ProviderError(self.source, "parse", "response is not valid Atom XML", status_code=response.status) from exc
        query_id = "q_" + hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:16]
        retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        records = [self._to_paper_doc(entry, query_id=query_id, page=page, retrieved_at=retrieved_at) for entry in root.findall(f"{{{_ATOM}}}entry")]
        # API 日期过滤用于减少网络返回；本地再次过滤，防止供应商/fixture忽略查询条件。
        filtered_out = 0
        if start is not None or end is not None:
            kept: list[dict[str, Any]] = []
            for record in records:
                year = ((record.get("bibliography") or {}).get("year"))
                if isinstance(year, int) and (start is None or year >= start) and (end is None or year <= end):
                    kept.append(record)
                else:
                    filtered_out += 1
            records = kept
        total_node = root.find(f"{{{_OPENSEARCH}}}totalResults")
        try:
            total = int(_text(total_node)) if total_node is not None else None
        except ValueError:
            total = None
        reported_total = total if total is not None else len(records)
        if date_filter and total is not None:
            reported_total = max(0, total - filtered_out)
        return ProviderResult(
            self.source,
            "search",
            records,
            total=reported_total,
            provenance={"endpoint": self.base_url, "page": page, "page_size": size, "execution_status": "live", "query_expression": expression, "date_filter": date_filter, "filtered_out": filtered_out},
        )

    def read(self, paper_id: str, *, cursor: str | None = None, **_: Any) -> ProviderResult:
        raise ProviderError(self.source, "unsupported", "arXiv content read is not configured")

    def relations(self, paper_id: str, *, relation: str = "all", cursor: str | None = None, **_: Any) -> ProviderResult:
        raise ProviderError(self.source, "unsupported", "arXiv relations endpoint is not configured")

    def _to_paper_doc(self, entry: ElementTree.Element, *, query_id: str, page: int, retrieved_at: str) -> dict[str, Any]:
        raw_id = _text(_find(entry, "id"))
        arxiv_id = _arxiv_id(raw_id)
        title = _text(_find(entry, "title"))
        summary = _text(_find(entry, "summary")) or None
        authors = [_text(author.find(f"{{{_ATOM}}}name")) for author in entry.findall(f"{{{_ATOM}}}author")]
        authors = [author for author in authors if author]
        published = _text(_find(entry, "published"))
        doi = _text(entry.find(f"{{{_ARXIV}}}doi")) or None
        pdf_url = None
        landing_url = raw_id or None
        for link in entry.findall(f"{{{_ATOM}}}link"):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
        status = "abstract" if summary else "metadata"
        paper_id = f"arxiv:{arxiv_id}" if arxiv_id else "arxiv:unknown-" + hashlib.sha256(title.encode()).hexdigest()[:16]
        doc = {
            "schema_version": "paperdoc.v1", "paper_id": paper_id,
            "identifiers": {"doi": doi, "arxiv_id": arxiv_id, "s2_id": None, "openalex_id": None, "pmid": None, "pmcid": None, "sciverse_doc_id": None, "unique_id": None},
            "bibliography": {"title": title, "authors": authors, "year": _year(published), "venue": "arXiv", "abstract": summary, "fields": []},
            "access": {"landing_url": landing_url, "pdf_url": pdf_url, "oa_url": pdf_url, "full_text_status": status, "content_type": "abstract" if summary else "metadata"},
            "content": {"content_ref": None, "chunks": [], "sections": [], "char_count": len(summary or "")},
            "relations": {"references": [], "citations": [], "related_works": []},
            "scores": {"retrieval": None, "relevance": None, "constraint": None, "quality": None, "evidence": None, "citation": None, "novelty": None, "final": None, "confidence": None},
            "provenance": {"sources": [self.source], "query_id": query_id, "subquery_id": None, "iteration": 0, "parent_node_id": None, "endpoints": [redact_url(self.base_url)], "retrieved_at": retrieved_at, "pages": [page], "reconciliation": {"complete": True}, "warnings": []},
            "evidence_refs": [], "status": {"hard_constraints_pass": None, "evidence_status": status, "provider_errors": []},
        }
        try:
            return validate_paper_doc(doc)
        except Exception as exc:
            raise ProviderError(self.source, "parse", "mapped PaperDoc failed validation") from exc


_OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"


__all__ = ["ArxivProvider", "DEFAULT_BASE_URL", "TransportResponse"]
