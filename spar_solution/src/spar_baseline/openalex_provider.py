"""OpenAlex ``/works`` provider for the P1 PaperDoc baseline.

The provider deliberately has no dependency on a HTTP client package.  A
transport can be injected in tests (or by a future provider runner), while
the default transport uses :mod:`urllib.request`.  Credentials are read from
the config object supplied by the caller and are never included in exception
messages or logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .paperdoc import validate_paper_doc
from .providers.base import ProviderError, ProviderResult


@dataclass(frozen=True)
class TransportResponse:
    """Small response object used by the injectable transport contract."""

    status: int
    body: bytes | str
    headers: Mapping[str, str] = field(default_factory=dict)


Transport = Callable[[str, str, Mapping[str, str], float], Any]


def _config_value(config: Any, *names: str) -> Any:
    """Read a value from either a mapping or a simple settings object."""

    if config is None:
        return None
    if isinstance(config, Mapping):
        for name in names:
            if name in config:
                return config[name]
        return None
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return None


def _normalise_base_url(value: Any) -> str:
    base = str(value or "https://api.openalex.org").strip().rstrip("/")
    if not base:
        return "https://api.openalex.org"
    parts = urlsplit(base)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ProviderError("openalex", "config", "base_url must be an http(s) URL")
    # A caller may provide the API root or a root ending in /works.  The
    # provider always appends exactly one /works below.
    path = parts.path.rstrip("/")
    if path == "/works":
        path = ""
    return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")


def _normalise_doi(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    doi = value.strip()
    if doi.lower().startswith("doi:"):
        doi = doi[4:].strip()
    parts = urlsplit(doi)
    if parts.scheme and parts.netloc and parts.path:
        doi = parts.path.lstrip("/")
    doi = doi.strip().rstrip(".").lower()
    return doi or None


def _normalise_openalex_id(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip().rstrip("/")
    # OpenAlex returns URLs for work IDs.  Keep the stable W... identifier in
    # PaperDoc so it is compact in Agent messages and easy to compare.
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    return value or None


def _reconstruct_abstract(value: Any) -> tuple[str | None, list[str]]:
    """Rebuild OpenAlex's inverted-index abstract without third-party code."""

    if value in (None, {}):
        return None, []
    if not isinstance(value, Mapping):
        return None, ["abstract_inverted_index_invalid"]
    positions: dict[int, list[str]] = {}
    try:
        for word, indices in value.items():
            if not isinstance(word, str) or not isinstance(indices, list):
                raise ValueError
            for index in indices:
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError
                positions.setdefault(index, []).append(word)
    except (TypeError, ValueError):
        return None, ["abstract_inverted_index_invalid"]
    if not positions:
        return None, []
    ordered = sorted(positions)
    warnings: list[str] = []
    if ordered[0] != 0:
        warnings.append("abstract_index_starts_nonzero")
    if len(ordered) != ordered[-1] - ordered[0] + 1:
        warnings.append("abstract_index_has_gaps")
    if any(len(words) > 1 for words in positions.values()):
        warnings.append("abstract_index_has_collisions")
    text = " ".join(" ".join(positions[index]) for index in ordered).strip()
    return (text or None), warnings


def _relation_items(values: Any) -> list[dict[str, str]]:
    """Return compact relation objects with their provider provenance."""

    if not isinstance(values, list):
        return []
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in values:
        identifier = _normalise_openalex_id(value)
        if identifier and identifier not in seen:
            items.append({"id": identifier, "relation_source": "openalex"})
            seen.add(identifier)
    return items


def _normalise_response(response: Any) -> TransportResponse:
    """Accept the documented response and a two-item tuple test double."""

    if isinstance(response, TransportResponse):
        return response
    if isinstance(response, tuple):
        if len(response) == 2:
            return TransportResponse(int(response[0]), response[1])
    raise ProviderError("openalex", "network", "transport returned an unsupported response")


def _default_transport(method: str, url: str, headers: Mapping[str, str], timeout: float) -> TransportResponse:
    request = Request(url, method=method, headers=dict(headers))
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310: configured API URL
            return TransportResponse(
                int(getattr(response, "status", response.getcode())),
                response.read(),
                dict(response.headers.items()),
            )
    except HTTPError as exc:
        # Preserve the body for the caller's status/error classification, but
        # never include the URL in the resulting error message.
        try:
            body = exc.read()
        except Exception:
            body = b""
        return TransportResponse(exc.code, body, dict(exc.headers.items()) if exc.headers else {})
    except (TimeoutError, URLError, OSError) as exc:
        raise ProviderError("openalex", "network", f"request failed: {type(exc).__name__}") from exc


class OpenAlexProvider:
    """Search OpenAlex works and convert records to PaperDoc v1 objects."""

    source = "openalex"

    def __init__(
        self,
        config: Any,
        *,
        transport: Transport | Any | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = _normalise_base_url(
            _config_value(config, "base_url", "OPENALEX_BASE_URL", "openalex_base_url")
        )
        self.api_key = _config_value(
            config, "api_key", "OPENALEX_API_KEY", "openalex_api_key"
        )
        self.api_key = str(self.api_key).strip() if self.api_key is not None else ""
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ProviderError(self.source, "config", "timeout must be positive")
        self.transport = transport or _default_transport

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> "OpenAlexProvider":
        """Build a provider from ``load_config`` or a provider settings mapping."""

        return cls(config, **kwargs)

    def _invoke_transport(self, url: str) -> TransportResponse:
        headers = {"Accept": "application/json", "User-Agent": "spar-p1/openalex"}
        transport = self.transport
        try:
            if not callable(transport):
                raise ProviderError(self.source, "config", "transport must be callable")
            response = transport("GET", url, headers, self.timeout)
            return _normalise_response(response)
        except ProviderError:
            raise
        except HTTPError as exc:
            code = "auth" if exc.code in {401, 403} else "rate" if exc.code == 429 else "network"
            raise ProviderError(
                self.source,
                code,
                f"HTTP status {exc.code}",
                status_code=exc.code,
                retryable=exc.code >= 500 or exc.code == 429,
            ) from exc
        except TimeoutError as exc:
            raise ProviderError(self.source, "timeout", "request timed out") from exc
        except (URLError, OSError) as exc:
            raise ProviderError(self.source, "network", f"request failed: {type(exc).__name__}") from exc
        except Exception as exc:
            raise ProviderError(self.source, "network", f"transport failed: {type(exc).__name__}") from exc

    def search(
        self,
        query: str,
        *,
        page_size: int = 10,
        cursor: str | None = None,
        per_page: int | None = None,
        page: int | None = None,
        **_: Any,
    ) -> ProviderResult:
        """Search ``/works`` and return a structured ProviderResult."""

        if not isinstance(query, str) or not query.strip():
            raise ProviderError(self.source, "config", "query must be a non-empty string")
        if per_page is None:
            per_page = page_size
        if not isinstance(per_page, int) or isinstance(per_page, bool) or not 1 <= per_page <= 200:
            raise ProviderError(self.source, "config", "per_page must be an integer between 1 and 200")
        if cursor not in (None, ""):
            try:
                cursor_page = int(cursor)
            except (TypeError, ValueError) as exc:
                raise ProviderError(self.source, "config", "cursor must be a numeric page") from exc
            if cursor_page < 1:
                raise ProviderError(self.source, "config", "cursor page must be >= 1")
            page = cursor_page
        if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page < 1):
            raise ProviderError(self.source, "config", "page must be a positive integer")

        params: dict[str, Any] = {"search": query.strip(), "per_page": per_page}
        if page is not None:
            params["page"] = page
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{self.base_url}/works?{urlencode(params)}"
        response = self._invoke_transport(url)
        if not 200 <= response.status < 300:
            code = "auth" if response.status in {401, 403} else "rate" if response.status == 429 else "network"
            raise ProviderError(
                self.source,
                code,
                f"HTTP status {response.status}",
                status_code=response.status,
                retryable=response.status >= 500 or response.status == 429,
            )

        try:
            raw = response.body.decode("utf-8") if isinstance(response.body, bytes) else response.body
            payload = json.loads(raw)
        except (UnicodeDecodeError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderError(self.source, "parse", "response is not valid JSON", status_code=response.status) from exc
        if not isinstance(payload, Mapping):
            raise ProviderError(self.source, "parse", "response JSON must be an object", status_code=response.status)
        if payload.get("error") or payload.get("errors"):
            detail = payload.get("error") or payload.get("errors")
            raise ProviderError(self.source, "business", "OpenAlex returned an API error", status_code=response.status)
        if "code" in payload and payload.get("code") not in (None, 0, "0"):
            raise ProviderError(self.source, "business", "OpenAlex returned a non-zero code", status_code=response.status)
        results = payload.get("results")
        if not isinstance(results, list):
            raise ProviderError(self.source, "parse", "response.results must be an array", status_code=response.status)
        if not results:
            raise ProviderError(self.source, "empty", "OpenAlex returned no works", status_code=response.status)

        retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        query_id = "q_" + hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:16]
        papers: list[dict[str, Any]] = []
        for index, record in enumerate(results):
            if not isinstance(record, Mapping):
                raise ProviderError(self.source, "parse", f"results[{index}] must be an object", status_code=response.status)
            paper = self._to_paper_doc(record, query_id=query_id, page=page or 1, retrieved_at=retrieved_at)
            validate_paper_doc(paper)
            papers.append(paper)
        meta = payload.get("meta")
        meta = meta if isinstance(meta, Mapping) else {}
        total = meta.get("count")
        total = total if isinstance(total, int) and total >= 0 else len(papers)
        return ProviderResult(
            self.source,
            "search",
            papers,
            total=total,
            provenance={
                "endpoint": f"{self.base_url}/works",
                "query_id": query_id,
                "page": page or 1,
                "per_page": per_page,
            },
        )

    def search_papers(self, query: str, *, per_page: int = 10, page: int | None = None) -> list[dict[str, Any]]:
        """Compatibility alias for provider runners using ``search_papers``."""

        return self.search(query, per_page=per_page, page=page)

    def read(self, paper_id: str, *, cursor: str | None = None, **_: Any) -> ProviderResult:
        raise ProviderError(self.source, "unsupported", "OpenAlex content read is not configured")

    def relations(
        self,
        paper_id: str,
        *,
        relation: str = "all",
        cursor: str | None = None,
        **_: Any,
    ) -> ProviderResult:
        raise ProviderError(self.source, "unsupported", "OpenAlex relations endpoint is not configured")

    def _to_paper_doc(
        self,
        record: Mapping[str, Any],
        *,
        query_id: str,
        page: int,
        retrieved_at: str,
    ) -> dict[str, Any]:
        openalex_id = _normalise_openalex_id(record.get("id"))
        doi = _normalise_doi(record.get("doi"))
        abstract, abstract_warnings = _reconstruct_abstract(record.get("abstract_inverted_index"))
        if doi:
            paper_id = f"doi:{doi}"
        elif openalex_id:
            paper_id = f"openalex:{openalex_id}"
        else:
            title = str(record.get("title") or "")
            paper_id = "openalex:unknown-" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]

        authors: list[str] = []
        for authorship in record.get("authorships") or []:
            if not isinstance(authorship, Mapping):
                continue
            author = authorship.get("author")
            if isinstance(author, Mapping) and isinstance(author.get("display_name"), str):
                name = author["display_name"].strip()
                if name and name not in authors:
                    authors.append(name)

        location = record.get("primary_location")
        location = location if isinstance(location, Mapping) else {}
        source = location.get("source")
        source = source if isinstance(source, Mapping) else {}
        landing_url = location.get("landing_page_url") or record.get("landing_page_url")
        pdf_url = location.get("pdf_url")
        oa = record.get("open_access")
        oa = oa if isinstance(oa, Mapping) else {}
        best_location = record.get("best_oa_location")
        best_location = best_location if isinstance(best_location, Mapping) else {}
        oa_url = oa.get("oa_url") or best_location.get("pdf_url") or best_location.get("landing_page_url")
        if not isinstance(landing_url, str):
            landing_url = None
        if not isinstance(pdf_url, str):
            pdf_url = None
        if not isinstance(oa_url, str):
            oa_url = None

        year = record.get("publication_year")
        if not isinstance(year, int):
            date = record.get("publication_date")
            try:
                year = int(str(date)[:4]) if date else None
            except (TypeError, ValueError):
                year = None
        fields: list[str] = []
        for group in (record.get("topics"), record.get("concepts")):
            for item in group or []:
                if isinstance(item, Mapping) and isinstance(item.get("display_name"), str):
                    field = item["display_name"].strip()
                    if field and field not in fields:
                        fields.append(field)

        full_text_status = "abstract" if abstract else "metadata"
        warnings = list(abstract_warnings)
        return {
            "schema_version": "paperdoc.v1",
            "paper_id": paper_id,
            "identifiers": {
                "doi": doi,
                "arxiv_id": None,
                "s2_id": None,
                "openalex_id": openalex_id,
                "pmid": None,
                "pmcid": None,
                "sciverse_doc_id": None,
                "unique_id": None,
            },
            "bibliography": {
                "title": str(record.get("title") or ""),
                "authors": authors,
                "year": year,
                "venue": source.get("display_name") if isinstance(source.get("display_name"), str) else None,
                "abstract": abstract,
                "fields": fields,
            },
            "access": {
                "landing_url": landing_url,
                "pdf_url": pdf_url,
                "oa_url": oa_url,
                "full_text_status": full_text_status,
                "content_type": "abstract" if abstract else "metadata",
            },
            "content": {
                "content_ref": None,
                "chunks": [],
                "sections": [],
                "char_count": len(abstract or ""),
            },
            "relations": {
                "references": _relation_items(record.get("referenced_works")),
                "citations": [],
                "related_works": _relation_items(record.get("related_works")),
            },
            "scores": {
                "retrieval": record.get("relevance_score") if isinstance(record.get("relevance_score"), (int, float)) else None,
                "relevance": None,
                "constraint": None,
                "quality": None,
                "evidence": None,
                "citation": None,
                "novelty": None,
                "final": None,
                "confidence": None,
            },
            "provenance": {
                "sources": [self.source],
                "query_id": query_id,
                "subquery_id": None,
                "iteration": 0,
                "parent_node_id": None,
                "endpoints": [f"{self.base_url}/works"],
                "retrieved_at": retrieved_at,
                "pages": [page],
                "reconciliation": {
                    "complete": True,
                    "expected": 1,
                    "received": 1,
                },
                "warnings": warnings,
            },
            "evidence_refs": [],
            "status": {
                "hard_constraints_pass": None,
                "evidence_status": full_text_status,
                "provider_errors": [],
            },
        }


__all__ = ["OpenAlexProvider", "ProviderError", "TransportResponse"]
