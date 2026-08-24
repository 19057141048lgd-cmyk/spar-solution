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
import re
from time import perf_counter
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
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

    @staticmethod
    def relation_api_cost(paper_id: str, relation: str) -> int:
        """Return the OpenAlex HTTP-call cost before relation expansion."""

        value = str(paper_id).strip()
        if value.casefold().startswith("openalex:"):
            value = value.split(":", 1)[1]
        base = {"citations": 1, "references": 2, "all": 3}.get(relation)
        if base is None:
            raise ValueError("relation must be references, citations or all")
        return base if re.fullmatch(r"W\d+", value, flags=re.IGNORECASE) else base + 1

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

    def _request_json(self, url: str) -> tuple[Mapping[str, Any], float]:
        started = perf_counter()
        response = self._invoke_transport(url)
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
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
            raise ProviderError(self.source, "business", "OpenAlex returned an API error", status_code=response.status)
        if "code" in payload and payload.get("code") not in (None, 0, "0"):
            raise ProviderError(self.source, "business", "OpenAlex returned a non-zero code", status_code=response.status)
        return payload, elapsed_ms

    def _url(self, path: str, params: Mapping[str, Any] | None = None) -> str:
        values = dict(params or {})
        if self.api_key:
            values["api_key"] = self.api_key
        query = urlencode(values)
        return f"{self.base_url}{path}" + (f"?{query}" if query else "")

    def _tracked_request(
        self, url: str, operation: str, calls: list[dict[str, Any]]
    ) -> Mapping[str, Any]:
        started = perf_counter()
        try:
            payload, latency_ms = self._request_json(url)
        except ProviderError as exc:
            calls.append(
                {
                    "operation": operation,
                    "latency_ms": round((perf_counter() - started) * 1000, 3),
                    "ok": False,
                    "error_code": exc.code,
                }
            )
            raise
        calls.append({"operation": operation, "latency_ms": latency_ms, "ok": True})
        return payload

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

        for name, value in (("start_year", start_year), ("end_year", end_year)):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or not 1900 <= value <= 2200
            ):
                raise ProviderError(self.source, "config", f"{name} must be an integer between 1900 and 2200")
        if start_year is not None and end_year is not None and start_year > end_year:
            raise ProviderError(self.source, "config", "start_year must not exceed end_year")

        params: dict[str, Any] = {"search": query.strip(), "per_page": per_page}
        if page is not None:
            params["page"] = page
        if start_year is not None and end_year is not None:
            params["filter"] = f"publication_year:{start_year}-{end_year}"
        elif start_year is not None:
            params["filter"] = f"publication_year:>={start_year}"
        elif end_year is not None:
            params["filter"] = f"publication_year:<={end_year}"
        payload, _ = self._request_json(self._url("/works", params))
        results = payload.get("results")
        if not isinstance(results, list):
            raise ProviderError(self.source, "parse", "response.results must be an array")
        retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        query_id = "q_" + hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:16]
        papers: list[dict[str, Any]] = []
        for index, record in enumerate(results):
            if not isinstance(record, Mapping):
                raise ProviderError(self.source, "parse", f"results[{index}] must be an object")
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
            warnings=["no_results"] if not papers else [],
            provenance={
                "endpoint": f"{self.base_url}/works",
                "query_id": query_id,
                "page": page or 1,
                "per_page": per_page,
                "filter": params.get("filter"),
                "no_results": not papers,
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
        page_size: int = 10,
        limit: int | None = None,
        **_: Any,
    ) -> ProviderResult:
        """Return cited and/or citing works as PaperDoc records."""

        if relation not in {"references", "citations", "all"}:
            raise ProviderError(self.source, "config", "relation must be references, citations or all")
        if cursor not in (None, ""):
            raise ProviderError(self.source, "config", "relations cursor is not supported")
        requested = page_size if limit is None else limit
        if not isinstance(requested, int) or isinstance(requested, bool) or requested < 1:
            raise ProviderError(self.source, "config", "relation limit must be a positive integer")
        effective_limit = min(requested, 50)
        warnings = ["relation_limit_truncated_to_50"] if requested > 50 else []
        calls: list[dict[str, Any]] = []
        source_errors: list[dict[str, Any]] = []

        seed_id = self._resolve_work_id(paper_id, calls)
        retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        query_id = "rel_" + hashlib.sha256(f"{seed_id}:{relation}".encode("utf-8")).hexdigest()[:16]
        records: list[dict[str, Any]] = []
        branch_successes = 0
        branch_errors: list[ProviderError] = []

        def fetch(kind: str) -> None:
            nonlocal branch_successes
            try:
                relation_records = self._relation_records(seed_id, kind, effective_limit, calls)
                branch_successes += 1
                for item in relation_records:
                    paper = self._to_paper_doc(item, query_id=query_id, page=1, retrieved_at=retrieved_at)
                    paper["relation_type"] = kind
                    paper["provenance"]["parent_node_id"] = f"openalex:{seed_id}"
                    paper["provenance"]["relation_source"] = f"openalex/{kind}"
                    validate_paper_doc(paper)
                    records.append(paper)
            except ProviderError as exc:
                if relation != "all":
                    raise
                warnings.append(f"{kind}_failed:{exc.code}")
                branch_errors.append(exc)
                source_errors.append(exc.to_dict())

        if relation in {"citations", "all"}:
            fetch("citations")
        if relation in {"references", "all"}:
            fetch("references")

        if branch_errors and branch_successes == 0:
            first = branch_errors[0]
            raise ProviderError(
                self.source,
                first.code,
                "all requested OpenAlex relation branches failed",
                retryable=any(error.retryable for error in branch_errors),
                status_code=first.status_code,
                details={"source_errors": source_errors},
            )
        if not records and not source_errors:
            warnings.append("no_results")
        return ProviderResult(
            self.source,
            "relations",
            records,
            total=len(records),
            warnings=warnings,
            provenance={
                "endpoint": f"{self.base_url}/works",
                "seed_openalex_id": seed_id,
                "relation": relation,
                "api_calls": len(calls),
                "calls": calls,
                "latency_ms": round(sum(call["latency_ms"] for call in calls), 3),
                "source_errors": source_errors,
                "no_results": not records,
            },
        )

    def _resolve_work_id(self, paper_id: str, calls: list[dict[str, Any]]) -> str:
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ProviderError(self.source, "config", "paper_id must be a non-empty string")
        value = paper_id.strip()
        if value.casefold().startswith("openalex:"):
            value = value.split(":", 1)[1]
        openalex_id = _normalise_openalex_id(value)
        if openalex_id and re.fullmatch(r"W\d+", openalex_id, flags=re.IGNORECASE):
            return openalex_id.upper()

        doi = _normalise_doi(value)
        if not doi or not doi.startswith("10."):
            raise ProviderError(self.source, "config", "paper_id must be an OpenAlex work ID or DOI")
        path = "/works/" + quote(f"https://doi.org/{doi}", safe=":/")
        payload = self._tracked_request(self._url(path, {"select": "id"}), "resolve_doi", calls)
        resolved = _normalise_openalex_id(payload.get("id"))
        if not resolved or not re.fullmatch(r"W\d+", resolved, flags=re.IGNORECASE):
            raise ProviderError(self.source, "parse", "DOI lookup returned no valid OpenAlex work ID")
        return resolved.upper()

    def _relation_records(
        self,
        seed_id: str,
        relation: str,
        limit: int,
        calls: list[dict[str, Any]],
    ) -> list[Mapping[str, Any]]:
        if relation == "citations":
            payload = self._tracked_request(
                self._url("/works", {"filter": f"cites:{seed_id}", "per_page": limit}),
                "citations",
                calls,
            )
            results = payload.get("results")
            if not isinstance(results, list):
                raise ProviderError(self.source, "parse", "response.results must be an array")
            if any(not isinstance(item, Mapping) for item in results):
                raise ProviderError(self.source, "parse", "citation results must contain objects")
            return results

        payload = self._tracked_request(
            self._url(f"/works/{seed_id}", {"select": "id,referenced_works"}),
            "reference_ids",
            calls,
        )
        raw_ids = payload.get("referenced_works")
        if not isinstance(raw_ids, list):
            raise ProviderError(self.source, "parse", "referenced_works must be an array")
        ids = list(
            dict.fromkeys(value for value in (_normalise_openalex_id(item) for item in raw_ids) if value)
        )[:limit]
        if not ids:
            return []
        batch = self._tracked_request(
            self._url("/works", {"filter": f"openalex_id:{'|'.join(ids)}", "per_page": len(ids)}),
            "reference_details",
            calls,
        )
        results = batch.get("results")
        if not isinstance(results, list):
            raise ProviderError(self.source, "parse", "response.results must be an array")
        if any(not isinstance(item, Mapping) for item in results):
            raise ProviderError(self.source, "parse", "reference results must contain objects")
        return results

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
