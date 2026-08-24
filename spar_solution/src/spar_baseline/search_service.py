"""P1 多 Provider 检索编排。

Provider 只负责调用自己的 API；本模块负责把不同返回形态收敛为
PaperDoc artifact。这样 OpenAlex 的旧 list 返回和 Bohrium 的
ProviderResult 可以在 P1 共存，后续再统一 Provider 实现。
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import re
import uuid
from typing import Any, Iterable, Mapping, Sequence

from .paperdoc import (
    canonical_paper_key,
    merge_paper_docs,
    validate_paper_doc,
)
from .providers.base import ProviderError, ProviderResult


_SECRET_TEXT_RE = re.compile(
    r"(?i)(authorization\s*:\s*bearer|bearer|api[_-]?key|access[_-]?key|token|password|secret|email|mailto)"
    r"\s*[:=]?\s*([^\s,;]+)"
)


def _safe_error_text(value: Any) -> str:
    text = str(value or "")
    return _SECRET_TEXT_RE.sub(lambda match: f"{match.group(1)}=***", text)


def _safe_details(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_details(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_details(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_details(item) for item in value]
    return _safe_error_text(value) if isinstance(value, str) else value


def _query_id(query: str) -> str:
    return "q_" + hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:16]


def _provider_name(provider: Any) -> str:
    value = getattr(provider, "name", None) or getattr(provider, "source", None)
    return str(value or provider.__class__.__name__).strip().casefold()


def _error_dict(source: str, error: Exception, *, code: str | None = None) -> dict[str, Any]:
    """将不同 Provider 的异常映射为不含密钥的结构化错误。"""

    if isinstance(error, ProviderError):
        result = error.to_dict()
        result.setdefault("source", source)
        result["message"] = _safe_error_text(result.get("message"))
        result["details"] = _safe_details(result.get("details") or {})
        return result
    result = {
        "source": str(getattr(error, "source", source)),
        "code": code or str(getattr(error, "code", "unknown")),
        "message": _safe_error_text(str(getattr(error, "message", str(error))) or type(error).__name__),
        "retryable": bool(getattr(error, "retryable", False)),
        "status_code": getattr(error, "status_code", getattr(error, "status", None)),
        "details": _safe_details(getattr(error, "details", {}) or {}),
    }
    return result


def _call_search(provider: Any, query: str, page_size: int) -> Any:
    """按 provider.search 的实际签名传递 page_size/per_page。"""

    method = getattr(provider, "search", None)
    if not callable(method):
        raise ProviderError(_provider_name(provider), "config", "provider has no search method")
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "page_size" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return method(query, page_size=page_size)
    if "per_page" in parameters:
        return method(query, per_page=page_size)
    return method(query)


def _records_from_result(provider: Any, result: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = _provider_name(provider)
    if isinstance(result, ProviderResult):
        if result.source != source or result.operation != "search":
            raise ProviderError(source, "parse", "provider result source/operation mismatch")
        return list(result.records), {
            "next_cursor": result.next_cursor,
            "total": result.total,
            "warnings": list(result.warnings),
            "provenance": dict(result.provenance),
        }
    if isinstance(result, list):
        return list(result), {"next_cursor": None, "total": len(result), "warnings": [], "provenance": {}}
    if isinstance(result, Mapping):
        records = result.get("records", result.get("data", result.get("results")))
        if not isinstance(records, list):
            raise ProviderError(source, "parse", "provider search result has no records array")
        return list(records), {
            "next_cursor": result.get("next_cursor"),
            "total": result.get("total", len(records)),
            "warnings": list(result.get("warnings") or []),
            "provenance": dict(result.get("provenance") or {}),
        }
    raise ProviderError(source, "parse", "provider returned an unsupported search result")


def _prepare_doc(doc: dict[str, Any], *, query_id: str, source: str) -> dict[str, Any]:
    prepared = deepcopy(doc)
    provenance = prepared.setdefault("provenance", {})
    # 当前搜索调用拥有最终 query_id，不能沿用 provider fixture 的占位值。
    provenance["query_id"] = query_id
    provenance.setdefault("subquery_id", None)
    provenance.setdefault("iteration", 0)
    provenance.setdefault("sources", [source])
    return validate_paper_doc(prepared)


def search(
    query: str,
    providers: Iterable[Any],
    *,
    page_size: int = 10,
    mode: str = "live",
    run_id: str | None = None,
) -> dict[str, Any]:
    """并行编排的 P1 最小版本（按传入顺序调用，保证 artifact 可复现）。"""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(page_size, int) or isinstance(page_size, bool) or not 1 <= page_size <= 200:
        raise ValueError("page_size must be an integer between 1 and 200")
    provider_list = list(providers)
    query = query.strip()
    query_id = _query_id(query)
    source_errors: list[dict[str, Any]] = []
    provider_stats: list[dict[str, Any]] = []
    merged: dict[str, dict[str, Any]] = {}
    input_records = 0
    valid_record_count = 0
    invalid_record_count = 0
    retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    for provider in provider_list:
        source = _provider_name(provider)
        try:
            raw_result = _call_search(provider, query, page_size)
            records, metadata = _records_from_result(provider, raw_result)
            valid_records = 0
            invalid_records = 0
            for record in records:
                input_records += 1
                try:
                    if not isinstance(record, dict):
                        raise ValueError("record must be an object")
                    doc = _prepare_doc(record, query_id=query_id, source=source)
                    result_provenance = metadata.get("provenance") or {}
                    for field in ("execution_status", "library_status"):
                        if field in result_provenance:
                            doc["provenance"][field] = result_provenance[field]
                    key = canonical_paper_key(doc)
                    if key.startswith("ambiguous:"):
                        key = f"{key}|record:{input_records}"
                    if key in merged:
                        merged[key] = merge_paper_docs(merged[key], doc)
                    else:
                        merged[key] = doc
                    valid_records += 1
                    valid_record_count += 1
                except Exception as exc:
                    invalid_records += 1
                    invalid_record_count += 1
                    source_errors.append(_error_dict(source, exc, code="parse"))
            provider_stats.append({
                "source": source,
                "ok": True,
                "records": valid_records,
                "invalid_records": invalid_records,
                "total": metadata.get("total"),
                "next_cursor": metadata.get("next_cursor"),
                "warnings": metadata.get("warnings", []),
            })
        except Exception as exc:
            source_errors.append(_error_dict(source, exc))
            provider_stats.append({"source": source, "ok": False, "records": 0})

    papers = list(merged.values())
    for paper in papers:
        paper["provenance"]["retrieved_at"] = paper["provenance"].get("retrieved_at") or retrieved_at
        paper["status"]["provider_errors"] = [
            error for error in source_errors if error.get("source") in paper["provenance"].get("sources", [])
        ]
        validate_paper_doc(paper)
    return {
        "schema_version": "spar.search.v1",
        "run_id": run_id or "run_" + uuid.uuid4().hex[:16],
        "mode": mode,
        "query": query,
        "query_id": query_id,
        "papers": papers,
        "source_errors": source_errors,
        "stats": {
            "provider_count": len(provider_list),
            "successful_providers": sum(1 for item in provider_stats if item["ok"]),
            "input_records": input_records,
            "valid_records": valid_record_count,
            "invalid_records": invalid_record_count,
            "merged_records": len(papers),
            "dedup_count": max(0, valid_record_count - len(papers)),
            "source_errors": source_errors,
            "providers": provider_stats,
        },
    }


__all__ = ["search"]
