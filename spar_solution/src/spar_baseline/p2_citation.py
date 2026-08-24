"""P2 的有界引用扩展与关闭引用消融。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from time import perf_counter
import inspect
import re
from typing import Any, Iterable, Mapping

from .paperdoc import REQUIRED_TOP_LEVEL, validate_paper_doc
from .providers.base import ProviderError, ProviderResult


_SECRET_RE = re.compile(r"(?i)(authorization\s*:\s*bearer|bearer|api[_-]?key|access[_-]?key|token|password|secret|email|mailto)\s*[:=]?\s*([^\s,;]+)")


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return _SECRET_RE.sub(lambda match: f"{match.group(1)}=***", str(value)) if isinstance(value, str) else value


def _source_name(provider: Any) -> str:
    return str(getattr(provider, "name", None) or getattr(provider, "source", None) or provider.__class__.__name__).strip().casefold()


def _error(source: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ProviderError):
        result = exc.to_dict()
        result["source"] = source
        result["message"] = _redact(result.get("message") or "provider error")
        result["details"] = _redact(result.get("details") or {})
        return result
    return {
        "source": source,
        "code": str(getattr(exc, "code", "unknown")),
        "message": _redact(str(getattr(exc, "message", str(exc))) or type(exc).__name__),
        "retryable": bool(getattr(exc, "retryable", False)),
        "status_code": getattr(exc, "status_code", None),
        "details": _redact(dict(getattr(exc, "details", {}) or {})),
    }


def _provider_sources(seed: Mapping[str, Any]) -> list[str]:
    sources = seed.get("provenance", {}).get("sources", [])
    return [str(source).casefold() for source in sources if str(source).casefold() != "merged"]


def _relation_call(provider: Any, paper_id: str, relation: str, page_size: int) -> ProviderResult:
    method = getattr(provider, "relations", None)
    if not callable(method):
        raise ProviderError(_source_name(provider), "config", "provider has no relations method")
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs: dict[str, Any] = {}
    if "relation" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        kwargs["relation"] = relation
    if "page_size" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        kwargs["page_size"] = page_size
    result = method(paper_id, **kwargs)
    source = _source_name(provider)
    if not isinstance(result, ProviderResult):
        raise ProviderError(source, "parse", "provider relations must return ProviderResult")
    if result.source.casefold() != source or result.operation != "relations":
        raise ProviderError(source, "parse", "provider result source/operation mismatch")
    return result


def _child_id(record: Mapping[str, Any]) -> str | None:
    for key in ("paper_id", "child_paper_id", "id", "unique_id", "arxiv_id", "doi"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    identifiers = record.get("identifiers")
    if isinstance(identifiers, Mapping):
        for key in ("doi", "arxiv_id", "s2_id", "openalex_id", "unique_id"):
            value = identifiers.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _relation_type(record: Mapping[str, Any], fallback: str) -> str:
    value = record.get("relation_type", record.get("relation", fallback))
    return str(value or fallback).casefold()


def _nested_paper(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("paper_doc", "paper", "full_doc"):
        nested = record.get(key)
        if isinstance(nested, Mapping):
            return nested
    return record if REQUIRED_TOP_LEVEL.issubset(record.keys()) else None


@dataclass
class CitationExpansionResult:
    papers: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    source_errors: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def records(self) -> list[dict[str, Any]]:
        return self.papers

    @property
    def expanded(self) -> list[dict[str, Any]]:
        return self.papers

    @property
    def relations(self) -> list[dict[str, Any]]:
        return self.edges

    def to_dict(self) -> dict[str, Any]:
        return {"papers": self.papers, "edges": self.edges, "source_errors": self.source_errors, "calls": self.calls, "stats": self.stats}

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class CitationExpander:
    """对通过门控的种子调用 Provider ``relations()``。

    种子必须明确通过硬约束、含摘要且 ``scores.relevance`` 达阈值。扩展深度
    固定不超过 1；关闭引用时返回显式 ablation 状态且不调用 Provider。
    """

    def __init__(
        self,
        providers: Mapping[str, Any] | Iterable[Any],
        *,
        enabled: bool = True,
        relevance_threshold: float = 0.6,
        max_depth: int = 1,
        max_seeds: int = 5,
        page_size: int = 10,
        max_workers: int = 4,
        relation: str = "all",
        max_api_calls: int | None = None,
    ) -> None:
        if max_depth not in {0, 1}:
            raise ValueError("P2 max_depth must be 0 or 1")
        if max_seeds < 0 or page_size < 1 or max_workers < 1:
            raise ValueError("max_seeds must be non-negative; page_size and max_workers must be positive")
        if not 0 <= relevance_threshold <= 1:
            raise ValueError("relevance_threshold must be between 0 and 1")
        if max_api_calls is not None and (isinstance(max_api_calls, bool) or not isinstance(max_api_calls, int) or max_api_calls < 0):
            raise ValueError("max_api_calls must be a non-negative integer or null")
        if isinstance(providers, Mapping):
            self.providers = {str(name).casefold(): provider for name, provider in providers.items()}
        else:
            self.providers = {_source_name(provider): provider for provider in providers}
        self.enabled = enabled
        self.relevance_threshold = relevance_threshold
        self.max_depth = max_depth
        self.max_seeds = max_seeds
        self.page_size = page_size
        self.max_workers = max_workers
        self.relation = relation
        self.max_api_calls = max_api_calls

    def eligible(self, seed: Mapping[str, Any]) -> bool:
        status = seed.get("status", {})
        bibliography = seed.get("bibliography", {})
        scores = seed.get("scores", {})
        relevance = scores.get("relevance")
        return (
            status.get("hard_constraints_pass") is True
            and isinstance(bibliography.get("abstract"), str)
            and bool(bibliography["abstract"].strip())
            and isinstance(relevance, (int, float))
            and not isinstance(relevance, bool)
            and relevance >= self.relevance_threshold
        )

    select_seeds = lambda self, papers: [paper for paper in papers if self.eligible(paper)][: self.max_seeds]

    def _provider_for(self, seed: Mapping[str, Any]) -> tuple[str, Any] | None:
        for source in _provider_sources(seed):
            if source in self.providers:
                return source, self.providers[source]
        return None

    def expand(self, papers: Iterable[dict[str, Any]], *, iteration: int = 0, enabled: bool | None = None) -> CitationExpansionResult:
        active = self.enabled if enabled is None else enabled
        all_papers = list(papers)
        if not active or self.max_depth == 0:
            return CitationExpansionResult(stats={"enabled": False, "ablation": "citation_disabled", "eligible_seeds": 0, "api_calls": 0, "papers": 0, "edges": 0, "source_errors": 0, "max_depth": self.max_depth})

        seeds = self.select_seeds(all_papers)
        tasks: list[tuple[int, dict[str, Any], str, Any, str, int]] = []
        immediate_errors: list[dict[str, Any]] = []
        reserved_calls = 0
        budget_skipped = 0
        for index, seed in enumerate(seeds):
            selected = self._provider_for(seed)
            if selected is None:
                sources = _provider_sources(seed)
                immediate_errors.append({"source": sources[0] if sources else "unknown", "code": "config", "message": "no relations provider for eligible seed", "retryable": False, "status_code": None, "details": {"paper_id": seed.get("paper_id")}})
                continue
            source, provider = selected
            provider_paper_id = str((seed.get("identifiers") or {}).get("openalex_id") or seed["paper_id"]) if source == "openalex" else str(seed["paper_id"])
            cost_fn = getattr(provider, "relation_api_cost", None)
            estimated_calls = int(cost_fn(provider_paper_id, self.relation)) if callable(cost_fn) else 1
            if self.max_api_calls is not None and reserved_calls + estimated_calls > self.max_api_calls:
                budget_skipped += 1
                continue
            reserved_calls += estimated_calls
            tasks.append((index, seed, source, provider, provider_paper_id, estimated_calls))

        completed: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]] = {}

        def execute(index: int, seed: dict[str, Any], source: str, provider: Any, provider_paper_id: str, estimated_calls: int) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
            started = perf_counter()
            parent_id = str(seed["paper_id"])
            try:
                result = _relation_call(provider, provider_paper_id, self.relation, self.page_size)
                expanded: list[dict[str, Any]] = []
                edges: list[dict[str, Any]] = []
                for record in result.records:
                    child_id = _child_id(record)
                    if not child_id:
                        raise ProviderError(source, "parse", "relation record has no stable child identifier")
                    relation_type = _relation_type(record, self.relation)
                    edges.append({"parent_paper_id": parent_id, "child_paper_id": child_id, "relation_type": relation_type, "source": source, "depth": 1})
                    nested = _nested_paper(record)
                    if nested is not None:
                        child = dict(nested)
                        provenance = child.setdefault("provenance", {})
                        provenance["parent_node_id"] = parent_id
                        provenance["iteration"] = iteration
                        provenance.setdefault("citation_depth", 1)
                        child.setdefault("relations", {}).setdefault("related_works", [])
                        validate_paper_doc(child)
                        expanded.append(child)
                api_calls = int(result.provenance.get("api_calls") or estimated_calls)
                call = {"paper_id": parent_id, "source": source, "relation": self.relation, "records": len(result.records), "ok": True, "api_calls": api_calls, "latency_ms": round((perf_counter() - started) * 1000, 3), "depth": 1}
                return index, expanded, edges, None, call
            except Exception as exc:
                call = {"paper_id": parent_id, "source": source, "relation": self.relation, "records": 0, "ok": False, "api_calls": estimated_calls, "latency_ms": round((perf_counter() - started) * 1000, 3), "depth": 1}
                return index, [], [], _error(source, exc), call

        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(tasks)))) as executor:
            futures = [executor.submit(execute, *task) for task in tasks]
            for future in as_completed(futures):
                index, expanded, edges, error, call = future.result()
                completed[index] = (expanded, edges, error, call)

        output = CitationExpansionResult(source_errors=immediate_errors)
        seen_papers: set[str] = set()
        seen_edges: set[tuple[str, str, str, str]] = set()
        for index, _, _, _, _, _ in tasks:
            expanded, edges, error, call = completed[index]
            output.calls.append(call)
            if error:
                output.source_errors.append(error)
            for paper in expanded:
                paper_id = str(paper["paper_id"])
                if paper_id not in seen_papers:
                    seen_papers.add(paper_id)
                    output.papers.append(paper)
            for edge in edges:
                key = (edge["parent_paper_id"], edge["child_paper_id"], edge["relation_type"], edge["source"])
                if key not in seen_edges:
                    seen_edges.add(key)
                    output.edges.append(edge)
        output.stats = {"enabled": True, "ablation": None, "eligible_seeds": len(seeds), "api_calls": sum(int(call.get("api_calls") or 1) for call in output.calls), "papers": len(output.papers), "edges": len(output.edges), "source_errors": len(output.source_errors), "max_depth": self.max_depth, "iteration": iteration, "budget_skipped": budget_skipped}
        return output

    run = expand


__all__ = ["CitationExpander", "CitationExpansionResult"]
