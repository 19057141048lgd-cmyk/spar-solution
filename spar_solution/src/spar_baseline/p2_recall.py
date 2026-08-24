"""P2 的来源路由和有界并发召回。

本模块只处理 QueryPlan 节点到 Provider 的结构化执行，不负责查询生成、去重
或相关性打分。所有 Provider 错误都保存在 ``source_errors``，不会被转换为低
相关论文。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from time import perf_counter
import inspect
import re
from typing import Any, Iterable, Mapping

from .providers.base import ProviderError, ProviderResult
from .paperdoc import validate_paper_doc


_SECRET_RE = re.compile(r"(?i)(authorization\s*:\s*bearer|bearer|api[_-]?key|access[_-]?key|token|password|secret|email|mailto)\s*[:=]?\s*([^\s,;]+)")


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return _SECRET_RE.sub(lambda match: f"{match.group(1)}=***", str(value)) if isinstance(value, str) else value


def _value(item: Any, *names: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if name in item:
                return item[name]
    else:
        for name in names:
            value = getattr(item, name, None)
            if value is not None:
                return value
    return default


def _source_name(provider: Any) -> str:
    return str(getattr(provider, "name", None) or getattr(provider, "source", None) or provider.__class__.__name__).strip().casefold()


def _safe_error(source: str, error: Exception, *, code: str | None = None) -> dict[str, Any]:
    if isinstance(error, ProviderError):
        result = error.to_dict()
        result["source"] = source
        result["message"] = _redact(str(result.get("message") or "provider error"))
        result["details"] = _redact(result.get("details") or {})
        return result
    return {
        "source": source,
        "code": code or str(getattr(error, "code", "unknown")),
        "message": _redact(str(getattr(error, "message", str(error))) or type(error).__name__),
        "retryable": bool(getattr(error, "retryable", False)),
        "status_code": getattr(error, "status_code", None),
        "details": _redact(dict(getattr(error, "details", {}) or {})),
    }


def _nodes(plan: Any) -> list[Any]:
    raw = _value(plan, "subqueries", "nodes", "queries", default=None)
    if raw is None and isinstance(plan, (list, tuple)):
        raw = plan
    if raw is None:
        raw = [plan]
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    return [item for item in raw if item is not None]


def _query(node: Any) -> str:
    query = _value(node, "query", "text", "query_text", "query_string", default="")
    return str(query or "").strip()


def _node_id(node: Any, index: int) -> str:
    value = _value(node, "subquery_id", "node_id", "id", default=None)
    return str(value or f"subquery_{index + 1}")


def _requested_sources(node: Any) -> list[str] | None:
    value = _value(node, "source_capabilities", "sources", "source", "providers", default=None)
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [value.casefold()]
    if isinstance(value, Iterable):
        return [str(item).casefold() for item in value if str(item).strip()]
    return None


@dataclass(frozen=True)
class RouteDecision:
    """一次可执行的子查询-来源映射。"""

    subquery_id: str
    query: str
    source: str
    provider: Any
    node_index: int
    iteration: int = 0
    parent_node_id: str | None = None
    reason: str = "capability_match"


class SourceRouter:
    """按 QueryPlan 节点的来源约束选择 Provider。

    ``providers`` 可为 mapping 或 provider 可迭代对象。没有显式来源约束时，
    选择所有拥有 ``search`` 方法且没有被标记 unavailable 的 Provider。
    """

    def __init__(self, providers: Mapping[str, Any] | Iterable[Any], *, capabilities: Mapping[str, Any] | None = None) -> None:
        if isinstance(providers, Mapping):
            items = list(providers.items())
        else:
            items = [(_source_name(provider), provider) for provider in providers]
        self.providers = {str(name).casefold(): provider for name, provider in items}
        self.capabilities = {str(name).casefold(): value for name, value in (capabilities or {}).items()}

    def route(self, node: Any) -> list[RouteDecision]:
        query = _query(node)
        node_index = int(_value(node, "node_index", default=0) or 0)
        subquery_id = str(_value(node, "subquery_id", "node_id", "id", default="subquery_1"))
        iteration = int(_value(node, "iteration", default=0) or 0)
        parent = _value(node, "parent_id", "parent_node_id", default=None)
        parent_node_id = str(parent) if parent is not None else None
        requested = _requested_sources(node)
        names = requested if requested is not None else list(self.providers)
        decisions: list[RouteDecision] = []
        for source in names:
            provider = self.providers.get(source.casefold())
            if provider is None or not callable(getattr(provider, "search", None)):
                continue
            if getattr(provider, "library_status", None) == "unavailable":
                continue
            capability = self.capabilities.get(source.casefold())
            if capability is False:
                continue
            decisions.append(RouteDecision(subquery_id, query, source.casefold(), provider, node_index, iteration, parent_node_id))
        return decisions

    def route_plan(self, plan: Any) -> list[RouteDecision]:
        decisions: list[RouteDecision] = []
        for index, node in enumerate(_nodes(plan)):
            # node_index is used only for stable replay; preserve an explicit index.
            if isinstance(node, Mapping) and "node_index" not in node:
                node = dict(node)
                node["node_index"] = index
            routed = self.route(node)
            decisions.extend(
                replace(item, node_index=index) if _value(node, "node_index", default=None) is None else item
                for item in routed
            )
        return decisions

    select_sources = route_plan


def _call_search(provider: Any, query: str, page_size: int) -> Any:
    method = getattr(provider, "search", None)
    if not callable(method):
        raise ProviderError(_source_name(provider), "config", "provider has no search method")
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "page_size" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return method(query, page_size=page_size)
    if "per_page" in parameters:
        return method(query, per_page=page_size)
    return method(query)


def _result_records(provider: Any, result: Any) -> list[dict[str, Any]]:
    source = _source_name(provider)
    if not isinstance(result, ProviderResult):
        raise ProviderError(source, "parse", "provider search must return ProviderResult")
    if result.source.casefold() != source or result.operation != "search":
        raise ProviderError(source, "parse", "provider result source/operation mismatch")
    records = list(result.records)
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ProviderError(source, "parse", f"records[{index}] must be an object")
        try:
            validate_paper_doc(dict(record))
        except Exception as exc:
            raise ProviderError(source, "parse", f"records[{index}] is not a valid PaperDoc") from exc
    return records


@dataclass
class RecallResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    source_errors: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    routes: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def results(self) -> list[dict[str, Any]]:
        return self.records

    @property
    def papers(self) -> list[dict[str, Any]]:
        return self.records

    @property
    def ok(self) -> bool:
        return bool(self.records) or not self.source_errors

    def to_dict(self) -> dict[str, Any]:
        return {"records": self.records, "source_errors": self.source_errors, "calls": self.calls, "routes": self.routes, "stats": self.stats}

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class RecallRunner:
    """并行执行 SourceRouter 产生的检索调用，并保持确定性输出顺序。"""

    def __init__(self, router: SourceRouter, *, max_workers: int = 4, page_size: int = 10, max_calls: int | None = None) -> None:
        if max_workers < 1 or page_size < 1:
            raise ValueError("max_workers and page_size must be positive")
        self.router = router
        self.max_workers = max_workers
        self.page_size = page_size
        self.max_calls = max_calls

    def run(self, plan: Any, *, iteration: int = 0, max_calls: int | None = None) -> RecallResult:
        routes = self.router.route_plan(plan)
        budget = self.max_calls if max_calls is None else max_calls
        if budget is not None:
            if budget < 0:
                raise ValueError("max_calls must be non-negative")
            routes = routes[:budget]
        route_dicts = [{"subquery_id": r.subquery_id, "query": r.query, "source": r.source, "node_index": r.node_index, "iteration": r.iteration, "parent_node_id": r.parent_node_id, "reason": r.reason} for r in routes]
        output: dict[int, tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]] = {}

        def execute(index: int, route: RouteDecision) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
            started = perf_counter()
            try:
                records = _result_records(route.provider, _call_search(route.provider, route.query, self.page_size))
                prepared: list[dict[str, Any]] = []
                for record in records:
                    item = dict(record)
                    provenance = item.setdefault("provenance", {})
                    provenance["subquery_id"] = route.subquery_id
                    provenance["iteration"] = route.iteration if route.iteration else iteration
                    provenance["parent_node_id"] = route.parent_node_id
                    prepared.append(item)
                call_iteration = route.iteration if route.iteration else iteration
                call = {"subquery_id": route.subquery_id, "source": route.source, "query": route.query, "records": len(prepared), "ok": True, "latency_ms": round((perf_counter() - started) * 1000, 3), "iteration": call_iteration}
                return index, prepared, None, call
            except Exception as exc:
                error = _safe_error(route.source, exc)
                call_iteration = route.iteration if route.iteration else iteration
                call = {"subquery_id": route.subquery_id, "source": route.source, "query": route.query, "records": 0, "ok": False, "latency_ms": round((perf_counter() - started) * 1000, 3), "iteration": call_iteration}
                return index, [], error, call

        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(routes)))) if routes else _NullExecutor() as executor:
            futures = [executor.submit(execute, index, route) for index, route in enumerate(routes)]
            for future in as_completed(futures):
                index, records, error, call = future.result()
                output[index] = (records, error, call)
        result = RecallResult()
        for index in range(len(routes)):
            records, error, call = output[index]
            result.records.extend(records)
            result.calls.append(call)
            if error:
                result.source_errors.append(error)
        result.routes = route_dicts
        result.stats = {"route_count": len(routes), "successful_calls": sum(1 for call in result.calls if call["ok"]), "api_calls": len(routes), "records": len(result.records), "source_errors": len(result.source_errors), "iteration": iteration, "max_workers": self.max_workers}
        return result

    recall = run


class _NullExecutor:
    def __enter__(self) -> "_NullExecutor":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        from concurrent.futures import Future
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


__all__ = ["RecallResult", "RecallRunner", "RouteDecision", "SourceRouter"]
