"""DeepSeek 理解/判断层。

DeepSeek 只负责把自然语言问题转换为结构化 ``QueryPlan``，以及对已由
论文 Provider 召回的 PaperDoc 做相关性/约束判断。论文事实不由本模块生成。
网络传输可注入，便于 fixture 测试；默认调用 DeepSeek Chat Completions API。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .config import load_config, redact_url
from .paperdoc import validate_paper_doc
from .providers.base import ProviderError
from .query_plan import GAP_CODES, QUERY_PLAN_SCHEMA, QueryPlan, QueryPlanValidationError, validate_query_plan
from .query_planner import QueryPlanner


DEEPSEEK_PLAN_SCHEMA = "deepseek_query_plan.v1"
DEEPSEEK_JUDGEMENT_SCHEMA = "deepseek_judgement.v1"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
SOURCE_CAPABILITIES = {"arxiv", "openalex", "bohrium", "local_library"}
SUBQUERY_KINDS = {"topic", "method", "dataset", "constraint", "comparison", "reference"}
JUDGEMENT_LABELS = {"relevant", "borderline", "irrelevant"}
CONSTRAINT_STATES = {"pass", "unknown", "fail"}


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: bytes | str | Mapping[str, Any]
    headers: Mapping[str, str] = field(default_factory=dict)


Transport = Callable[[str, str, Mapping[str, str], bytes, float], Any]


class DeepSeekSchemaError(ValueError):
    """DeepSeek 返回的数据未满足规定的结构化协议。"""


class DeepSeekCallError(ProviderError):
    """DeepSeek 调用失败；错误消息不得包含 key 或服务端原文。"""

    def __init__(self, code: str, message: str, *, retryable: bool = False, status_code: int | None = None) -> None:
        super().__init__("deepseek", code, message, retryable=retryable, status_code=status_code)


def _default_transport(method: str, url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> TransportResponse:
    request = Request(url, data=body, method=method, headers=dict(headers))
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310: configured DeepSeek endpoint
            return TransportResponse(int(getattr(response, "status", response.getcode())), response.read(), dict(response.headers.items()))
    except HTTPError as exc:
        # 不读取/传播响应体，避免错误响应意外回显凭证或提示词。
        return TransportResponse(exc.code, b"", dict(exc.headers.items()) if exc.headers else {})
    except (TimeoutError, URLError, OSError) as exc:
        raise DeepSeekCallError("network", f"request failed: {type(exc).__name__}", retryable=True) from exc


def _normalise_response(value: Any) -> TransportResponse:
    if isinstance(value, TransportResponse):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return TransportResponse(int(value[0]), value[1])
    raise DeepSeekCallError("network", "transport returned an unsupported response")


def _decode_json(value: Any, *, context: str) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, bytes):
        try:
            return json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeepSeekCallError("parse", f"{context} is not valid JSON") from exc
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise DeepSeekCallError("parse", f"{context} is not valid JSON") from exc
    raise DeepSeekCallError("parse", f"{context} is not valid JSON")


def _extract_content(payload: Mapping[str, Any]) -> Any:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise DeepSeekCallError("parse", "response choices are missing")
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or "content" not in message:
        raise DeepSeekCallError("parse", "response message content is missing")
    content = message["content"]
    if isinstance(content, Mapping):
        return dict(content)
    if not isinstance(content, str):
        raise DeepSeekCallError("parse", "response content is not JSON text")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    return _decode_json(text, context="response content")


def _required_string(value: Any, path: str, *, max_length: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeepSeekSchemaError(f"{path} must be a non-empty string")
    if len(value) > max_length:
        raise DeepSeekSchemaError(f"{path} is too long")
    return value.strip()


def _string_list(value: Any, path: str, *, required: bool = False, max_items: int = 32) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise DeepSeekSchemaError(f"{path} must be an array of non-empty strings")
    if required and not value:
        raise DeepSeekSchemaError(f"{path} must not be empty")
    if len(value) > max_items:
        raise DeepSeekSchemaError(f"{path} has too many items")
    return [item.strip() for item in value]


def _score(value: Any, path: str) -> float:
    if isinstance(value, bool):
        return 0.5
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if 1 < number <= 100:
        number /= 100
    return round(number, 6) if 0 <= number <= 1 else 0.5


def _years(value: Any) -> dict[str, int | None]:
    if not isinstance(value, Mapping):
        raise DeepSeekSchemaError("time_range must be an object")
    result: dict[str, int | None] = {}
    for name in ("start_year", "end_year"):
        item = value.get(name)
        if item is not None and (isinstance(item, bool) or not isinstance(item, int) or not 1800 <= item <= 2200):
            raise DeepSeekSchemaError(f"time_range.{name} must be a year or null")
        result[name] = item
    if result["start_year"] is not None and result["end_year"] is not None and result["start_year"] > result["end_year"]:
        raise DeepSeekSchemaError("time_range.start_year must not exceed end_year")
    return result


def _constraints(value: Any, path: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) > 24:
        raise DeepSeekSchemaError(f"{path} must be an array")
    result: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise DeepSeekSchemaError(f"{path}[{index}] must be an object")
        result.append({
            "name": _required_string(item.get("name"), f"{path}[{index}].name", max_length=100),
            "operator": _required_string(item.get("operator"), f"{path}[{index}].operator", max_length=40),
            "value": _required_string(item.get("value"), f"{path}[{index}].value", max_length=500),
        })
    return result


def _source_list(value: Any, path: str) -> list[str]:
    values = _string_list(value, path, required=True, max_items=8)
    aliases = {"semantic scholar": "openalex", "semantic_scholar": "openalex", "s2": "openalex", "local": "local_library"}
    normalized = [aliases.get(item.casefold(), item.casefold()) for item in values]
    accepted = [item for item in normalized if item in SOURCE_CAPABILITIES]
    if not accepted:
        raise DeepSeekSchemaError(f"{path} contains no supported sources")
    return list(dict.fromkeys(accepted))


def _plan_payload(raw_query: str, response: Mapping[str, Any], *, history: Sequence[Mapping[str, Any]] | None = None) -> QueryPlan:
    # DeepSeek 有时会遵守任务语义但省略 schema_version 和空字段。允许安全
    # 补全这些非事实字段，保留 PaperDoc/来源/ID 等关键边界，避免整个流程无效降级。
    required = {"topic", "keywords", "synonyms", "methods", "datasets", "tasks", "time_range", "hard_constraints", "soft_constraints", "source_capabilities", "queries", "gaps"}
    if response.get("schema_version") != DEEPSEEK_PLAN_SCHEMA or required - set(response):
        response = _coerce_compact_plan(raw_query, response)
    missing = required - set(response)
    if missing:
        raise DeepSeekSchemaError(f"plan response missing fields: {sorted(missing)}")
    topic = _required_string(response.get("topic"), "topic", max_length=1000)
    keywords = _string_list(response.get("keywords"), "keywords", required=True)
    synonyms = _string_list(response.get("synonyms"), "synonyms")
    methods = _string_list(response.get("methods"), "methods")
    datasets = _string_list(response.get("datasets"), "datasets")
    tasks = _string_list(response.get("tasks"), "tasks", required=True)
    time_range = _years(response.get("time_range"))
    hard = _constraints(response.get("hard_constraints"), "hard_constraints")
    soft = _constraints(response.get("soft_constraints"), "soft_constraints")
    sources = _source_list(response.get("source_capabilities"), "source_capabilities")
    gaps = _string_list(response.get("gaps"), "gaps")
    unknown_gaps = set(gaps) - GAP_CODES
    if unknown_gaps:
        raise DeepSeekSchemaError(f"gaps contains unsupported codes: {sorted(unknown_gaps)}")
    queries = response.get("queries")
    if not isinstance(queries, list) or not 1 <= len(queries) <= 12:
        raise DeepSeekSchemaError("queries must contain between 1 and 12 items")
    subqueries: list[dict[str, Any]] = []
    for index, item in enumerate(queries, 1):
        if not isinstance(item, Mapping):
            raise DeepSeekSchemaError(f"queries[{index - 1}] must be an object")
        kind = str(item.get("kind") or "topic").casefold()
        kind = {"research_question": "topic", "question": "topic", "approach": "method", "technique": "method", "data": "dataset", "use_case": "comparison"}.get(kind, kind)
        if kind not in SUBQUERY_KINDS:
            kind = "topic"
        text = _required_string(item.get("query_text"), f"queries[{index - 1}].query_text", max_length=500)
        query_sources = _source_list(item.get("source_capabilities", sources), f"queries[{index - 1}].source_capabilities")
        subqueries.append({"subquery_id": f"sq_deepseek_{index:02d}", "parent_id": None, "kind": kind, "query_text": text, "source_capabilities": query_sources, "iteration": 0})
    # DeepSeek 返回的字段进入 canonical QueryPlan；预算和停止策略仍由代码固定，
    # 防止模型通过响应无限扩大 API 消耗。
    planner = QueryPlanner()
    deterministic = planner.plan(raw_query, history=list(history or []))
    payload = deterministic.to_dict()
    payload.update({
        "raw_query": raw_query,
        "topic": topic,
        "methods": methods,
        "datasets": datasets,
        "tasks": tasks,
        "time_range": {**time_range, "source": "deepseek"},
        "hard_constraints": hard,
        "soft_constraints": soft,
        "source_capabilities": sources,
        "subqueries": subqueries,
        "gaps": gaps,
        "keywords": keywords,
        "synonyms": synonyms,
        "planner": "deepseek",
    })
    try:
        validate_query_plan(payload)
    except QueryPlanValidationError as exc:
        raise DeepSeekSchemaError(f"generated QueryPlan failed validation: {exc}") from exc
    return QueryPlan(payload)


def _coerce_compact_plan(raw_query: str, response: Mapping[str, Any]) -> dict[str, Any]:
    """把模型返回的短 JSON 补成可校验计划；无法识别查询时仍失败。"""

    raw_queries = response.get("queries") or response.get("subqueries") or response.get("search_queries") or response.get("query")
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]
    if not isinstance(raw_queries, list):
        raise DeepSeekSchemaError(f"schema_version must be {DEEPSEEK_PLAN_SCHEMA}")
    query_items: list[dict[str, Any]] = []
    for item in raw_queries[:12]:
        if isinstance(item, str) and item.strip():
            query_items.append({"kind": "topic", "query_text": item.strip()})
        elif isinstance(item, Mapping) and str(item.get("query_text") or item.get("query") or item.get("search_query") or "").strip():
            query_items.append({"kind": str(item.get("kind") or "topic"), "query_text": str(item.get("query_text") or item.get("query") or item.get("search_query")).strip(), "source_capabilities": item.get("source_capabilities")})
    if not query_items:
        raise DeepSeekSchemaError("compact plan has no usable queries")
    deterministic = QueryPlanner().plan(raw_query)
    sources = response.get("source_capabilities") or response.get("sources") or response.get("databases") or deterministic.get("source_capabilities")
    if isinstance(sources, str):
        sources = [sources]
    sources = [str(item).strip() for item in sources if str(item).strip()] if isinstance(sources, list) else list(deterministic["source_capabilities"])
    sources = [item for item in sources if item in SOURCE_CAPABILITIES] or list(deterministic["source_capabilities"])
    topic = str(response.get("topic") or response.get("objective") or raw_query).strip()
    keywords = response.get("keywords")
    if not isinstance(keywords, list) or not keywords:
        keywords = [item for item in re.findall(r"[\\w-]+", topic, flags=re.UNICODE) if len(item) > 1][:12] or [topic]
    return {
        "schema_version": DEEPSEEK_PLAN_SCHEMA,
        "topic": topic,
        "keywords": [str(item) for item in keywords if str(item).strip()][:24],
        "synonyms": response.get("synonyms") if isinstance(response.get("synonyms"), list) else [],
        "methods": response.get("methods") if isinstance(response.get("methods"), list) else [],
        "datasets": response.get("datasets") if isinstance(response.get("datasets"), list) else [],
        "tasks": response.get("tasks") if isinstance(response.get("tasks"), list) and response.get("tasks") else response.get("research_questions") if isinstance(response.get("research_questions"), list) and response.get("research_questions") else deterministic.get("tasks") or [topic],
        "time_range": response.get("time_range") if isinstance(response.get("time_range"), Mapping) else response.get("timeline") if isinstance(response.get("timeline"), Mapping) else {"start_year": None, "end_year": None},
        "hard_constraints": response.get("hard_constraints") if isinstance(response.get("hard_constraints"), list) else [],
        "soft_constraints": response.get("soft_constraints") if isinstance(response.get("soft_constraints"), list) else [],
        "source_capabilities": sources,
        "queries": [{**item, "source_capabilities": item.get("source_capabilities") or sources} for item in query_items],
        "gaps": response.get("gaps") if isinstance(response.get("gaps"), list) else [],
    }


def _paper_summary(paper: Mapping[str, Any]) -> dict[str, Any]:
    validate_paper_doc(paper)
    bib = paper.get("bibliography") or {}
    identifiers = paper.get("identifiers") or {}
    abstract = str(bib.get("abstract") or "")[:5000]
    return {
        "paper_id": str(paper["paper_id"]),
        "identifiers": {key: identifiers.get(key) for key in ("doi", "arxiv_id", "openalex_id", "s2_id", "unique_id") if identifiers.get(key)},
        "title": str(bib.get("title") or ""),
        "authors": [str(item) for item in bib.get("authors") or []][:12],
        "year": bib.get("year"),
        "venue": str(bib.get("venue") or ""),
        "abstract": abstract,
    }


def _validate_judgements(response: Mapping[str, Any], expected_ids: Sequence[str]) -> list[dict[str, Any]]:
    values = response.get("results") or response.get("judgements") or response.get("judgments")
    if values is None and response.get("paper_id"):
        values = [response]
    if response.get("schema_version") != DEEPSEEK_JUDGEMENT_SCHEMA:
        response = {**response, "schema_version": DEEPSEEK_JUDGEMENT_SCHEMA}
    if values is None:
        values = []
    if not isinstance(values, list):
        raise DeepSeekSchemaError("results must be an array")
    supplied_ids = [str(item.get("paper_id")) for item in values if isinstance(item, Mapping) and item.get("paper_id")]
    if len(set(supplied_ids)) != len(supplied_ids) or any(item not in expected_ids for item in supplied_ids):
        raise DeepSeekSchemaError("results contain duplicate or unknown candidate PaperDoc")
    if len(values) < len(expected_ids):
        existing = set(supplied_ids)
        values = list(values) + [{"paper_id": paper_id, "relevance_score": 0.5, "relevance_label": "uncertain", "hard_constraint_state": "unknown", "reason": "candidate was not explicitly judged", "evidence_needed": [], "confidence": 0.0} for paper_id in expected_ids if paper_id not in existing]
    if len(values) != len(expected_ids):
        raise DeepSeekSchemaError("results contain more items than candidate PaperDocs")
    expected = list(expected_ids)
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise DeepSeekSchemaError(f"results[{index}] must be an object")
        paper_id = _required_string(item.get("paper_id"), f"results[{index}].paper_id", max_length=500)
        if paper_id not in expected or paper_id in seen:
            raise DeepSeekSchemaError(f"results[{index}].paper_id does not match candidates")
        seen.add(paper_id)
        score = item.get("relevance_score", item.get("score", item.get("relevance", 0.5)))
        label = item.get("relevance_label") or item.get("label") or ("relevant" if float(score) >= 0.6 else "borderline")
        label = {"yes": "relevant", "match": "relevant", "high": "relevant", "true": "relevant", "maybe": "borderline", "uncertain": "borderline", "no": "irrelevant", "low": "irrelevant", "false": "irrelevant"}.get(str(label).casefold(), str(label).casefold())
        if label not in JUDGEMENT_LABELS:
            label = "borderline"
        state = item.get("hard_constraint_state") or item.get("constraint_state") or "unknown"
        state = {"yes": "pass", "true": "pass", "pass": "pass", "no": "fail", "false": "fail", "fail": "fail"}.get(str(state).casefold(), "unknown")
        if label not in JUDGEMENT_LABELS:
            raise DeepSeekSchemaError(f"results[{index}].relevance_label is invalid")
        if state not in CONSTRAINT_STATES:
            raise DeepSeekSchemaError(f"results[{index}].hard_constraint_state is invalid")
        raw_evidence = item.get("evidence_needed", [])
        evidence = _string_list(raw_evidence if isinstance(raw_evidence, list) else [], f"results[{index}].evidence_needed", max_items=12)
        output.append({
            "paper_id": paper_id,
            "relevance_score": _score(score, f"results[{index}].relevance_score"),
            "relevance_label": label,
            "hard_constraint_state": state,
            "reason": _required_string(item.get("reason") or "model judgement without additional explanation", f"results[{index}].reason", max_length=1500),
            "evidence_needed": evidence,
            "confidence": _score(item.get("confidence", 0.5), f"results[{index}].confidence"),
        })
    if set(seen) != set(expected):
        raise DeepSeekSchemaError("results are missing candidate PaperDocs")
    return output


class DeepSeekClient:
    """只负责一次 DeepSeek JSON 调用的最小客户端。"""

    def __init__(self, api_key: str | None = None, *, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL, transport: Transport | None = None, timeout: float = 45.0) -> None:
        config = load_config()
        self.api_key = api_key if api_key is not None else config.get("DEEPSEEK_API_KEY", "")
        configured_url = config.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        configured_model = config.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.base_url = str(configured_url if base_url == DEFAULT_BASE_URL else base_url).strip().rstrip("/")
        self.model = str(configured_model if model == DEFAULT_MODEL else model).strip() or DEFAULT_MODEL
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise DeepSeekCallError("config", "timeout must be positive")
        self.transport = transport or _default_transport
        self._transport_injected = transport is not None

    def complete_json(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1600) -> dict[str, Any]:
        if not self.api_key and not self._transport_injected:
            raise DeepSeekCallError("config", "DEEPSEEK_API_KEY is missing")
        endpoint = urljoin(self.base_url + "/", "chat/completions")
        body = json.dumps({"model": self.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0, "max_tokens": max_tokens, "response_format": {"type": "json_object"}}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        started = time.perf_counter()
        try:
            response = _normalise_response(self.transport("POST", endpoint, headers, body, self.timeout))
        except DeepSeekCallError:
            raise
        except Exception as exc:
            raise DeepSeekCallError("network", f"transport failed: {type(exc).__name__}", retryable=True) from exc
        if not 200 <= response.status < 300:
            raise DeepSeekCallError("auth" if response.status in {401, 403} else "rate" if response.status == 429 else "network", f"HTTP status {response.status}", retryable=response.status >= 500 or response.status == 429, status_code=response.status)
        payload = _decode_json(response.body, context="DeepSeek response")
        if not isinstance(payload, Mapping):
            raise DeepSeekCallError("parse", "DeepSeek response must be an object")
        try:
            content = _extract_content(payload)
        except DeepSeekCallError:
            raise
        if not isinstance(content, Mapping):
            raise DeepSeekCallError("parse", "DeepSeek content must be a JSON object")
        # latency is intentionally not embedded in model output; callers may log it
        # separately without including request/response bodies or credentials.
        _ = time.perf_counter() - started
        return dict(content)


class DeepSeekUnderstandingLayer:
    """DeepSeek 的前置查询规划和召回后判断。"""

    def __init__(self, client: DeepSeekClient | None = None) -> None:
        self.client = client or DeepSeekClient()

    def plan(self, query: str, *, history: Sequence[Mapping[str, Any]] | None = None) -> QueryPlan:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        system = "You are an academic search query planner. Return JSON only; never invent paper facts."
        user = json.dumps({"task": "decompose_query", "schema_version": DEEPSEEK_PLAN_SCHEMA, "query": query, "history": list(history or []), "required": {"topic": "string", "keywords": "string[]", "synonyms": "string[]", "methods": "string[]", "datasets": "string[]", "tasks": "string[]", "time_range": {"start_year": "integer|null", "end_year": "integer|null"}, "hard_constraints": "constraint[]", "soft_constraints": "constraint[]", "source_capabilities": sorted(SOURCE_CAPABILITIES), "queries": "1-12 query objects with kind/query_text/source_capabilities", "gaps": sorted(GAP_CODES)}}, ensure_ascii=False)
        return _plan_payload(query, self.client.complete_json(system, user), history=history)

    def judge(self, query_plan: Mapping[str, Any], papers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        validate_query_plan(query_plan)
        summaries = [_paper_summary(paper) for paper in papers]
        ids = [item["paper_id"] for item in summaries]
        if len(set(ids)) != len(ids):
            raise DeepSeekSchemaError("candidate PaperDoc paper_id values must be unique")
        system = "You are an academic paper relevance judge. Return JSON only. Judge only supplied PaperDoc facts; do not invent metadata."
        user = json.dumps({"task": "judge_candidates", "schema_version": DEEPSEEK_JUDGEMENT_SCHEMA, "query_plan": {"query_id": query_plan["query_id"], "raw_query": query_plan["raw_query"], "topic": query_plan["topic"], "methods": query_plan["methods"], "datasets": query_plan["datasets"], "tasks": query_plan["tasks"], "time_range": query_plan["time_range"], "hard_constraints": query_plan["hard_constraints"]}, "candidates": summaries, "required_result_fields": ["paper_id", "relevance_score", "relevance_label", "hard_constraint_state", "reason", "evidence_needed", "confidence"]}, ensure_ascii=False)
        return _validate_judgements(self.client.complete_json(system, user), ids)


__all__ = ["DEEPSEEK_JUDGEMENT_SCHEMA", "DEEPSEEK_PLAN_SCHEMA", "DeepSeekCallError", "DeepSeekClient", "DeepSeekSchemaError", "DeepSeekUnderstandingLayer", "TransportResponse"]
