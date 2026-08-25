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
        raise DeepSeekSchemaError(f"{path} must be a number between 0 and 1")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DeepSeekSchemaError(f"{path} must be a number between 0 and 1") from exc
    if 1 < number <= 100:
        number /= 100
    if not 0 <= number <= 1:
        raise DeepSeekSchemaError(f"{path} must be a number between 0 and 1")
    return round(number, 6)


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
    """解析约束列表；单条畸形约束跳过，不拖垮整个计划。

    真实运行中模型偶尔产出 name 为空或类型错误的约束；旧逻辑整计划拒绝并
    回退规则规划器，导致整轮检索质量下降（见 live-benchmark-q4-fixed-v2）。
    """

    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        operator = item.get("operator")
        constraint_value = item.get("value")
        if (
            not isinstance(name, str) or not name.strip()
            or not isinstance(operator, str) or not operator.strip()
            or not isinstance(constraint_value, str) or not constraint_value.strip()
        ):
            continue
        result.append({"name": name.strip()[:100], "operator": operator.strip()[:40], "value": constraint_value.strip()[:500]})
        if len(result) >= 24:
            break
    return result


def _source_list(value: Any, path: str) -> list[str]:
    values = _string_list(value, path, required=True, max_items=8)
    aliases = {"semantic scholar": "openalex", "semantic_scholar": "openalex", "s2": "openalex", "local": "local_library"}
    normalized = [aliases.get(item.casefold(), item.casefold()) for item in values]
    accepted = [item for item in normalized if item in SOURCE_CAPABILITIES]
    if not accepted:
        raise DeepSeekSchemaError(f"{path} contains no supported sources")
    return list(dict.fromkeys(accepted))


def _string_list_tolerant(value: Any, *, max_items: int = 32) -> list[str]:
    """列表元素级宽容解析：坏元素过滤，不整列表拒绝。"""

    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()][:max_items]


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
    # 列表字段元素级宽容：畸形元素过滤；必需列表为空时从 topic 兜底推导，
    # 避免一条坏数据把整个计划打回规则规划器。
    keywords = _string_list_tolerant(response.get("keywords"), max_items=24)
    synonyms = _string_list_tolerant(response.get("synonyms"))
    methods = _string_list_tolerant(response.get("methods"))
    datasets = _string_list_tolerant(response.get("datasets"))
    tasks = _string_list_tolerant(response.get("tasks")) or _string_list_tolerant(response.get("research_questions"))
    if not keywords:
        keywords = [item for item in re.findall(r"[\w-]+", topic, flags=re.UNICODE) if len(item) > 1][:12] or [topic]
    if not tasks:
        tasks = [topic]
    time_range = _years(response.get("time_range"))
    hard = _constraints(response.get("hard_constraints"), "hard_constraints")
    soft = _constraints(response.get("soft_constraints"), "soft_constraints")
    sources = _source_list(response.get("source_capabilities"), "source_capabilities")
    gaps = [gap for gap in _string_list_tolerant(response.get("gaps")) if gap in GAP_CODES]
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
    # 判断层只需标题+摘要前段锚定五档量规；5000 字符截断使 judge prompt
    # 均值 ~5万 token/题压效率红线，2000 已覆盖量规判分所需信息。
    abstract = str(bib.get("abstract") or "")[:2000]
    return {
        "paper_id": str(paper["paper_id"]),
        "identifiers": {key: identifiers.get(key) for key in ("doi", "arxiv_id", "openalex_id", "s2_id", "unique_id") if identifiers.get(key)},
        "title": str(bib.get("title") or ""),
        "authors": [str(item) for item in bib.get("authors") or []][:12],
        "year": bib.get("year"),
        "venue": str(bib.get("venue") or ""),
        "abstract": abstract,
    }


def _tolerant_score(value: Any, default: float = 0.5) -> float:
    """判断条目用的分数解析：畸形值退到中性 default，不丢弃整条。"""

    try:
        return _score(value, "score")
    except DeepSeekSchemaError:
        return default


def _normalise_judgement_item(item: Mapping[str, Any], expected_id: str) -> dict[str, Any]:
    paper_id = _required_string(item.get("paper_id"), "results.paper_id", max_length=500)
    if paper_id != expected_id:
        raise DeepSeekSchemaError("results.paper_id does not match candidate")
    score = item.get("relevance_score", item.get("score", item.get("relevance")))
    if score is None:
        raise DeepSeekSchemaError("results.relevance_score is required")
    normalized_score = _tolerant_score(score)
    raw_label = item.get("relevance_label") or item.get("label")
    label = {"yes": "relevant", "match": "relevant", "high": "relevant", "true": "relevant", "maybe": "borderline", "uncertain": "borderline", "no": "irrelevant", "low": "irrelevant", "false": "irrelevant"}.get(str(raw_label or "").casefold(), "")
    if not label:
        # 模型偶尔产出自由文本标签；分数是唯一必需信号，按分数兜底而不是丢弃整条。
        label = "relevant" if normalized_score >= 0.6 else "irrelevant" if normalized_score < 0.3 else "borderline"
    raw_state = item.get("hard_constraint_state") or item.get("constraint_state")
    state = {"yes": "pass", "true": "pass", "pass": "pass", "no": "fail", "false": "fail", "fail": "fail", "unknown": "unknown"}.get(str(raw_state or "").casefold(), "unknown")
    raw_evidence = item.get("evidence_needed", [])
    evidence = _string_list(raw_evidence if isinstance(raw_evidence, list) else [], "results.evidence_needed", max_items=12)
    raw_reason = item.get("reason")
    reason = str(raw_reason).strip()[:1500] if raw_reason is not None else ""
    return {
        "paper_id": paper_id,
        "relevance_score": normalized_score,
        "relevance_label": label,
        "hard_constraint_state": state,
        "reason": reason or "model judgement without additional explanation",
        "evidence_needed": evidence,
        "confidence": _tolerant_score(item.get("confidence", 0.5)),
    }


def _judgement_values(response: Mapping[str, Any]) -> Any:
    values = response.get("results") or response.get("judgements") or response.get("judgments")
    if values is None and response.get("paper_id"):
        values = [response]
    return values if values is not None else []


def _validate_judgements(response: Mapping[str, Any], expected_ids: Sequence[str]) -> list[dict[str, Any]]:
    values = _judgement_values(response)
    if not isinstance(values, list):
        raise DeepSeekSchemaError("results must be an array")
    supplied_ids = [str(item.get("paper_id")) for item in values if isinstance(item, Mapping) and item.get("paper_id")]
    if len(set(supplied_ids)) != len(supplied_ids) or any(item not in expected_ids for item in supplied_ids):
        raise DeepSeekSchemaError("results contain duplicate or unknown candidate PaperDoc")
    if len(values) != len(expected_ids):
        raise DeepSeekSchemaError("results must contain exactly one item per candidate PaperDoc")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, Mapping):
            raise DeepSeekSchemaError("results items must be objects")
        paper_id = _required_string(item.get("paper_id"), "results.paper_id", max_length=500)
        if paper_id not in expected_ids or paper_id in seen:
            raise DeepSeekSchemaError("results.paper_id does not match candidates")
        seen.add(paper_id)
        output.append(_normalise_judgement_item(item, paper_id))
    if set(seen) != set(expected_ids):
        raise DeepSeekSchemaError("results are missing candidate PaperDocs")
    return output


def _parse_judgements_partial(response: Mapping[str, Any], expected_ids: Sequence[str]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """逐条接受合法判断；畸形条目只记 issue，不再拖垮整批。

    返回 (合法条目, 未覆盖的候选 ID, 问题清单)。零合法条目由调用方按
    整批失败处理（触发减半重试）。
    """

    values = _judgement_values(response)
    if not isinstance(values, list):
        raise DeepSeekSchemaError("results must be an array")
    expected = list(expected_ids)
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    issues: list[str] = []
    for item in values:
        if not isinstance(item, Mapping):
            issues.append("non_object_result_item")
            continue
        paper_id = str(item.get("paper_id") or "")
        if not paper_id:
            issues.append("result_item_without_paper_id")
            continue
        if paper_id not in expected:
            issues.append(f"unknown_paper_id:{paper_id[:48]}")
            continue
        if paper_id in seen:
            issues.append(f"duplicate_paper_id:{paper_id[:48]}")
            continue
        try:
            valid.append(_normalise_judgement_item(item, paper_id))
            seen.add(paper_id)
        except DeepSeekSchemaError as exc:
            issues.append(f"invalid_item:{paper_id[:48]}:{exc}")
    missing = [paper_id for paper_id in expected if paper_id not in seen]
    return valid, missing, issues


def blend_relevance(llm_score: float | None, lexical_score: float | None, *, llm_weight: float = 0.65, disagreement_penalty: float = 0.10, penalty_threshold: float = 0.5) -> float | None:
    """LLM 相关性分与词法相关性分的保守融合（供管线集成使用）。

    设计意图：判断层运行间不稳定——同一论文的 LLM 分会在 0.05 与 1.0 之间
    跳变。词法分作为独立第二信号拉住 LLM 的极端输出：blended =
    llm_weight*llm + (1-llm_weight)*lexical；两侧严重打架
    （|llm-lexical| > penalty_threshold）时再减 disagreement_penalty*差值，
    抑制任一侧过度自信，而不是简单取平均。

    - 双侧均 None 返回 None；只有一侧有值时直接返回该值（缺失侧按无信息
      处理，不用 0.5 之类的中性值顶替加权份额）。
    - 结果 clamp 到 [0, 1] 并 round 到 6 位小数；llm_weight 必须在 [0, 1]。
    """

    if not 0 <= llm_weight <= 1:
        raise ValueError("llm_weight must be between 0 and 1")
    if llm_score is None and lexical_score is None:
        return None
    if llm_score is None or lexical_score is None:
        available = llm_score if llm_score is not None else lexical_score
        return round(max(0.0, min(1.0, float(available))), 6)
    blended = llm_weight * llm_score + (1 - llm_weight) * lexical_score
    disagreement = abs(llm_score - lexical_score)
    if disagreement > penalty_threshold:
        blended -= disagreement_penalty * disagreement
    return round(max(0.0, min(1.0, blended)), 6)


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
        self.reset_usage()

    def reset_usage(self, *, max_calls: int | None = None) -> None:
        if max_calls is not None and (isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 0):
            raise DeepSeekCallError("config", "max_calls must be a non-negative integer or null")
        self._max_calls = max_calls
        self._usage = {
            "calls": 0,
            "failures": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0.0,
        }

    @property
    def usage(self) -> dict[str, int | float]:
        return dict(self._usage)

    def complete_json(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1600) -> dict[str, Any]:
        if not self.api_key and not self._transport_injected:
            raise DeepSeekCallError("config", "DEEPSEEK_API_KEY is missing")
        endpoint = urljoin(self.base_url + "/", "chat/completions")
        body = json.dumps({"model": self.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0, "max_tokens": max_tokens, "response_format": {"type": "json_object"}}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response: TransportResponse | None = None
        for attempt in range(2):
            if self._max_calls is not None and self._usage["calls"] >= self._max_calls:
                raise DeepSeekCallError("budget", "LLM call budget exhausted")
            started = time.perf_counter()
            self._usage["calls"] += 1
            try:
                response = _normalise_response(self.transport("POST", endpoint, headers, body, self.timeout))
            except DeepSeekCallError as exc:
                self._usage["failures"] += 1
                self._usage["latency_ms"] = round(float(self._usage["latency_ms"]) + (time.perf_counter() - started) * 1000, 3)
                if exc.retryable and attempt == 0:
                    continue
                raise
            except Exception as exc:
                self._usage["failures"] += 1
                self._usage["latency_ms"] = round(float(self._usage["latency_ms"]) + (time.perf_counter() - started) * 1000, 3)
                if attempt == 0:
                    continue
                raise DeepSeekCallError("network", f"transport failed: {type(exc).__name__}", retryable=True) from exc
            self._usage["latency_ms"] = round(float(self._usage["latency_ms"]) + (time.perf_counter() - started) * 1000, 3)
            if 200 <= response.status < 300:
                break
            self._usage["failures"] += 1
            retryable = response.status >= 500 or response.status == 429
            if retryable and attempt == 0:
                continue
            raise DeepSeekCallError("auth" if response.status in {401, 403} else "rate" if response.status == 429 else "network", f"HTTP status {response.status}", retryable=retryable, status_code=response.status)
        if response is None:
            raise DeepSeekCallError("network", "DeepSeek returned no response")
        try:
            payload = _decode_json(response.body, context="DeepSeek response")
        except DeepSeekCallError:
            self._usage["failures"] += 1
            raise
        if not isinstance(payload, Mapping):
            self._usage["failures"] += 1
            raise DeepSeekCallError("parse", "DeepSeek response must be an object")
        raw_usage = payload.get("usage")
        if isinstance(raw_usage, Mapping):
            prompt = raw_usage.get("prompt_tokens", 0)
            completion = raw_usage.get("completion_tokens", 0)
            total = raw_usage.get("total_tokens")
            prompt = prompt if isinstance(prompt, int) and not isinstance(prompt, bool) and prompt >= 0 else 0
            completion = completion if isinstance(completion, int) and not isinstance(completion, bool) and completion >= 0 else 0
            total = total if isinstance(total, int) and not isinstance(total, bool) and total >= 0 else prompt + completion
            self._usage["prompt_tokens"] += prompt
            self._usage["completion_tokens"] += completion
            self._usage["total_tokens"] += total
        try:
            content = _extract_content(payload)
        except DeepSeekCallError:
            self._usage["failures"] += 1
            raise
        if not isinstance(content, Mapping):
            self._usage["failures"] += 1
            raise DeepSeekCallError("parse", "DeepSeek content must be a JSON object")
        return dict(content)


# 判断层 system prompt：带评分量规（rubric）锚定分数档位，抑制运行间抖动
# （真实运行中同一论文曾在 1.0 与 0.05 之间跳变，综述稳定挤掉原始研究）。
_JUDGE_SYSTEM_PROMPT = (
    "You are an academic paper relevance judge. Return JSON only. Judge only supplied PaperDoc facts; do not invent metadata. Keep each reason under 40 words.\n"
    "Score every candidate strictly against this rubric:\n"
    "- 0.90-1.00: primary research that directly answers the research question (its methods/objects precisely match what the question asks).\n"
    "- 0.70-0.89: same core area but NOT a direct answer to THIS question. Within this band use 0.85-0.89 only if the paper plausibly answers part of the question, 0.70-0.79 for adjacent work. Do NOT cluster many papers at the same score: the question asks which papers actually answer it.\n"

    "- 0.40-0.69: partially relevant: shares methods or datasets but studies a different research question.\n"
    "- 0.10-0.39: weakly relevant: only broad field overlap (e.g. both are NLP).\n"
    "- 0.00-0.09: irrelevant.\n"
    "Prefer the specific primary research papers that answer the question over broad surveys or textbooks. A survey only scores >=0.7 if the question explicitly asks for surveys/overviews. Judge against the raw question, not just topic keywords. Return exactly one item per candidate with the given paper_id."
)


class DeepSeekUnderstandingLayer:
    """DeepSeek 的前置查询规划和召回后判断。"""

    MAX_JUDGE_BATCH = 10

    def __init__(self, client: DeepSeekClient | None = None) -> None:
        self.client = client or DeepSeekClient()
        self.last_judge_issues: list[str] = []

    def plan(self, query: str, *, history: Sequence[Mapping[str, Any]] | None = None) -> QueryPlan:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        system = "You are an academic search query planner. Return JSON only; never invent paper facts."
        user = json.dumps({"task": "decompose_query", "schema_version": DEEPSEEK_PLAN_SCHEMA, "query": query, "history": list(history or []), "required": {"topic": "string", "keywords": "string[]", "synonyms": "string[]", "methods": "string[]", "datasets": "string[]", "tasks": "string[]", "time_range": {"start_year": "integer|null", "end_year": "integer|null"}, "hard_constraints": "constraint[]", "soft_constraints": "constraint[]", "source_capabilities": sorted(SOURCE_CAPABILITIES), "queries": "1-12 query objects with kind/query_text/source_capabilities", "gaps": sorted(GAP_CODES)}}, ensure_ascii=False)
        return _plan_payload(query, self.client.complete_json(system, user), history=history)

    def _judge_request(self, query_plan: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        # 输出预算随候选数放大：逐篇结论 + 固定开销，避免大批次被截断。
        max_tokens = min(8000, 150 * len(summaries) + 400)
        system = _JUDGE_SYSTEM_PROMPT
        user = json.dumps({"task": "judge_candidates", "schema_version": DEEPSEEK_JUDGEMENT_SCHEMA, "query_plan": {"query_id": query_plan["query_id"], "raw_query": query_plan["raw_query"], "topic": query_plan["topic"], "methods": query_plan["methods"], "datasets": query_plan["datasets"], "tasks": query_plan["tasks"], "time_range": query_plan["time_range"], "hard_constraints": query_plan["hard_constraints"]}, "candidates": list(summaries), "required_result_fields": ["paper_id", "relevance_score", "relevance_label", "hard_constraint_state", "reason", "evidence_needed", "confidence"]}, ensure_ascii=False)
        return self.client.complete_json(system, user, max_tokens=max_tokens)

    def judge(self, query_plan: Mapping[str, Any], papers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """批量判断候选论文，逐条校验并部分接受。

        失败策略：整批无效时按减半批次重试；部分有效时只重试缺失的候选；
        单篇仍失败则放弃该篇（保留词法分），不再让一篇畸形响应拖垮整批。
        `last_judge_issues` 记录本次调用的全部问题，供 manifest 审计。
        """

        validate_query_plan(query_plan)
        summaries = [_paper_summary(paper) for paper in papers]
        ids = [item["paper_id"] for item in summaries]
        if len(set(ids)) != len(ids):
            raise DeepSeekSchemaError("candidate PaperDoc paper_id values must be unique")
        by_id = dict(zip(ids, summaries))
        results: dict[str, dict[str, Any]] = {}
        issues: list[str] = []
        self.last_judge_issues = issues
        queue: list[str] = list(ids)
        # cap 是当前批次上限：失败时对同一队头减半重试，成功后恢复满批。
        cap = self.MAX_JUDGE_BATCH
        while queue:
            batch = queue[:cap]
            rest = queue[cap:]
            try:
                response = self._judge_request(query_plan, [by_id[paper_id] for paper_id in batch])
                valid, missing, batch_issues = _parse_judgements_partial(response, batch)
            except DeepSeekCallError as exc:
                if exc.code == "budget":
                    issues.append(f"budget_exhausted:{len(queue)}_candidates_left_lexical")
                    break
                if len(batch) == 1:
                    issues.append(f"judge_failed:{batch[0][:48]}:{exc.code}")
                    queue = rest
                    cap = self.MAX_JUDGE_BATCH
                else:
                    cap = max(1, len(batch) // 2)
                continue
            except DeepSeekSchemaError as exc:
                if len(batch) == 1:
                    issues.append(f"judge_failed:{batch[0][:48]}:schema:{exc}")
                    queue = rest
                    cap = self.MAX_JUDGE_BATCH
                else:
                    cap = max(1, len(batch) // 2)
                continue
            issues.extend(batch_issues)
            if valid:
                for item in valid:
                    results[item["paper_id"]] = item
                queue = missing + list(rest)
                cap = self.MAX_JUDGE_BATCH
            elif len(batch) == 1:
                issues.append(f"judge_failed:{batch[0][:48]}:no_valid_result")
                queue = rest
                cap = self.MAX_JUDGE_BATCH
            else:
                cap = max(1, len(batch) // 2)
        return [results[paper_id] for paper_id in ids if paper_id in results]


__all__ = ["DEEPSEEK_JUDGEMENT_SCHEMA", "DEEPSEEK_PLAN_SCHEMA", "DeepSeekCallError", "DeepSeekClient", "DeepSeekSchemaError", "DeepSeekUnderstandingLayer", "TransportResponse", "blend_relevance"]
