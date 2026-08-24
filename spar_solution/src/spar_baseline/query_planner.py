"""P2 查询规划器。

默认实现只使用规则和标准库，因此 AutoScholarQuery 这类问句不会被原样
拆成 ``AND`` 词袋。LLM 只作为可注入的 JSON 提供者：结果仍须经过同一套
``QueryPlan`` 校验，失败不会悄悄变成一个无效计划。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from .query_plan import GAP_CODES, QueryPlan, QueryPlanValidationError, validate_query_plan


_FILLER = {
    "a", "an", "the", "some", "any", "papers", "paper", "studies", "study", "works", "work",
    "resources", "resource", "information", "please", "tell", "me", "can", "could", "would",
    "you", "provide", "providing", "list", "give", "about", "what", "which", "that", "are",
    "is", "there", "there", "any", "have", "has", "do", "did", "does", "explored", "explore",
    "focused", "focus", "attempts", "attempt", "based", "field", "following", "related", "find",
    "identify", "identifying", "information", "on", "of", "in", "for", "to", "and", "or", "with",
    "from", "using", "used", "use", "through", "via", "into", "that", "where", "how", "why",
    "were", "was", "published", "between", "since", "after", "before", "until",
}
_METHOD_HINTS = (
    "architecture", "architectures", "technique", "techniques", "method", "methods", "algorithm",
    "network", "networks", "learning", "model", "models", "transformer", "attention", "q-learning",
    "reinforcement", "causal", "intervention", "experimentation", "calibration", "calibrate",
    "reconstruction", "representation", "adversarial", "probabilistic", "sequential",
)
_DATASET_HINTS = ("dataset", "datasets", "benchmark", "benchmarks", "corpus", "corpora", "data set", "data")
_APPLICATION_HINTS = ("health", "medical", "robot", "cyber", "security", "finance", "peer review", "review")
_TASK_HINTS = (
    "anomaly detection", "image segmentation", "heart rate monitoring", "peer review",
    "detection", "detect", "segmentation", "reconstruction", "monitoring", "measurement",
    "prediction", "calibration", "classification", "estimation", "generation", "retrieval",
)
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")


def _clean_query(query: str) -> str:
    """保留科学术语，删除礼貌/疑问壳；不对剩余词做强制 AND。"""

    value = re.sub(r"[^\w\s+\-/]", " ", query.casefold(), flags=re.UNICODE)
    # 连字符术语（例如 q-learning）需要保留；下划线只作为空格。
    tokens = [
        token for token in value.replace("_", " ").split()
        if token not in _FILLER and not _YEAR_RE.fullmatch(token)
    ]
    return " ".join(tokens).strip(" -")


def _phrases(query: str, hints: tuple[str, ...]) -> list[str]:
    lowered = query.casefold()
    found: list[str] = []
    for hint in sorted(hints, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(hint)}(?!\w)", lowered) and not any(hint in item for item in found):
            found.append(hint)
    return found


def _query_id(query: str) -> str:
    return "qplan_" + hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:12]


def _constraint(name: str, operator: str, value: str) -> dict[str, str]:
    return {"name": name, "operator": operator, "value": value}


def _subquery(subquery_id: str, kind: str, text: str, iteration: int = 0, parent_id: str | None = None) -> dict[str, Any]:
    return {
        "subquery_id": subquery_id,
        "parent_id": parent_id,
        "kind": kind,
        "query_text": text,
        "source_capabilities": ["arxiv", "openalex", "local_library"],
        "iteration": iteration,
    }


def _parse_years(query: str) -> tuple[int | None, int | None]:
    years = [int(item) for item in _YEAR_RE.findall(query)]
    if not years:
        return None, None
    if len(years) == 1:
        if re.search(r"since|after|from", query.casefold()):
            return years[0], None
        if re.search(r"before|until|through", query.casefold()):
            return None, years[0]
    return min(years), max(years) if len(years) > 1 else None


def _parse_json_object(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise QueryPlanValidationError("LLM plan must be a JSON object or JSON string")
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QueryPlanValidationError("LLM plan is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise QueryPlanValidationError("LLM plan must be a JSON object")
    return parsed


class QueryPlanner:
    """确定性 QueryPlan 生成器，支持可选的受校验 LLM JSON 注入。"""

    def __init__(self, llm_json_provider: Callable[[str], str | Mapping[str, Any]] | None = None) -> None:
        self.llm_json_provider = llm_json_provider

    @staticmethod
    def from_llm_json(value: str | Mapping[str, Any], *, raw_query: str | None = None) -> QueryPlan:
        payload = _parse_json_object(value)
        if raw_query is not None:
            payload["raw_query"] = raw_query
            payload["query_id"] = _query_id(raw_query)
        validate_query_plan(payload)
        return QueryPlan(payload)

    def plan(
        self,
        query: str,
        *,
        history: list[Mapping[str, Any]] | None = None,
        llm_json: str | Mapping[str, Any] | None = None,
    ) -> QueryPlan:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if llm_json is not None:
            return self.from_llm_json(llm_json, raw_query=query)
        if self.llm_json_provider is not None:
            return self.from_llm_json(self.llm_json_provider(query), raw_query=query)
        return self._deterministic(query, history=history)

    def _deterministic(self, query: str, *, history: list[Mapping[str, Any]] | None = None) -> QueryPlan:
        cleaned = _clean_query(query)
        if not cleaned:
            raise ValueError("query contains no searchable terms")
        start, end = _parse_years(query)
        methods = _phrases(cleaned, _METHOD_HINTS)
        datasets = _phrases(cleaned, _DATASET_HINTS)
        applications = _phrases(cleaned, _APPLICATION_HINTS)
        tasks = _phrases(cleaned, _TASK_HINTS)
        # 任务保留一个短的语义查询，而不是把疑问句逐词拼进 Provider。
        task = cleaned
        gaps: list[str] = []
        if not methods:
            gaps.append("missing_method")
        if not datasets:
            gaps.append("missing_dataset")
        if start is None and end is None:
            gaps.append("missing_time_range")
        if not applications:
            gaps.append("missing_application")
        if history:
            if "citation_neighbor_gain" not in gaps:
                gaps.append("citation_neighbor_gain")

        hard: list[dict[str, str]] = []
        soft: list[dict[str, str]] = [_constraint("evidence", "prefer", "abstract_or_fulltext")]
        if start is not None or end is not None:
            hard.append(_constraint("time_range", "between", f"{start or ''}:{end or ''}"))
        if "only open access" in query.casefold() or "open-access" in query.casefold():
            hard.append(_constraint("access", "equals", "open_access"))
        if applications:
            soft.append(_constraint("application", "prefer", ", ".join(applications)))

        subqueries = [_subquery("sq_topic_01", "topic", task)]
        if methods:
            subqueries.append(_subquery("sq_method_01", "method", " ".join(methods)))
        if datasets:
            subqueries.append(_subquery("sq_dataset_01", "dataset", " ".join(datasets)))
        if start is not None or end is not None:
            subqueries.append(_subquery("sq_constraint_01", "constraint", f"publication {start or ''} {end or ''}"))
        return QueryPlan({
            "schema_version": "query_plan.v1",
            "query_id": _query_id(query),
            "raw_query": query,
            "topic": task,
            "methods": methods,
            "datasets": datasets,
            "tasks": tasks or [task],
            "time_range": {"start_year": start, "end_year": end, "source": "explicit" if start or end else "unspecified"},
            "hard_constraints": hard,
            "soft_constraints": soft,
            "source_capabilities": ["arxiv", "openalex", "local_library"],
            "budget": {"max_iterations": 2, "max_citation_depth": 1, "max_subqueries_per_gap": 2, "max_provider_calls": 20},
            "stop_strategy": {"min_new_relevant": 2, "min_subquery_coverage": 0.8, "min_evidence_coverage": 0.7},
            "subqueries": subqueries,
            "gaps": gaps,
            "history_size": len(history or []),
        })

    def next_iteration(self, plan: QueryPlan | Mapping[str, Any], *, gaps: list[str] | None = None) -> QueryPlan:
        """按 gap 生成下一轮子查询；每个 gap 最多两条，最多两轮。"""

        current = QueryPlan.from_dict(plan)
        iteration = max(item["iteration"] for item in current["subqueries"]) + 1
        if iteration >= current["budget"]["max_iterations"]:
            return current
        selected = [gap for gap in (gaps if gaps is not None else current["gaps"]) if gap in GAP_CODES]
        topic = str(current.get("topic") or current.get("raw_query") or "").strip()
        gap_queries = {
            "missing_method": ("method", f"{topic} methods techniques algorithms"),
            "missing_dataset": ("dataset", f"{topic} datasets benchmarks"),
            "missing_time_range": ("constraint", f"{topic} publication year date"),
            # query_plan.v1 没有单独的 application kind；用 comparison
            # 保留应用场景扩展，同时让下一轮计划继续满足严格 schema。
            "missing_application": ("comparison", f"{topic} applications use cases"),
            "citation_neighbor_gain": ("reference", f"{topic} related references citations"),
        }
        additions = []
        for gap in dict.fromkeys(selected):
            kind, text = gap_queries[gap]
            additions.append(_subquery(f"sq_{gap}_{iteration:02d}", kind, text, iteration, current["subqueries"][-1]["subquery_id"]))
        payload = current.to_dict()
        payload["subqueries"].extend(additions)
        payload["gaps"] = selected
        return QueryPlan(payload)


def plan_query(query: str, **kwargs: Any) -> QueryPlan:
    return QueryPlanner().plan(query, **kwargs)


__all__ = ["QueryPlanner", "plan_query"]
