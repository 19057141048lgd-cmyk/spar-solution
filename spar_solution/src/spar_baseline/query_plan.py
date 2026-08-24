"""P2 结构化查询协议。

查询计划是普通 ``dict`` 的受限子类：既保留 JSON/JSONL artifact 的兼容性，
又提供少量字段访问和校验能力。PaperDoc v1 仍是论文对象的唯一协议，本模块
只描述检索意图，不复制论文正文。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


QUERY_PLAN_SCHEMA = "query_plan.v1"
SUBQUERY_KINDS = {"topic", "method", "dataset", "constraint", "comparison", "reference"}
GAP_CODES = {
    "missing_method",
    "missing_dataset",
    "missing_time_range",
    "missing_application",
    "citation_neighbor_gain",
}


class QueryPlanValidationError(ValueError):
    """查询计划不满足 query_plan.v1。"""


def _string_list(value: Any, path: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise QueryPlanValidationError(f"{path} must be an array of strings")
    if not allow_empty and not value:
        raise QueryPlanValidationError(f"{path} must not be empty")
    return value


def _nonnegative_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise QueryPlanValidationError(f"{path} must be a non-negative integer")
    return value


def _number(value: Any, path: str, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise QueryPlanValidationError(f"{path} must be a number")
    number = float(value)
    if number < minimum or (maximum is not None and number > maximum):
        raise QueryPlanValidationError(f"{path} is outside the allowed range")
    return number


def validate_subquery(subquery: Mapping[str, Any]) -> dict[str, Any]:
    """校验单条结构化子查询并返回原始字段的副本。"""

    if not isinstance(subquery, Mapping):
        raise QueryPlanValidationError("subquery must be an object")
    required = {"subquery_id", "parent_id", "kind", "query_text", "source_capabilities", "iteration"}
    missing = required - set(subquery)
    if missing:
        raise QueryPlanValidationError(f"subquery missing fields: {sorted(missing)}")
    if not isinstance(subquery["subquery_id"], str) or not subquery["subquery_id"].strip():
        raise QueryPlanValidationError("subquery.subquery_id must be a non-empty string")
    if subquery["parent_id"] is not None and not isinstance(subquery["parent_id"], str):
        raise QueryPlanValidationError("subquery.parent_id must be a string or null")
    if subquery["kind"] not in SUBQUERY_KINDS:
        raise QueryPlanValidationError(f"subquery.kind must be one of {sorted(SUBQUERY_KINDS)}")
    if not isinstance(subquery["query_text"], str) or not subquery["query_text"].strip():
        raise QueryPlanValidationError("subquery.query_text must be a non-empty string")
    _string_list(subquery["source_capabilities"], "subquery.source_capabilities", allow_empty=False)
    _nonnegative_int(subquery["iteration"], "subquery.iteration")
    return dict(subquery)


def validate_query_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """严格校验 query_plan.v1；不对自然语言字段做隐式修复。"""

    if not isinstance(plan, Mapping):
        raise QueryPlanValidationError("query_plan must be an object")
    required = {
        "schema_version",
        "query_id",
        "raw_query",
        "topic",
        "methods",
        "datasets",
        "tasks",
        "time_range",
        "hard_constraints",
        "soft_constraints",
        "source_capabilities",
        "budget",
        "stop_strategy",
        "subqueries",
        "gaps",
    }
    missing = required - set(plan)
    if missing:
        raise QueryPlanValidationError(f"query_plan missing fields: {sorted(missing)}")
    if plan["schema_version"] != QUERY_PLAN_SCHEMA:
        raise QueryPlanValidationError(f"schema_version must be {QUERY_PLAN_SCHEMA}")
    for field in ("query_id", "raw_query", "topic"):
        if not isinstance(plan[field], str) or not plan[field].strip():
            raise QueryPlanValidationError(f"{field} must be a non-empty string")
    for field in ("methods", "datasets", "tasks", "source_capabilities"):
        _string_list(plan[field], field)
    if not isinstance(plan["time_range"], Mapping):
        raise QueryPlanValidationError("time_range must be an object")
    start, end = plan["time_range"].get("start_year"), plan["time_range"].get("end_year")
    for value, path in ((start, "time_range.start_year"), (end, "time_range.end_year")):
        if value is not None:
            if not isinstance(value, int) or isinstance(value, bool) or not 1800 <= value <= 2200:
                raise QueryPlanValidationError(f"{path} must be a year or null")
    if start is not None and end is not None and start > end:
        raise QueryPlanValidationError("time_range.start_year must not exceed end_year")
    for field in ("hard_constraints", "soft_constraints"):
        if not isinstance(plan[field], list) or any(not isinstance(item, Mapping) for item in plan[field]):
            raise QueryPlanValidationError(f"{field} must be an array of objects")
        for index, item in enumerate(plan[field]):
            if not isinstance(item.get("name"), str) or not item["name"].strip():
                raise QueryPlanValidationError(f"{field}[{index}].name must be a non-empty string")
            if "value" not in item or not isinstance(item["value"], str):
                raise QueryPlanValidationError(f"{field}[{index}].value must be a string")
    if not isinstance(plan["budget"], Mapping):
        raise QueryPlanValidationError("budget must be an object")
    for field in ("max_iterations", "max_citation_depth", "max_subqueries_per_gap", "max_provider_calls"):
        _nonnegative_int(plan["budget"].get(field), f"budget.{field}")
    if not isinstance(plan["stop_strategy"], Mapping):
        raise QueryPlanValidationError("stop_strategy must be an object")
    for field in ("min_new_relevant",):
        _nonnegative_int(plan["stop_strategy"].get(field), f"stop_strategy.{field}")
    for field in ("min_subquery_coverage", "min_evidence_coverage"):
        _number(plan["stop_strategy"].get(field), f"stop_strategy.{field}", maximum=1.0)
    if not isinstance(plan["subqueries"], list) or not plan["subqueries"]:
        raise QueryPlanValidationError("subqueries must be a non-empty array")
    ids: set[str] = set()
    for item in plan["subqueries"]:
        validated = validate_subquery(item)
        if validated["subquery_id"] in ids:
            raise QueryPlanValidationError("subquery_id values must be unique")
        ids.add(validated["subquery_id"])
        parent = validated["parent_id"]
        if parent is not None and parent not in ids:
            raise QueryPlanValidationError("subquery parent_id must refer to an earlier subquery")
    _string_list(plan["gaps"], "gaps")
    invalid_gaps = set(plan["gaps"]) - GAP_CODES
    if invalid_gaps:
        raise QueryPlanValidationError(f"unknown gap codes: {sorted(invalid_gaps)}")
    return dict(plan)


class QueryPlan(dict[str, Any]):
    """可直接 ``json.dumps`` 的 QueryPlan。"""

    def __init__(self, data: Mapping[str, Any] | None = None, **fields: Any) -> None:
        payload = dict(data or {})
        payload.update(fields)
        validate_query_plan(payload)
        super().__init__(deepcopy(payload))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QueryPlan":
        return cls(data)

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self))

    @property
    def subqueries(self) -> list[dict[str, Any]]:
        return self["subqueries"]

    @property
    def gaps(self) -> list[str]:
        return self["gaps"]


__all__ = [
    "GAP_CODES",
    "QUERY_PLAN_SCHEMA",
    "QueryPlan",
    "QueryPlanValidationError",
    "SUBQUERY_KINDS",
    "validate_query_plan",
    "validate_subquery",
]
