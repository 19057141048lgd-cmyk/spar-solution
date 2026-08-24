"""P1 可执行论文检索评测指标。

本模块把身份匹配和指标计算分开：论文先按固定身份规则去重，再在
截断后的预测集合和 Gold 集合之间做一次一配匹配。Provider 错误是运行
事件，不是 PaperDoc，因此只能通过 ``provider_errors``/运行元数据传入，
不会被计算为论文 FP。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .identity import match_papers


def _record_ref(record: Mapping[str, Any]) -> dict[str, Any]:
    """返回不含正文的可追溯论文引用，便于写入评测 artifact。"""

    identifiers = record.get("identifiers")
    bibliography = record.get("bibliography")
    return {
        "paper_id": record.get("paper_id"),
        "identifiers": dict(identifiers) if isinstance(identifiers, Mapping) else {},
        "title": bibliography.get("title") if isinstance(bibliography, Mapping) else record.get("title"),
    }


def deduplicate_papers(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """按 ``identity.match_papers`` 的规则去重。

    只有身份状态为 ``matched`` 才会去重；``ambiguous`` 记录保留，并在
    诊断字段中记录，避免把无法确认的论文静默合并。
    """

    unique: list[Mapping[str, Any]] = []
    duplicate_indices: list[int] = []
    ambiguous_indices: list[int] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"paper record at index {index} must be an object")
        duplicate = False
        ambiguous = False
        for existing in unique:
            try:
                result = match_papers(record, existing)
            except (TypeError, ValueError, KeyError):
                result = {"status": "ambiguous", "reason": "invalid_identity_metadata"}
            if result.get("status") == "matched":
                duplicate = True
                break
            if result.get("status") == "ambiguous":
                ambiguous = True
        if duplicate:
            duplicate_indices.append(index)
            continue
        if ambiguous:
            ambiguous_indices.append(index)
        unique.append(record)
    return {
        "records": list(unique),
        "duplicates_removed": len(duplicate_indices),
        "duplicate_indices": duplicate_indices,
        "ambiguous_indices": ambiguous_indices,
    }


# Short alias used by experiment runners.
deduplicate_records = deduplicate_papers


def _precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if tp + fp else 0.0


def _recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if tp + fn else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = _precision(tp, fp)
    recall = _recall(tp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _match_at_k(predictions: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]], k: int) -> dict[str, Any]:
    prediction_dedup = deduplicate_papers(predictions)
    gold_dedup = deduplicate_papers(gold)
    predicted = prediction_dedup["records"][:k]
    gold_records = gold_dedup["records"]
    used_gold: set[int] = set()
    matches: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for prediction_index, prediction in enumerate(predicted):
        matched: tuple[int, dict[str, Any]] | None = None
        ambiguous_candidates: list[tuple[int, dict[str, Any]]] = []
        for gold_index, gold_record in enumerate(gold_records):
            if gold_index in used_gold:
                continue
            try:
                result = match_papers(prediction, gold_record)
            except (TypeError, ValueError, KeyError):
                result = {"status": "ambiguous", "reason": "invalid_identity_metadata"}
            if result.get("status") == "matched":
                matched = (gold_index, result)
                break
            if result.get("status") == "ambiguous":
                ambiguous_candidates.append((gold_index, result))
        if matched is not None:
            gold_index, result = matched
            used_gold.add(gold_index)
            matches.append({
                "prediction_index": prediction_index,
                "gold_index": gold_index,
                "prediction": _record_ref(prediction),
                "gold": _record_ref(gold_records[gold_index]),
                "matched_by": result.get("matched_by"),
                "identity_key": result.get("identity_key"),
                "reason": result.get("reason"),
            })
        elif ambiguous_candidates:
            ambiguous.append({
                "prediction_index": prediction_index,
                "prediction": _record_ref(prediction),
                "gold_indices": [index for index, _ in ambiguous_candidates],
                "reasons": [result.get("reason") for _, result in ambiguous_candidates],
            })

    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(gold_records) - len(used_gold)
    return {
        **_counts(tp, fp, fn),
        "cutoff": k,
        "predicted_count": len(predicted),
        "gold_count": len(gold_records),
        "matches": matches,
        "ambiguous": ambiguous,
        "duplicates_removed": prediction_dedup["duplicates_removed"],
        "gold_duplicates_removed": gold_dedup["duplicates_removed"],
        "prediction_ambiguous_count": len(prediction_dedup["ambiguous_indices"]),
    }


def evaluate_at_k(
    predictions: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    k: int = 10,
    provider_errors: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """计算一个查询在指定 K 的指标。

    ``provider_errors`` 只作为审计计数返回；不会进入 predicted 集合，也
    不会增加 FP。无 Gold 或无预测时，除法结果按协议安全返回 0.0。
    """

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    result = _match_at_k(predictions, gold, k)
    result["source_errors_count"] = len(provider_errors or [])
    return result


def _split_run(value: Any) -> tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]:
    """支持直接传论文列表，或传 search artifact。"""

    if isinstance(value, Mapping) and "papers" in value:
        metadata = dict(value.get("stats") or {})
        metadata.setdefault("source_errors", value.get("source_errors") or [])
        return value.get("papers") or [], metadata
    return value or [], {}


def _source_counts(metadata: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    providers = metadata.get("providers")
    if isinstance(providers, Sequence) and not isinstance(providers, (str, bytes)):
        for item in providers:
            if not isinstance(item, Mapping):
                continue
            source = str(item.get("source") or "unknown")
            counts[source] = counts.get(source, 0) + int(item.get("records") or 0)
    direct = metadata.get("source_counts")
    if isinstance(direct, Mapping):
        for source, count in direct.items():
            counts[str(source)] = int(count or 0)
    return counts


def evaluate_queries(
    predictions_by_query: Mapping[str, Any],
    gold_by_query: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    k_values: Sequence[int] = (10, 20),
    run_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """聚合多个查询，输出每个 K 的 Macro-F1、Micro-F1 和运行统计。"""

    ks = tuple(k_values)
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("k_values must contain positive integers")
    query_ids = list(dict.fromkeys([*gold_by_query.keys(), *predictions_by_query.keys()]))
    per_query: dict[str, dict[str, Any]] = {}
    aggregate: dict[int, dict[str, Any]] = {}
    for query_id in query_ids:
        predictions, artifact_metadata = _split_run(predictions_by_query.get(query_id, []))
        gold_records = gold_by_query.get(query_id, []) or []
        metadata = dict(artifact_metadata)
        metadata.update((run_metadata or {}).get(query_id) or {})
        errors = metadata.get("source_errors") or []
        per_query[query_id] = {
            "metrics": {str(k): evaluate_at_k(predictions, gold_records, k=k, provider_errors=errors) for k in ks},
            "latency_ms": metadata.get("latency_ms", metadata.get("latency")),
            "api_calls": int(metadata.get("api_calls") or metadata.get("api_call_count") or 0),
            "source_counts": _source_counts(metadata),
            "dedup_count": int(metadata.get("dedup_count") or 0),
            "source_errors_count": len(errors) if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)) else int(errors or 0),
        }
    for k in ks:
        items = [per_query[qid]["metrics"][str(k)] for qid in query_ids]
        tp = sum(item["tp"] for item in items)
        fp = sum(item["fp"] for item in items)
        fn = sum(item["fn"] for item in items)
        values = _counts(tp, fp, fn)
        values["cutoff"] = k
        values["macro_f1"] = sum(item["f1"] for item in items) / len(items) if items else 0.0
        values["query_count"] = len(items)
        aggregate[k] = values

    latencies = [item["latency_ms"] for item in per_query.values() if isinstance(item["latency_ms"], (int, float))]
    source_totals: dict[str, int] = {}
    for item in per_query.values():
        for source, count in item["source_counts"].items():
            source_totals[source] = source_totals.get(source, 0) + count
    return {
        "per_query": per_query,
        "by_cutoff": {str(k): aggregate[k] for k in ks},
        "macro_f1": {str(k): aggregate[k]["macro_f1"] for k in ks},
        "micro_f1": {str(k): aggregate[k]["f1"] for k in ks},
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "api_call_count": sum(item["api_calls"] for item in per_query.values()),
        "source_return_counts": source_totals,
        # 优先使用检索层统计的跨来源合并数；评测层重复预测作为兜底。
        "dedup_count": sum(
            item["dedup_count"] or item["metrics"][str(max(ks))]["duplicates_removed"]
            for item in per_query.values()
        ),
        "source_errors_count": sum(item["source_errors_count"] for item in per_query.values()),
    }


def evaluate_modes(
    mode_results: Mapping[str, Mapping[str, Any]],
    gold_by_query: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    k_values: Sequence[int] = (10, 20),
) -> dict[str, Any]:
    """对 A/B/C/D 等检索模式返回同一套可比较指标。"""

    return {
        mode: evaluate_queries(results, gold_by_query, k_values=k_values)
        for mode, results in mode_results.items()
    }


__all__ = [
    "deduplicate_papers",
    "deduplicate_records",
    "evaluate_at_k",
    "evaluate_queries",
    "evaluate_modes",
]
