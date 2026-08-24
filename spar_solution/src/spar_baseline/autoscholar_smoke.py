"""AutoScholarQuery 的可复现 arXiv smoke 评测。"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .experiment import gold_paper_doc
from .identity import normalize_arxiv_id
from .metrics import evaluate_at_k, evaluate_queries
from .providers.arxiv import ArxivProvider
from .providers.base import ProviderError, ProviderResult
from .query_planner import QueryPlanner


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_rows(path: str | Path, *, offset: int, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"dataset line {line_number} must be an object")
        rows.append(value)
    return rows[offset:offset + limit]


def _cutoff_year(row: Mapping[str, Any]) -> int | None:
    source_meta = row.get("source_meta")
    value = source_meta.get("published_time") if isinstance(source_meta, Mapping) else None
    text = str(value or "")
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else None


def _gold_docs(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for value in row.get("answer_arxiv_id") or []:
        arxiv_id = normalize_arxiv_id(value)
        if arxiv_id:
            docs.append(gold_paper_doc({
                "paper_id": f"arxiv:{arxiv_id}",
                "identifiers": {"arxiv_id": arxiv_id},
            }, source="autoscholar_gold"))
    return docs


def _predicted_ids(records: list[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for record in records:
        identifiers = record.get("identifiers")
        value = identifiers.get("arxiv_id") if isinstance(identifiers, Mapping) else None
        arxiv_id = normalize_arxiv_id(value)
        if arxiv_id and arxiv_id not in values:
            values.append(arxiv_id)
    return values


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_autoscholar_smoke(
    dataset: str | Path,
    output: str | Path,
    *,
    offset: int = 0,
    limit: int = 5,
    page_size: int = 10,
    sleep_seconds: float = 3.1,
    provider: Any | None = None,
    planner: QueryPlanner | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """运行固定预算的 arXiv smoke；失败查询不参与相关性指标。"""

    if offset < 0 or limit <= 0 or page_size <= 0 or sleep_seconds < 0:
        raise ValueError("offset/sleep must be non-negative; limit/page_size must be positive")
    rows = _read_rows(dataset, offset=offset, limit=limit)
    if not rows:
        raise ValueError("no dataset rows selected")
    provider = provider or ArxivProvider()
    planner = planner or QueryPlanner()
    started_at = _utc_now()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    predictions: dict[str, list[Mapping[str, Any]]] = {}
    gold: dict[str, list[Mapping[str, Any]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    api_calls = 0

    for index, row in enumerate(rows):
        qid = str(row.get("qid") or f"row_{offset + index}")
        question = str(row.get("question") or "").strip()
        row_gold = _gold_docs(row)
        detail: dict[str, Any] = {
            "query_id": qid,
            "question": question,
            "gold_arxiv_ids": [doc["identifiers"]["arxiv_id"] for doc in row_gold],
            "status": "pending",
            "api_calls": 0,
        }
        try:
            plan = planner.plan(question)
            query = str(plan["subqueries"][0]["query_text"])
            detail["planned_query"] = query
        except Exception as exc:
            error = {"query_id": qid, "stage": "query_plan", "code": "plan_error", "message": f"{type(exc).__name__}: {str(exc)[:160]}"}
            errors.append(error)
            detail.update({"status": "plan_error", "error": error})
            results.append(detail)
            continue

        if api_calls:
            sleep_fn(sleep_seconds)
        started = clock()
        api_calls += 1
        detail["api_calls"] = 1
        try:
            response = provider.search(query, page_size=page_size, cutoff_year=_cutoff_year(row))
            if not isinstance(response, ProviderResult) or not response.ok:
                raise ProviderError("arxiv", "parse", "provider returned an invalid result")
            latency_ms = round((clock() - started) * 1000, 3)
            metric = evaluate_at_k(response.records, row_gold, k=10)
            detail.update({
                "status": "ok",
                "query_expression": response.provenance.get("query_expression"),
                "returned_count": len(response.records),
                "latency_ms": latency_ms,
                "predicted_arxiv_ids": _predicted_ids(response.records),
                "metrics_at_10": metric,
            })
            predictions[qid] = response.records
            gold[qid] = row_gold
            metadata[qid] = {
                "latency_ms": latency_ms,
                "api_calls": 1,
                "source_counts": {"arxiv": len(response.records)},
            }
        except Exception as exc:
            if isinstance(exc, ProviderError):
                error = {"query_id": qid, "stage": "arxiv", **exc.to_dict()}
            else:
                error = {"query_id": qid, "stage": "arxiv", "source": "arxiv", "code": "provider_error", "message": f"{type(exc).__name__}: {str(exc)[:160]}"}
            errors.append(error)
            detail.update({
                "status": "provider_error",
                "latency_ms": round((clock() - started) * 1000, 3),
                "error": error,
                "metrics_at_10": None,
            })
        results.append(detail)

    evaluation = evaluate_queries(predictions, gold, k_values=(10,), run_metadata=metadata)
    query_summaries = [{
        "query_id": item["query_id"],
        "status": item["status"],
        "query_expression": item.get("query_expression"),
        "returned_count": item.get("returned_count"),
        "latency_ms": item.get("latency_ms"),
    } for item in results]
    summary = {
        "schema_version": "autoscholar.smoke.summary.v1",
        "selected_queries": len(rows),
        "evaluated_queries": len(predictions),
        "failed_queries": len(rows) - len(predictions),
        "metrics_at_10": evaluation["by_cutoff"]["10"],
        "average_latency_ms": evaluation["average_latency_ms"],
        "api_calls": sum(item["api_calls"] for item in results),
        "returned_count": sum(int(item.get("returned_count") or 0) for item in results),
        "queries": query_summaries,
    }
    manifest = {
        "schema_version": "autoscholar.smoke.run.v1",
        "dataset": str(Path(dataset).resolve()),
        "output": str(Path(output).resolve()),
        "offset": offset,
        "limit": limit,
        "page_size": page_size,
        "sleep_seconds": sleep_seconds,
        "provider": "arxiv",
        "provider_revision": "or_wide_recall_2026-08-24",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "query_expressions": [{"query_id": item["query_id"], "query_expression": item.get("query_expression")} for item in results],
        "prior_baseline": {
            "status": "invalid",
            "reason": "旧产物由已废弃的逐词 AND 查询生成，且部分 Gold 曾继承 mock DOI；不得用于效果对比。",
        },
    }
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / "summary.json", summary)
    _write_json(output_path / "results.json", {"schema_version": "autoscholar.smoke.results.v1", "queries": results})
    _write_json(output_path / "errors.json", {"schema_version": "autoscholar.smoke.errors.v1", "errors": errors})
    _write_json(output_path / "run_manifest.json", manifest)
    return {"summary": summary, "results": results, "errors": errors, "manifest": manifest}


__all__ = ["run_autoscholar_smoke"]
