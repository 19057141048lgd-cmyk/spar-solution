"""P2 可回放运行的指标与 citation 消融比较。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _paper_id(item: Mapping[str, Any]) -> str:
    return str(item.get("paper_id") or "")


def _f1(precision: float, recall: float) -> float:
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


def _cutoff(papers: list[Mapping[str, Any]], gold: set[str], k: int) -> dict[str, Any]:
    predicted = {_paper_id(item) for item in papers[:k] if _paper_id(item)}
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"k": k, "tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": _f1(precision, recall)}


def _run_stats(run: Mapping[str, Any]) -> dict[str, Any]:
    recall = list(run.get("recall") or [])
    citation = list(run.get("citation") or run.get("citations") or [])
    recall_calls = sum((list(item.get("calls") or []) for item in recall), [])
    citation_calls = sum((list(item.get("calls") or []) for item in citation), [])
    calls = [*recall_calls, *citation_calls]
    return {
        "api_calls": sum(1 for call in calls if call.get("ok") is not None),
        "recall_api_calls": len(recall_calls),
        "citation_api_calls": len(citation_calls),
        "successful_calls": sum(1 for call in calls if call.get("ok") is True),
        "source_errors": len(run.get("errors") or []),
        "latency_ms": round(sum(float(call.get("latency_ms") or 0) for call in calls), 3),
        "citation_edges": sum(len(item.get("edges") or []) for item in citation),
        "citation_papers": sum(len(item.get("papers") or []) for item in citation),
    }


def evaluate_p2_run(run: Mapping[str, Any] | Any, *, gold_ids: Iterable[str] = (), cutoffs: tuple[int, ...] = (10, 20)) -> dict[str, Any]:
    """计算单个 P2 run 的检索、证据、引用和成本指标。"""
    payload = run.to_dict() if hasattr(run, "to_dict") else dict(run)
    papers = list((payload.get("papers") or {}).get("papers", [])) if isinstance(payload.get("papers"), Mapping) else list(payload.get("papers") or [])
    gold = {str(item) for item in gold_ids if str(item)}
    by_cutoff = {str(k): _cutoff(papers, gold, k) for k in cutoffs}
    verdicts = list(payload.get("verdicts") or [])
    evidence = list(payload.get("evidence") or [])
    citation = list(payload.get("citation") or payload.get("citations") or [])
    edges = sum(len(item.get("edges") or []) for item in citation)
    evidence_covered_ids = {
        str(item.get("paper_id"))
        for item in evidence
        if item.get("evidence_status") not in {None, "unavailable"} and item.get("paper_id")
    }
    evidence_coverage = len(evidence_covered_ids) / max(1, len(papers))
    citation_covered_ids = {str(edge.get("parent_paper_id")) for item in citation for edge in item.get("edges") or []}
    citation_coverage = len(citation_covered_ids) / max(1, len(papers))
    mrr = 0.0
    if gold:
        for rank, paper in enumerate(papers, 1):
            if _paper_id(paper) in gold:
                mrr = 1.0 / rank
                break
    return {"papers": len(papers), "verdicts": len(verdicts), "by_cutoff": by_cutoff, "mrr": round(mrr, 6), "citation_coverage": round(min(1.0, citation_coverage), 6), "evidence_coverage": round(min(1.0, evidence_coverage), 6), "stats": {**_run_stats(payload), "citation_edges": edges}}


def compare_citation_ablation(enabled_run: Mapping[str, Any] | Any, disabled_run: Mapping[str, Any] | Any) -> dict[str, Any]:
    enabled = evaluate_p2_run(enabled_run)
    disabled = evaluate_p2_run(disabled_run)
    return {
        "citation_enabled": enabled,
        "citation_disabled": disabled,
        "delta": {
            "citation_edges": enabled["stats"]["citation_edges"] - disabled["stats"]["citation_edges"],
            "citation_papers": enabled["stats"]["citation_papers"] - disabled["stats"]["citation_papers"],
            "papers": enabled["papers"] - disabled["papers"],
            "evidence_coverage": round(enabled["evidence_coverage"] - disabled["evidence_coverage"], 6),
        },
        "acceptance": {"citation_called": enabled["stats"]["citation_api_calls"] > disabled["stats"]["citation_api_calls"], "citation_artifact_present": enabled["stats"]["citation_edges"] > 0},
    }


__all__ = ["compare_citation_ablation", "evaluate_p2_run"]
