"""P2 可回放运行的指标与 citation 消融比较。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any

from .metrics import evaluate_at_k


def _identity_record(item: str | Mapping[str, Any]) -> dict[str, Any]:
    """把旧式字符串 ID 补成可供统一 identity 层匹配的最小记录。"""

    record = dict(item) if isinstance(item, Mapping) else {"paper_id": str(item)}
    paper_id = str(record.get("paper_id") or "").strip()
    identifiers = dict(record.get("identifiers") or {})
    lowered = paper_id.casefold()
    if lowered.startswith("doi:"):
        identifiers["doi"] = identifiers.get("doi") or paper_id[4:]
    elif lowered.startswith("arxiv:"):
        identifiers["arxiv_id"] = identifiers.get("arxiv_id") or paper_id[6:]
    elif lowered.startswith("openalex:"):
        identifiers["openalex_id"] = identifiers.get("openalex_id") or paper_id[9:]
    elif re.fullmatch(r"10\.\d{4,9}/\S+", paper_id, flags=re.IGNORECASE):
        identifiers["doi"] = identifiers.get("doi") or paper_id
    elif re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", paper_id, flags=re.IGNORECASE):
        identifiers["arxiv_id"] = identifiers.get("arxiv_id") or paper_id
    elif re.fullmatch(r"W\d+", paper_id, flags=re.IGNORECASE):
        identifiers["openalex_id"] = identifiers.get("openalex_id") or paper_id
    elif paper_id:
        # P2 旧 artifact 的 fixture/local paper_id 视作来源稳定 ID，保持旧签名兼容。
        identifiers["unique_id"] = identifiers.get("unique_id") or paper_id
    record["identifiers"] = identifiers
    return record


def _run_stats(run: Mapping[str, Any]) -> dict[str, Any]:
    recall = list(run.get("recall") or [])
    citation = list(run.get("citation") or run.get("citations") or [])
    recall_calls = sum((list(item.get("calls") or []) for item in recall), [])
    citation_calls = sum((list(item.get("calls") or []) for item in citation), [])
    calls = [*recall_calls, *citation_calls]
    manifest = run.get("manifest") or run.get("run_manifest") or {}
    cost = (manifest if isinstance(manifest, Mapping) else {}).get("cost") or run.get("cost") or {}
    cost = cost if isinstance(cost, Mapping) else {}
    return {
        "api_calls": sum(int(call.get("api_calls") or 1) for call in calls if call.get("ok") is not None),
        "recall_api_calls": sum(int(call.get("api_calls") or 1) for call in recall_calls),
        "citation_api_calls": sum(int(call.get("api_calls") or 1) for call in citation_calls),
        "successful_calls": sum(int(call.get("api_calls") or 1) for call in calls if call.get("ok") is True),
        "source_errors": len(run.get("errors") or []),
        "latency_ms": round(sum(float(call.get("latency_ms") or 0) for call in calls), 3),
        "citation_edges": sum(len(item.get("edges") or []) for item in citation),
        "citation_papers": sum(len(item.get("papers") or []) for item in citation),
        "provider_calls": dict(cost.get("provider_calls") or {}),
        "llm_calls": int(cost.get("llm_calls") or 0),
        "prompt_tokens": int(cost.get("prompt_tokens") or 0),
        "completion_tokens": int(cost.get("completion_tokens") or 0),
        "total_tokens": int(cost.get("total_tokens") or 0),
        "llm_failures": int(cost.get("llm_failures") or 0),
        "wall_ms": float(cost.get("wall_ms") or 0),
        "per_stage_ms": dict(cost.get("per_stage_ms") or {}),
    }


def evaluate_p2_run(run: Mapping[str, Any] | Any, *, gold_ids: Iterable[str | Mapping[str, Any]] = (), cutoffs: tuple[int, ...] = (10, 20)) -> dict[str, Any]:
    """使用统一 identity 规则计算单个 P2 run 的检索、证据、引用和成本指标。"""
    payload = run.to_dict() if hasattr(run, "to_dict") else dict(run)
    raw_papers = list((payload.get("papers") or {}).get("papers", [])) if isinstance(payload.get("papers"), Mapping) else list(payload.get("papers") or [])
    papers = [_identity_record(item) for item in raw_papers]
    gold = [_identity_record(item) for item in gold_ids if isinstance(item, Mapping) or str(item).strip()]
    by_cutoff = {str(k): evaluate_at_k(papers, gold, k=k, provider_errors=payload.get("errors") or []) for k in cutoffs}
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
    if gold and papers:
        full_ranking = evaluate_at_k(papers, gold, k=len(papers))
        if full_ranking["matches"]:
            mrr = 1.0 / (min(item["prediction_index"] for item in full_ranking["matches"]) + 1)
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
