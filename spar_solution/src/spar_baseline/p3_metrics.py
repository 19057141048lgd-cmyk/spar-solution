"""P3 圆桌流程的可复现指标。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .p3_protocol import estimate_bytes, estimate_tokens


def evaluate_p3_run(run: Mapping[str, Any] | Any, *, gold_ids: Iterable[str] = (), k: int = 10) -> dict[str, Any]:
    payload = run.to_dict() if hasattr(run, "to_dict") else dict(run)
    papers = list(payload.get("selected") or payload.get("papers") or [])
    gold = {str(item) for item in gold_ids if str(item)}
    predicted = {str(item.get("paper_id")) for item in papers[:k] if item.get("paper_id")}
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    messages = list(payload.get("messages") or [])
    return {
        "k": k,
        "papers": len(papers),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "agent_count": len({str(item.get("sender")) for item in messages} | {str(item.get("receiver")) for item in messages}),
        "message_count": len(messages),
        "message_bytes": sum(estimate_bytes(item) for item in messages),
        "message_tokens_estimate": sum(estimate_tokens(item) for item in messages),
        "artifact_count": int((payload.get("stats") or {}).get("artifact_count", 0)),
        "source_errors": len(payload.get("errors") or []),
        "stop_reason": (payload.get("stop") or {}).get("reason_code"),
    }


def compare_p3_ablation(enabled_run: Mapping[str, Any] | Any, disabled_run: Mapping[str, Any] | Any) -> dict[str, Any]:
    enabled = evaluate_p3_run(enabled_run)
    disabled = evaluate_p3_run(disabled_run)
    enabled_payload = enabled_run.to_dict() if hasattr(enabled_run, "to_dict") else dict(enabled_run)
    disabled_payload = disabled_run.to_dict() if hasattr(disabled_run, "to_dict") else dict(disabled_run)
    enabled_edges = len((enabled_payload.get("citation") or {}).get("edges") or [])
    disabled_edges = len((disabled_payload.get("citation") or {}).get("edges") or [])
    return {"citation_enabled": enabled, "citation_disabled": disabled, "delta": {"citation_edges": enabled_edges - disabled_edges, "selected": enabled["papers"] - disabled["papers"]}, "acceptance": {"citation_called": bool((enabled_payload.get("citation") or {}).get("stats", {}).get("enabled")) and not bool((disabled_payload.get("citation") or {}).get("stats", {}).get("enabled")), "citation_artifact_present": enabled_edges > 0}}


__all__ = ["compare_p3_ablation", "evaluate_p3_run"]
