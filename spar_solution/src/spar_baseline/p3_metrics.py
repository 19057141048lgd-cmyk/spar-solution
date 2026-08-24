"""P3 圆桌流程的可复现指标。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import re

from .identity import normalize_arxiv_id, normalize_doi
from .metrics import evaluate_at_k
from .p3_protocol import estimate_bytes, estimate_tokens


def _gold_record(value: Any, predictions: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Convert a compact Gold identifier to the canonical PaperDoc identity shape."""

    text = str(value or "").strip()
    for prediction in predictions:
        if str(prediction.get("paper_id") or "") == text:
            identifiers = dict(prediction.get("identifiers") or {})
            for key in ("doi", "arxiv_id", "openalex_id"):
                if prediction.get(key) and not identifiers.get(key):
                    identifiers[key] = prediction[key]
            bibliography = dict(prediction.get("bibliography") or {})
            for key in ("title", "year", "venue"):
                if prediction.get(key) is not None and key not in bibliography:
                    bibliography[key] = prediction[key]
            return {"paper_id": text, "identifiers": identifiers, "bibliography": bibliography}
    lowered = text.casefold()
    identifiers: dict[str, str] = {}
    if lowered.startswith(("doi:", "https://doi.org/")) or lowered.startswith("10."):
        identifiers["doi"] = normalize_doi(text) or text
    elif lowered.startswith(("arxiv:", "https://arxiv.org/", "http://arxiv.org/")) or re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7})(?:v\d+)?", text, re.IGNORECASE):
        identifiers["arxiv_id"] = normalize_arxiv_id(text) or text
    elif lowered.startswith(("openalex:", "https://openalex.org/")) or re.fullmatch(r"w\d+", text, re.IGNORECASE):
        identifiers["openalex_id"] = text.split(":", 1)[-1].rstrip("/")
    elif ":" in text:
        field, value_part = text.split(":", 1)
        if field.casefold() in {"s2", "s2_id", "local", "local_id", "unique", "unique_id", "sciverse", "sciverse_doc_id"}:
            identifiers["s2_id" if field.casefold() in {"s2", "s2_id"} else "local_id" if field.casefold() in {"local", "local_id"} else "unique_id"] = value_part
        else:
            identifiers["unique_id"] = text
    else:
        identifiers["unique_id"] = text
    return {"paper_id": text, "identifiers": identifiers, "bibliography": {"title": text}}


def _as_paper_doc(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize spar.final.v1 rows back to the PaperDoc shape for matching."""

    if isinstance(value.get("bibliography"), Mapping):
        return dict(value)
    identifiers = dict(value.get("identifiers") or {})
    for key in ("doi", "arxiv_id", "openalex_id"):
        if value.get(key):
            identifiers[key] = value[key]
    return {
        "schema_version": "paperdoc.v1",
        "paper_id": str(value.get("paper_id") or ""),
        "identifiers": identifiers,
        "bibliography": {key: value.get(key) for key in ("title", "year", "venue") if value.get(key) is not None},
        "status": {"hard_constraints_pass": True},
    }


def evaluate_p3_run(run: Mapping[str, Any] | Any, *, gold_ids: Iterable[str] = (), k: int = 10) -> dict[str, Any]:
    payload = run.to_dict() if hasattr(run, "to_dict") else dict(run)
    final = payload.get("final_selection") or {}
    papers = [_as_paper_doc(item) for item in (final.get("results") or []) if isinstance(item, Mapping)] if isinstance(final, Mapping) else []
    if not papers:
        papers = list(payload.get("selected") or payload.get("papers") or [])
    metric = evaluate_at_k(papers, [_gold_record(item, papers) for item in gold_ids if str(item)], k=k, provider_errors=payload.get("errors") or [])
    messages = list(payload.get("messages") or [])
    cost = payload.get("cost") or (payload.get("manifest") or {}).get("cost") or {}
    stats = payload.get("stats") or {}
    provider_calls = cost.get("provider_calls", 0) if isinstance(cost, Mapping) else 0
    if isinstance(provider_calls, Mapping):
        provider_calls_total = sum(int(value or 0) for value in provider_calls.values())
    else:
        provider_calls_total = int(provider_calls or 0)
    return {
        "k": k,
        "papers": len(papers),
        "tp": metric["tp"],
        "fp": metric["fp"],
        "fn": metric["fn"],
        "precision": round(metric["precision"], 6),
        "recall": round(metric["recall"], 6),
        "f1": round(metric["f1"], 6),
        "agent_count": len(({str(item.get("sender")) for item in messages} | {str(item.get("receiver")) for item in messages}) & {"planner", "retriever", "citation_explorer", "evidence_judge", "arbiter"}),
        "message_count": len(messages),
        "message_bytes": sum(estimate_bytes(item) for item in messages),
        "message_tokens_estimate": sum(estimate_tokens(item) for item in messages),
        "artifact_count": int(stats.get("artifact_count", 0)),
        "provider_calls": provider_calls_total,
        "llm_calls": int(cost.get("llm_calls", 0) or 0) if isinstance(cost, Mapping) else 0,
        "prompt_tokens": int(cost.get("prompt_tokens", 0) or 0) if isinstance(cost, Mapping) else 0,
        "completion_tokens": int(cost.get("completion_tokens", 0) or 0) if isinstance(cost, Mapping) else 0,
        "total_tokens": int(cost.get("total_tokens", 0) or 0) if isinstance(cost, Mapping) else 0,
        "latency_ms": float(cost.get("wall_ms", cost.get("latency_ms", 0.0)) or 0.0) if isinstance(cost, Mapping) else 0.0,
        "source_errors": metric["source_errors_count"],
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


def compare_pipeline_runs(runs: Mapping[str, Mapping[str, Any] | Any], *, gold_ids: Iterable[str] = (), k: int = 10) -> dict[str, Any]:
    """Compare P2/P3/communication variants with one identity/metric path."""

    return {name: evaluate_p3_run(run, gold_ids=gold_ids, k=k) for name, run in runs.items()}


def compare_p2_p3(p2_run: Mapping[str, Any] | Any, p3_run: Mapping[str, Any] | Any, *, gold_ids: Iterable[str] = (), k: int = 10) -> dict[str, Any]:
    """统一身份和成本口径比较 P2 与 P3，禁止只比较一个 F1 数字。"""

    values = compare_pipeline_runs({"p2": p2_run, "p3": p3_run}, gold_ids=gold_ids, k=k)
    return {"runs": values, "delta": {key: values["p3"].get(key, 0) - values["p2"].get(key, 0) for key in ("recall", "f1", "latency_ms", "provider_calls", "total_tokens")}}


def compare_communication(structured_run: Mapping[str, Any] | Any, long_text_run: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Report structured versus long-text communication cost without inventing gains."""

    def snapshot(value: Mapping[str, Any] | Any) -> dict[str, Any]:
        payload = value.to_dict() if hasattr(value, "to_dict") else dict(value)
        stats = payload.get("stats") or {}
        embedded = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
        return {
            "message_bytes": int(stats.get("protocol_message_bytes") or stats.get("message_bytes") or 0),
            "message_tokens_estimate": int(stats.get("protocol_message_tokens_estimate") or stats.get("message_tokens_estimate") or 0),
            "llm_calls": int((payload.get("cost") or {}).get("llm_calls") or 0),
            "total_tokens": int((payload.get("cost") or {}).get("total_tokens") or 0),
            "recall": float(embedded.get("recall", 0.0) or 0.0),
            "f1": float(embedded.get("f1", 0.0) or 0.0),
            "latency_ms": float(embedded.get("latency_ms", (payload.get("cost") or {}).get("wall_ms", 0.0)) or 0.0),
            "provider_calls": int(embedded.get("provider_calls", (payload.get("cost") or {}).get("provider_calls", 0)) or 0) if not isinstance((payload.get("cost") or {}).get("provider_calls"), Mapping) else sum(int(item or 0) for item in (payload.get("cost") or {}).get("provider_calls", {}).values()),
        }

    structured = snapshot(structured_run)
    long_text = snapshot(long_text_run)
    return {"structured": structured, "long_text": long_text, "delta": {key: long_text[key] - structured[key] for key in structured}}


def long_text_baseline(run: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Construct a deterministic counterfactual that repeats PaperDocs in messages."""

    payload = run.to_dict() if hasattr(run, "to_dict") else dict(run)
    papers = list(payload.get("papers") or [])
    messages = list(payload.get("messages") or [])
    repeated = sum(estimate_bytes({"message_type": item.get("type"), "papers": papers}) for item in messages)
    return {"stats": {"message_bytes": repeated, "message_tokens_estimate": estimate_tokens({"papers": papers}) * max(1, len(messages))}, "cost": dict(payload.get("cost") or {}), "metrics": evaluate_p3_run(payload)}


__all__ = ["compare_communication", "compare_p2_p3", "compare_pipeline_runs", "compare_p3_ablation", "evaluate_p3_run", "long_text_baseline"]
