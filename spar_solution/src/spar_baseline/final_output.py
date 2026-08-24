"""Build the submission-facing structured result from a P2 run."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


FINAL_SCHEMA = "spar.final.v1"
COMPONENTS = ("relevance", "constraint", "evidence", "quality", "citation", "novelty")
ZONES = {"high", "partial", "reserve"}


def _payload(run: Mapping[str, Any] | Any) -> dict[str, Any]:
    return run.to_dict() if hasattr(run, "to_dict") else deepcopy(dict(run))


def _papers(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("papers") or []
    if isinstance(value, Mapping):
        value = value.get("papers") or []
    return [deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]


def _citations(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = payload.get("citation") or payload.get("citations") or []
    if isinstance(value, Mapping):
        rounds = value.get("rounds")
        if isinstance(rounds, list):
            return [item for item in rounds if isinstance(item, Mapping)]
        return [value]
    return [item for item in value if isinstance(item, Mapping)]


def _zone(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "partial"
    return "reserve"


def build_final_selection(run: Mapping[str, Any] | Any, *, top_k: int = 20) -> dict[str, Any]:
    """Return ``spar.final.v1``; hard-excluded papers never enter results."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    payload = _payload(run)
    papers = _papers(payload)
    eligible = [
        paper for paper in papers
        if paper.get("status", {}).get("hard_constraints_pass") is not False
        and isinstance(paper.get("scores", {}).get("final"), (int, float))
    ]
    eligible.sort(key=lambda item: (float(item["scores"]["final"]), str(item.get("paper_id") or "")), reverse=True)
    selected = eligible[:top_k]
    verdicts = {
        str(item.get("paper_id")): item
        for item in payload.get("verdicts") or []
        if isinstance(item, Mapping) and item.get("paper_id")
    }
    results: list[dict[str, Any]] = []
    for rank, paper in enumerate(selected, 1):
        bib = paper.get("bibliography") or {}
        identifiers = paper.get("identifiers") or {}
        access = paper.get("access") or {}
        scores = paper.get("scores") or {}
        verdict = verdicts.get(str(paper["paper_id"]), {})
        final_score = float(scores["final"])
        results.append({
            "rank": rank,
            "paper_id": str(paper["paper_id"]),
            "title": str(bib.get("title") or ""),
            "year": bib.get("year"),
            "venue": bib.get("venue"),
            "doi": identifiers.get("doi"),
            "arxiv_id": identifiers.get("arxiv_id"),
            "landing_url": access.get("landing_url"),
            "relevance_zone": _zone(final_score),
            "final_score": final_score,
            "component_scores": {name: scores.get(name) for name in COMPONENTS},
            "evidence_refs": list(paper.get("evidence_refs") or []),
            "reason_codes": list(verdict.get("reason_codes") or []),
            "llm_judgement": deepcopy(verdict.get("llm_judgement")) if isinstance(verdict.get("llm_judgement"), Mapping) else None,
        })

    selected_ids = {item["paper_id"] for item in results}
    paper_by_id = {str(item.get("paper_id")): item for item in papers if item.get("paper_id")}
    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[str, str, str]] = set()
    graph_ids = set(selected_ids)
    for batch in _citations(payload):
        for raw in batch.get("edges") or []:
            if not isinstance(raw, Mapping):
                continue
            parent = str(raw.get("parent_paper_id") or "")
            child = str(raw.get("child_paper_id") or "")
            relation = str(raw.get("relation_type") or "related")
            if not parent or not child or not ({parent, child} & selected_ids):
                continue
            key = (parent, child, relation)
            if key in edge_keys:
                continue
            edge_keys.add(key)
            graph_ids.update((parent, child))
            edges.append({
                "parent_paper_id": parent,
                "child_paper_id": child,
                "relation_type": relation,
                "source": raw.get("source"),
                "depth": raw.get("depth"),
            })
    nodes = []
    for paper_id in sorted(graph_ids):
        paper = paper_by_id.get(paper_id) or {}
        nodes.append({
            "paper_id": paper_id,
            "title": str((paper.get("bibliography") or {}).get("title") or ""),
            "outside_topk": paper_id not in selected_ids,
        })

    manifest = payload.get("manifest") or payload.get("run_manifest") or {}
    cost = payload.get("cost") or (manifest.get("cost") if isinstance(manifest, Mapping) else {}) or {}
    zones = {name: sum(item["relevance_zone"] == name for item in results) for name in ("high", "partial", "reserve")}
    output = {
        "schema_version": FINAL_SCHEMA,
        "query": str(payload.get("query") or (manifest.get("query") if isinstance(manifest, Mapping) else "") or ""),
        "query_id": str((payload.get("query_plan") or {}).get("query_id") or (manifest.get("query_id") if isinstance(manifest, Mapping) else "") or ""),
        "results": results,
        "relation_graph": {"nodes": nodes, "edges": edges},
        "summary": {"selected": len(results), "excluded": len(papers) - len(eligible), "zones": zones, "citation_edges": len(edges)},
        "degraded": bool((isinstance(manifest, Mapping) and manifest.get("status") == "degraded") or payload.get("errors")),
        "cost": deepcopy(dict(cost)) if isinstance(cost, Mapping) else {},
    }
    return validate_final_selection(output)


def validate_final_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != FINAL_SCHEMA:
        raise ValueError(f"schema_version must be {FINAL_SCHEMA}")
    if not isinstance(value.get("results"), list):
        raise ValueError("results must be an array")
    for rank, item in enumerate(value["results"], 1):
        if not isinstance(item, Mapping) or item.get("rank") != rank or not item.get("paper_id"):
            raise ValueError("results must have contiguous ranks and paper_id")
        if item.get("relevance_zone") not in ZONES:
            raise ValueError("invalid relevance_zone")
        score = item.get("final_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= float(score) <= 1:
            raise ValueError("final_score must be between 0 and 1")
        if set(item.get("component_scores") or {}) != set(COMPONENTS):
            raise ValueError("component_scores must contain all six components")
    graph = value.get("relation_graph")
    if not isinstance(graph, Mapping) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise ValueError("relation_graph must contain nodes and edges arrays")
    if not isinstance(value.get("summary"), Mapping) or not isinstance(value.get("cost"), Mapping):
        raise ValueError("summary and cost must be objects")
    return deepcopy(dict(value))


__all__ = ["FINAL_SCHEMA", "build_final_selection", "validate_final_selection"]
