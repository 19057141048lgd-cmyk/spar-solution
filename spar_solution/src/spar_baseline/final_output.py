"""Build the submission-facing structured result (spar.final.v2).

v2 按公开测试集（PaSa 官方口径）校准交卷方式：
- ``results`` 是**提交集合**：按 ``select_threshold`` 选出的论文（非固定
  top-K），对应官方 document-level Precision/Recall/F1 的被测对象；
- ``ranked_pool`` 是按分数排序的完整候选前 ``pool_k`` 篇，供官方
  Recall@20/50/100 计分；
- ``selection_rule`` 记录阈值与规则，可审计、可复现；
- 分数基准兼容两种管线：P2 用 ``scores.final``，搜索树用 ``scores.relevance``。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


FINAL_SCHEMA = "spar.final.v2"
LEGACY_SCHEMA = "spar.final.v1"
COMPONENTS = ("relevance", "constraint", "evidence", "quality", "citation", "novelty")
ZONES = {"high", "partial", "reserve"}
# 阈值按分数基准区分：搜索树 relevance 是 LLM 五档量规分（0.9+ 常见），
# P2 final 是六分量加权分（现实上限 ~0.6）。同一阈值会清空另一种基准的
# 提交集合。0.9/8 来自 50 题存档池离线扫描（selected_f1 0.081→0.174）。
DEFAULT_THRESHOLD_BY_BASIS = {"final": 0.55, "relevance": 0.9}
DEFAULT_SELECT_THRESHOLD = 0.9
DEFAULT_MAX_SELECTED = 8
DEFAULT_POOL_K = 50


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


def _score_of(paper: Mapping[str, Any]) -> float | None:
    """提交分数基准：优先 final（P2 管线），退 relevance（搜索树管线）。"""

    scores = paper.get("scores") or {}
    for key in ("final", "relevance"):
        value = scores.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def build_final_selection(
    run: Mapping[str, Any] | Any,
    *,
    top_k: int | None = None,
    select_threshold: float | None = None,
    max_selected: int = DEFAULT_MAX_SELECTED,
    pool_k: int = DEFAULT_POOL_K,
) -> dict[str, Any]:
    """返回 ``spar.final.v2`` 交付物。

    默认按阈值选集合（官方口径）；``select_threshold=None`` 时按分数基准
    自动取默认（final→0.55，relevance→0.9，见 DEFAULT_THRESHOLD_BY_BASIS）。
    传入 ``top_k`` 时退回旧 top-K 模式（兼容历史调用），``selection_rule.mode``
    会如实记录。
    """

    if select_threshold is not None and not 0 <= select_threshold <= 1:
        raise ValueError("select_threshold must be between 0 and 1")
    if max_selected < 1 or pool_k < 1:
        raise ValueError("max_selected and pool_k must be positive")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    payload = _payload(run)
    papers = _papers(payload)
    eligible = [
        paper for paper in papers
        if paper.get("status", {}).get("hard_constraints_pass") is not False
        and _score_of(paper) is not None
    ]
    basis = "final" if any(isinstance((paper.get("scores") or {}).get("final"), (int, float)) and not isinstance((paper.get("scores") or {}).get("final"), bool) for paper in eligible) else "relevance"
    if select_threshold is None:
        select_threshold = DEFAULT_THRESHOLD_BY_BASIS[basis]
    eligible.sort(key=lambda item: (_score_of(item) or 0.0, str(item.get("paper_id") or "")), reverse=True)
    if top_k is not None:
        selected = eligible[:top_k]
        rule = {"mode": "legacy_top_k", "top_k": top_k, "basis": "final_or_relevance"}
    else:
        selected = [paper for paper in eligible if (_score_of(paper) or 0.0) >= select_threshold][:max_selected]
        rule = {"mode": "threshold", "select_threshold": select_threshold, "max_selected": max_selected, "basis": basis}

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
        score = _score_of(paper) or 0.0
        results.append({
            "rank": rank,
            "paper_id": str(paper["paper_id"]),
            "title": str(bib.get("title") or ""),
            "year": bib.get("year"),
            "venue": bib.get("venue"),
            "doi": identifiers.get("doi"),
            "arxiv_id": identifiers.get("arxiv_id"),
            "landing_url": access.get("landing_url"),
            "relevance_zone": _zone(score),
            "final_score": score,
            "component_scores": {name: scores.get(name) for name in COMPONENTS},
            "evidence_refs": list(paper.get("evidence_refs") or []),
            "reason_codes": list(verdict.get("reason_codes") or []),
            "llm_judgement": deepcopy(verdict.get("llm_judgement")) if isinstance(verdict.get("llm_judgement"), Mapping) else None,
        })
    # 官方 Recall@K 计分用的排序池：收录全部未被判死的论文；未打分的
    # （如末层引用捞回、LLM 预算耗尽未判）排在已打分之后——它们丢了会直接
    # 吃掉 recall@K（hybrid-5 实测：test_47 的 3 篇 Gold 因此消失）。
    pool_papers = [paper for paper in papers if paper.get("status", {}).get("hard_constraints_pass") is not False]
    pool_papers.sort(key=lambda item: (_score_of(item) is not None, _score_of(item) or 0.0, str(item.get("paper_id") or "")), reverse=True)
    ranked_pool = [
        {"rank": rank, "paper_id": str(paper.get("paper_id") or ""), "title": str((paper.get("bibliography") or {}).get("title") or ""), "score": _score_of(paper)}
        for rank, paper in enumerate(pool_papers[:pool_k], 1)
    ]

    selected_ids = {item["paper_id"] for item in results}
    paper_by_id = {str(item.get("paper_id")): item for item in papers if item.get("paper_id")}
    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[str, str, str]] = set()
    graph_ids = set(selected_ids)
    for batch in _citations(payload):
        for raw in batch.get("edges") or []:
            if not isinstance(raw, Mapping):
                continue
            parent = str(raw.get("parent_paper_id") or raw.get("parent") or "")
            child = str(raw.get("child_paper_id") or raw.get("child") or "")
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
            "outside_selection": paper_id not in selected_ids,
        })

    manifest = payload.get("manifest") or payload.get("run_manifest") or {}
    cost = payload.get("cost") or (manifest.get("cost") if isinstance(manifest, Mapping) else {}) or {}
    zones = {name: sum(item["relevance_zone"] == name for item in results) for name in ("high", "partial", "reserve")}
    output = {
        "schema_version": FINAL_SCHEMA,
        "query": str(payload.get("query") or (manifest.get("query") if isinstance(manifest, Mapping) else "") or ""),
        "query_id": str((payload.get("query_plan") or {}).get("query_id") or (manifest.get("query_id") if isinstance(manifest, Mapping) else "") or ""),
        "results": results,
        "ranked_pool": ranked_pool,
        "selection_rule": rule,
        "relation_graph": {"nodes": nodes, "edges": edges},
        "summary": {
            "selected": len(results),
            "pool_size": len(ranked_pool),
            "excluded": len(papers) - len(eligible),
            "zones": zones,
            "citation_edges": len(edges),
        },
        "degraded": bool((isinstance(manifest, Mapping) and manifest.get("status") == "degraded") or payload.get("errors")),
        "cost": deepcopy(dict(cost)) if isinstance(cost, Mapping) else {},
    }
    return validate_final_selection(output)


def validate_final_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("final selection must be an object")
    schema = value.get("schema_version")
    if schema not in (FINAL_SCHEMA, LEGACY_SCHEMA):
        raise ValueError(f"schema_version must be {FINAL_SCHEMA} or {LEGACY_SCHEMA}")
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
    if schema == FINAL_SCHEMA:
        if not isinstance(value.get("ranked_pool"), list) or not isinstance(value.get("selection_rule"), Mapping):
            raise ValueError("v2 requires ranked_pool and selection_rule")
        for rank, item in enumerate(value.get("ranked_pool") or [], 1):
            if not isinstance(item, Mapping) or item.get("rank") != rank or not item.get("paper_id"):
                raise ValueError("ranked_pool must have contiguous ranks and paper_id")
        rule = value["selection_rule"]
        mode = rule.get("mode")
        if mode == "threshold":
            if not 0 <= float(rule.get("select_threshold") or 0) <= 1 or int(rule.get("max_selected") or 0) < 1:
                raise ValueError("invalid threshold selection rule")
        elif mode == "legacy_top_k":
            if int(rule.get("top_k") or 0) < 1:
                raise ValueError("invalid legacy top_k rule")
        else:
            raise ValueError("invalid selection rule mode")
    graph = value.get("relation_graph")
    if not isinstance(graph, Mapping) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise ValueError("relation_graph must contain nodes and edges arrays")
    if not isinstance(value.get("summary"), Mapping) or not isinstance(value.get("cost"), Mapping):
        raise ValueError("summary and cost must be objects")
    return deepcopy(dict(value))


__all__ = ["DEFAULT_MAX_SELECTED", "DEFAULT_POOL_K", "DEFAULT_SELECT_THRESHOLD", "DEFAULT_THRESHOLD_BY_BASIS", "FINAL_SCHEMA", "LEGACY_SCHEMA", "build_final_selection", "validate_final_selection"]
