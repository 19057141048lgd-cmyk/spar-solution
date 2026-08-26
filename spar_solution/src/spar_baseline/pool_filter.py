"""按读题时的目的过滤检索结果：错领域丢掉，对领域的留下（含综述）。"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


FILTER_SYSTEM_PROMPT = (
    "You filter academic search results. Return JSON only. Do not invent papers. "
    "You are given the intended research field and what we are looking for. "
    "KEEP a paper if it belongs to that field, including literature surveys and reviews in that field. "
    "DROP a paper if it is clearly a different field (astronomy sky surveys, opinion questionnaires, "
    "unrelated hardware/engineering) even if the title shares a word such as survey or review. "
    "If unsure but it could be in-field, KEEP. "
    'Return {"results": [{"paper_id": "...", "keep": true, "reason": "short"}]}. '
    "Return every given paper_id exactly once."
)

FILTER_BATCH = 10


def _paper_card(paper: Mapping[str, Any]) -> dict[str, str]:
    bib = paper.get("bibliography") or {}
    return {
        "paper_id": str(paper.get("paper_id") or ""),
        "title": str(bib.get("title") or "")[:300],
        "abstract": str(bib.get("abstract") or "")[:800],
        "year": str(bib.get("year") or ""),
    }


def _keep_map(payload: Mapping[str, Any] | None, expected_ids: Sequence[str]) -> dict[str, bool]:
    expected = {str(item) for item in expected_ids}
    mapping: dict[str, bool] = {}
    values = (payload or {}).get("results") if isinstance(payload, Mapping) else None
    if not isinstance(values, list):
        return {paper_id: True for paper_id in expected}
    for item in values:
        if not isinstance(item, Mapping):
            continue
        paper_id = str(item.get("paper_id") or "")
        if paper_id not in expected or paper_id in mapping:
            continue
        mapping[paper_id] = bool(item.get("keep"))
    for paper_id in expected:
        mapping.setdefault(paper_id, True)
    return mapping


def filter_papers(
    client: Any,
    understanding: Mapping[str, Any],
    papers: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = FILTER_BATCH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """用读题结果过滤论文。单批失败则该批全留，避免误删。"""

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    if not papers:
        return kept, dropped
    intent = {
        "field": understanding.get("field"),
        "alt_fields": understanding.get("alt_fields") or [],
        "looking_for": understanding.get("answer_looks_like") or "",
        "family_name": understanding.get("family_name") or "",
    }
    size = max(1, int(batch_size))
    complete = getattr(client, "complete_json", None)
    if not callable(complete):
        return [dict(paper) for paper in papers], []

    for start in range(0, len(papers), size):
        batch = [dict(paper) for paper in papers[start : start + size]]
        ids = [str(paper.get("paper_id") or "") for paper in batch]
        user = json.dumps(
            {
                "task": "filter_search_results",
                "intent": intent,
                "candidates": [_paper_card(paper) for paper in batch],
            },
            ensure_ascii=False,
        )
        try:
            payload = complete(FILTER_SYSTEM_PROMPT, user, max_tokens=min(4000, 120 * len(batch) + 200))
        except Exception:
            kept.extend(batch)
            continue
        flags = _keep_map(payload if isinstance(payload, Mapping) else None, ids)
        for paper in batch:
            paper_id = str(paper.get("paper_id") or "")
            if flags.get(paper_id, True):
                kept.append(paper)
            else:
                dropped.append(paper)
    return kept, dropped


__all__ = ["FILTER_SYSTEM_PROMPT", "filter_papers"]
