"""搜完看整池：对不上原题就换街，不顺着错的论文继续挖。

Ai2/PaSa 的后半段是「用已判定相关的论文再生搜索词」。若第一遍理解错了，
那些论文会看起来很相关，再生成只会越走越偏。这里只做纠偏，不跟簇。
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .question_understanding import collect_search_queries, parse_understanding


REFLECT_POOL_SYSTEM_PROMPT = (
    "You already ran an academic search from a first reading of the question. "
    "You are given that reading and a sample of titles that came back. Return JSON only. "
    "Do not invent paper titles. "
    "Decide whether the result set is answering the question, or whether it is a self-consistent "
    "cluster for a DIFFERENT reading of the same words. "
    "Internal similarity of the titles is NOT proof that the first reading was right. "
    "Ambiguous words (reconstruction, hybrid, alignment, survey, stationary, context) often "
    "belong to more than one field. If almost every title follows one reading, and the question "
    "did not name that object of study, mark wrong_street. "
    "If wrong_street or mixed, do NOT propose more queries in the same street as the titles. "
    "Propose queries for an alternative field that could still answer the question. "
    "JSON: {\"verdict\": \"on_track\"|\"wrong_street\"|\"mixed\", \"why\": \"short\", "
    "\"field\": \"...\", \"queries\": [\"short keyword searches\"], \"survey_queries\": [\"...\"]}."
)


def parse_course_correction(payload: Mapping[str, Any] | None, query: str) -> dict[str, Any]:
    raw = dict(payload) if isinstance(payload, Mapping) else {}
    verdict = str(raw.get("verdict") or "on_track").strip().casefold()
    if verdict not in {"on_track", "wrong_street", "mixed"}:
        verdict = "on_track"
    parsed = parse_understanding(raw, query)
    queries = collect_search_queries(parsed) if verdict in {"wrong_street", "mixed"} else []
    return {
        "verdict": verdict,
        "why": str(raw.get("why") or "")[:400],
        "field": parsed.get("field") or "",
        "queries": queries,
    }


def reflect_on_pool(
    client: Any,
    query: str,
    understanding: Mapping[str, Any] | None,
    papers: Sequence[Mapping[str, Any]],
    *,
    sample: int = 15,
) -> dict[str, Any]:
    """对照原题看池子。失败则视为 on_track，不打断后续。"""

    complete = getattr(client, "complete_json", None)
    if not callable(complete):
        return {"verdict": "on_track", "why": "no_llm", "field": "", "queries": []}
    titles = []
    for paper in papers[:sample]:
        title = str((paper.get("bibliography") or {}).get("title") or "").strip()
        if title:
            titles.append(title[:180])
    user = json.dumps(
        {
            "task": "reflect_on_pool",
            "query": query,
            "first_reading": {
                "field": (understanding or {}).get("field"),
                "alt_fields": (understanding or {}).get("alt_fields") or [],
                "answer_looks_like": (understanding or {}).get("answer_looks_like") or "",
            },
            "retrieved_titles": titles,
        },
        ensure_ascii=False,
    )
    try:
        payload = complete(REFLECT_POOL_SYSTEM_PROMPT, user, max_tokens=700)
    except Exception:
        return {"verdict": "on_track", "why": "reflect_failed", "field": "", "queries": []}
    return parse_course_correction(payload if isinstance(payload, Mapping) else None, query)


__all__ = ["REFLECT_POOL_SYSTEM_PROMPT", "parse_course_correction", "reflect_on_pool"]
