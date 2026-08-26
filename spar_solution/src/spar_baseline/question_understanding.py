"""读题：先理解题目，再出搜索词。

这一层不搜论文、不发明论文事实。失败时返回空理解，调用方退回原查询计划。
"""

from __future__ import annotations

from typing import Any, Mapping


UNDERSTANDING_SCHEMA = "question_understanding.v1"

DRAFT_SYSTEM_PROMPT = (
    "You read an academic search question BEFORE any search. Return JSON only. "
    "Do not invent paper titles or claim that a specific paper exists. "
    "The wording often comes from a survey's method taxonomy: answering papers may never use these exact words. "
    "First decide the research field (and plausible alternatives). Then say what an answering paper would actually be about, "
    "in that field's own terminology. Then write search queries in THAT terminology, plus one or two queries whose purpose "
    "is to find surveys/reviews in the field. "
    "JSON fields: field (string), alt_fields (string[]), jargon_from_survey (boolean), "
    "family_name (string, the taxonomy phrase or empty), answer_looks_like (short string), "
    "keywords (string[]), queries (3-5 short keyword searches, not a copy of the question), "
    "survey_queries (1-2 survey/review searches), confidence (0-1)."
)

def _string_list(value: Any, *, limit: int = 8) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()][:limit]
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            output.append(item.strip())
        elif isinstance(item, Mapping):
            text = str(item.get("query") or item.get("query_text") or item.get("search_query") or "").strip()
            if text:
                output.append(text)
        if len(output) >= limit:
            break
    return output


def _bounded_confidence(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if 1 < number <= 100:
        number /= 100
    if not 0 <= number <= 1:
        return default
    return round(number, 6)


def parse_understanding(payload: Mapping[str, Any] | None, query: str, *, fallback: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """把模型 JSON 收成可校验对象；缺字段用 fallback 或空值，不抛。"""

    raw = dict(payload) if isinstance(payload, Mapping) else {}
    base = dict(fallback) if isinstance(fallback, Mapping) else {}
    field = str(raw.get("field") or base.get("field") or "").strip()
    family_name = str(raw.get("family_name") or base.get("family_name") or "").strip()
    answer_looks_like = str(raw.get("answer_looks_like") or base.get("answer_looks_like") or "").strip()
    queries = _string_list(raw.get("queries")) or _string_list(base.get("queries"))
    survey_queries = _string_list(raw.get("survey_queries"), limit=4) or _string_list(base.get("survey_queries"), limit=4)
    alt_fields = _string_list(raw.get("alt_fields")) or _string_list(base.get("alt_fields"))
    keywords = _string_list(raw.get("keywords")) or _string_list(base.get("keywords"))
    jargon = raw.get("jargon_from_survey")
    if not isinstance(jargon, bool):
        jargon = bool(base.get("jargon_from_survey") or family_name)
    return {
        "schema_version": UNDERSTANDING_SCHEMA,
        "query": query,
        "field": field,
        "alt_fields": alt_fields,
        "jargon_from_survey": jargon,
        "family_name": family_name,
        "answer_looks_like": answer_looks_like,
        "keywords": keywords,
        "queries": queries,
        "survey_queries": survey_queries,
        "confidence": _bounded_confidence(raw.get("confidence", base.get("confidence", 0.5))),
        "source": "llm",
    }


def collect_search_queries(understanding: Mapping[str, Any] | None, *, limit: int = 6) -> list[str]:
    """综述定位词在前，再跟领域检索词。空理解返回 []。"""

    if not isinstance(understanding, Mapping):
        return []
    seen: set[str] = set()
    output: list[str] = []
    for text in list(understanding.get("survey_queries") or []) + list(understanding.get("queries") or []):
        cleaned = " ".join(str(text or "").split())
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
        if len(output) >= limit:
            break
    return output


__all__ = [
    "DRAFT_SYSTEM_PROMPT",
    "UNDERSTANDING_SCHEMA",
    "collect_search_queries",
    "parse_understanding",
]
