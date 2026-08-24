"""PaperDoc v1 的最小校验、身份归一化和跨源合并实现。

这里只使用 Python 标准库，避免 P1 的协议测试依赖任何真实 API 或第三方包。
"""

from __future__ import annotations

from copy import deepcopy
import re
import unicodedata
from typing import Any

from .identity import normalize_arxiv_id, normalize_doi


SCHEMA_VERSION = "paperdoc.v1"
FULL_TEXT_STATUSES = {"metadata", "abstract", "partial_text", "fulltext", "unavailable"}
EVIDENCE_STATUSES = FULL_TEXT_STATUSES
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "paper_id",
    "identifiers",
    "bibliography",
    "access",
    "content",
    "relations",
    "scores",
    "provenance",
    "evidence_refs",
    "status",
}


class PaperDocValidationError(ValueError):
    """PaperDoc 不满足协议时抛出。"""


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PaperDocValidationError(f"{path} must be an object")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise PaperDocValidationError(f"{path} must be an array")
    return value


def _require_string(value: Any, path: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise PaperDocValidationError(f"{path} must be a string")
    return value


def validate_paper_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """校验 PaperDoc v1，并返回原对象。

    校验器刻意拒绝列表字符串化，确保不同 Agent 之间不会悄悄退化为自然语言字段。
    """

    root = _require_mapping(doc, "paper_doc")
    missing = REQUIRED_TOP_LEVEL - root.keys()
    if missing:
        raise PaperDocValidationError(f"missing top-level fields: {sorted(missing)}")
    if root["schema_version"] != SCHEMA_VERSION:
        raise PaperDocValidationError("schema_version must be paperdoc.v1")
    _require_string(root["paper_id"], "paper_id", allow_empty=False)

    identifiers = _require_mapping(root["identifiers"], "identifiers")
    bibliography = _require_mapping(root["bibliography"], "bibliography")
    access = _require_mapping(root["access"], "access")
    content = _require_mapping(root["content"], "content")
    relations = _require_mapping(root["relations"], "relations")
    scores = _require_mapping(root["scores"], "scores")
    provenance = _require_mapping(root["provenance"], "provenance")
    status = _require_mapping(root["status"], "status")

    for field in ("authors", "fields"):
        _require_list(bibliography.get(field), f"bibliography.{field}")
    for field in ("references", "citations", "related_works"):
        _require_list(relations.get(field), f"relations.{field}")
    _require_list(root["evidence_refs"], "evidence_refs")
    _require_list(provenance.get("sources"), "provenance.sources")
    _require_list(provenance.get("endpoints"), "provenance.endpoints")
    _require_list(provenance.get("pages"), "provenance.pages")
    _require_list(provenance.get("warnings"), "provenance.warnings")
    _require_list(status.get("provider_errors"), "status.provider_errors")

    full_text_status = access.get("full_text_status")
    if full_text_status not in FULL_TEXT_STATUSES:
        raise PaperDocValidationError(
            f"access.full_text_status must be one of {sorted(FULL_TEXT_STATUSES)}"
        )
    evidence_status = status.get("evidence_status")
    if evidence_status not in EVIDENCE_STATUSES:
        raise PaperDocValidationError(
            f"status.evidence_status must be one of {sorted(EVIDENCE_STATUSES)}"
        )
    if content.get("chunks") is None or not isinstance(content.get("chunks"), list):
        raise PaperDocValidationError("content.chunks must be an array")
    if not isinstance(provenance.get("iteration"), int) or provenance["iteration"] < 0:
        raise PaperDocValidationError("provenance.iteration must be a non-negative integer")
    if not isinstance(status.get("hard_constraints_pass"), (bool, type(None))):
        raise PaperDocValidationError("status.hard_constraints_pass must be boolean or null")
    return doc


def _norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def canonical_paper_key(doc: dict[str, Any]) -> str:
    """按 DOI/稳定来源 ID/标题身份生成可复现的合并键。"""

    validate_paper_doc(doc)
    identifiers = doc["identifiers"]
    for field in ("doi", "arxiv_id", "s2_id", "openalex_id", "pmid", "pmcid", "sciverse_doc_id", "unique_id"):
        value = identifiers.get(field)
        if isinstance(value, str) and value.strip():
            if field == "doi":
                value = normalize_doi(value)
            elif field == "arxiv_id":
                value = normalize_arxiv_id(value)
            elif field == "openalex_id":
                value = re.sub(r"^https?://openalex\.org/", "", value, flags=re.IGNORECASE)
            if value:
                return f"{field}:{_norm_text(value)}"
    bibliography = doc["bibliography"]
    title = _norm_text(bibliography.get("title") or "")
    year = bibliography.get("year")
    authors = bibliography.get("authors") or []
    first_author = _norm_text(str(authors[0])) if authors else ""
    if not title or year is None or not first_author:
        # ambiguous 记录必须由上层按记录序号隔离，禁止强行合并。
        return f"ambiguous:{_norm_text(str(doc['paper_id']))}"
    return f"title:{title}|year:{year}|author:{first_author}"


def _merge_unique(left: list[Any], right: list[Any]) -> list[Any]:
    result = list(left)
    seen = {repr(item) for item in result}
    for item in right:
        marker = repr(item)
        if marker not in seen:
            result.append(deepcopy(item))
            seen.add(marker)
    return result


def merge_paper_docs(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    """合并同一论文的两个来源，保留来源和字段 provenance。"""

    validate_paper_doc(primary)
    validate_paper_doc(secondary)
    primary_key = canonical_paper_key(primary)
    secondary_key = canonical_paper_key(secondary)
    if primary_key.startswith("ambiguous:") or secondary_key.startswith("ambiguous:"):
        raise PaperDocValidationError("cannot merge PaperDoc records with ambiguous identities")
    if primary_key != secondary_key:
        raise PaperDocValidationError("cannot merge PaperDoc records with different identities")

    merged = deepcopy(primary)
    p_bib = merged["bibliography"]
    s_bib = secondary["bibliography"]
    p_ids = merged["identifiers"]
    s_ids = secondary["identifiers"]
    for key, value in s_ids.items():
        if not p_ids.get(key) and value:
            p_ids[key] = deepcopy(value)
    if len(str(s_bib.get("abstract") or "")) > len(str(p_bib.get("abstract") or "")):
        p_bib["abstract"] = s_bib["abstract"]
    if len(s_bib.get("authors") or []) > len(p_bib.get("authors") or []):
        p_bib["authors"] = deepcopy(s_bib["authors"])
    p_bib["fields"] = _merge_unique(p_bib.get("fields") or [], s_bib.get("fields") or [])

    for key in ("references", "citations", "related_works"):
        merged["relations"][key] = _merge_unique(
            merged["relations"].get(key) or [], secondary["relations"].get(key) or []
        )
    merged["provenance"]["sources"] = _merge_unique(
        merged["provenance"].get("sources") or [], secondary["provenance"].get("sources") or []
    )
    merged["provenance"]["endpoints"] = _merge_unique(
        merged["provenance"].get("endpoints") or [], secondary["provenance"].get("endpoints") or []
    )
    merged["provenance"]["warnings"] = _merge_unique(
        merged["provenance"].get("warnings") or [], secondary["provenance"].get("warnings") or []
    )
    merged["evidence_refs"] = _merge_unique(merged["evidence_refs"], secondary["evidence_refs"])
    if merged["access"].get("full_text_status") in {"metadata", "abstract", "unavailable"}:
        candidate_status = secondary["access"].get("full_text_status")
        order = {"unavailable": 0, "metadata": 1, "abstract": 2, "partial_text": 3, "fulltext": 4}
        if order[candidate_status] > order[merged["access"].get("full_text_status")]:
            merged["access"] = deepcopy(secondary["access"])
            merged["content"] = deepcopy(secondary["content"])
            merged["status"]["evidence_status"] = secondary["status"]["evidence_status"]
    merged["provenance"]["sources"] = _merge_unique(
        merged["provenance"]["sources"], ["merged"]
    )
    return validate_paper_doc(merged)


def provider_error(source: str, code: str, message: str) -> dict[str, str]:
    """生成不与论文相关性分数混淆的 Provider 错误事件。"""

    if not source or not code or not message:
        raise ValueError("source, code and message are required")
    return {"source": source, "code": code, "message": message}
