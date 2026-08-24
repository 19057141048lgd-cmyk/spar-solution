"""P1 paper identity matching with auditable reason codes."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal, Mapping


MatchStatus = Literal["matched", "unmatched", "ambiguous"]
STABLE_ID_FIELDS = (
    "s2_id",
    "openalex_id",
    "local_id",
    "unique_id",
    "sciverse_doc_id",
)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def normalize_doi(value: Any) -> str | None:
    value = _text(value)
    if not value:
        return None
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^doi\s*:\s*", "", value, flags=re.IGNORECASE)
    return value.strip().casefold() or None


def normalize_arxiv_id(value: Any) -> str | None:
    value = _text(value)
    if not value:
        return None
    value = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^arxiv\s*:\s*", "", value, flags=re.IGNORECASE)
    return re.sub(r"v\d+$", "", value.strip(), flags=re.IGNORECASE).casefold() or None


def normalize_title(value: Any) -> str | None:
    value = _text(value)
    if not value:
        return None
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split()) or None


_ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)


def arxiv_id_from_doi(value: Any) -> str | None:
    """Recognise the deterministic arXiv DOI prefix and return the arXiv ID.

    arXiv registers every preprint as ``10.48550/arxiv.<id>``; providers such
    as OpenAlex expose that DOI without a separate arXiv ID field. Records that
    only differ in this representation are the same paper and must match.
    """

    doi = normalize_doi(value)
    if not doi:
        return None
    match = _ARXIV_DOI_RE.match(doi)
    if not match:
        return None
    return normalize_arxiv_id(match.group(1))


def _normalize_stable_id(field: str, value: Any) -> str | None:
    value = _text(value)
    if not value:
        return None
    if field == "openalex_id":
        value = re.sub(r"^https?://openalex\.org/", "", value, flags=re.IGNORECASE)
    return value.casefold()


def _identifiers(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("identifiers")
    return value if isinstance(value, Mapping) else {}


def _bibliography(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("bibliography")
    return value if isinstance(value, Mapping) else record


def _first_author(record: Mapping[str, Any]) -> str | None:
    bibliography = _bibliography(record)
    explicit = bibliography.get("first_author")
    if explicit:
        return normalize_title(explicit)
    authors = bibliography.get("authors")
    if not isinstance(authors, list) or not authors:
        return None
    author = authors[0]
    if isinstance(author, Mapping):
        author = author.get("name") or author.get("display_name")
    return normalize_title(author)


def _result(status: MatchStatus, matched_by: str | None, key: str | None, reason: str) -> dict[str, Any]:
    return {"status": status, "matched_by": matched_by, "identity_key": key, "reason": reason}


def match_papers(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two PaperDoc/Gold records by the frozen P1 priority rules.

    Conflicting high-priority identifiers are not allowed to fall through to a
    title match. Missing fallback fields produce ``ambiguous`` rather than TP.
    """

    left_ids = _identifiers(left)
    right_ids = _identifiers(right)

    for field, normalizer in (("doi", normalize_doi), ("arxiv_id", normalize_arxiv_id)):
        left_value = normalizer(left_ids.get(field))
        right_value = normalizer(right_ids.get(field))
        if left_value and right_value:
            if left_value == right_value:
                return _result("matched", field, f"{field}:{left_value}", f"same_{field}")
            return _result("ambiguous", None, None, f"conflicting_{field}")

    # arXiv preprint DOIs are equivalent to arXiv IDs. OpenAlex-sourced records
    # usually only carry the DOI, while AutoScholarQuery-style gold only carries
    # the bare arXiv ID; without this rule those pairs fall through to title
    # matching and count as ambiguous (not TP) even when both sides exist.
    left_arxiv = normalize_arxiv_id(left_ids.get("arxiv_id")) or arxiv_id_from_doi(left_ids.get("doi"))
    right_arxiv = normalize_arxiv_id(right_ids.get("arxiv_id")) or arxiv_id_from_doi(right_ids.get("doi"))
    if left_arxiv and right_arxiv:
        if left_arxiv == right_arxiv:
            return _result("matched", "arxiv_doi_equivalence", f"arxiv_id:{left_arxiv}", "doi_arxiv_equivalent")
        return _result("ambiguous", None, None, "conflicting_arxiv_identity")

    stable_matches: list[tuple[str, str]] = []
    stable_conflicts: list[str] = []
    for field in STABLE_ID_FIELDS:
        left_value = _normalize_stable_id(field, left_ids.get(field))
        right_value = _normalize_stable_id(field, right_ids.get(field))
        if left_value and right_value:
            if left_value == right_value:
                stable_matches.append((field, left_value))
            else:
                stable_conflicts.append(field)
    if stable_matches and stable_conflicts:
        return _result("ambiguous", None, None, "conflicting_stable_identifiers")
    if stable_matches:
        field, value = stable_matches[0]
        return _result("matched", field, f"{field}:{value}", f"same_{field}")
    if stable_conflicts:
        return _result("ambiguous", None, None, "conflicting_stable_identifiers")

    left_bib = _bibliography(left)
    right_bib = _bibliography(right)
    left_title = normalize_title(left_bib.get("title"))
    right_title = normalize_title(right_bib.get("title"))
    if not left_title or not right_title:
        return _result("ambiguous", None, None, "insufficient_title_metadata")
    if left_title != right_title:
        return _result("unmatched", None, None, "different_normalized_title")

    left_year = left_bib.get("year")
    right_year = right_bib.get("year")
    left_author = _first_author(left)
    right_author = _first_author(right)
    if left_year is None or right_year is None or not left_author or not right_author:
        return _result("ambiguous", None, None, "insufficient_title_year_author_metadata")
    if str(left_year) != str(right_year) or left_author != right_author:
        return _result("unmatched", None, None, "different_year_or_first_author")
    key = f"title:{left_title}|year:{left_year}|author:{left_author}"
    return _result("matched", "title_year_first_author", key, "same_title_year_first_author")


__all__ = [
    "MatchStatus",
    "STABLE_ID_FIELDS",
    "arxiv_id_from_doi",
    "match_papers",
    "normalize_arxiv_id",
    "normalize_doi",
    "normalize_title",
]
