"""Loader and validator for provisional P1 retrieval Gold data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


GOLD_SCHEMA_VERSION = "spar.gold.v1"
ANNOTATION_STATUSES = {"provisional", "verified", "official"}
IDENTIFIER_FIELDS = {
    "doi", "arxiv_id", "s2_id", "openalex_id", "local_id", "unique_id",
    "sciverse_doc_id", "pmid", "pmcid",
}


class GoldValidationError(ValueError):
    pass


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GoldValidationError(f"{path} must be a non-empty string")
    return value


def _validate_annotation(item: Mapping[str, Any], path: str) -> None:
    _nonempty_string(item.get("annotation_source"), f"{path}.annotation_source")
    status = item.get("annotation_status")
    if status not in ANNOTATION_STATUSES:
        raise GoldValidationError(f"{path}.annotation_status must be one of {sorted(ANNOTATION_STATUSES)}")
    _nonempty_string(item.get("annotated_at"), f"{path}.annotated_at")
    if not isinstance(item.get("notes"), str):
        raise GoldValidationError(f"{path}.notes must be a string")


def validate_gold(data: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise GoldValidationError("gold must be an object")
    if data.get("schema_version") != GOLD_SCHEMA_VERSION:
        raise GoldValidationError(f"schema_version must be {GOLD_SCHEMA_VERSION}")
    _validate_annotation(data, "gold")
    queries = data.get("queries")
    if not isinstance(queries, list):
        raise GoldValidationError("gold.queries must be an array")

    query_ids: set[str] = set()
    for index, query in enumerate(queries):
        path = f"gold.queries[{index}]"
        if not isinstance(query, Mapping):
            raise GoldValidationError(f"{path} must be an object")
        query_id = _nonempty_string(query.get("query_id"), f"{path}.query_id")
        if query_id in query_ids:
            raise GoldValidationError(f"duplicate query_id: {query_id}")
        query_ids.add(query_id)
        _nonempty_string(query.get("query"), f"{path}.query")
        _validate_annotation(query, path)
        relevant = query.get("relevant_papers")
        if not isinstance(relevant, list):
            raise GoldValidationError(f"{path}.relevant_papers must be an array")
        for paper_index, paper in enumerate(relevant):
            paper_path = f"{path}.relevant_papers[{paper_index}]"
            if not isinstance(paper, Mapping):
                raise GoldValidationError(f"{paper_path} must be an object")
            _nonempty_string(paper.get("paper_id"), f"{paper_path}.paper_id")
            identifiers = paper.get("identifiers")
            if not isinstance(identifiers, Mapping) or not any(
                key in IDENTIFIER_FIELDS and isinstance(value, str) and value.strip()
                for key, value in identifiers.items()
            ):
                raise GoldValidationError(f"{paper_path}.identifiers must contain a stable identifier")
            _nonempty_string(paper.get("title"), f"{paper_path}.title")
            if not isinstance(paper.get("year"), int):
                raise GoldValidationError(f"{paper_path}.year must be an integer")
            _nonempty_string(paper.get("first_author"), f"{paper_path}.first_author")
            _nonempty_string(paper.get("judgment_basis"), f"{paper_path}.judgment_basis")
    return data


def load_gold(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_gold(data)
    return data


__all__ = ["ANNOTATION_STATUSES", "GOLD_SCHEMA_VERSION", "GoldValidationError", "load_gold", "validate_gold"]
