"""P1 无网络 mock 闭环：两来源合并 + 显式 Provider 错误。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .paperdoc import merge_paper_docs, provider_error, validate_paper_doc


def _paper(source: str, abstract: str, *, full_text_status: str = "abstract") -> dict[str, Any]:
    return {
        "schema_version": "paperdoc.v1",
        "paper_id": "doi:10.1234/mock.paper",
        "identifiers": {"doi": "10.1234/mock.paper", "arxiv_id": None, "s2_id": None, "openalex_id": None, "pmid": None, "pmcid": None, "sciverse_doc_id": None, "unique_id": None},
        "bibliography": {"title": "Mock scientific retrieval paper", "authors": ["Ada Lovelace"], "year": 2025, "venue": "Mock Venue", "abstract": abstract, "fields": ["computer science"]},
        "access": {"landing_url": "https://example.invalid/paper", "pdf_url": None, "oa_url": None, "full_text_status": full_text_status, "content_type": "abstract"},
        "content": {"content_ref": None, "chunks": [], "sections": [], "char_count": len(abstract)},
        "relations": {"references": [], "citations": [], "related_works": []},
        "scores": {"retrieval": 0.8, "relevance": None, "constraint": None, "quality": None, "evidence": None, "citation": None, "novelty": None, "final": None, "confidence": None},
        "provenance": {"sources": [source], "query_id": "mock-query", "subquery_id": "mock-subquery", "iteration": 0, "parent_node_id": None, "endpoints": [f"mock://{source}/search"], "retrieved_at": "2026-08-24T00:00:00Z", "pages": [1], "reconciliation": {"complete": True}, "warnings": []},
        "evidence_refs": [],
        "status": {"hard_constraints_pass": None, "evidence_status": full_text_status, "provider_errors": []},
    }


def run_mock() -> dict[str, Any]:
    first = _paper("mock_a", "Short abstract from source A.")
    second = _paper("mock_b", "A longer abstract from source B with complementary metadata.")
    merged = merge_paper_docs(first, second)
    errors = [provider_error("bohrium", "config_missing", "mock intentionally avoids real credentials")]
    merged["status"]["provider_errors"] = errors
    validate_paper_doc(merged)
    return {
        "ok": True,
        "papers": [merged],
        "stats": {"input_records": 2, "merged_records": 1, "source_errors": errors},
    }
