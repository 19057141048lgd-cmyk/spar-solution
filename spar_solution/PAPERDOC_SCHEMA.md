# PaperDoc 固定结构（方案版）

这是 P1 开始实现前必须冻结的协议。字段名和类型先固定；后续只能向后兼容地增加可选字段，不能把数组改成字符串。

```json
{
  "schema_version": "paperdoc.v1",
  "paper_id": "canonical-id",
  "identifiers": {
    "doi": null,
    "arxiv_id": null,
    "s2_id": null,
    "openalex_id": null,
    "pmid": null,
    "pmcid": null,
    "sciverse_doc_id": null,
    "unique_id": null
  },
  "bibliography": {
    "title": "",
    "authors": [],
    "year": null,
    "venue": null,
    "abstract": null,
    "fields": []
  },
  "access": {
    "landing_url": null,
    "pdf_url": null,
    "oa_url": null,
    "full_text_status": "metadata",
    "content_type": null
  },
  "content": {
    "content_ref": null,
    "chunks": [],
    "sections": [],
    "char_count": 0
  },
  "relations": {
    "references": [],
    "citations": [],
    "related_works": []
  },
  "scores": {
    "retrieval": null,
    "relevance": null,
    "constraint": null,
    "quality": null,
    "evidence": null,
    "citation": null,
    "novelty": null,
    "final": null,
    "confidence": null
  },
  "provenance": {
    "sources": [],
    "query_id": null,
    "subquery_id": null,
    "iteration": 0,
    "parent_node_id": null,
    "endpoints": [],
    "retrieved_at": null,
    "pages": [],
    "reconciliation": null,
    "warnings": []
  },
  "evidence_refs": [],
  "status": {
    "hard_constraints_pass": null,
    "evidence_status": "metadata",
    "provider_errors": []
  }
}
```

## 约束

- `full_text_status` 只允许 `metadata`、`abstract`、`partial_text`、`fulltext`、`unavailable`。
- `content.chunks` 保存 `{chunk_id, content_ref, offset, section, page}`，正文原文放 artifact，不放 Agent 消息。
- `relations` 保存关系 ID 和 `relation_source`，不只保存引用计数。
- `scores` 允许为空；Provider 失败不能写成 `relevance=0`。
- `provenance.reconciliation.complete=false` 时，必须保留 `warnings`。
- `evidence_refs` 只引用 `EvidenceItem` ID；证据文本由 EvidenceJudge 按 ID读取。
