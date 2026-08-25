"""多源检索结果的排名融合：RRF 逆序名融合与 SPAR 风格分桶排序。

思路参考（未复制代码）：
- openreview_search 的 RRF：融合分 = Σ weight * 1/(k + rank + 1)，k 默认 60，rank 从 0 起；
  双路命中的论文会被标注出来。
- SPAR 的 _rank_query_doc_list：相似度分按 0.05 分桶，桶内按 citationCount、year 降序。

本模块只依赖 Python 标准库，输入输出均为 PaperDoc dict。
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Callable, Mapping, Optional, Sequence

from .paperdoc import canonical_paper_key

DEFAULT_RRF_K = 60
DEFAULT_BUCKET = 0.05

MergeCallback = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _merge_by_repr(left: list[Any], right: list[Any]) -> list[Any]:
    """按出现顺序合并两个列表并去重，用 repr 做身份以兼容不可哈希元素。"""

    result = list(left)
    seen = {repr(item) for item in result}
    for item in right:
        marker = repr(item)
        if marker not in seen:
            result.append(item)
            seen.add(marker)
    return result


def _default_merge(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """默认合并策略：保留先出现的文档本体，合并 provenance 列表与标识字段。

    identifiers 取并集（不覆盖已有值）：arXiv 副本带来 arxiv_id、OpenAlex
    副本带来 openalex_id/DOI。合并后引用扩展与身份匹配两边都能用——否则
    arXiv 首现的论文丢失 W-id，OpenAlex 引用接口无法扩展它。
    """

    merged = deepcopy(base)
    for field in ("sources", "endpoints", "warnings", "pages"):
        merged["provenance"][field] = _merge_by_repr(
            merged["provenance"].get(field) or [], other["provenance"].get(field) or []
        )
    merged_ids = merged.setdefault("identifiers", {})
    for key, value in (other.get("identifiers") or {}).items():
        if value and not merged_ids.get(key):
            merged_ids[key] = deepcopy(value)
    # 首现副本缺失摘要/作者时用另一份补齐（不按长度覆盖，保持"保留第一份"契约）。
    for bib_field in ("abstract", "authors"):
        base_value = (merged.get("bibliography") or {}).get(bib_field)
        other_value = (other.get("bibliography") or {}).get(bib_field)
        if other_value and not base_value:
            merged.setdefault("bibliography", {})[bib_field] = deepcopy(other_value)
    return merged


def _check_positive_number(value: Any, name: str) -> float:
    """校验参数是正数（拒绝 bool），返回 float 形式。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return float(value)


def rrf_fuse(
    ranked_lists: Mapping[str, Sequence[dict[str, Any]]],
    *,
    k: int | float = DEFAULT_RRF_K,
    weights: Optional[Mapping[str, float]] = None,
    merge: Optional[MergeCallback] = None,
) -> list[dict[str, Any]]:
    """对多源已排序结果做 Reciprocal Rank Fusion。

    Args:
        ranked_lists: 源名 -> 已按相关性降序的 PaperDoc 列表。
        k: RRF 平滑常数，必须为正数，默认 60。
        weights: 源名 -> 权重，缺省源权重为 1.0。
        merge: 可选合并回调 merge(accumulated_doc, incoming_doc) -> doc，
            用于同一论文多源出现时定制字段取舍；缺省保留第一份并合并 provenance。

    Returns:
        按 rrf 分降序的新文档列表，每篇附带：
        scores.rrf / provenance.rrf_sources / provenance.rrf_best_rank，
        多源命中时另加 provenance.rrf_multi_source=True。空输入返回 []。
    """

    k_value = _check_positive_number(k, "k")
    weight_map = dict(weights) if weights is not None else {}
    for source_name, weight in weight_map.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(
                f"weight for source {source_name!r} must be a number, got {weight!r}"
            )

    # key -> {doc: 当前合并结果, score: 累计 rrf 分, sources: 命中来源, best_rank: 各源最好名次}
    accumulated: dict[str, dict[str, Any]] = {}
    for source_name, doc_list in (ranked_lists or {}).items():
        weight = float(weight_map.get(source_name, 1.0))
        for rank, doc in enumerate(doc_list):
            key = canonical_paper_key(doc)
            entry = accumulated.get(key)
            if entry is None:
                entry = {
                    "doc": deepcopy(doc),
                    "score": 0.0,
                    "sources": [],
                    "best_rank": {},
                }
                accumulated[key] = entry
            else:
                # 同一论文再次出现：按回调或默认策略合并成一份记录。
                if merge is not None:
                    entry["doc"] = merge(entry["doc"], deepcopy(doc))
                else:
                    entry["doc"] = _default_merge(entry["doc"], doc)
            entry["score"] += weight * (1.0 / (k_value + rank + 1))
            if source_name not in entry["sources"]:
                entry["sources"].append(source_name)
            previous_rank = entry["best_rank"].get(source_name)
            if previous_rank is None or rank < previous_rank:
                entry["best_rank"][source_name] = rank

    # 平分时按合并键升序破平，保证输出确定。
    ordered = sorted(accumulated.items(), key=lambda item: (-item[1]["score"], item[0]))

    results: list[dict[str, Any]] = []
    for _, entry in ordered:
        doc = entry["doc"]
        doc.setdefault("scores", {})["rrf"] = entry["score"]
        provenance = doc.setdefault("provenance", {})
        provenance["rrf_sources"] = list(entry["sources"])
        provenance["rrf_best_rank"] = dict(entry["best_rank"])
        if len(entry["sources"]) > 1:
            provenance["rrf_multi_source"] = True
        results.append(doc)
    return results


def _sim_value(paper: dict[str, Any], sim_key: str) -> float:
    """从 scores 里取相似度分，缺失或 None 按 0 处理。"""

    scores = paper.get("scores") or {}
    value = scores.get(sim_key)
    return 0.0 if value is None else value


def _citation_estimate(paper: dict[str, Any]) -> int:
    """用 relations 里 citations + references 的条数估算引用规模，缺失按 0。"""

    relations = paper.get("relations") or {}
    citations = relations.get("citations") or []
    references = relations.get("references") or []
    return len(citations) + len(references)


def spar_rank(
    papers: Sequence[dict[str, Any]],
    *,
    sim_key: str = "relevance",
    bucket: float = DEFAULT_BUCKET,
) -> list[dict[str, Any]]:
    """SPAR 风格排序：相似度分桶，桶内按引用数、年份降序。

    排序键为 (桶号 desc, 引用数 desc, 年份 desc, paper_id asc)：
    - 桶号 = floor(sim / bucket)，sim 取 paper["scores"][sim_key]（None 当 0）；
    - 引用数由 relations 的 citations + references 数量估算（没有则 0）；
    - 年份取 bibliography.year（None 当 0）。

    不修改输入，返回排好序的新列表。
    """

    bucket_value = _check_positive_number(bucket, "bucket")

    def sort_key(paper: dict[str, Any]) -> tuple[int, int, int, str]:
        bucket_index = math.floor(_sim_value(paper, sim_key) / bucket_value)
        year = (paper.get("bibliography") or {}).get("year") or 0
        return (
            -bucket_index,
            -_citation_estimate(paper),
            -year,
            str(paper.get("paper_id") or ""),
        )

    return sorted(papers, key=sort_key)
