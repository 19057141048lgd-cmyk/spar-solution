"""PaSa 官方口径的评测（对齐 bytedance/pasa 的 metrics.py）。

协议要点（来自 repos/pasa/metrics.py + utils.py）：
- 匹配方式：``keep_letters`` 规范化标题（只保留字母并小写）后做集合交集；
- 指标（对每道题计算后做宏观平均）：
  * crawler_recall：全部爬取论文的召回；
  * selected_precision / selected_recall：选择分 > 0.5 的子集的精确/召回；
  * recall@20 / @50 / @100：按选择分降序取前 K 篇的召回；
- Gold 来自数据集的 ``answer`` 标题列表。

我们的 PaperDoc 用 ``scores.relevance`` 对应其 ``select_score``。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


SELECT_THRESHOLD = 0.5


def keep_letters(value: Any) -> str:
    """PaSa 的标题归一化：只保留字母并小写。"""

    return "".join(char for char in str(value or "") if char.isalpha()).lower()


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_pasa_style(
    papers: Sequence[Mapping[str, Any]],
    gold_titles: Sequence[str],
    *,
    select_threshold: float = SELECT_THRESHOLD,
) -> dict[str, Any]:
    """按 PaSa 协议计算单题指标（宏观平均由调用方汇总）。"""

    answer = {keep_letters(title) for title in gold_titles if keep_letters(title)}
    crawled = []
    seen: set[str] = set()
    for paper in papers:
        title = keep_letters((paper.get("bibliography") or {}).get("title"))
        if not title or title in seen:
            continue
        seen.add(title)
        score = (paper.get("scores") or {}).get("relevance")
        crawled.append((title, float(score) if isinstance(score, (int, float)) else 0.0))
    selected = {title for title, score in crawled if score > select_threshold}
    crawled_sorted = sorted(crawled, key=lambda item: (-item[1], item[0]))
    top = {k: {title for title, _ in crawled_sorted[:k]} for k in (20, 50, 100)}

    def pr(pred: set[str]) -> tuple[float, float]:
        if not answer:
            return 0.0, 0.0
        tp = len(pred & answer)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(answer)
        return precision, recall

    output: dict[str, Any] = {"gold_count": len(answer), "crawled_count": len(crawled), "selected_count": len(selected)}
    for name, pred in (
        ("crawler", seen),
        ("selected", selected),
        ("recall_20", top[20]),
        ("recall_50", top[50]),
        ("recall_100", top[100]),
    ):
        precision, recall = pr(pred)
        output[f"{name}_tp"] = len(pred & answer)
        output[f"{name}_precision"] = round(precision, 6)
        output[f"{name}_recall"] = round(recall, 6)
    precision, recall = output["selected_precision"], output["selected_recall"]
    output["selected_f1"] = round(_f1(precision, recall), 6)
    return output


def aggregate_pasa_style(per_query: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """PaSa 的宏观平均（每题等权）。"""

    rows = [item for item in per_query if isinstance(item, Mapping)]
    if not rows:
        return {}
    keys = ("crawler_recall", "selected_precision", "selected_recall", "selected_f1", "recall_20_recall", "recall_50_recall", "recall_100_recall")
    output = {"queries": len(rows)}
    for key in keys:
        output[key] = round(sum(float(item.get(key) or 0.0) for item in rows) / len(rows), 6)
    return output


__all__ = ["aggregate_pasa_style", "evaluate_pasa_style", "keep_letters"]
