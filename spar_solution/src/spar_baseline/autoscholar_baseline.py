"""AutoScholarQuery 基线评测。

该模块只用于在 P2 优化前测量当前基线，不改变 P2 主流程。两路使用同一
arXiv Provider：``current`` 使用现有确定性 QueryPlanner，``deepseek`` 只
让 DeepSeek 生成一个短检索式，随后仍由 Provider 负责事实召回。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .providers.arxiv import ArxivProvider
from .query_planner import QueryPlanner


def _norm_arxiv(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", text, flags=re.I)
    text = re.sub(r"\.pdf$", "", text, flags=re.I)
    text = re.sub(r"^arxiv:", "", text, flags=re.I)
    return re.sub(r"v\d+$", "", text, flags=re.I).casefold()


def load_rows(path: str | Path, *, offset: int = 0, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines()[offset:offset + limit]:
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("dataset rows must be objects")
            rows.append(item)
    return rows


def _metric(predicted: list[str], gold: set[str], k: int = 10) -> dict[str, Any]:
    predicted_set = set(predicted[:k])
    tp = len(predicted_set & gold)
    fp = len(predicted_set - gold)
    fn = len(gold - predicted_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"k": k, "tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def _deepseek_query(question: str, *, api_key: str, base_url: str, model: str, timeout: float) -> tuple[str, dict[str, Any]]:
    prompt = (
        "Return only valid JSON in this exact shape: {\"queries\":[\"short query\"]}. "
        "Generate exactly one concise academic search query with at most 8 meaningful terms. "
        "Preserve important multi-word phrases. Do not include AND, OR, explanation, or markdown.\n\n"
        f"Question: {question}"
    )
    endpoint = urljoin(base_url.rstrip("/") + "/", "chat/completions")
    request = Request(
        endpoint,
        data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 120}).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310: configured DeepSeek endpoint
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"DeepSeek HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"DeepSeek request failed: {type(exc).__name__}") from exc
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        parsed = json.loads(str(content).strip().strip("`"))
        queries = parsed.get("queries") if isinstance(parsed, Mapping) else None
        query = str(queries[0]).strip() if isinstance(queries, list) and queries else ""
    except (json.JSONDecodeError, IndexError, TypeError, AttributeError):
        query = ""
    if not query or " AND " in query.upper() or " OR " in query.upper():
        raise RuntimeError("DeepSeek returned invalid query JSON")
    return query, {"latency_ms": round((time.perf_counter() - started) * 1000, 3), "model": model}


def _arxiv_ids(records: list[Mapping[str, Any]]) -> list[str]:
    ids: list[str] = []
    for record in records:
        value = (record.get("identifiers") or {}).get("arxiv_id") if isinstance(record.get("identifiers"), Mapping) else None
        normalized = _norm_arxiv(value)
        if normalized and normalized not in ids:
            ids.append(normalized)
    return ids


def run_baseline(
    dataset: str | Path,
    output: str | Path,
    *,
    offset: int = 0,
    limit: int = 20,
    page_size: int = 10,
    arxiv_sleep: float = 3.1,
    deepseek_key: str | None = None,
    deepseek_base_url: str = "https://api.deepseek.com",
    deepseek_model: str = "deepseek-chat",
) -> dict[str, Any]:
    rows = load_rows(dataset, offset=offset, limit=limit)
    if not rows:
        raise ValueError("no dataset rows selected")
    key = deepseek_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for this comparison")
    provider = ArxivProvider()
    planner = QueryPlanner()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    calls = {"deepseek": 0, "arxiv": 0}

    for index, row in enumerate(rows):
        question = str(row.get("question") or "").strip()
        qid = str(row.get("qid") or f"row_{offset + index}")
        gold = {_norm_arxiv(item) for item in row.get("answer_arxiv_id") or [] if _norm_arxiv(item)}
        item: dict[str, Any] = {"qid": qid, "question": question, "gold_arxiv_ids": sorted(gold), "current": {}, "deepseek": {}}
        try:
            current_plan = planner.plan(question)
            current_query = str(current_plan["subqueries"][0]["query_text"])
            item["current"]["query"] = current_query
        except Exception as exc:
            item["current"] = {"error": f"planner:{type(exc).__name__}"}
            current_query = ""

        deepseek_started = time.perf_counter()
        try:
            calls["deepseek"] += 1
            deep_query, deep_meta = _deepseek_query(question, api_key=key, base_url=deepseek_base_url, model=deepseek_model, timeout=45)
            item["deepseek"].update({"query": deep_query, **deep_meta})
        except Exception as exc:
            item["deepseek"] = {"error": str(exc)[:160], "latency_ms": round((time.perf_counter() - deepseek_started) * 1000, 3)}
            errors.append({"qid": qid, "stage": "deepseek", "code": "provider_error", "message": str(exc)[:160]})
            deep_query = ""

        for mode, query in (("current", current_query), ("deepseek", deep_query)):
            if not query:
                item[mode]["predicted_arxiv_ids"] = []
                item[mode]["metrics_at_10"] = _metric([], gold)
                continue
            if calls["arxiv"]:
                time.sleep(max(0.0, arxiv_sleep))
            started = time.perf_counter()
            try:
                calls["arxiv"] += 1
                response = provider.search(query, page_size=page_size)
                predicted = _arxiv_ids(response.records)
                item[mode].update({"predicted_arxiv_ids": predicted, "metrics_at_10": _metric(predicted, gold), "latency_ms": round((time.perf_counter() - started) * 1000, 3), "records": len(response.records)})
            except Exception as exc:
                item[mode] = {**item[mode], "predicted_arxiv_ids": [], "metrics_at_10": _metric([], gold), "error": f"arxiv:{type(exc).__name__}:{str(exc)[:120]}"}
                errors.append({"qid": qid, "stage": f"arxiv_{mode}", "code": "provider_error", "message": str(exc)[:160]})
        results.append(item)

    def aggregate(mode: str) -> dict[str, Any]:
        metrics = [item[mode]["metrics_at_10"] for item in results]
        total_tp = sum(item["tp"] for item in metrics)
        total_fp = sum(item["fp"] for item in metrics)
        total_fn = sum(item["fn"] for item in metrics)
        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"queries": len(metrics), "tp": total_tp, "fp": total_fp, "fn": total_fn, "micro_precision": round(precision, 6), "micro_recall": round(recall, 6), "micro_f1": round(f1, 6), "macro_f1": round(sum(item["f1"] for item in metrics) / max(1, len(metrics)), 6), "mean_latency_ms": round(sum(float(results[i][mode].get("latency_ms") or 0) for i in range(len(results))) / max(1, len(results)), 3), "mean_records": round(sum(int(results[i][mode].get("records") or 0) for i in range(len(results))) / max(1, len(results)), 3)}

    payload = {"schema_version": "autoscholar.baseline.v1", "dataset": str(Path(dataset)), "offset": offset, "limit": limit, "page_size": page_size, "execution": "live_arxiv_plus_deepseek", "deepseek_model": deepseek_model, "results": {"current": aggregate("current"), "deepseek": aggregate("deepseek")}, "calls": calls, "errors": errors, "rows": results, "limitations": ["仅评测所选样本，不代表 1000 条全集", "arXiv API 返回元数据/摘要，未读取全文", "DeepSeek 仅用于评测期查询生成，尚未接入 P2 主流程", "Gold 来自数据集答案，身份按 arXiv ID 精确归一化"]}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate current AutoScholar arXiv baseline")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--arxiv-sleep", type=float, default=3.1)
    args = parser.parse_args(argv)
    payload = run_baseline(args.dataset, args.output, offset=args.offset, limit=args.limit, page_size=args.page_size, arxiv_sleep=args.arxiv_sleep)
    print(json.dumps({"output": args.output, "calls": payload["calls"], "metrics": payload["results"], "errors": len(payload["errors"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

