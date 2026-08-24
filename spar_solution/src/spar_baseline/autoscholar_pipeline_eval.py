"""AutoScholarQuery 全管线评测。

每道题运行完整 P2 live 管线（DeepSeek 规划 + arXiv/OpenAlex 召回 + 引用
扩展 + LLM 相关性判断），按数据集自带的 ``answer_arxiv_id`` Gold 计分。
与 ``autoscholar_baseline``（单查询对照）共用同一数据集，结果可直接对比。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from .autoscholar_baseline import _norm_arxiv, load_rows
from .p2_cli import build_live_pipeline
from .p2_metrics import evaluate_p2_run


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def run_pipeline_eval(
    dataset: str | Path,
    output: str | Path,
    *,
    offset: int = 0,
    limit: int = 10,
    page_size: int = 10,
    sleep_seconds: float = 3.0,
    exclude_sources: tuple[str, ...] = ("bohrium",),
) -> dict[str, Any]:
    rows = load_rows(dataset, offset=offset, limit=limit)
    if not rows:
        raise ValueError("no dataset rows selected")
    pipeline = build_live_pipeline(citation_enabled=True, page_size=page_size)
    for source in exclude_sources:
        pipeline.providers.pop(source, None)

    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    per_query: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, row in enumerate(rows):
        question = str(row.get("question") or "").strip()
        qid = str(row.get("qid") or f"row_{offset + index}")
        gold = sorted({_norm_arxiv(item) for item in row.get("answer_arxiv_id") or [] if _norm_arxiv(item)})
        run_dir = output_root / qid
        item: dict[str, Any] = {"qid": qid, "question": question, "gold_count": len(gold)}
        try:
            run = pipeline.run(question, output_dir=run_dir)
            result = evaluate_p2_run(run, gold_ids=gold, cutoffs=(10, 20))
            manifest = run.manifest
            cost = manifest.get("cost") or {}
            item.update({
                "planner_source": manifest.get("planner_source"),
                "iterations": manifest.get("iterations"),
                "candidates": len(run.papers),
                "metrics_at_10": result["by_cutoff"]["10"],
                "metrics_at_20": result["by_cutoff"]["20"],
                "provider_calls": sum((cost.get("provider_calls") or {}).values()),
                "llm_calls": cost.get("llm_calls"),
                "total_tokens": cost.get("total_tokens"),
                "wall_ms": cost.get("wall_ms"),
                "errors": len(run.errors),
            })
        except Exception as exc:  # 单题失败不终止整批评测。
            item["error"] = f"{type(exc).__name__}: {exc}"[:200]
        per_query.append(item)
        print(f"[{index + 1}/{len(rows)}] {qid}: tp@10={item.get('metrics_at_10', {}).get('tp')} candidates={item.get('candidates')}", flush=True)
        if index + 1 < len(rows):
            time.sleep(max(0.0, sleep_seconds))

    def aggregate(cutoff: int) -> dict[str, Any]:
        metrics = [item[f"metrics_at_{cutoff}"] for item in per_query if f"metrics_at_{cutoff}" in item]
        tp = sum(m["tp"] for m in metrics)
        fp = sum(m["fp"] for m in metrics)
        fn = sum(m["fn"] for m in metrics)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            "cutoff": cutoff,
            "queries": len(metrics),
            "tp": tp, "fp": fp, "fn": fn,
            "micro_precision": round(precision, 6),
            "micro_recall": round(recall, 6),
            "micro_f1": round(_f1(precision, recall), 6),
            "macro_f1": round(sum(m["f1"] for m in metrics) / max(1, len(metrics)), 6),
        }

    payload = {
        "schema_version": "autoscholar.pipeline_eval.v1",
        "dataset": str(dataset),
        "offset": offset,
        "limit": limit,
        "page_size": page_size,
        "excluded_sources": list(exclude_sources),
        "execution": "live_p2_pipeline",
        "results": {"at_10": aggregate(10), "at_20": aggregate(20)},
        "totals": {
            "wall_ms": round((time.perf_counter() - started) * 1000, 3),
            "provider_calls": sum(int(item.get("provider_calls") or 0) for item in per_query),
            "llm_calls": sum(int(item.get("llm_calls") or 0) for item in per_query),
            "total_tokens": sum(int(item.get("total_tokens") or 0) for item in per_query),
            "failed_queries": sum(1 for item in per_query if "error" in item),
        },
        "rows": per_query,
        "limitations": [
            "仅评测所选样本，不代表 1000 条全集",
            "引用扩展仅 OpenAlex；Bohrium 因密钥无效被排除",
            "身份匹配含 arXiv-DOI↔arXiv-ID 等价（2026-08-25 修复后口径）",
        ],
    }
    output_path = Path(output)
    output_path = output_path / "summary.json" if output_path.is_dir() else output_path
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the full P2 pipeline on AutoScholarQuery")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=3.0)
    args = parser.parse_args(argv)
    payload = run_pipeline_eval(args.dataset, args.output, offset=args.offset, limit=args.limit, page_size=args.page_size, sleep_seconds=args.sleep)
    print(json.dumps({"output": args.output, "results": payload["results"], "totals": payload["totals"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
