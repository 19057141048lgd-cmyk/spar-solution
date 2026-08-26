"""只跑前两步：读题搜索 → LLM 按目的过滤，池子不够再翻页补搜。

不扩引用、不打「是不是答案」的分。一次一题。
  python spar_solution/scripts_probe_filter.py --qid AutoScholarQuery_test_9
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SOLUTION = Path(__file__).resolve().parent
REPO = SOLUTION.parent
sys.path.insert(0, str(SOLUTION))

from src.spar_baseline.autoscholar_baseline import load_rows
from src.spar_baseline.deepseek_layer import DeepSeekClient
from src.spar_baseline.p2_cli import build_live_pipeline
from src.spar_baseline.pasa_metrics import keep_letters
from src.spar_baseline.pool_filter import filter_papers
from src.spar_baseline.pool_reflect import reflect_on_pool
from src.spar_baseline.question_understanding import (
    DRAFT_SYSTEM_PROMPT,
    collect_search_queries,
    parse_understanding,
)
from src.spar_baseline.search_tree import _is_survey, _sanitize_query

DATASET = REPO / "repos" / "SPAR" / "benchmark" / "AutoScholarQuery_test.jsonl"
ALLOWED = {
    "AutoScholarQuery_test_0",
    "AutoScholarQuery_test_7",
    "AutoScholarQuery_test_8",
    "AutoScholarQuery_test_9",
    "AutoScholarQuery_test_15",
    "AutoScholarQuery_test_88",
}
TARGET_POOL = 50
MAX_PAGES = 3
PAGE_SIZE = 10


def _title_key(paper):
    return keep_letters((paper.get("bibliography") or {}).get("title"))


def _search_page(providers, queries, page, seen):
    fresh = []
    errors = 0
    for text in queries:
        for name, provider in providers.items():
            if not callable(getattr(provider, "search", None)):
                continue
            try:
                result = provider.search(text, page_size=PAGE_SIZE, page=page)
            except Exception:
                errors += 1
                continue
            for paper in result.records:
                key = _title_key(paper)
                if not key or key in seen:
                    continue
                seen.add(key)
                fresh.append(paper)
    return fresh, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", required=True)
    args = parser.parse_args()
    qid = args.qid.strip()
    if qid not in ALLOWED:
        raise SystemExit(f"本阶段只允许 {sorted(ALLOWED)}")

    row = None
    for item in load_rows(DATASET, offset=0, limit=200):
        if str(item.get("qid")) == qid:
            row = item
            break
    if row is None:
        raise SystemExit(f"找不到题目 {qid}")
    question = str(row["question"])
    gold = [keep_letters(title) for title in (row.get("answer") or []) if keep_letters(title)]

    pipeline = build_live_pipeline(citation_enabled=False, page_size=PAGE_SIZE)
    pipeline.providers.pop("bohrium", None)
    client = DeepSeekClient()
    started = time.perf_counter()

    draft = client.complete_json(
        DRAFT_SYSTEM_PROMPT,
        json.dumps({"task": "understand_question", "query": question}, ensure_ascii=False),
        max_tokens=800,
    )
    understanding = parse_understanding(draft, question)
    queries = [_sanitize_query(text) for text in collect_search_queries(understanding)]
    queries = [text for text in queries if text and len(text.split()) >= 2]

    print(f"题: {qid}")
    print(f"问: {question}")
    print(f"理解领域: {understanding.get('field')} | 备选: {understanding.get('alt_fields')}")
    print(f"搜索词: {queries}")

    seen: set[str] = set()
    pool: list[dict] = []
    dropped_all: list[dict] = []
    searched = 0
    for page in range(1, MAX_PAGES + 1):
        if len(pool) >= TARGET_POOL:
            break
        fresh, errors = _search_page(pipeline.providers, queries, page, seen)
        searched += len(fresh)
        if errors:
            print(f"  第{page}页有 {errors} 次搜索失败")
        if not fresh:
            print(f"  第{page}页没有新论文，停止补搜")
            break
        kept, dropped = filter_papers(client, understanding, fresh)
        pool.extend(kept)
        dropped_all.extend(dropped)
        print(f"  第{page}页：新搜到 {len(fresh)}，留下 {len(kept)}，丢掉 {len(dropped)}，池子现在 {len(pool)}")

    course = reflect_on_pool(client, question, understanding, pool)
    print(f"纠偏: {course.get('verdict')} | {course.get('why')}")
    if course.get("verdict") in {"wrong_street", "mixed"} and course.get("queries"):
        print(f"换街搜索词: {course['queries']}")
        if course.get("field"):
            understanding = dict(understanding)
            understanding["field"] = course["field"]
        extra, errors = _search_page(pipeline.providers, [_sanitize_query(t) for t in course["queries"] if _sanitize_query(t)], 1, seen)
        searched += len(extra)
        if extra:
            kept, dropped = filter_papers(client, understanding, extra)
            pool.extend(kept)
            dropped_all.extend(dropped)
            print(f"  换街后又搜到 {len(extra)}，留下 {len(kept)}，丢掉 {len(dropped)}，池子现在 {len(pool)}")

    gold_hits = [
        str((paper.get("bibliography") or {}).get("title") or "")
        for paper in pool
        if _title_key(paper) in gold
    ]
    survey_hits = [
        str((paper.get("bibliography") or {}).get("title") or "")
        for paper in pool
        if _is_survey(paper)
    ]
    dropped_titles = [str((paper.get("bibliography") or {}).get("title") or "") for paper in dropped_all[:8]]
    usage = client.usage
    print(f"池子 {len(pool)} 篇（目标 {TARGET_POOL}），一共搜到过 {searched} 篇，丢掉 {len(dropped_all)} 篇")
    print(f"池子里金标 {len(gold_hits)}/{len(gold)}: {gold_hits[:5]}")
    print(f"池子里综述 {len(survey_hits)} 篇，例如: {survey_hits[:5]}")
    print(f"丢掉的例子: {dropped_titles}")
    print(f"llm={usage.get('calls')} tokens={usage.get('total_tokens')} 秒={round(time.perf_counter()-started, 1)}")

    out = SOLUTION / "artifacts" / "autoscholar" / "probe-filter" / qid
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "qid": qid,
                "understanding": understanding,
                "queries": queries,
                "pool_size": len(pool),
                "searched": searched,
                "dropped": len(dropped_all),
                "gold_hits": gold_hits,
                "survey_hits": survey_hits[:20],
                "dropped_examples": dropped_titles,
                "pool_titles": [str((paper.get("bibliography") or {}).get("title") or "") for paper in pool[:50]],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
