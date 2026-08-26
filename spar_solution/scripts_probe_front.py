"""只验证「读题 → 搜索词 → 第一页里有没有金标或综述」。

不跑整棵搜索树。一次只跑一题。用法：
  python spar_solution/scripts_probe_front.py --qid AutoScholarQuery_test_9
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
from src.spar_baseline.question_understanding import (
    DRAFT_SYSTEM_PROMPT,
    REFLECT_SYSTEM_PROMPT,
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
}


def _titles(records):
    return [str((paper.get("bibliography") or {}).get("title") or "") for paper in records]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", required=True)
    args = parser.parse_args()
    qid = args.qid.strip()
    if qid not in ALLOWED:
        raise SystemExit(f"本阶段只允许 {sorted(ALLOWED)}")

    rows = {str(row["qid"]): row for row in load_rows(DATASET, offset=0, limit=50)}
    row = rows[qid]
    question = str(row["question"])
    gold = [keep_letters(title) for title in (row.get("answer") or []) if keep_letters(title)]

    pipeline = build_live_pipeline(citation_enabled=False, page_size=10)
    pipeline.providers.pop("bohrium", None)
    client = DeepSeekClient()

    started = time.perf_counter()
    draft = client.complete_json(DRAFT_SYSTEM_PROMPT, json.dumps({"task": "understand_question", "query": question}, ensure_ascii=False), max_tokens=800)
    parsed = parse_understanding(draft, question)
    revised = client.complete_json(
        REFLECT_SYSTEM_PROMPT,
        json.dumps({"task": "reflect_understanding", "query": question, "draft": parsed}, ensure_ascii=False),
        max_tokens=800,
    )
    understanding = parse_understanding(revised, question, fallback=parsed)
    queries = [_sanitize_query(text) for text in collect_search_queries(understanding)]
    queries = [text for text in queries if text and len(text.split()) >= 2]

    print(f"题: {qid}")
    print(f"问: {question}")
    print(f"理解领域: {understanding.get('field')}")
    print(f"备选: {understanding.get('alt_fields')}")
    print(f"像综述黑话: {understanding.get('jargon_from_survey')}")
    print(f"答案大概长什么样: {understanding.get('answer_looks_like')}")
    print(f"质疑: {understanding.get('doubts')}")
    print(f"搜索词: {queries}")

    gold_hits = []
    survey_hits = []
    seen = set()
    for text in queries:
        for name, provider in pipeline.providers.items():
            if not callable(getattr(provider, "search", None)):
                continue
            try:
                result = provider.search(text, page_size=10)
            except Exception as exc:
                print(f"  搜失败 {name} / {text[:60]}: {type(exc).__name__}")
                continue
            for paper in result.records:
                title = str((paper.get("bibliography") or {}).get("title") or "")
                key = keep_letters(title)
                if not title or key in seen:
                    continue
                seen.add(key)
                if key in gold:
                    gold_hits.append(title)
                if _is_survey(paper):
                    survey_hits.append(title)

    usage = client.usage
    print(f"第一页见到金标 {len(gold_hits)}/{len(gold)}: {gold_hits[:5]}")
    print(f"第一页见到综述 {len(survey_hits)} 篇，例如: {survey_hits[:5]}")
    print(f"去重后第一页论文 {len(seen)} 篇 | llm={usage.get('calls')} tokens={usage.get('total_tokens')} 秒={round(time.perf_counter()-started, 1)}")
    out = SOLUTION / "artifacts" / "autoscholar" / "probe-front" / qid
    out.mkdir(parents=True, exist_ok=True)
    (out / "understanding.json").write_text(json.dumps({"understanding": understanding, "queries": queries, "gold_hits": gold_hits, "survey_hits": survey_hits[:20], "pool_size": len(seen)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
