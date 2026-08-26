"""对照各家「前半段」：只出搜索词并搜第一页。不扩引用。

方法都用原文提示词，模型统一 DeepSeek。一次一题。
  python spar_solution/scripts_compare_fronts.py --qid AutoScholarQuery_test_9
"""

from __future__ import annotations

import argparse
import json
import re
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
    collect_search_queries,
    parse_understanding,
)
from src.spar_baseline.search_tree import _sanitize_query

DATASET = REPO / "repos" / "SPAR" / "benchmark" / "AutoScholarQuery_test.jsonl"
ALLOWED = {
    "AutoScholarQuery_test_0",
    "AutoScholarQuery_test_9",
}

# PaSa agent_prompt.json generate_query
PASA_PROMPT = (
    "Please generate some mutually exclusive queries in a list to search the relevant papers "
    "according to the User Query. Searching for survey papers would be better.\n"
    "Return JSON only: {\"queries\": [\"...\"]}.\n"
    "User Query: USER_QUERY"
)

# SPAR instruction.py template_extract_keywords
SPAR_PROMPT = (
    "Suggest OpenAlex or SemanticScholar search API queries to retrieve relevant papers addressing "
    "the most recent research on the given question. The search queries should be concise, "
    "comma-separated, and highly relevant. Format your response as follows:\n\n"
    "**Example:**\n\n"
    "Question: How have prior works incorporated personality attributes to train personalized "
    "dialogue generation models?\n"
    "Response:[Start] personalized dialogue generation, personalized language models, personalized dialogue[End]\n\n"
    "Now, generate search queries for the following question:\n"
    "Question: USER_QUERY\n"
    "Response:"
)

# Ai2 broad_search_by_keyword_prompts.py
AI2_KEYWORD_PROMPT = (
    "Given a user-provided natural language description of desired scientific papers, "
    "reformulate the query as an alternative search query for the Semantic Scholar search engine. "
    "Focus on formulating the natural language query into a keyword search query, that is, "
    "remove unnecessary descriptive words that wont show up in the content itself and keep the keywords to look for. "
    "Use plain text for queries, as Semantic Scholar does not support special syntax. "
    "Semantic Scholar does not support hyphens in queries, so avoid hyphens. "
    "When building the queries, try to use only content-keywords, that is, do emit metadata or non keyword-y wordings! "
    "Return JSON: {\"queries\": [\"one keyword query\", \"optional second\"]}.\n"
    "Input description: ```USER_QUERY```"
)

# Ai2 formulation_prompts.py alternative dense: wording used in the domain
AI2_DOMAIN_WORDING_PROMPT = (
    "Your task is to come up with up to 5 alternative search queries that will help find passages "
    "that answer the following search query. The queries will be run on a dense index that contains "
    "passages from academic research papers. I am NOT looking for simple synonym paraphrases of common words. "
    "Try using some reasoning to come up with interesting new ways to answer the original query. "
    "Make sure you use wording that is actually used within the searched for domain. Don't just give arbitrary synonyms. "
    "Drop phrases like \"study about...\", \"research showing...\". "
    "Return JSON: {\"queries\": [\"...\"]}."
)

# sciagent query_rewriter.py
SCIAGENT_PROMPT = (
    "You are an academic search query optimizer. Given a user's request for research papers, "
    "extract a focused search query for academic databases.\n"
    "Rules:\n"
    "- Output ONLY topic keywords suitable for Semantic Scholar / OpenAlex search APIs\n"
    "- Do NOT include year constraints, citation counts, limits, or meta-instructions\n"
    "- Use standard academic terminology\n"
    "- Be concise: 2-5 keywords or a short phrase\n"
    "- Think about what terms would appear in paper titles and abstracts\n"
    "Return JSON: {\"queries\": [\"...\", \"...\"]} (2-5 short queries)."
)


def _parse_queries(payload, raw_text: str) -> list[str]:
    texts: list[str] = []
    if isinstance(payload, dict):
        for key in ("queries", "alternative_queries"):
            values = payload.get(key)
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, str) and item.strip():
                        texts.append(item.strip())
                    elif isinstance(item, dict):
                        t = str(item.get("query") or item.get("query_text") or item.get("search_query") or "").strip()
                        if t:
                            texts.append(t)
        for key in ("keyword_query", "search_query"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
        understanding = parse_understanding(payload, "")
        texts.extend(collect_search_queries(understanding))
    if not texts and raw_text:
        match = re.search(r"\[Start\](.*?)\[End\]", raw_text, flags=re.I | re.S)
        if match:
            texts.extend(part.strip() for part in match.group(1).split(",") if part.strip())
    cleaned = []
    seen = set()
    for text in texts:
        text = _sanitize_query(text)
        key = text.casefold()
        if text and len(text.split()) >= 2 and key not in seen:
            seen.add(key)
            cleaned.append(text)
    return cleaned[:6]


def _generate(client: DeepSeekClient, method: str, question: str) -> tuple[list[str], dict]:
    if method == "ours":
        payload = client.complete_json(
            DRAFT_SYSTEM_PROMPT,
            json.dumps({"task": "understand_question", "query": question}, ensure_ascii=False),
            max_tokens=800,
        )
        parsed = parse_understanding(payload, question)
        return collect_search_queries(parsed), {"understanding": parsed}
    if method == "pasa":
        payload = client.complete_json(
            "Return JSON only. Never invent paper titles.",
            PASA_PROMPT.replace("USER_QUERY", question),
            max_tokens=400,
        )
        return _parse_queries(payload, json.dumps(payload)), {}
    if method == "spar":
        payload = client.complete_json(
            "Follow the response format exactly. You may wrap the [Start]...[End] block in JSON as {\"raw\": \"...\"}.",
            SPAR_PROMPT.replace("USER_QUERY", question),
            max_tokens=400,
        )
        raw = json.dumps(payload, ensure_ascii=False)
        return _parse_queries(payload, raw), {}
    if method == "ai2_keyword":
        payload = client.complete_json("Return JSON only.", AI2_KEYWORD_PROMPT.replace("USER_QUERY", question), max_tokens=400)
        return _parse_queries(payload, ""), {}
    if method == "ai2_domain_wording":
        payload = client.complete_json(AI2_DOMAIN_WORDING_PROMPT, question, max_tokens=500)
        return _parse_queries(payload, ""), {}
    if method == "sciagent":
        payload = client.complete_json(SCIAGENT_PROMPT, question, max_tokens=400)
        return _parse_queries(payload, ""), {}
    raise ValueError(method)


def _search(providers, queries):
    seen = set()
    titles = []
    for text in queries:
        for provider in providers.values():
            if not callable(getattr(provider, "search", None)):
                continue
            try:
                result = provider.search(text, page_size=10, page=1)
            except Exception:
                continue
            for paper in result.records:
                title = str((paper.get("bibliography") or {}).get("title") or "")
                key = keep_letters(title)
                if not title or key in seen:
                    continue
                seen.add(key)
                titles.append(title)
    return titles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", required=True)
    args = parser.parse_args()
    qid = args.qid.strip()
    if qid not in ALLOWED:
        raise SystemExit(f"本对照只跑 {sorted(ALLOWED)}")

    row = None
    for item in load_rows(DATASET, offset=0, limit=20):
        if str(item.get("qid")) == qid:
            row = item
            break
    if row is None:
        raise SystemExit(f"找不到 {qid}")
    question = str(row["question"])
    gold = [keep_letters(t) for t in (row.get("answer") or []) if keep_letters(t)]
    gold_titles = list(row.get("answer") or [])

    pipeline = build_live_pipeline(citation_enabled=False, page_size=10)
    pipeline.providers.pop("bohrium", None)
    client = DeepSeekClient()
    methods = ["ours", "pasa", "spar", "ai2_keyword", "ai2_domain_wording", "sciagent"]
    started = time.perf_counter()
    rows_out = []

    print(f"题: {qid}")
    print(f"问: {question}")
    print(f"金标: {gold_titles}")
    print()

    for method in methods:
        t0 = time.perf_counter()
        try:
            queries, extra = _generate(client, method, question)
        except Exception as exc:
            print(f"[{method}] 出词失败: {type(exc).__name__}: {str(exc)[:120]}")
            rows_out.append({"method": method, "error": str(exc)[:200], "queries": [], "gold_hits": []})
            continue
        titles = _search(pipeline.providers, queries)
        hits = [title for title in titles if keep_letters(title) in gold]
        print(f"[{method}] 搜索词: {queries}")
        print(f"  第一页 {len(titles)} 篇，金标 {len(hits)}/{len(gold)}: {hits[:3]}")
        print(f"  用时 {round(time.perf_counter()-t0, 1)}s")
        print()
        rows_out.append(
            {
                "method": method,
                "queries": queries,
                "pool": len(titles),
                "gold_hits": hits,
                "sample_titles": titles[:8],
                "extra": extra.get("understanding") if extra.get("understanding") else extra,
            }
        )

    usage = client.usage
    print(f"合计 llm={usage.get('calls')} tokens={usage.get('total_tokens')} 秒={round(time.perf_counter()-started, 1)}")
    out = SOLUTION / "artifacts" / "autoscholar" / "compare-fronts" / qid
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps({"qid": qid, "question": question, "gold": gold_titles, "methods": rows_out, "usage": dict(usage)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
