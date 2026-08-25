"""新流程 5 题随机抽测：与旧搜索树流程同题对比。

用法：python spar_solution/scripts_test_flow_5.py [随机种子，默认 42]

从已有 tree 基线的前 50 题里随机抽 5 题（有对照），逐题跑正文驱动新流程，
输出：机制统计（正文/引用/查回）+ Gold 命中 + 双评审结论。
"""

import json
import random
import sys
from pathlib import Path

SOLUTION = Path(__file__).resolve().parent
REPO = SOLUTION.parent
sys.path.insert(0, str(SOLUTION))

from src.spar_baseline.config import load_config
from src.spar_baseline.deepseek_layer import DeepSeekClient, DeepSeekUnderstandingLayer
from src.spar_baseline.fulltext_flow import (
    dual_review,
    load_paper_fulltext,
    pick_references,
    select_relevant_sections,
)
from src.spar_baseline.identity import normalize_title
from src.spar_baseline.openalex_provider import OpenAlexProvider
from src.spar_baseline.providers.arxiv import ArxivProvider
from src.spar_baseline.search_tree import _sanitize_query

DATASET = REPO / "repos" / "SPAR" / "benchmark" / "AutoScholarQuery_test.jsonl"
TREE_BASELINE = SOLUTION / "artifacts" / "autoscholar" / "tree-n50"


def run_one(row, layer, arxiv, openalex, cache_dir) -> dict:
    question = row["question"]
    gold_ids = [g.rstrip("v0123456789") for g in row["answer_arxiv_id"]]
    gold_titles = [normalize_title(t) for t in row["answer"]]

    def is_gold(paper):
        blob = " ".join(str(v) for v in (paper.get("identifiers") or {}).values() if v)
        return any(g in blob for g in gold_ids) or normalize_title(paper["bibliography"].get("title") or "") in gold_titles

    # ① 查询（计划 + 领域多种角）
    queries: list[str] = []
    try:
        plan = layer.plan(question)
        queries = [q for q in dict.fromkeys(_sanitize_query(s["query_text"]) for s in plan["subqueries"]) if len(q.split()) >= 2][:3]
    except Exception:
        pass
    try:
        angles = layer.client.complete_json(
            "You are an academic search strategist. First infer the likely research FIELD of the question "
            "(e.g. bandits/RL vs queueing theory vs NLP) to avoid terminology collisions, then produce 3 short "
            "keyword queries from different angles in that field. Return JSON only: "
            '{"queries": [{"query_text": "..."}]}.',
            json.dumps({"question": question}, ensure_ascii=False),
            max_tokens=500,
        )
        extra = [q for q in dict.fromkeys(_sanitize_query(a.get("query_text") or "") for a in angles.get("queries") or []) if len(q.split()) >= 2]
        queries = list(dict.fromkeys(queries + extra))[:5]
    except Exception:
        pass
    # ② 检索种子（按词法初筛排序取前 3）
    seeds: dict[str, dict] = {}
    for query in queries[:4]:
        for provider in (arxiv, openalex):
            try:
                result = provider.search(query, page_size=5)
            except Exception:
                continue
            for paper in result.records[:3]:
                key = str(paper.get("identifiers", {}).get("arxiv_id") or paper.get("paper_id"))
                seeds.setdefault(key, paper)
    seed_list = list(seeds.values())[:3]

    # ③④ 每篇种子一个"扩展员"：正文→章节→引用点名
    resolved: dict[str, dict] = {}
    stats = {"seeds": len(seed_list), "fulltext_ok": 0, "refs_picked": 0}
    for seed in seed_list:
        fulltext = load_paper_fulltext(seed, cache_dir=cache_dir)
        if fulltext.source == "none":
            continue
        stats["fulltext_ok"] += 1
        select_relevant_sections(layer.client, question, fulltext)
        for pick in pick_references(layer.client, question, fulltext):
            stats["refs_picked"] += 1
            for provider in (arxiv, openalex):
                try:
                    result = provider.search(pick["query"], page_size=3)
                except Exception:
                    continue
                match = next((c for c in result.records[:2] if normalize_title(c["bibliography"].get("title") or "") == normalize_title(pick["query"])), None)
                if match:
                    match["_reason"] = pick.get("reason", "")
                    resolved.setdefault(str(match.get("paper_id")), match)
                    break
    candidates = list(resolved.values())

    # ⑤ 双评审
    reviews = dual_review(layer.client, question, candidates) if candidates else []
    kept = [c for c, r in zip(candidates, reviews) if r["status"] == "keep"]
    gold_in_cand = [c for c in candidates if is_gold(c)]
    gold_in_keep = [c for c in kept if is_gold(c)]
    return {
        "qid": row["qid"],
        "question": question[:60],
        "queries": queries,
        "seed_titles": [p["bibliography"]["title"][:40] for p in seed_list],
        **stats,
        "resolved": len(candidates),
        "gold_total": len(gold_ids),
        "gold_in_candidates": len(gold_in_cand),
        "gold_kept": len(gold_in_keep),
        "keep_count": len(kept),
        "review_failed": sum(1 for r in reviews if r["status"] == "review_failed"),
    }


def main() -> int:
    seed_value = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()][:50]
    sample = random.Random(seed_value).sample(rows, 5)
    config = dict(load_config())
    layer = DeepSeekUnderstandingLayer(DeepSeekClient(api_key=config.get("DEEPSEEK_API_KEY", "")))
    arxiv = ArxivProvider.from_config(config)
    openalex = OpenAlexProvider(config)
    cache = SOLUTION / "artifacts" / "flow-cache"

    print(f"随机种子 {seed_value} 抽中: {[r['qid'][-2:] for r in sample]}\n")
    totals = {"gold_total": 0, "gold_in_candidates": 0, "gold_kept": 0, "resolved": 0, "fulltext_ok": 0, "refs_picked": 0}
    for row in sample:
        result = run_one(row, layer, arxiv, openalex, cache)
        # 对照：旧搜索树流程同题命中（按标题/ID 匹配 tree 基线产物）
        tree_hit = 0
        tree_file = TREE_BASELINE / row["qid"] / "papers.json"
        if tree_file.is_file():
            papers = json.loads(tree_file.read_text(encoding="utf-8"))["papers"]
            gold_ids = [g.rstrip("v0123456789") for g in row["answer_arxiv_id"]]
            gold_titles = [normalize_title(t) for t in row["answer"]]
            tree_hit = sum(
                1
                for p in papers
                if any(g in " ".join(str(v) for v in (p.get("identifiers") or {}).values() if v) for g in gold_ids)
                or normalize_title(p["bibliography"].get("title") or "") in gold_titles
            )
        print(f"[{result['qid']}] {result['question']}")
        print(f"  种子({result['seeds']}): {result['seed_titles']}")
        print(f"  正文OK {result['fulltext_ok']}/{result['seeds']} | 引用点名 {result['refs_picked']} | 查回 {result['resolved']} 篇")
        print(f"  Gold: 候选中 {result['gold_in_candidates']}/{result['gold_total']} | 双评审保留 {result['gold_kept']} | 评审失败 {result['review_failed']} | 旧树基线命中 {tree_hit}")
        print()
        for key in totals:
            if key in result:
                totals[key] += result[key]
    print("=== 汇总（新流程 5 题） ===")
    print(f"Gold 总数 {totals['gold_total']} | 进候选 {totals['gold_in_candidates']} | 双评审保留 {totals['gold_kept']}")
    print(f"正文获取 {totals['fulltext_ok']} 篇 | 引用点名 {totals['refs_picked']} 条 | 查回 {totals['resolved']} 篇")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
