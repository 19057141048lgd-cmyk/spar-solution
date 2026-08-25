"""单题端到端验证：正文驱动的检索流程是否真实可行。

用法：python spar_solution/scripts_validate_flow.py [题目行号，默认 9]

验证链条：问题 → 检索选种子 → 下载正文（HTML/PDF）→ 切章节 → LLM 挑章节
→ 引用句解析 → 挑扩展引用 → 按标题查回真论文 → 双评审 → 对 Gold 计分。
每一步打印证据（下载了多少、解析出几节几条引用、挑了哪几条、查回了什么），
证明机制不是自嗨。
"""

import json
import sys
from pathlib import Path

SOLUTION = Path(__file__).resolve().parent
REPO = SOLUTION.parent
sys.path.insert(0, str(SOLUTION))

from src.spar_baseline.config import load_config
from src.spar_baseline.deepseek_layer import DeepSeekClient, DeepSeekUnderstandingLayer
from src.spar_baseline.fulltext_flow import (
    dual_review,
    extract_reference_title,
    load_paper_fulltext,
    pick_citations,
    pick_references,
    select_relevant_sections,
)
from src.spar_baseline.identity import normalize_title
from src.spar_baseline.openalex_provider import OpenAlexProvider
from src.spar_baseline.providers.arxiv import ArxivProvider
from src.spar_baseline.query_planner import QueryPlanner

ROW = int(sys.argv[1]) if len(sys.argv) > 1 else 9
DATASET = REPO / "repos" / "SPAR" / "benchmark" / "AutoScholarQuery_test.jsonl"


def main() -> int:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    row = rows[ROW]
    question = row["question"]
    gold_ids = row["answer_arxiv_id"]
    gold_titles = [normalize_title(t) for t in row["answer"]]
    print(f"[题目 {row['qid']}] {question}")
    print(f"[Gold] {len(gold_ids)} 篇: {gold_ids}")
    print()

    layer = DeepSeekUnderstandingLayer(DeepSeekClient(api_key=load_config().get("DEEPSEEK_API_KEY", "")))
    arxiv = ArxivProvider.from_config(dict(load_config()))
    openalex = OpenAlexProvider(dict(load_config()))

    # ① 计划 + 检索 → 种子（多种角查询：先识别领域，再出关键词式查询）
    from src.spar_baseline.search_tree import _sanitize_query

    try:
        plan = layer.plan(question)
        queries = [q for q in dict.fromkeys(_sanitize_query(s["query_text"]) for s in plan["subqueries"]) if len(q.split()) >= 2][:4]
        planner_source = "deepseek"
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
    except Exception as exc:
        plan = QueryPlanner().plan(question)
        queries = [s["query_text"] for s in plan["subqueries"]][:2]
        planner_source = f"rules({exc.__class__.__name__})"
    print(f"[① 规划:{planner_source}] 查询: {queries}")
    seeds: list[dict] = []
    for query in queries[:2]:
        for provider in (arxiv, openalex):
            try:
                result = provider.search(query, page_size=5)
                seeds.extend(result.records[:3])
            except Exception as exc:
                print(f"    检索失败 {provider.source}: {exc}")
    # 去重 + 初筛种子
    unique: dict[str, dict] = {}
    for paper in seeds:
        key = str(paper.get("identifiers", {}).get("arxiv_id") or paper.get("identifiers", {}).get("doi") or paper.get("paper_id"))
        unique.setdefault(key, paper)
    seeds = list(unique.values())[:3]
    print(f"[① 检索] 去重后种子 {len(seeds)} 篇: {[p['bibliography']['title'][:45] for p in seeds]}")
    print()

    # ② 正文获取 + ③ 章节挑选 + ④ 引用挑选（每篇种子一个"扩展员"）
    resolved: list[dict] = []
    for seed in seeds:
        title = seed["bibliography"]["title"][:50]
        fulltext = load_paper_fulltext(seed, cache_dir=SOLUTION / "artifacts" / "flow-cache")
        print(f"[② 正文:{fulltext.source}] {title}")
        print(f"    章节 {len(fulltext.sections)} 个 | 参考文献条目 {len(fulltext.references)} 条 | 引用句 {len(fulltext.citation_contexts)} 处")
        if fulltext.source == "none":
            print("    ✗ 正文获取失败，跳过该种子")
            continue
        indices = select_relevant_sections(layer.client, question, fulltext)
        picked = pick_references(layer.client, question, fulltext)
        print(f"[③ 章节挑选] 相关章节索引: {indices}")
        print(f"[④ 引用挑选] {len(picked)} 条:")
        for pick in picked:
            print(f"    - {pick['query'][:70]} ({pick.get('reason', '')[:40]})")
        for pick in picked:
            ref_title = pick["query"]
            # ⑤ 按标题查回真论文
            for provider in (arxiv, openalex):
                try:
                    result = provider.search(ref_title, page_size=3)
                except Exception:
                    continue
                for candidate in result.records[:2]:
                    if normalize_title(candidate["bibliography"].get("title") or "") == normalize_title(ref_title):
                        candidate["_pick_reason"] = pick.get("reason", "")
                        candidate["_from_seed"] = title
                        resolved.append(candidate)
                        break
                else:
                    continue
                break
        print()
    unique_resolved: dict[str, dict] = {}
    for paper in resolved:
        key = str(paper.get("identifiers", {}).get("arxiv_id") or paper.get("paper_id"))
        unique_resolved.setdefault(key, paper)
    resolved = list(unique_resolved.values())
    print(f"[⑤ 查回] 引用解析后实际查回论文 {len(resolved)} 篇:")
    for paper in resolved:
        print(f"    - {paper['bibliography']['title'][:60]} (来自: {paper.get('_from_seed', '')[:30]})")
    print()

    # ⑥ 双评审 + Gold 对照
    reviews = dual_review(layer.client, question, resolved) if resolved else []
    print("[⑥ 双评审] 严格派 / 召回派 / 结论:")
    hits = 0
    for paper, review in zip(resolved, reviews):
        blob = " ".join(str(v) for v in (paper.get("identifiers") or {}).values() if v)
        is_gold = any(g in blob for g in gold_ids) or normalize_title(paper["bibliography"].get("title") or "") in gold_titles
        hits += bool(is_gold)
        print(f"    {paper['bibliography']['title'][:44]:46} A={review['score_a']:.2f} B={review['score_b']:.2f} -> {review['status']}" + ("   ◆ GOLD!" if is_gold else ""))
    print()
    print(f"[结果] 查回 {len(resolved)} 篇，其中命中 Gold {hits} 篇 / 共 {len(gold_ids)} 篇")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
