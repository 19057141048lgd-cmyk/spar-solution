# -*- coding: utf-8 -*-
"""小测试：联网搜索做术语接地（用户提议，2026-08-26）。

对三道顽固零分题：Bing 网页搜索题面 → 抓标题+摘要 → DeepSeek 推断
"题目黑话出自哪个领域的综述分类表"并给出该领域的 arXiv 检索词。
全程不给任何领域提示，验证联网接地的独立价值。

成本：3 次 Bing 抓取（免费）+ 3 次 DeepSeek（每次 ~3k token）。
"""

import json
import re
import sys
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

SOLUTION = Path(__file__).resolve().parent
sys.path.insert(0, str(SOLUTION))

from src.spar_baseline.autoscholar_baseline import load_rows
from src.spar_baseline.config import load_config
from src.spar_baseline.deepseek_layer import DeepSeekClient

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
QUESTIONS = {
    "AutoScholarQuery_test_0": "异常检测（时序/自编码重构误差），金标=Graph Attention Network 异常检测",
    "AutoScholarQuery_test_9": "上下文老虎机（contextual bandits 非参数/平滑性），金标=7篇经典 contextual bandit 论文",
    "AutoScholarQuery_test_7": "医学图像分割不确定性（Probabilistic U-Net 家族），金标=4篇分割不确定性论文",
}


def strip_tags(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", text).strip()


def bing_search(query: str, timeout: float = 15.0) -> list[dict]:
    # urlopen 的 TLS 指纹被 Bing 识别为机器人（同 URL curl 有 9 条结果、
    # urlopen 返回 JS 壳 0 条），改走 curl 子进程。
    import subprocess
    url = "https://cn.bing.com/search?q=" + quote_plus(query) + "&count=10"
    html = subprocess.run(
        ["curl", "-s", "-m", str(int(timeout)), "-L", "-A", UA, url],
        capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=timeout + 5,
    ).stdout
    results = []
    for block in re.findall(r'<li class="b_algo".*?</li>', html, flags=re.DOTALL)[:8]:
        title = re.search(r"<h2[^>]*>.*?<a[^>]*>(.*?)</a>", block, flags=re.DOTALL)
        snippet = re.search(r"<p[^>]*>(.*?)</p>", block, flags=re.DOTALL)
        if title:
            results.append({
                "title": strip_tags(title.group(1))[:150],
                "snippet": strip_tags(snippet.group(1))[:300] if snippet else "",
            })
    return results


SYSTEM = (
    "You are an academic search strategist. The question below uses phrasing that likely comes "
    "from a research survey's method taxonomy - the answering papers may never use these exact "
    "words. You are given WEB SEARCH RESULTS for the question. Use them as evidence to infer: "
    "which specific research field/subfield does this phrasing come from, and what would papers "
    "in that field call this? Return JSON only: "
    '{"field": "...", "evidence": "one sentence citing the web results", '
    '"arxiv_queries": ["2-3 short keyword queries in that field\'s own terminology"]}. '
    "Never invent facts."
)


def main() -> int:
    rows = {str(r["qid"]): r for r in load_rows(SOLUTION.parent / "repos" / "SPAR" / "benchmark" / "AutoScholarQuery_test.jsonl", offset=0, limit=50)}
    client = DeepSeekClient(api_key=load_config().get("DEEPSEEK_API_KEY"))
    hits = 0
    for qid, truth in QUESTIONS.items():
        question = str(rows[qid]["question"])
        # 整句搜索会命中中文词典垃圾（CAN 总线/studies 释义）；改为
        # 剥疑问壳 + 核心短语加引号精确匹配。
        from src.spar_baseline.search_tree import _sanitize_query
        cleaned = _sanitize_query(question)
        words = cleaned.split()
        phrase = " ".join(words[:4])
        web = bing_search(f'"{phrase}"')
        print(f"\n===== {qid}\n题目: {question}\n(正确答案领域: {truth})")
        print(f"Bing 返回 {len(web)} 条:")
        for item in web[:4]:
            print(f"  - {item['title'][:80]}")
            if item["snippet"]:
                print(f"    {item['snippet'][:110]}")
        if not web:
            print("  (Bing 无结果，跳过 LLM 判定)")
            continue
        user = json.dumps({"question": question, "web_results": web[:8]}, ensure_ascii=False)
        try:
            payload = client.complete_json(SYSTEM, user, max_tokens=500)
        except Exception as exc:
            print(f"  DeepSeek 失败: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        print(f"LLM 判定: field={payload.get('field')}")
        print(f"  证据: {str(payload.get('evidence'))[:140]}")
        print(f"  建议查询: {payload.get('arxiv_queries')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
