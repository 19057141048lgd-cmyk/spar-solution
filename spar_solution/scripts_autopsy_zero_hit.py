"""零命中题验尸：Gold 是否藏在我们候选池论文的一跳 references 里。

对每道零命中题：
1. 把 Gold arXiv ID 解析成 OpenAlex W-id（/works/doi:10.48550/arxiv.<id>）；
2. 取候选池相关性 top-8 且有 openalex_id 的论文，拉它们的 referenced_works；
3. 判定：Gold 是否在池内论文的一跳引用内（reachable）、池子是否同领域
   （用 top-1 标题人工 sanity）、Gold 是否有 OpenAlex 记录。
"""

import json
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "repos" / "SPAR" / "benchmark" / "AutoScholarQuery_test.jsonl"
ART = ROOT / "spar_solution" / "artifacts" / "autoscholar" / "tree-n50"
ZERO_HIT = [0, 9, 16, 24, 37]
HEADERS = {"User-Agent": "spar-autopsy/1.0", "Accept": "application/json"}


def get(url: str):
    request = Request(url, headers=HEADERS)
    with urlopen(request, timeout=25) as response:
        return json.loads(response.read().decode("utf-8"))


def gold_openalex_id(arxiv_id: str):
    doi = f"10.48550/arxiv.{arxiv_id}"
    url = "https://api.openalex.org/works/https://doi.org/" + quote(doi, safe="") + "?select=id"
    try:
        payload = get(url)
        return (payload.get("id") or "").rsplit("/", 1)[-1]
    except Exception as exc:
        return f"ERR:{type(exc).__name__}"


def referenced(work_id: str):
    try:
        return get(f"https://api.openalex.org/works/{work_id}?select=id,referenced_works").get("referenced_works") or []
    except Exception:
        return []


def main():
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index in ZERO_HIT:
        row = rows[index]
        qid = row["qid"]
        question = row["question"][:70]
        papers = json.loads((ART / qid / "papers.json").read_text(encoding="utf-8"))["papers"]
        papers.sort(key=lambda p: p.get("scores", {}).get("relevance") or 0.0, reverse=True)
        print(f"== [{qid}] {question}")
        top1 = (papers[0]["bibliography"].get("title") or "")[:60] if papers else "-"
        print(f"   池子 top1: {top1} (rel={papers[0]['scores'].get('relevance') if papers else '-'})")
        gold_ids = row["answer_arxiv_id"]
        gold_w = {}
        for g in gold_ids:
            wid = gold_openalex_id(g)
            gold_w[g] = wid
            time.sleep(0.3)
        print(f"   Gold 解析: { {g: (w[:12] if not w.startswith('ERR') else w) for g, w in gold_w.items()} }")
        valid_w = {w for w in gold_w.values() if w.startswith("W")}
        pool_refs = set()
        seeds = []
        for p in papers[:8]:
            wid = (p.get("identifiers") or {}).get("openalex_id")
            if not wid:
                continue
            seeds.append(wid)
            pool_refs.update(r.rsplit("/", 1)[-1] for r in referenced(wid))
            time.sleep(0.3)
        hit = valid_w & pool_refs
        print(f"   一跳可达: {len(hit)}/{len(valid_w)} Gold 在池内 {len(seeds)} 篇论文的 references 里 {sorted(hit)[:3]}")
        print()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
