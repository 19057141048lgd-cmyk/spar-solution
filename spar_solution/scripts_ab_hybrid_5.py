"""hybrid 模式 5 题 A/B：与 v2 基线（openalex）同题对比。

用法：python spar_solution/scripts_ab_hybrid_5.py
固定用 seed 42 抽的 5 题（test_40/7/1/47/17），和 scripts_test_flow_5 同款，
三方可比：v2 基线（搜索树+openalex） vs 新流程独立版 vs 搜索树+hybrid。
"""

import json
import random
import sys
from pathlib import Path

SOLUTION = Path(__file__).resolve().parent
REPO = SOLUTION.parent
sys.path.insert(0, str(SOLUTION))

from src.spar_baseline.autoscholar_baseline import _norm_arxiv
from src.spar_baseline.config import load_config
from src.spar_baseline.p2_cli import build_live_pipeline
from src.spar_baseline.p2_metrics import evaluate_p2_run
from src.spar_baseline.pasa_metrics import evaluate_pasa_style
from src.spar_baseline.search_tree import SearchTreeRunner

DATASET = REPO / "repos" / "SPAR" / "benchmark" / "AutoScholarQuery_test.jsonl"
BASELINE_DIRS = [SOLUTION / "artifacts" / "autoscholar" / "tree-n50-v2", SOLUTION / "artifacts" / "autoscholar" / "tree-n50-v2-resume"]


def baseline_papers(index: int, qid: str):
    for base in BASELINE_DIRS:
        path = base / qid / "papers.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))["papers"]
    return []


def main() -> int:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()][:50]
    sample = random.Random(42).sample(rows, 5)
    pipeline = build_live_pipeline(citation_enabled=True, page_size=10)
    pipeline.providers.pop("bohrium", None)
    runner = SearchTreeRunner(pipeline.providers, pipeline.understanding_layer, page_size=10, expand_mode="hybrid")

    out_dir = SOLUTION / "artifacts" / "autoscholar" / "hybrid-5"
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = {"tp": 0, "gold": 0, "base_tp": 0, "hybrid_crawl": 0.0, "base_crawl": 0.0, "edges_ft": 0}
    for row in sample:
        qid = row["qid"]
        gold = sorted({_norm_arxiv(g) for g in row["answer_arxiv_id"] if _norm_arxiv(g)})
        result = runner.run(row["question"])
        papers = result["papers"]
        (out_dir / qid).mkdir(exist_ok=True)
        (out_dir / qid / "papers.json").write_text(json.dumps({"papers": papers}, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / qid / "stats.json").write_text(json.dumps(result["stats"], ensure_ascii=False, indent=2), encoding="utf-8")
        m = evaluate_p2_run({"papers": papers}, gold_ids=gold, cutoffs=(10,))["by_cutoff"]["10"]
        pasa = evaluate_pasa_style(papers, row.get("answer") or [])
        base = baseline_papers(0, qid)
        base_m = evaluate_p2_run({"papers": base}, gold_ids=gold, cutoffs=(10,))["by_cutoff"]["10"] if base else {"tp": 0}
        base_pasa = evaluate_pasa_style(base, row.get("answer") or []) if base else {"crawler_recall": 0.0, "recall_20_recall": 0.0}
        ft_edges = sum(1 for e in result.get("edges", []) if e.get("source") == "fulltext")
        print(f"[{qid}] {row['question'][:52]}")
        print(f"  hybrid: tp@10={m['tp']}/{len(gold)} recall20={pasa['recall_20_recall']:.2f} crawl={pasa['crawler_recall']:.2f} 候选={len(papers)} 正文边={ft_edges}")
        print(f"  基线:   tp@10={base_m['tp']}/{len(gold)} recall20={base_pasa['recall_20_recall']:.2f} crawl={base_pasa['crawler_recall']:.2f}")
        totals["tp"] += m["tp"]; totals["gold"] += len(gold); totals["base_tp"] += base_m["tp"]
        totals["hybrid_crawl"] += pasa["crawler_recall"]; totals["base_crawl"] += base_pasa["crawler_recall"]; totals["edges_ft"] += ft_edges
    n = len(sample)
    print()
    print(f"=== 5 题汇总：hybrid tp@10={totals['tp']}/{totals['gold']} vs 基线 {totals['base_tp']}/{totals['gold']} ===")
    print(f"总检回: hybrid {totals['hybrid_crawl']/n:.2f} vs 基线 {totals['base_crawl']/n:.2f} | 正文引用边 {totals['edges_ft']} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
