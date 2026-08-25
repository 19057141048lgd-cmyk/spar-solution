"""哨兵 5 题验证（HANDOVER §7 唯一合法联网验证方式）。

固定 5 题覆盖四种故障形态：排序失败×2（test_4/8，盯 P0-1 末层补判）、
零召回错领域×1（test_0，防修复伤召回）、高分×1（test_3，防回退）、
部分分排序敏感×1（test_20，判分校准探针）。

用法：python spar_solution/scripts_sentinel_5.py [--run-name NAME] [--expand-mode openalex]
只跑这 5 题（上限硬编码 5），与 tree-n50-v2* 存档基线同题配对比较。
成本预算：约 15 分钟、≤10 万 token、约 100-150 次 API。
"""

import argparse
import json
import sys
import time
from pathlib import Path

SOLUTION = Path(__file__).resolve().parent
REPO = SOLUTION.parent
sys.path.insert(0, str(SOLUTION))

from src.spar_baseline.autoscholar_baseline import _norm_arxiv, load_rows
from src.spar_baseline.p2_cli import build_live_pipeline
from src.spar_baseline.p2_metrics import evaluate_p2_run
from src.spar_baseline.pasa_metrics import evaluate_pasa_style
from src.spar_baseline.search_tree import SearchTreeRunner

DATASET = REPO / "repos" / "SPAR" / "benchmark" / "AutoScholarQuery_test.jsonl"
BASELINE_DIRS = [SOLUTION / "artifacts" / "autoscholar" / "tree-n50-v2", SOLUTION / "artifacts" / "autoscholar" / "tree-n50-v2-resume"]
MAX_QUERIES = 5  # 铁律：单轮永远 ≤5 题

SENTINEL_QIDS = (
    "AutoScholarQuery_test_4",   # 排序失败：crawl=1.0 → r20=0（P0-1 守卫）
    "AutoScholarQuery_test_8",   # 排序失败第二例：crawl=1.0 → r20=0
    "AutoScholarQuery_test_0",   # 零召回+错领域：防修复伤召回
    "AutoScholarQuery_test_3",   # 最高分段 r20=0.6：防高分回退
    "AutoScholarQuery_test_20",  # 部分分：crawl=0.4/r20=0.2，排序校准探针
)


def baseline_papers(qid: str):
    for base in BASELINE_DIRS:
        path = base / qid / "papers.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))["papers"]
    return []


def score(papers, row):
    gold_ids = sorted({_norm_arxiv(g) for g in row.get("answer_arxiv_id") or [] if _norm_arxiv(g)})
    return {
        "pasa": evaluate_pasa_style(papers, row.get("answer") or []),
        "f1_10": {k: v for k, v in evaluate_p2_run({"papers": papers}, gold_ids=gold_ids, cutoffs=(10,))["by_cutoff"]["10"].items() if k in ("tp", "fp", "fn", "f1", "recall")},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expand-mode", default="openalex", choices=("openalex", "hybrid", "fulltext"))
    parser.add_argument("--sleep", type=float, default=3.0)
    args = parser.parse_args()
    assert len(SENTINEL_QIDS) <= MAX_QUERIES

    rows = {str(r["qid"]): r for r in load_rows(DATASET, offset=0, limit=50)}
    pipeline = build_live_pipeline(citation_enabled=True, page_size=10)
    pipeline.providers.pop("bohrium", None)
    runner = SearchTreeRunner(pipeline.providers, pipeline.understanding_layer, page_size=10, expand_mode=args.expand_mode)

    out_root = SOLUTION / "artifacts" / "autoscholar" / f"sentinel-{args.run_name}"
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    per_query, totals = [], {"api": 0, "llm": 0, "tokens": 0, "wall_ms": 0.0}
    for index, qid in enumerate(SENTINEL_QIDS):
        row = rows[qid]
        result = runner.run(str(row["question"]))
        qdir = out_root / qid
        qdir.mkdir(exist_ok=True)
        for name, payload in (("papers", {"papers": result["papers"]}), ("nodes", result["nodes"]), ("edges", result["edges"]), ("stats", result["stats"]), ("errors", result["errors"])):
            (qdir / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        new = score(result["papers"], row)
        base_papers = baseline_papers(qid)
        old = score(base_papers, row) if base_papers else None
        stats = result["stats"]
        totals["api"] += int(stats.get("provider_calls") or 0)
        totals["llm"] += int(stats.get("llm_calls") or 0)
        totals["tokens"] += int(stats.get("llm_total_tokens") or 0)
        totals["wall_ms"] += float(stats.get("wall_ms") or 0)
        delta = None if old is None else round(new["pasa"]["recall_20_recall"] - old["pasa"]["recall_20_recall"], 3)
        per_query.append({"qid": qid, "new": new, "old": old, "delta_r20": delta, "stats": stats, "stop_reason": result.get("stop_reason")})
        base_r20 = f"{old['pasa']['recall_20_recall']:.2f}" if old else "-"
        print(f"[{index+1}/5] {qid}: r20 {new['pasa']['recall_20_recall']:.2f} (基线 {base_r20}) crawl {new['pasa']['crawler_recall']:.2f} | api={stats.get('provider_calls')} llm={stats.get('llm_calls')} tokens={stats.get('llm_total_tokens')}", flush=True)
        if index + 1 < len(SENTINEL_QIDS):
            time.sleep(max(0.0, args.sleep))

    def macro(items, path):
        values = [item[path] for item in items if item is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    summary = {
        "schema_version": "sentinel.v1",
        "run_name": args.run_name,
        "expand_mode": args.expand_mode,
        "qids": list(SENTINEL_QIDS),
        "new_macro": {
            "recall_20": macro([q["new"]["pasa"] for q in per_query], "recall_20_recall"),
            "recall_50": macro([q["new"]["pasa"] for q in per_query], "recall_50_recall"),
            "crawler_recall": macro([q["new"]["pasa"] for q in per_query], "crawler_recall"),
            "selected_precision": macro([q["new"]["pasa"] for q in per_query], "selected_precision"),
        },
        "old_macro": {
            "recall_20": macro([q["old"]["pasa"] for q in per_query if q["old"]], "recall_20_recall"),
            "recall_50": macro([q["old"]["pasa"] for q in per_query if q["old"]], "recall_50_recall"),
            "crawler_recall": macro([q["old"]["pasa"] for q in per_query if q["old"]], "crawler_recall"),
        },
        "win_loss": [{"qid": q["qid"], "delta_r20": q["delta_r20"]} for q in per_query],
        "totals": {**totals, "wall_min": round(totals["wall_ms"] / 60000, 1), "real_min": round((time.perf_counter() - started) / 60, 1)},
        "rows": per_query,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("new_macro", "old_macro", "win_loss", "totals")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
