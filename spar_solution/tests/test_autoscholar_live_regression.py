"""可选的真实 AutoScholar 回归测试（联网 + 需要 DEEPSEEK_API_KEY）。

默认跳过；显式设置环境变量后运行：

    SPAR_LIVE_REGRESSION=1 python -m unittest spar_solution.tests.test_autoscholar_live_regression

回归目标：arXiv-DOI ↔ arXiv ID 身份互认（Bug 1）和 DeepSeek 判断部分接受
（Bug 2）修复后，完整 P2 管线在真实 AutoScholarQuery 题目上必须命中 Gold
（修复前 Q4 实测 tp=0，尽管 4/5 Gold 已被检回）。
"""

import json
import os
import unittest
from pathlib import Path


DATASET = Path(__file__).resolve().parents[2].parent / "repos" / "SPAR" / "benchmark" / "AutoScholarQuery_test.jsonl"
REQUIRED_ENV = "SPAR_LIVE_REGRESSION"
LIVE_QUERIES = (3,)  # Q4: foundation models NLP, gold 5 篇（BERT/GPT-3/T5/PaLM/LLaMA）


@unittest.skipUnless(
    os.environ.get(REQUIRED_ENV) == "1" and DATASET.is_file(),
    f"set {REQUIRED_ENV}=1 to run the live AutoScholar regression",
)
class AutoScholarLiveRegressionTests(unittest.TestCase):
    def test_pipeline_hits_gold_on_foundation_models_question(self):
        from spar_solution.src.spar_baseline.p2_cli import build_live_pipeline
        from spar_solution.src.spar_baseline.p2_metrics import evaluate_p2_run

        rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
        row = rows[LIVE_QUERIES[0]]
        run_dir = Path("spar_solution/artifacts/p2/live-regression-q4")
        pipeline = build_live_pipeline(citation_enabled=True)
        pipeline.run(row["question"], output_dir=run_dir)
        papers = json.loads((run_dir / "papers.json").read_text(encoding="utf-8"))["papers"]
        result = evaluate_p2_run({"papers": papers}, gold_ids=row["answer_arxiv_id"], cutoffs=(10,))
        metrics = result["by_cutoff"]["10"]
        # 修复前 tp=0；修复后 OpenAlex 检回的 arXiv-DOI 论文必须能计 TP。
        self.assertGreaterEqual(
            metrics["tp"], 2,
            f"expected >=2 gold hits in top-10, got tp={metrics['tp']} (recall={metrics['recall']})",
        )


if __name__ == "__main__":
    unittest.main()
