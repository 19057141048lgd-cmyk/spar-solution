"""PaSa 官方口径评测的单元测试。"""

import unittest

from spar_solution.src.spar_baseline.pasa_metrics import aggregate_pasa_style, evaluate_pasa_style, keep_letters


def _paper(title, relevance):
    return {"bibliography": {"title": title}, "scores": {"relevance": relevance}}


class PasaMetricsTests(unittest.TestCase):
    def test_keep_letters_matches_pasa(self):
        self.assertEqual(keep_letters("A Primer in BERTology: What We Know!"), "aprimerinbertologywhatweknow")
        self.assertEqual(keep_letters("  "), "")
        self.assertEqual(keep_letters(None), "")

    def test_single_query_metrics(self):
        papers = [
            _paper("Deep Contextual Bandits!", 0.9),
            _paper("Contextual Bandits, Revisited.", 0.7),
            _paper("A Survey of Everything", 0.4),
            _paper("Deep Contextual Bandits!", 0.8),  # 重复标题只计一次
        ]
        gold = ["deep contextual bandits", "Contextual Bandits Revisited"]
        result = evaluate_pasa_style(papers, gold)
        self.assertEqual(result["gold_count"], 2)
        self.assertEqual(result["crawled_count"], 3)
        self.assertEqual(result["selected_count"], 2)  # >0.5 的两篇
        self.assertAlmostEqual(result["crawler_recall"], 1.0)
        self.assertAlmostEqual(result["selected_precision"], 1.0)
        self.assertAlmostEqual(result["selected_recall"], 1.0)
        self.assertAlmostEqual(result["selected_f1"], 1.0)
        self.assertAlmostEqual(result["recall_20_recall"], 1.0)
        # 打分低的综述不在 selected 里。
        survey = evaluate_pasa_style([_paper("A Survey of Everything", 0.4)], ["a survey of everything"])
        self.assertEqual(survey["selected_count"], 0)
        self.assertAlmostEqual(survey["selected_recall"], 0.0)
        self.assertAlmostEqual(survey["crawler_recall"], 1.0)

    def test_recall_at_k_respects_score_order(self):
        # keep_letters 会滤掉数字，标题必须用字母区分。
        names = [f"Paper {'abcdefghijklmnopqrstuvwxyz'[i % 26]}{chr(ord('a') + i // 26)}" for i in range(25)]
        papers = [_paper(names[i], 0.9 - 0.1 * i) for i in range(25)]
        gold = names[15:25]  # Gold 都在低分段
        result = evaluate_pasa_style(papers, gold)
        self.assertEqual(result["recall_20_tp"], 5)  # 前 20 覆盖前 20 篇，含 Gold 的 15-19
        self.assertAlmostEqual(result["recall_20_recall"], 0.5)
        self.assertAlmostEqual(result["recall_50_recall"], 1.0)
        self.assertAlmostEqual(result["recall_100_recall"], 1.0)

    def test_macro_aggregation(self):
        first = evaluate_pasa_style([_paper("Alpha Paper", 0.9)], ["alpha paper"])
        second = evaluate_pasa_style([_paper("Beta Paper", 0.2)], ["beta paper"])  # 未被选中
        aggregated = aggregate_pasa_style([first, second])
        self.assertEqual(aggregated["queries"], 2)
        self.assertAlmostEqual(aggregated["crawler_recall"], 1.0)
        self.assertAlmostEqual(aggregated["selected_recall"], 0.5)
        self.assertAlmostEqual(aggregated["selected_precision"], 0.5)  # 空选择集的精确按 0 计
        self.assertAlmostEqual(aggregated["recall_20_recall"], 1.0)  # recall@K 按爬取排名，低分 Gold 仍计入

    def test_empty_gold(self):
        result = evaluate_pasa_style([_paper("Any", 0.9)], [])
        self.assertEqual(result["gold_count"], 0)
        self.assertAlmostEqual(result["selected_precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
