"""fulltext_flow 的离线测试：HTML 缓存命中与点名标题模糊匹配。"""

import tempfile
import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.fulltext_flow import fetch_arxiv_html_markdown
from spar_solution.src.spar_baseline.search_tree import _title_matches


class HtmlCacheTests(unittest.TestCase):
    def test_cache_hit_skips_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            cached = cache / "html" / "2401.12345.txt"
            cached.parent.mkdir(parents=True)
            cached.write_text("cached body with References marker", encoding="utf-8")
            # 命中缓存时不得发起任何网络请求（无 opener 注入也安全）。
            self.assertEqual(fetch_arxiv_html_markdown("2401.12345", cache_dir=cache), "cached body with References marker")

    def test_invalid_id_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(fetch_arxiv_html_markdown("not-an-id", cache_dir=Path(tmp)))


class TitleMatchTests(unittest.TestCase):
    def test_exact_normalized_match(self):
        self.assertTrue(_title_matches("WiFi CSI Heart-Rate Monitoring!", "wifi csi heart rate monitoring"))

    def test_subtitle_superset_matches(self):
        # LLM 清洗出的标题常比真标题多出副标题/说明词。
        self.assertTrue(_title_matches("Causal Bandits: Learning Good Interventions", "Learning Good Interventions via Causal Inference"))

    def test_below_overlap_threshold_rejects(self):
        self.assertFalse(_title_matches("Graph Neural Networks for Molecular Property Prediction", "Transformer Models for Machine Translation"))

    def test_empty_inputs_reject(self):
        self.assertFalse(_title_matches("", "some title"))
        self.assertFalse(_title_matches("some title", ""))


if __name__ == "__main__":
    unittest.main()
