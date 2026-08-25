"""rank_fusion 的单元测试：RRF 公式、权重、跨源合并、分桶排序与容错。"""

import unittest
from copy import deepcopy

from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.rank_fusion import rrf_fuse, spar_rank


def _doc(source: str, name: str, *, abstract: str | None = None) -> dict:
    """基于共享 mock 构造 PaperDoc，并显式设置互不相同的 DOI 身份。"""

    paper = _paper(source, abstract if abstract is not None else f"Abstract for {name}.")
    paper["identifiers"]["doi"] = f"10.1234/{name}"
    paper["paper_id"] = f"doi:10.1234/{name}"
    return paper


def _set_sim(paper: dict, value) -> dict:
    paper["scores"]["relevance"] = value
    return paper


def _set_citations(paper: dict, citations: int = 0, references: int = 0) -> dict:
    paper["relations"]["citations"] = [
        {"paper_id": f"doi:10.0/c-{paper['paper_id']}-{i}"} for i in range(citations)
    ]
    paper["relations"]["references"] = [
        {"paper_id": f"doi:10.0/r-{paper['paper_id']}-{i}"} for i in range(references)
    ]
    return paper


def _set_year(paper: dict, year) -> dict:
    paper["bibliography"]["year"] = year
    return paper


def _by_paper_id(results: list, paper_id: str) -> dict:
    for doc in results:
        if doc["paper_id"] == paper_id:
            return doc
    raise AssertionError(f"paper {paper_id!r} not found in results")


# 手算基准（k=60，rank 从 0 起）：
# 1/61 = 0.016393442622950821, 1/62 = 0.016129032258064516, 1/63 = 0.015873015873015872
class RrfFuseTests(unittest.TestCase):
    def test_rrf_formula_matches_hand_computation(self):
        paper_a = _doc("arxiv", "alpha")
        paper_b = _doc("arxiv", "bravo")
        paper_c = _doc("arxiv", "charlie")
        results = rrf_fuse({"arxiv": [paper_a, paper_b, paper_c], "openalex": [paper_b]})

        self.assertEqual(
            [doc["paper_id"] for doc in results],
            ["doi:10.1234/bravo", "doi:10.1234/alpha", "doi:10.1234/charlie"],
        )
        self.assertAlmostEqual(
            _by_paper_id(results, "doi:10.1234/alpha")["scores"]["rrf"], 1 / 61, places=15
        )
        self.assertAlmostEqual(
            _by_paper_id(results, "doi:10.1234/bravo")["scores"]["rrf"],
            1 / 62 + 1 / 61,
            places=15,
        )
        self.assertAlmostEqual(
            _by_paper_id(results, "doi:10.1234/charlie")["scores"]["rrf"], 1 / 63, places=15
        )
        # 各源最好名次与命中来源记录正确。
        self.assertEqual(
            _by_paper_id(results, "doi:10.1234/bravo")["provenance"]["rrf_best_rank"],
            {"arxiv": 1, "openalex": 0},
        )
        self.assertEqual(
            _by_paper_id(results, "doi:10.1234/alpha")["provenance"]["rrf_sources"], ["arxiv"]
        )

    def test_rrf_weights_flip_order(self):
        paper_a = _doc("arxiv", "alpha")
        paper_b = _doc("arxiv", "bravo")
        paper_c = _doc("arxiv", "charlie")
        ranked_lists = {"arxiv": [paper_a, paper_b, paper_c], "openalex": [paper_b]}

        # 默认权重全 1：双路命中的 bravo 第一。
        default_order = [doc["paper_id"] for doc in rrf_fuse(ranked_lists)]
        self.assertEqual(
            default_order,
            ["doi:10.1234/bravo", "doi:10.1234/alpha", "doi:10.1234/charlie"],
        )

        # openalex 降到 0.01 后，单路榜首 alpha 反超（0.016393 > 0.016293）。
        weighted = rrf_fuse(ranked_lists, weights={"arxiv": 1.0, "openalex": 0.01})
        self.assertEqual(
            [doc["paper_id"] for doc in weighted],
            ["doi:10.1234/alpha", "doi:10.1234/bravo", "doi:10.1234/charlie"],
        )
        self.assertAlmostEqual(
            _by_paper_id(weighted, "doi:10.1234/bravo")["scores"]["rrf"],
            1 / 62 + 0.01 / 61,
            places=15,
        )

    def test_rrf_merges_same_paper_across_sources(self):
        shared_arxiv = _doc("arxiv", "shared", abstract="Short arXiv abstract.")
        shared_openalex = _doc("openalex", "shared", abstract="A much longer OpenAlex abstract.")
        only_arxiv = _doc("arxiv", "only-a")
        only_openalex = _doc("openalex", "only-o")
        results = rrf_fuse(
            {
                "arxiv": [shared_arxiv, only_arxiv],
                "openalex": [only_openalex, shared_openalex],
            }
        )

        # 同一 DOI 只保留一条记录。
        self.assertEqual(len(results), 3)
        merged = _by_paper_id(results, "doi:10.1234/shared")
        self.assertTrue(merged["provenance"]["rrf_multi_source"])
        self.assertEqual(merged["provenance"]["rrf_sources"], ["arxiv", "openalex"])
        self.assertEqual(merged["provenance"]["rrf_best_rank"], {"arxiv": 0, "openalex": 1})
        # 默认合并保留第一份本体，但 provenance 来源列表合并了双源。
        self.assertEqual(merged["bibliography"]["abstract"], "Short arXiv abstract.")
        self.assertIn("arxiv", merged["provenance"]["sources"])
        self.assertIn("openalex", merged["provenance"]["sources"])
        # 融合分 = 1/61 + 1/62。
        self.assertAlmostEqual(merged["scores"]["rrf"], 1 / 61 + 1 / 62, places=15)
        # 单源命中的论文不带 multi_source 标注。
        self.assertNotIn("rrf_multi_source", _by_paper_id(results, "doi:10.1234/only-a")["provenance"])

    def test_rrf_custom_merge_callback(self):
        shared_arxiv = _doc("arxiv", "shared", abstract="Short arXiv abstract.")
        shared_openalex = _doc(
            "openalex", "shared", abstract="A much longer OpenAlex abstract with details."
        )
        calls: list[str] = []

        def keep_richer(base: dict, incoming: dict) -> dict:
            calls.append("called")
            if len(incoming["bibliography"]["abstract"]) > len(base["bibliography"]["abstract"]):
                return incoming
            return base

        results = rrf_fuse(
            {"arxiv": [shared_arxiv], "openalex": [shared_openalex]}, merge=keep_richer
        )
        self.assertEqual(calls, ["called"])
        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["bibliography"]["abstract"],
            "A much longer OpenAlex abstract with details.",
        )

    def test_rrf_empty_input_returns_empty(self):
        self.assertEqual(rrf_fuse({}), [])
        self.assertEqual(rrf_fuse({"arxiv": [], "openalex": []}), [])

    def test_rrf_invalid_k_and_weights_raise(self):
        papers = {"arxiv": [_doc("arxiv", "alpha")]}
        for bad_k in (0, -60, "60", None):
            with self.assertRaises(ValueError):
                rrf_fuse(papers, k=bad_k)
        with self.assertRaises(ValueError):
            rrf_fuse(papers, weights={"arxiv": "heavy"})

    def test_rrf_does_not_mutate_inputs(self):
        paper_a = _doc("arxiv", "alpha")
        paper_b = _doc("openalex", "bravo")
        ranked_lists = {"arxiv": [paper_a], "openalex": [paper_b]}
        snapshot = deepcopy(ranked_lists)
        rrf_fuse(ranked_lists)
        self.assertEqual(ranked_lists, snapshot)
        self.assertNotIn("rrf", paper_a["scores"])
        self.assertNotIn("rrf_sources", paper_a["provenance"])


class SparRankTests(unittest.TestCase):
    def test_spar_rank_same_bucket_orders_by_citations_then_year(self):
        # 0.22/0.24/0.21 都落在 floor(x/0.05)=4 号桶。
        low_sim_high_cite = _set_citations(_set_sim(_doc("arxiv", "low"), 0.24), citations=10)
        high_sim_low_cite = _set_year(
            _set_citations(_set_sim(_doc("arxiv", "high"), 0.22), citations=3), 2018
        )
        same_cite_newer = _set_year(
            _set_citations(_set_sim(_doc("arxiv", "newer"), 0.21), citations=3), 2024
        )
        same_cite_older = _set_year(
            _set_citations(_set_sim(_doc("arxiv", "older"), 0.22), citations=3), 2019
        )
        results = spar_rank(
            [high_sim_low_cite, low_sim_high_cite, same_cite_older, same_cite_newer]
        )
        self.assertEqual(
            [doc["paper_id"] for doc in results],
            [
                "doi:10.1234/low",  # 同桶先比引用数：10 > 3
                "doi:10.1234/newer",  # 引用数相同再比年份：2024 > 2019
                "doi:10.1234/older",
                "doi:10.1234/high",  # 引用数最少垫底，尽管 sim 最高
            ],
        )

    def test_spar_rank_cross_bucket_beats_citations(self):
        # 0.26 -> 桶 5，0.24 -> 桶 4：桶号优先，引用数无法跨桶翻盘。
        higher_bucket = _set_sim(_doc("arxiv", "bucket5"), 0.26)
        lower_bucket = _set_citations(_set_sim(_doc("arxiv", "bucket4"), 0.24), citations=100)
        results = spar_rank([lower_bucket, higher_bucket])
        self.assertEqual(
            [doc["paper_id"] for doc in results], ["doi:10.1234/bucket5", "doi:10.1234/bucket4"]
        )

    def test_spar_rank_treats_none_sim_as_zero(self):
        none_low_cite = _set_citations(_doc("arxiv", "none-low"), citations=2)
        none_high_cite = _set_citations(_doc("arxiv", "none-high"), citations=9)
        scored = _set_sim(_doc("arxiv", "scored"), 0.1)  # 桶 2
        results = spar_rank([none_low_cite, scored, none_high_cite])
        self.assertEqual(
            [doc["paper_id"] for doc in results],
            ["doi:10.1234/scored", "doi:10.1234/none-high", "doi:10.1234/none-low"],
        )

    def test_spar_rank_paper_id_tiebreak_deterministic_and_pure(self):
        zeta = _set_citations(_set_sim(_doc("arxiv", "zeta"), 0.12), citations=2, references=2)
        alpha = _set_citations(_set_sim(_doc("arxiv", "alpha"), 0.13), citations=2, references=2)
        papers = [zeta, alpha]
        snapshot = deepcopy(papers)

        first = spar_rank(papers)
        second = spar_rank(papers)
        self.assertEqual(
            [doc["paper_id"] for doc in first], ["doi:10.1234/alpha", "doi:10.1234/zeta"]
        )
        self.assertEqual([doc["paper_id"] for doc in first], [doc["paper_id"] for doc in second])
        # 输入列表与文档本身都未被修改。
        self.assertEqual(papers, snapshot)
        self.assertIs(papers[0], zeta)

    def test_spar_rank_tolerates_missing_fields(self):
        minimal = [
            {"paper_id": "doi:10.1234/bare-b", "scores": {"relevance": 0.3}, "bibliography": {}},
            {"paper_id": "doi:10.1234/bare-a", "scores": {}, "bibliography": {}},
        ]
        results = spar_rank(minimal)
        # relevance 缺失按 0 处理：bare-b 落在正桶，排在无分（桶 0）的 bare-a 之前。
        self.assertEqual(
            [doc["paper_id"] for doc in results], ["doi:10.1234/bare-b", "doi:10.1234/bare-a"]
        )

    def test_spar_rank_invalid_bucket_raises(self):
        with self.assertRaises(ValueError):
            spar_rank([_doc("arxiv", "alpha")], bucket=0)
        with self.assertRaises(ValueError):
            spar_rank([_doc("arxiv", "alpha")], bucket=-0.05)


if __name__ == "__main__":
    unittest.main()


class IdentifierUnionMergeTests(unittest.TestCase):
    """跨源合并必须保留两侧标识字段（arxiv_id + openalex_id 并存）。"""

    def test_default_merge_unions_identifiers(self):
        arxiv_copy = _paper("arxiv", "WiFi heart rate abstract text here")
        arxiv_copy["paper_id"] = "arxiv:2301.12345"
        arxiv_copy["identifiers"] = dict.fromkeys(arxiv_copy["identifiers"], None)
        arxiv_copy["identifiers"]["arxiv_id"] = "2301.12345"
        arxiv_copy["identifiers"]["doi"] = None
        openalex_copy = _paper("openalex", "WiFi heart rate abstract text here")
        openalex_copy["paper_id"] = "doi:10.48550/arxiv.2301.12345"
        openalex_copy["identifiers"]["doi"] = "10.48550/arxiv.2301.12345"
        openalex_copy["identifiers"]["openalex_id"] = "W42"
        fused = rrf_fuse({"arxiv": [arxiv_copy], "openalex": [openalex_copy]})
        self.assertEqual(len(fused), 1)
        ids = fused[0]["identifiers"]
        self.assertEqual(ids["arxiv_id"], "2301.12345")
        self.assertEqual(ids["openalex_id"], "W42")
        self.assertEqual(ids["doi"], "10.48550/arxiv.2301.12345")
