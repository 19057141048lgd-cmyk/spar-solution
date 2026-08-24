"""全文第 1 层（本地 PDF 抽取）的离线测试：用 fitz 现造一个文本型 PDF。"""

import tempfile
import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.fulltext import (
    apply_fulltext_nudge,
    augment_fulltext,
    augment_topk,
    download_pdf,
    extract_fulltext_chunks,
    fulltext_coverage,
    fulltext_text,
    query_terms,
)
from spar_solution.src.spar_baseline.mock_pipeline import _paper


def _make_text_pdf(path: Path, text: str) -> None:
    import fitz

    document = fitz.open()
    page = document.new_page()
    for index, line in enumerate(text.split("\n")):
        page.insert_text((72, 72 + 14 * index), line)
    document.save(str(path))
    document.close()


def _fake_opener(data: bytes):
    return lambda url, timeout: data


class FulltextLayerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        body = ("WiFi CSI heart rate monitoring via deep learning. " * 6
                + "\nWe evaluate on a public dataset with 40 participants. " * 4
                + "\nExperiments show competitive error rates for respiration and heart rate estimation. " * 4)
        self.pdf = self.root / "fixture.pdf"
        _make_text_pdf(self.pdf, body)

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_chunks_from_text_pdf(self):
        chunks = extract_fulltext_chunks(self.pdf)
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 0)
        self.assertLessEqual(len(chunks), 8)
        self.assertIn("offset", chunks[0])
        text = fulltext_text(self.pdf)
        self.assertGreater(len(text), 200)
        self.assertIn("heart rate", text.casefold())

    def test_download_rejects_non_pdf_and_bad_scheme(self):
        self.assertIsNone(download_pdf("file:///etc/passwd", self.root / "d1"))
        self.assertIsNone(download_pdf("", self.root / "d2"))
        self.assertIsNone(download_pdf("https://example.invalid/x.pdf", self.root / "d3", opener=_fake_opener(b"<html>not a pdf</html>")))
        path = download_pdf("https://example.invalid/x.pdf", self.root / "d4", opener=_fake_opener(b"%PDF-1.4 fake"))
        self.assertIsNotNone(path)
        self.assertTrue(path.read_bytes().startswith(b"%PDF"))

    def test_download_size_cap(self):
        def opener(url, timeout):
            return b"%PDF" + b"0" * 100

        self.assertIsNone(download_pdf("https://example.invalid/big.pdf", self.root / "d5", max_bytes=10, opener=opener))

    def test_coverage_and_terms(self):
        terms = query_terms("papers about WiFi heart rate monitoring using deep learning")
        self.assertIn("wifi", terms)
        self.assertNotIn("about", terms)
        coverage = fulltext_coverage(fulltext_text(self.pdf), terms)
        self.assertIsNotNone(coverage)
        self.assertGreaterEqual(coverage, 0.8)
        self.assertIsNone(fulltext_coverage("", terms))
        self.assertIsNone(fulltext_coverage("text", []))

    def test_nudge_math(self):
        paper = _paper("arxiv", "abstract")
        paper["scores"]["relevance"] = 0.7
        apply_fulltext_nudge(paper, 0.9)
        self.assertAlmostEqual(paper["scores"]["relevance"], 0.75)
        apply_fulltext_nudge(paper, 0.1)
        self.assertAlmostEqual(paper["scores"]["relevance"], 0.70)
        mid = _paper("arxiv", "abstract")
        mid["scores"]["relevance"] = 0.7
        apply_fulltext_nudge(mid, 0.5)
        self.assertAlmostEqual(mid["scores"]["relevance"], 0.7)
        capped = _paper("arxiv", "abstract")
        capped["scores"]["relevance"] = 0.98
        apply_fulltext_nudge(capped, 0.95)
        self.assertLessEqual(capped["scores"]["relevance"], 1.0)

    def test_augment_fulltext_end_to_end_with_injected_opener(self):
        paper = _paper("arxiv", "WiFi CSI heart rate monitoring abstract.")
        paper["paper_id"] = "fixture:fulltext"
        paper["identifiers"]["doi"] = "10.1234/fulltext.test"
        paper["access"]["pdf_url"] = "https://example.invalid/fixture.pdf"
        result = augment_fulltext(paper, self.root, terms=["wifi", "heart", "rate"], opener=_fake_opener(self.pdf.read_bytes()))
        self.assertIsNotNone(result)
        doc = result["paper"]
        self.assertEqual(doc["access"]["full_text_status"], "fulltext")
        self.assertEqual(doc["status"]["evidence_status"], "fulltext")
        self.assertTrue(doc["content"]["chunks"])
        self.assertTrue(all(chunk.get("content_ref") for chunk in doc["content"]["chunks"]))
        self.assertIsNotNone(result["coverage"])
        # 下载失败（返回 HTML）时保持原状。
        self.assertIsNone(augment_fulltext(paper, self.root, opener=_fake_opener(b"<html/>")))

    def test_augment_topk_only_touches_head(self):
        papers = []
        for index in range(4):
            paper = _paper("arxiv", f"WiFi study {index}")
            paper["paper_id"] = f"fixture:ft:{index}"
            paper["identifiers"]["doi"] = f"10.1234/ft.{index}"
            paper["access"]["pdf_url"] = "https://example.invalid/fixture.pdf"
            papers.append(paper)
        papers[3]["access"]["pdf_url"] = None  # 无链接
        out, stats = augment_topk(papers, self.root, ["wifi"], top_k=2, opener=_fake_opener(self.pdf.read_bytes()))
        self.assertEqual(stats["attempted"], 2)  # 只尝试前 2 篇（第 3 篇有链接但在 top_k 外）
        self.assertEqual(stats["succeeded"], 2)
        self.assertEqual(out[3]["access"]["full_text_status"], "abstract")


if __name__ == "__main__":
    unittest.main()
