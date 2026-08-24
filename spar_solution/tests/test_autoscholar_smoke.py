import json
import tempfile
import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.autoscholar_smoke import run_autoscholar_smoke
from spar_solution.src.spar_baseline.eval_cli import build_parser
from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.providers.base import ProviderError, ProviderResult


def _doc(arxiv_id):
    doc = _paper("arxiv", "abstract")
    doc["paper_id"] = f"arxiv:{arxiv_id}"
    doc["identifiers"] = {key: None for key in doc["identifiers"]}
    doc["identifiers"]["arxiv_id"] = arxiv_id
    return doc


class FakeProvider:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.fail:
            raise ProviderError("arxiv", "network", "offline", retryable=True)
        return ProviderResult(
            "arxiv",
            "search",
            [_doc("2009.02040")],
            total=1,
            provenance={"query_expression": "(all:hybrid OR all:architectures)"},
        )


class AutoScholarSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "input.jsonl"
        rows = [
            {"qid": "q1", "question": "Papers about hybrid architectures?", "answer_arxiv_id": ["arXiv:2009.02040v2"], "source_meta": {"published_time": "20231024"}},
            {"qid": "q2", "question": "Papers about anomaly detection?", "answer_arxiv_id": ["9999.00001"]},
        ]
        self.dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_success_writes_auditable_artifacts_without_mock_doi(self):
        provider = FakeProvider()
        sleeps = []
        payload = run_autoscholar_smoke(
            self.dataset,
            self.root / "out",
            limit=2,
            provider=provider,
            sleep_seconds=0.25,
            sleep_fn=sleeps.append,
        )
        self.assertEqual(payload["summary"]["api_calls"], 2)
        self.assertEqual(payload["summary"]["evaluated_queries"], 2)
        self.assertEqual(sleeps, [0.25])
        self.assertEqual(provider.calls[0][1]["cutoff_year"], 2023)
        self.assertEqual(payload["results"][0]["metrics_at_10"]["tp"], 1)
        self.assertEqual(payload["results"][0]["gold_arxiv_ids"], ["2009.02040"])
        self.assertIn("query_expression", payload["summary"]["queries"][0])
        self.assertEqual(payload["manifest"]["prior_baseline"]["status"], "invalid")
        for name in ("summary.json", "results.json", "errors.json", "run_manifest.json"):
            self.assertTrue((self.root / "out" / name).is_file())
        serialized = (self.root / "out" / "results.json").read_text(encoding="utf-8")
        self.assertNotIn("10.1234/mock.paper", serialized)

    def test_provider_failure_is_error_and_not_a_false_negative(self):
        payload = run_autoscholar_smoke(
            self.dataset,
            self.root / "failed",
            limit=1,
            provider=FakeProvider(fail=True),
            sleep_seconds=0,
        )
        self.assertEqual(payload["summary"]["evaluated_queries"], 0)
        self.assertEqual(payload["summary"]["failed_queries"], 1)
        self.assertEqual(payload["summary"]["metrics_at_10"]["fn"], 0)
        self.assertEqual(payload["results"][0]["metrics_at_10"], None)
        self.assertEqual(payload["errors"][0]["code"], "network")

    def test_cli_exposes_required_arguments_and_defaults(self):
        args = build_parser().parse_args([
            "auto-scholar-smoke", "--dataset", str(self.dataset), "--output", str(self.root / "out")
        ])
        self.assertEqual((args.offset, args.limit, args.page_size, args.sleep), (0, 5, 10, 3.1))


if __name__ == "__main__":
    unittest.main()
