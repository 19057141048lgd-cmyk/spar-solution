import unittest

from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.p2_evidence import ConstraintGate, EvidenceLoader


class P2EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "hard_constraints": [
                {"name": "year", "value": "2020-2025"},
                {"name": "evidence", "value": "abstract"},
            ]
        }

    def test_constraint_gate_has_pass_fail_unknown(self):
        paper = _paper("mock", "WiFi CSI heart rate monitoring method", full_text_status="abstract")
        paper["bibliography"]["year"] = 2024
        verdict = ConstraintGate().evaluate(self.plan, paper)
        self.assertEqual(verdict.state, "pass")

        paper["bibliography"]["year"] = 2018
        self.assertEqual(ConstraintGate().evaluate(self.plan, paper).state, "fail")

        paper["bibliography"]["year"] = None
        self.assertEqual(ConstraintGate().evaluate(self.plan, paper).state, "unknown")

    def test_fulltext_gate_does_not_promote_abstract(self):
        paper = _paper("mock", "WiFi abstract")
        plan = {"hard_constraints": [{"name": "full_text", "value": "fulltext"}]}
        verdict = ConstraintGate().evaluate(plan, paper)
        self.assertEqual(verdict.state, "fail")
        self.assertIn("evidence_insufficient", verdict.reason_codes)

    def test_loader_marks_metadata_unavailable_and_abstract_traceable(self):
        metadata = _paper("mock", "", full_text_status="metadata")
        metadata["bibliography"]["abstract"] = None
        items = EvidenceLoader().load(metadata, "abstract")
        self.assertEqual(items[0].evidence_status, "unavailable")
        self.assertEqual(items[0].unavailable_reason, "insufficient_evidence:metadata<abstract")

        abstract = _paper("mock", "A short abstract")
        items = EvidenceLoader().load(abstract, "abstract")
        self.assertEqual(items[0].evidence_status, "abstract")
        self.assertTrue(items[0].evidence_ref)
        self.assertNotIn("A short abstract", items[0].to_dict().__repr__())

    def test_loader_uses_chunk_refs_without_copying_body(self):
        paper = _paper("mock", "abstract", full_text_status="fulltext")
        paper["content"]["chunks"] = [{"chunk_id": "c1", "content_ref": "artifact://c1", "offset": 0, "section": "Methods", "page": 2}]
        items = EvidenceLoader().load(paper, "fulltext")
        self.assertEqual(items[0].evidence_ref, "artifact://c1")
        self.assertEqual(items[0].section, "Methods")
        self.assertEqual(items[0].page, 2)


if __name__ == "__main__":
    unittest.main()
