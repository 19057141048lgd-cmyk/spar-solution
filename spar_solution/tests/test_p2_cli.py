import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from spar_solution.src.spar_baseline.p2_cli import build_live_pipeline, main
from spar_solution.src.spar_baseline.p2_pipeline import run_p2_fixture


class P2CliTests(unittest.TestCase):
    def test_live_pipeline_without_llm_key_uses_rules(self):
        pipeline = build_live_pipeline({})
        self.assertEqual(set(pipeline.providers), {"arxiv", "openalex"})
        self.assertIsNone(pipeline.understanding_layer)

    def test_live_pipeline_enables_configured_fact_and_reasoning_providers(self):
        pipeline = build_live_pipeline({"BOHR_ACCESS_KEY": "test", "DEEPSEEK_API_KEY": "test"})
        self.assertEqual(set(pipeline.providers), {"arxiv", "openalex", "bohrium"})
        self.assertIsNotNone(pipeline.understanding_layer)

    def test_finalize_rebuilds_an_invalid_existing_final_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            run_p2_fixture(output_dir=directory)
            final_path = Path(directory) / "final_selection.json"
            final_path.write_text('{"schema_version":"old"}', encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["finalize", "--input", directory]), 0)
            self.assertEqual(json.loads(final_path.read_text(encoding="utf-8"))["schema_version"], "spar.final.v2")


if __name__ == "__main__":
    unittest.main()
