import json
import tempfile
import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.providers.base import ProviderError
from spar_solution.src.spar_baseline.providers.local_library import (
    FixtureLocalLibraryProvider,
    LocalLibraryProvider,
)


class LocalLibraryProviderTests(unittest.TestCase):
    def test_fixture_is_explicitly_mock(self):
        result = FixtureLocalLibraryProvider([_paper("local", "fixture")]).search("wifi")
        self.assertEqual(result.provenance["library_status"], "mock")
        self.assertEqual(len(result.records), 1)

    def test_missing_path_is_unavailable_and_not_empty_success(self):
        provider = LocalLibraryProvider()
        self.assertEqual(provider.library_status, "unavailable")
        with self.assertRaises(ProviderError) as raised:
            provider.search("wifi")
        self.assertEqual(raised.exception.code, "config_missing")
        self.assertEqual(raised.exception.details["library_status"], "unavailable")

    def test_configured_json_path_is_explicitly_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "papers.json"
            path.write_text(json.dumps({"papers": [_paper("local", "configured")]}), encoding="utf-8")
            provider = LocalLibraryProvider(path=path)
            result = provider.search("wifi")
        self.assertEqual(provider.library_status, "configured")
        self.assertEqual(result.provenance["library_status"], "configured")


if __name__ == "__main__":
    unittest.main()
