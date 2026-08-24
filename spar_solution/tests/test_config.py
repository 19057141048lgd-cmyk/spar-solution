import tempfile
import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.config import (
    get_provider_config,
    load_config,
    redact_config,
    redact_url,
    read_env_file,
)


class ConfigTests(unittest.TestCase):
    def test_file_is_loaded_and_environment_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env.local"
            path.write_text('BOHR_ACCESS_KEY="file-secret"\nOPENALEX_API_KEY=file-key\n', encoding="utf-8")
            config = load_config(path, {"BOHR_ACCESS_KEY": "env-secret", "CUSTOM_FLAG": "1"})
        self.assertEqual(config["BOHR_ACCESS_KEY"], "env-secret")
        self.assertEqual(config["OPENALEX_API_KEY"], "file-key")
        self.assertEqual(config["CUSTOM_FLAG"], "1")

    def test_provider_settings_and_redaction(self):
        config = load_config(environ={"BOHR_ACCESS_KEY": "secret-1234"})
        settings = get_provider_config(config, "bohrium")
        self.assertTrue(settings.configured)
        self.assertNotIn("secret-1234", str(settings.redacted()))
        self.assertEqual(redact_config({"BOHR_ACCESS_KEY": "secret-1234", "BOHRIUM_BASE_URL": "https://x"})["BOHR_ACCESS_KEY"], "se***34")

    def test_url_query_secrets_are_hidden(self):
        safe = redact_url("https://example.test/p?api_key=secret&search=wifi")
        self.assertEqual(safe, "https://example.test/p?api_key=%2A%2A%2A&search=wifi")


if __name__ == "__main__":
    unittest.main()
