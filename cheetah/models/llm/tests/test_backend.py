from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cheetah.models.llm import backend


class TestBackendFingerprints(unittest.TestCase):
    def test_model_config_fingerprint_ignores_generation_only_overrides(self) -> None:
        base = {
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 16,
            "num_hidden_layers": 16,
            "rope_theta": 10000.0,
        }
        runtime_overrides = {
            **base,
            "temperature": 1.0,
            "max_new_tokens": 400,
            "top_k": 0,
            "top_p": 0.8,
            "repetition_penalty": 1.0,
        }

        self.assertEqual(
            backend.model_config_fingerprint(base),
            backend.model_config_fingerprint(runtime_overrides),
        )

    def test_runtime_asset_fingerprints_prefer_config_file_hash(self) -> None:
        runtime_model_config = {
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "temperature": 1.0,
            "max_new_tokens": 400,
        }
        config_file_payload = {
            "hidden_size": 2048,
            "num_hidden_layers": 16,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            (model_path / "config.json").write_text(
                json.dumps(config_file_payload, sort_keys=True),
                encoding="utf-8",
            )
            (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
            expected_config_fingerprint = backend.model_config_file_fingerprint(model_path)

            fingerprints = backend.runtime_asset_fingerprints(
                model_config=runtime_model_config,
                model_path=model_path,
            )

        self.assertEqual(
            fingerprints["config_fingerprint"],
            expected_config_fingerprint,
        )
        self.assertNotEqual(
            fingerprints["config_fingerprint"],
            backend.model_config_fingerprint(runtime_model_config),
        )


if __name__ == "__main__":
    unittest.main()
