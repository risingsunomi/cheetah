from pathlib import Path
import tempfile
import unittest

from tiny_cheetah.tui.widget.model_picker_screen import discover_cached_models


class TestModelPickerScreen(unittest.TestCase):
    def test_discover_cached_models_merges_hf_and_repo_custom_caches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hf_cache_dir = tmp_path / "hf"
            custom_cache_dir = tmp_path / "custom"

            hf_snapshot = hf_cache_dir / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / "abc123"
            hf_snapshot.mkdir(parents=True)
            (hf_snapshot / "config.json").write_text("{}", encoding="utf-8")
            (hf_snapshot / "model.safetensors").write_bytes(b"hf")

            duplicate_custom = custom_cache_dir / "Qwen__Qwen2.5-0.5B-Instruct"
            duplicate_custom.mkdir(parents=True)
            (duplicate_custom / "config.json").write_text("{}", encoding="utf-8")
            (duplicate_custom / "model.safetensors").write_bytes(b"custom")

            custom_model = custom_cache_dir / "unsloth__Llama-3.2-1B-Instruct"
            custom_model.mkdir(parents=True)
            (custom_model / "config.json").write_text("{}", encoding="utf-8")
            (custom_model / "model.safetensors").write_bytes(b"custom")

            incomplete_model = custom_cache_dir / "openai__gpt-oss-20b"
            incomplete_model.mkdir(parents=True)
            (incomplete_model / "config.json").write_text("{}", encoding="utf-8")

            models = discover_cached_models(
                hf_cache_dir=hf_cache_dir,
                custom_cache_dir=custom_cache_dir,
            )

        self.assertEqual(
            set(models),
            {
                "Qwen/Qwen2.5-0.5B-Instruct",
                "unsloth/Llama-3.2-1B-Instruct",
            },
        )
        self.assertEqual(len(models), 2)
        self.assertEqual(models, sorted(models, key=str.lower))


if __name__ == "__main__":
    unittest.main()
