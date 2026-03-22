import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ..repo_custom import RepoCustom

class TestRepo(unittest.IsolatedAsyncioTestCase):
    async def test_custom_download(self):
        repo = RepoCustom("unsloth/Llama-3.2-1B-Instruct")
        model_path, model_config, messages = await repo.download()
        self.assertTrue(model_path.exists())
        self.assertEqual(model_config, repo.model_config.config)
        self.assertIsInstance(messages, list)

    # def test_hf_download(self):
    #     repo = RepoHuggingFace("unsloth/Llama-3.2-1B-Instruct")
    #     model_path = repo.download()
    #     self.assertTrue(model_path.exists())


class TestRepoCustomCacheFiles(unittest.TestCase):
    def test_gpt_oss_cache_requires_chat_template(self):
        payload = {
            "model_type": "gpt_oss",
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 64,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "vocab_size": 128,
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo = RepoCustom("openai/gpt-oss-20b", cache_root=Path(tmp))
            (repo.base_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
            repo._load_configs()

            self.assertEqual(repo._missing_cached_files(), ["chat_template.jinja"])

            (repo.base_dir / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
            self.assertEqual(repo._missing_cached_files(), [])

    def test_repo_custom_can_load_torch_model_config(self):
        payload = {
            "model_type": "gpt_oss",
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 64,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "vocab_size": 128,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo = RepoCustom("openai/gpt-oss-20b", cache_root=Path(tmp), backend="torch")
            (repo.base_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
            repo._load_configs()

            self.assertTrue(repo.model_config.config.get("moe"))
            self.assertEqual(repo.model_config.config.get("num_local_experts"), 8)
            self.assertEqual(repo.model_config.config.get("num_experts_per_tok"), 2)


class TestRepoCustomProgress(unittest.IsolatedAsyncioTestCase):
    async def test_download_emits_progress_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = RepoCustom("demo/model", cache_root=Path(tmp))
            messages: list[str] = []

            async def fake_download_file(*args, **kwargs):
                callback = kwargs["progress_callback"]
                progress = kwargs["progress"]
                await callback(
                    progress.render(
                        "Downloading",
                        kwargs["current"],
                        kwargs["total"],
                        args[0],
                        downloaded_bytes=5,
                        total_bytes=10,
                    )
                )
                return 10, 10

            with (
                mock.patch.object(
                    repo,
                    "_fetch_file_list",
                    new=mock.AsyncMock(return_value=[{"path": "config.json", "type": "file"}]),
                ),
                mock.patch.object(repo, "_download_file", new=mock.AsyncMock(side_effect=fake_download_file)),
                mock.patch.object(repo, "_load_configs"),
            ):
                model_path, model_config, emitted = await repo.download(progress_callback=messages.append)

            self.assertEqual(model_path, repo.base_dir)
            self.assertEqual(model_config, repo.model_config.config)
            self.assertEqual(emitted, messages)
            self.assertTrue(messages[0].startswith("[download "))
            self.assertIn("Downloading 1/1: config.json", messages[0])
            self.assertTrue(any("50.0%" in message for message in messages))
            self.assertTrue(any("Finished 1/1: config.json" in message for message in messages))
            self.assertEqual(messages[-1], "Download complete.")
