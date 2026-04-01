import json
from pathlib import Path
import tempfile
import unittest

from cheetah.models.llm.tinygrad.model_config import ModelConfig


class TestTinygradModelConfig(unittest.TestCase):
    def test_load_generation_config_keeps_missing_sampling_fields_unset(self) -> None:
        payload = {
            "model_type": "qwen2",
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 64,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "vocab_size": 1024,
        }
        gen_payload = {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            gen_cfg = Path(tmp) / "generation_config.json"
            cfg.write_text(json.dumps(payload), encoding="utf-8")
            gen_cfg.write_text(json.dumps(gen_payload), encoding="utf-8")

            model_config = ModelConfig()
            model_config.load(cfg)
            model_config.load_generation_config(gen_cfg)

        c = model_config.config
        self.assertIsNone(c["temperature"])
        self.assertIsNone(c["max_new_tokens"])
        self.assertIsNone(c["top_k"])
        self.assertIsNone(c["top_p"])
        self.assertIsNone(c["repetition_penalty"])
        self.assertEqual(c["eos_token_id"], 2)

    def test_load_generation_config_reads_generation_limits(self) -> None:
        payload = {
            "model_type": "qwen2",
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 64,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "vocab_size": 1024,
        }
        gen_payload = {
            "temperature": 0.7,
            "max_new_tokens": 512,
            "top_k": 20,
            "top_p": 0.8,
            "repetition_penalty": 1.05,
        }

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            gen_cfg = Path(tmp) / "generation_config.json"
            cfg.write_text(json.dumps(payload), encoding="utf-8")
            gen_cfg.write_text(json.dumps(gen_payload), encoding="utf-8")

            model_config = ModelConfig()
            model_config.load(cfg)
            model_config.load_generation_config(gen_cfg)

        self.assertEqual(model_config.config["max_new_tokens"], 512)
        self.assertEqual(model_config.config["repetition_penalty"], 1.05)


if __name__ == "__main__":
    unittest.main()
