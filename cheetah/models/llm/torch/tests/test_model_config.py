import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from cheetah.models.llm.torch.model_config import ModelConfig


@unittest.skipIf(torch is None, "torch is not installed")
class TestModelConfig(unittest.TestCase):
    def test_load_qwen2_defaults_match_transformers_bias_and_qk_norm_behavior(self):
        payload = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 3584,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "max_position_embeddings": 32768,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "vocab_size": 152064,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
        }

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps(payload), encoding="utf-8")
            model_config = ModelConfig()
            model_config.load(cfg)

        c = model_config.config
        self.assertEqual(c["model_type"], "qwen2")
        self.assertFalse(c["qk_norm"])
        self.assertTrue(c["qkv_bias"])
        self.assertFalse(c["o_proj_bias"])
        self.assertFalse(c["tie_word_embeddings"])

    def test_load_gpt_oss_moe_fields(self):
        payload = {
            "model_type": "gpt_oss",
            "architectures": ["GptOssForCausalLM"],
            "hidden_size": 2880,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "max_position_embeddings": 131072,
            "intermediate_size": 2880,
            "num_hidden_layers": 24,
            "vocab_size": 201088,
            "experts_per_token": 4,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "rope_theta": 150000.0,
            "initial_context_length": 4096,
            "router_aux_loss_coef": 0.9,
            "output_router_logits": False,
            "use_cache": True,
            "torch_dtype": "bfloat16",
        }

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps(payload), encoding="utf-8")
            model_config = ModelConfig()
            model_config.load(cfg)

        c = model_config.config
        self.assertEqual(c["model_type"], "gpt_oss")
        self.assertTrue(c["moe"])
        self.assertEqual(c["num_local_experts"], 32)
        self.assertEqual(c["experts_per_token"], 4)
        self.assertEqual(c["num_experts_per_tok"], 4)
        self.assertEqual(c["initial_context_length"], 4096)
        self.assertEqual(c["router_aux_loss_coef"], 0.9)
        self.assertFalse(c["output_router_logits"])
        self.assertTrue(c["use_cache"])
        self.assertTrue(c["attn_bias"])
        self.assertTrue(c["mlp_bias"])

    def test_load_nemotron_h_mamba_fields(self):
        payload = {
            "model_type": "nemotron_h",
            "architectures": ["NemotronHForCausalLM"],
            "hidden_size": 3136,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "attention_head_dim": 128,
            "max_position_embeddings": 131072,
            "intermediate_size": 12544,
            "num_hidden_layers": 6,
            "vocab_size": 256000,
            "layer_norm_epsilon": 1e-5,
            "hybrid_override_pattern": "M*-M*-",
            "mamba_num_heads": 96,
            "mamba_head_dim": 80,
            "ssm_state_size": 128,
            "n_groups": 8,
            "conv_kernel": 4,
            "expand": 2,
            "mamba_hidden_act": "silu",
            "mlp_hidden_act": "relu2",
            "use_conv_bias": True,
            "mamba_proj_bias": False,
            "residual_in_fp32": False,
            "use_bias": False,
            "chunk_size": 256,
            "torch_dtype": "bfloat16",
        }

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps(payload), encoding="utf-8")
            model_config = ModelConfig()
            model_config.load(cfg)

        c = model_config.config
        self.assertEqual(c["model_type"], "nemotron_h")
        self.assertTrue(c["mamba"])
        self.assertEqual(
            c["layers_block_type"],
            ["mamba", "attention", "mlp", "mamba", "attention", "mlp"],
        )
        self.assertEqual(c["attention_head_dim"], 128)
        self.assertEqual(c["mamba_num_heads"], 96)
        self.assertEqual(c["mamba_head_dim"], 80)
        self.assertEqual(c["ssm_state_size"], 128)
        self.assertEqual(c["conv_kernel"], 4)
        self.assertEqual(c["chunk_size"], 256)
        self.assertFalse(c["use_bias"])

    def test_load_generation_config_keeps_missing_sampling_fields_unset(self):
        payload = {
            "model_type": "gpt_oss",
            "architectures": ["GptOssForCausalLM"],
            "hidden_size": 2880,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "max_position_embeddings": 131072,
            "intermediate_size": 2880,
            "num_hidden_layers": 24,
            "vocab_size": 201088,
        }
        gen_payload = {
            "bos_token_id": 199998,
            "eos_token_id": [200002, 199999, 200012],
            "pad_token_id": 199999,
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
        self.assertIsNone(c["top_k"])
        self.assertIsNone(c["top_p"])
        self.assertIsNone(c["repetition_penalty"])
        self.assertEqual(c["eos_token_id"], [200002, 199999, 200012])

    def test_load_generation_config_reads_repetition_penalty(self):
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

        self.assertEqual(model_config.config["repetition_penalty"], 1.05)


if __name__ == "__main__":
    unittest.main()
