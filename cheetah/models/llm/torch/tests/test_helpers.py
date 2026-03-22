import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from safetensors.numpy import save_file
from safetensors.torch import save_file as save_torch_file

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from cheetah.models.llm.torch.helpers import generate, load_model, load_safetensors, sample
    from cheetah.models.llm.torch.helpers import permute as helpers_permute


if torch is not None:
    class _DummyQProjModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 8, bias=False)


    class _DummyExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj_blocks = torch.nn.Parameter(
                torch.zeros((2, 8, 1, 16), dtype=torch.uint8),
                requires_grad=False,
            )


    class _DummyMlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = torch.nn.Linear(4, 2, bias=True)
            self.experts = _DummyExperts()


    class _DummyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _DummyMlp()

    class _DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(4, 4)


    class _DummyMoEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_DummyLayer()])


    class _DummyBackboneModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _DummyBackbone()


    class _DummyGenerateModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.position_ids_seen: list[torch.Tensor] = []
            self.reset_calls = 0

        def reset_kv_cache(self) -> None:
            self.reset_calls += 1

        def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            hidden_state: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del attention_mask, hidden_state
            if position_ids is None:
                raise AssertionError("position_ids should be provided")
            self.position_ids_seen.append(position_ids.detach().cpu().clone())
            batch_size, seq_len = x.shape
            logits = torch.zeros((batch_size, seq_len, 4), dtype=torch.float32)
            logits[..., 1] = 1.0
            return logits


    class _DummyTokenizer:
        eos_token_id = 3


def _write_model_index(model_dir: Path, weight_map: dict[str, str]) -> None:
    index_payload = {
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index_payload))


@unittest.skipIf(torch is None, "torch is not installed")
class TestHelpersLoader(unittest.TestCase):
    def test_load_model_sets_eval_mode_for_incremental_decode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["LlamaForCausalLM"],
                        "attention_bias": False,
                        "attention_dropout": 0.0,
                        "bos_token_id": 1,
                        "eos_token_id": 2,
                        "hidden_act": "silu",
                        "hidden_size": 8,
                        "intermediate_size": 16,
                        "max_position_embeddings": 32,
                        "model_type": "llama",
                        "num_attention_heads": 2,
                        "num_hidden_layers": 1,
                        "num_key_value_heads": 1,
                        "pad_token_id": 0,
                        "rms_norm_eps": 1e-5,
                        "rope_theta": 10000.0,
                        "tie_word_embeddings": True,
                        "torch_dtype": "float32",
                        "use_cache": True,
                        "vocab_size": 16,
                    }
                )
            )

            dummy_model = _DummyGenerateModel()
            dummy_tokenizer = object()

            with (
                mock.patch("cheetah.models.llm.torch.helpers.build_model", return_value=dummy_model),
                mock.patch("cheetah.models.llm.torch.helpers.load_safetensors"),
                mock.patch(
                    "cheetah.models.llm.torch.helpers.AutoTokenizer.from_pretrained",
                    return_value=dummy_tokenizer,
                ),
            ):
                model, _, tokenizer, resolved = asyncio.run(
                    load_model(
                        model_id=str(model_dir),
                        offline_mode=True,
                    )
                )

            self.assertIs(model, dummy_model)
            self.assertIs(tokenizer, dummy_tokenizer)
            self.assertEqual(resolved, model_dir)
            self.assertFalse(model.training)

    def test_load_safetensors_applies_qproj_permute_for_non_gpt_oss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "model.q_proj.weight"

            raw_weight = np.arange(32, dtype=np.float32).reshape(8, 4)
            save_file({key: raw_weight}, str(model_dir / weight_file))
            _write_model_index(model_dir, {key: weight_file})

            model = _DummyQProjModel()
            load_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1, "model_type": "llama"},
                weight_device="cpu",
                use_tied=False,
            )

            expected = helpers_permute(torch.from_numpy(raw_weight), 2).to(dtype=model.q_proj.weight.dtype)
            torch.testing.assert_close(model.q_proj.weight.detach().cpu(), expected)

    def test_load_safetensors_skips_qproj_permute_for_gpt_oss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "model.q_proj.weight"

            raw_weight = np.arange(32, dtype=np.float32).reshape(8, 4)
            save_file({key: raw_weight}, str(model_dir / weight_file))
            _write_model_index(model_dir, {key: weight_file})

            model = _DummyQProjModel()
            load_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1, "model_type": "gpt_oss"},
                weight_device="cpu",
                use_tied=False,
            )

            expected = torch.from_numpy(raw_weight).to(dtype=model.q_proj.weight.dtype)
            torch.testing.assert_close(model.q_proj.weight.detach().cpu(), expected)

    def test_load_safetensors_loads_backbone_key_without_double_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "backbone.embeddings.weight"

            raw_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            save_torch_file({key: raw_weight}, str(model_dir / weight_file))
            _write_model_index(model_dir, {key: weight_file})

            model = _DummyBackboneModel()
            load_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1, "model_type": "nemotron_h"},
                weight_device="cpu",
                use_tied=False,
            )

            torch.testing.assert_close(
                model.backbone.embeddings.weight.detach().cpu(),
                raw_weight.to(dtype=model.backbone.embeddings.weight.dtype),
            )

    def test_load_safetensors_dequantizes_fp8_weight_with_scale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "backbone.embeddings.weight"

            fp32_values = torch.tensor(
                [[1.0, -2.0, 3.5, -4.0], [0.5, 1.5, -1.0, 2.0], [3.0, -0.5, 0.25, -0.75], [1.25, -1.25, 2.5, -2.5]],
                dtype=torch.float32,
            )
            scale = torch.tensor(0.5, dtype=torch.float32)
            packed = (fp32_values / scale).to(torch.float8_e4m3fn)
            expected = packed.to(torch.float32) * scale
            save_torch_file(
                {
                    key: packed,
                    f"{key}_scale": scale,
                },
                str(model_dir / weight_file),
            )
            _write_model_index(model_dir, {key: weight_file})

            model = _DummyBackboneModel()
            load_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1, "model_type": "nemotron_h"},
                weight_device="cpu",
                use_tied=False,
            )

            torch.testing.assert_close(
                model.backbone.embeddings.weight.detach().cpu(),
                expected.to(dtype=model.backbone.embeddings.weight.dtype),
            )


@unittest.skipIf(torch is None, "torch is not installed")
class TestHelpersGenerate(unittest.TestCase):
    def test_sample_applies_repetition_penalty_before_argmax_shortcut(self):
        logits = torch.tensor([1.0, 0.9, 0.1], dtype=torch.float32)
        tok = sample(
            logits,
            temp=0.0,
            k=0,
            p=1.0,
            seen_tokens=[0],
            repetition_penalty=2.0,
        )
        self.assertEqual(int(tok.item()), 1)

    def test_generate_uses_absolute_decode_positions_after_prefill(self):
        model = _DummyGenerateModel()
        tokenizer = _DummyTokenizer()

        out = generate(
            model,
            input_ids=torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            tokenizer=tokenizer,
            max_new_tokens=3,
            temp=0.0,
            top_k=0,
            top_p=1.0,
        )

        self.assertEqual(out, [1, 1, 1])
        self.assertEqual(model.reset_calls, 1)
        self.assertEqual(len(model.position_ids_seen), 3)
        self.assertTrue(torch.equal(model.position_ids_seen[0], torch.tensor([[0, 1, 2, 3]])))
        self.assertTrue(torch.equal(model.position_ids_seen[1], torch.tensor([4])))
        self.assertTrue(torch.equal(model.position_ids_seen[2], torch.tensor([5])))

    def test_load_safetensors_loads_gpt_oss_moe_expert_tensor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "model.layers.0.mlp.experts.gate_up_proj_blocks"

            raw_weight = np.full((2, 8, 1, 16), 0x22, dtype=np.uint8)
            save_file({key: raw_weight}, str(model_dir / weight_file))
            _write_model_index(model_dir, {key: weight_file})

            model = _DummyMoEModel()
            load_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1, "model_type": "gpt_oss"},
                weight_device="cpu",
                use_tied=False,
            )

            expected = torch.from_numpy(raw_weight)
            self.assertTrue(
                torch.equal(
                    model.layers[0].mlp.experts.gate_up_proj_blocks.detach().cpu(),
                    expected,
                )
            )

    def test_load_safetensors_without_index_scans_all_shards(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            key = "model.q_proj.weight"
            raw_weight = np.arange(32, dtype=np.float32).reshape(8, 4)

            save_file({"model.embed_tokens.weight": np.zeros((4, 4), dtype=np.float32)}, str(model_dir / "a.safetensors"))
            save_file({key: raw_weight}, str(model_dir / "b.safetensors"))

            model = _DummyQProjModel()
            load_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1, "model_type": "gpt_oss"},
                weight_device="cpu",
                use_tied=False,
            )

            expected = torch.from_numpy(raw_weight).to(dtype=model.q_proj.weight.dtype)
            torch.testing.assert_close(model.q_proj.weight.detach().cpu(), expected)

    def test_load_safetensors_with_index_infers_model_prefix_even_if_first_key_is_lm_head(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            qproj_key = "model.q_proj.weight"
            raw_weight = np.arange(32, dtype=np.float32).reshape(8, 4)

            save_file(
                {
                    "lm_head.weight": np.zeros((4, 4), dtype=np.float32),
                    qproj_key: raw_weight,
                },
                str(model_dir / weight_file),
            )
            _write_model_index(
                model_dir,
                {
                    "lm_head.weight": weight_file,
                    qproj_key: weight_file,
                },
            )

            model = _DummyQProjModel()
            load_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1, "model_type": "gpt_oss"},
                weight_device="cpu",
                use_tied=False,
            )

            expected = torch.from_numpy(raw_weight).to(dtype=model.q_proj.weight.dtype)
            torch.testing.assert_close(model.q_proj.weight.detach().cpu(), expected)


if __name__ == "__main__":
    unittest.main()
