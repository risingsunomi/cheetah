import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from cheetah.models.llm.torch.quantize import (
        _dequantize_bnb_nf4,
        _dequantize_bnb_nf4_simple,
        _permute,
        is_quantized_model_config,
        load_quantized_safetensors,
    )


if torch is not None:
    class _DummyTiedModel(torch.nn.Module):
        def __init__(self, vocab_size: int = 4, embed_dim: int = 4):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(vocab_size, embed_dim)
            self.output = torch.nn.Linear(embed_dim, vocab_size, bias=False)


    class _DummyQProjModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 8, bias=False)


def _pack_nibbles(nibble_values: np.ndarray) -> np.ndarray:
    values = nibble_values.reshape(-1).astype(np.uint8, copy=False)
    if values.size % 2 != 0:
        raise ValueError("Expected an even number of nibble values")

    packed = np.empty(values.size // 2, dtype=np.uint8)
    packed[:] = ((values[0::2] & 0x0F) << 4) | (values[1::2] & 0x0F)
    return packed


def _encode_quant_state(quant_state: dict) -> np.ndarray:
    payload = json.dumps(quant_state).encode("utf-8")
    return np.frombuffer(payload, dtype=np.uint8)


def _write_model_index(model_dir: Path, weight_map: dict[str, str]) -> None:
    index_payload = {
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index_payload))


@unittest.skipIf(torch is None, "torch is not installed")
class TestQuantizedLoader(unittest.TestCase):
    def test_is_quantized_model_config_true_for_bnb(self):
        config = {
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
            }
        }
        self.assertTrue(is_quantized_model_config(config))

    def test_is_quantized_model_config_false_without_quantization(self):
        self.assertFalse(is_quantized_model_config({}))
        self.assertFalse(is_quantized_model_config({"quantization_config": None}))

    def test_dequantize_bnb_nf4(self):
        blocksize = 64
        nested_blocksize = 256

        nibble_idx = np.arange(128, dtype=np.uint8) % 16
        packed_weight = _pack_nibbles(nibble_idx)

        packed_absmax = np.array([1, 2], dtype=np.uint8)
        nested_absmax = np.array([1.0], dtype=np.float32)
        nested_quant_map = np.zeros(256, dtype=np.float32)
        nested_quant_map[1] = 1.0
        nested_quant_map[2] = 2.0
        nested_offset = 0.5

        quant_map = (np.arange(16, dtype=np.float32) - 8.0) / 8.0
        quant_state = {
            "shape": [16, 8],
            "blocksize": blocksize,
            "nested_blocksize": nested_blocksize,
            "nested_offset": nested_offset,
            "quant_type": "nf4",
        }

        out = _dequantize_bnb_nf4(
            packed_weight=packed_weight,
            packed_absmax=packed_absmax,
            quant_map=quant_map,
            nested_absmax=nested_absmax,
            nested_quant_map=nested_quant_map,
            quant_state=quant_state,
        )

        self.assertEqual(out.shape, (16, 8))

        block_scales = np.array([1.5, 2.5], dtype=np.float32)
        expected = quant_map[nibble_idx.astype(np.int64)] * np.repeat(block_scales, blocksize)
        np.testing.assert_allclose(out.reshape(-1), expected, rtol=1e-6, atol=1e-6)

    def test_dequantize_bnb_nf4_simple(self):
        blocksize = 64
        nibble_idx = np.arange(64, dtype=np.uint8) % 16
        packed_weight = _pack_nibbles(nibble_idx)

        quant_map = (np.arange(16, dtype=np.float32) - 8.0) / 8.0
        absmax = np.array([2.0], dtype=np.float32)
        quant_state = {
            "shape": [8, 8],
            "blocksize": blocksize,
        }

        out = _dequantize_bnb_nf4_simple(
            packed_weight=packed_weight,
            absmax=absmax,
            quant_map=quant_map,
            quant_state=quant_state,
        )

        expected = quant_map[nibble_idx.astype(np.int64)] * 2.0
        np.testing.assert_allclose(out.reshape(-1), expected, rtol=1e-6, atol=1e-6)

    def test_load_quantized_safetensors_loads_nf4_weight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "model.output.weight"

            shape = (4, 4)
            blocksize = 8
            nibble_idx = np.arange(np.prod(shape), dtype=np.uint8) % 16
            packed_weight = _pack_nibbles(nibble_idx)
            absmax = np.array([1.5, 2.5], dtype=np.float32)
            quant_map = (np.arange(16, dtype=np.float32) - 8.0) / 8.0
            quant_state = {
                "shape": list(shape),
                "blocksize": blocksize,
                "quant_type": "nf4",
            }

            tensors = {
                key: packed_weight,
                f"{key}.absmax": absmax,
                f"{key}.quant_map": quant_map,
                f"{key}.quant_state.bitsandbytes__nf4": _encode_quant_state(quant_state),
            }
            save_file(tensors, str(model_dir / weight_file))
            _write_model_index(model_dir, {key: weight_file})

            model = _DummyTiedModel(vocab_size=4, embed_dim=4)
            load_quantized_safetensors(
                model,
                model_dir,
                model_config={},
                weight_device="cpu",
                use_tied=False,
            )

            expected = quant_map[nibble_idx.astype(np.int64)] * np.repeat(absmax, blocksize)
            expected = torch.tensor(expected.reshape(shape), dtype=model.output.weight.dtype)
            torch.testing.assert_close(model.output.weight.detach().cpu(), expected)

    def test_load_quantized_safetensors_uses_tied_embed_for_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "model.embed_tokens.weight"

            shape = (4, 4)
            blocksize = 8
            nibble_idx = np.arange(np.prod(shape), dtype=np.uint8) % 16
            packed_weight = _pack_nibbles(nibble_idx)
            absmax = np.array([1.0, 2.0], dtype=np.float32)
            quant_map = (np.arange(16, dtype=np.float32) - 8.0) / 8.0
            quant_state = {
                "shape": list(shape),
                "blocksize": blocksize,
                "quant_type": "nf4",
            }

            tensors = {
                key: packed_weight,
                f"{key}.absmax": absmax,
                f"{key}.quant_map": quant_map,
                f"{key}.quant_state.bitsandbytes__nf4": _encode_quant_state(quant_state),
            }
            save_file(tensors, str(model_dir / weight_file))
            _write_model_index(model_dir, {key: weight_file})

            model = _DummyTiedModel(vocab_size=4, embed_dim=4)
            load_quantized_safetensors(
                model,
                model_dir,
                model_config={},
                weight_device="cpu",
                use_tied=True,
            )

            expected = quant_map[nibble_idx.astype(np.int64)] * np.repeat(absmax, blocksize)
            expected = torch.tensor(expected.reshape(shape), dtype=model.embed_tokens.weight.dtype)
            torch.testing.assert_close(model.embed_tokens.weight.detach().cpu(), expected)
            torch.testing.assert_close(model.output.weight.detach().cpu(), expected)

    def test_load_quantized_safetensors_falls_back_to_plain_tensor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "model.output.weight"

            plain_weight = np.arange(16, dtype=np.float32).reshape(4, 4)
            save_file({key: plain_weight}, str(model_dir / weight_file))
            _write_model_index(model_dir, {key: weight_file})

            model = _DummyTiedModel(vocab_size=4, embed_dim=4)
            load_quantized_safetensors(
                model,
                model_dir,
                model_config={},
                weight_device="cpu",
                use_tied=False,
            )

            expected = torch.from_numpy(plain_weight).to(dtype=model.output.weight.dtype)
            torch.testing.assert_close(model.output.weight.detach().cpu(), expected)

    def test_load_quantized_safetensors_applies_qproj_permute(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "model.q_proj.weight"

            raw_weight = np.arange(32, dtype=np.float32).reshape(8, 4)
            save_file({key: raw_weight}, str(model_dir / weight_file))
            _write_model_index(model_dir, {key: weight_file})

            model = _DummyQProjModel()
            load_quantized_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1},
                weight_device="cpu",
                use_tied=False,
            )

            expected = _permute(torch.from_numpy(raw_weight), 2).to(dtype=model.q_proj.weight.dtype)
            torch.testing.assert_close(model.q_proj.weight.detach().cpu(), expected)

    def test_load_quantized_safetensors_skips_qproj_permute_for_gpt_oss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "model.q_proj.weight"

            raw_weight = np.arange(32, dtype=np.float32).reshape(8, 4)
            save_file({key: raw_weight}, str(model_dir / weight_file))
            _write_model_index(model_dir, {key: weight_file})

            model = _DummyQProjModel()
            load_quantized_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1, "model_type": "gpt_oss"},
                weight_device="cpu",
                use_tied=False,
            )

            expected = torch.from_numpy(raw_weight).to(dtype=model.q_proj.weight.dtype)
            torch.testing.assert_close(model.q_proj.weight.detach().cpu(), expected)

    def test_load_quantized_safetensors_without_index_scans_all_shards(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            key = "model.output.weight"

            save_file({"model.embed_tokens.weight": np.zeros((4, 4), dtype=np.float32)}, str(model_dir / "a.safetensors"))
            plain_weight = np.arange(16, dtype=np.float32).reshape(4, 4)
            save_file({key: plain_weight}, str(model_dir / "b.safetensors"))

            model = _DummyTiedModel(vocab_size=4, embed_dim=4)
            load_quantized_safetensors(
                model,
                model_dir,
                model_config={},
                weight_device="cpu",
                use_tied=False,
            )

            expected = torch.from_numpy(plain_weight).to(dtype=model.output.weight.dtype)
            torch.testing.assert_close(model.output.weight.detach().cpu(), expected)

    def test_load_quantized_safetensors_with_index_infers_model_prefix_even_if_first_key_is_lm_head(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "model.q_proj.weight"
            raw_weight = np.arange(32, dtype=np.float32).reshape(8, 4)

            save_file(
                {
                    "lm_head.weight": np.zeros((4, 4), dtype=np.float32),
                    key: raw_weight,
                },
                str(model_dir / weight_file),
            )
            _write_model_index(
                model_dir,
                {
                    "lm_head.weight": weight_file,
                    key: weight_file,
                },
            )

            model = _DummyQProjModel()
            load_quantized_safetensors(
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
