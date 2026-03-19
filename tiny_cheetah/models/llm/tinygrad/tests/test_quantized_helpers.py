import unittest
import numpy as np

from tiny_cheetah.models.llm.tinygrad.quantize import _dequantize_bnb_nf4, _dequantize_bnb_nf4_simple, is_quantized_model_config


class TestQuantizedLoader(unittest.TestCase):
    def test_is_quantized_model_config_true_for_unsloth_style_bnb(self):
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

        # 2 blocks => 128 values => 64 packed bytes.
        nibble_idx = np.arange(128, dtype=np.uint8) % 16
        packed_weight = np.empty(64, dtype=np.uint8)
        packed_weight[:] = (nibble_idx[0::2] << 4) | nibble_idx[1::2]

        # Two block scales encoded through nested quantization path.
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
        packed_weight = np.empty(32, dtype=np.uint8)
        packed_weight[:] = (nibble_idx[0::2] << 4) | nibble_idx[1::2]

        quant_map = (np.arange(16, dtype=np.float32) - 8.0) / 8.0
        absmax = np.array([2.0], dtype=np.float32)
        quant_state = {"shape": [8, 8], "blocksize": blocksize}

        out = _dequantize_bnb_nf4_simple(
            packed_weight=packed_weight,
            absmax=absmax,
            quant_map=quant_map,
            quant_state=quant_state,
        )

        expected = quant_map[nibble_idx.astype(np.int64)] * 2.0
        np.testing.assert_allclose(out.reshape(-1), expected, rtol=1e-6, atol=1e-6)
