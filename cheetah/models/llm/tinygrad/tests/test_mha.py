import unittest

import numpy as np
import tinygrad as tg

from cheetah.models.llm.tinygrad.attention import MultiHeadAttention


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.config = {
            "embed_dim": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 8,
            "max_seq_len": 16,
            "attn_bias": False,
            "qkv_bias": False,
            "o_proj_bias": False,
            "attn_dropout": 0.0,
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "rope_scaling_factor": None,
            "rope_low_freq_factor": 0.0,
            "rope_high_freq_factor": 0.0,
            "rope_original_max_pos_embeddings": 0,
            "qk_norm": False,
        }

    def test_forward_shape(self):
        mha = MultiHeadAttention(config=self.config)
        x = tg.Tensor.randn(3, 5, self.config["embed_dim"])
        out = mha(x).realize()
        self.assertEqual(tuple(out.shape), (3, 5, self.config["embed_dim"]))

    def test_padding_mask_does_not_disable_causal_prefill(self):
        config = dict(self.config)
        config.update(
            {
                "embed_dim": 2,
                "num_heads": 1,
                "num_kv_heads": 1,
                "head_dim": 2,
                "qkv_bias": False,
                "o_proj_bias": False,
            }
        )
        mha = MultiHeadAttention(config=config, is_causal=True)
        mha.q_proj.weight.assign(tg.Tensor.zeros((2, 2)))
        mha.k_proj.weight.assign(tg.Tensor.zeros((2, 2)))
        mha.v_proj.weight.assign(tg.Tensor.eye(2))
        mha.o_proj.weight.assign(tg.Tensor.eye(2))

        x = tg.Tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]]])
        mask = tg.Tensor([[1, 1, 1]])
        pos = tg.Tensor([[0, 0, 0]])

        out = mha(x, attention_mask=mask, position_ids=pos).realize().numpy()
        np.testing.assert_allclose(out[0, 0], np.array([1.0, 0.0], dtype=np.float32), atol=1e-5, rtol=1e-5)

    def test_training_mode_bypasses_and_clears_kv_cache(self):
        mha = MultiHeadAttention(config=self.config, is_causal=True)
        original_training = tg.Tensor.training
        try:
            tg.Tensor.training = False
            x = tg.Tensor.randn(2, 4, self.config["embed_dim"])
            mask = tg.Tensor.ones((2, 4))
            pos = tg.Tensor.arange(4).reshape(1, 4).expand((2, 4))
            _ = mha(x, attention_mask=mask, position_ids=pos).realize()

            self.assertIsNotNone(mha.kv_cache)
            assert mha.kv_cache is not None
            self.assertGreater(mha.kv_cache.cache_pos, 0)

            tg.Tensor.training = True
            train_out = mha(x, attention_mask=mask, position_ids=pos).realize()

            self.assertEqual(tuple(train_out.shape), (2, 4, self.config["embed_dim"]))
            assert mha.kv_cache is not None
            self.assertEqual(mha.kv_cache.cache_pos, 0)
        finally:
            tg.Tensor.training = original_training


if __name__ == "__main__":
    unittest.main()
