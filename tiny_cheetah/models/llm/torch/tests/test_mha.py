import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from tiny_cheetah.models.llm.torch.attention import MultiHeadAttention


@unittest.skipIf(torch is None, "torch is not installed")
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
        x = torch.randn(3, 5, self.config["embed_dim"])
        out = mha(x)
        self.assertEqual(tuple(out.shape), (3, 5, self.config["embed_dim"]))

    def test_prefill_then_decode_step(self):
        mha = MultiHeadAttention(config=self.config, is_causal=True)
        x = torch.randn(2, 4, self.config["embed_dim"])
        mask = torch.ones((2, 4), dtype=torch.long)
        pos = torch.arange(4).unsqueeze(0).expand(2, 4)

        prefill_out = mha(x, attention_mask=mask, position_ids=pos)
        self.assertEqual(tuple(prefill_out.shape), (2, 4, self.config["embed_dim"]))

        x_step = torch.randn(2, 1, self.config["embed_dim"])
        mask_step = torch.ones((2, 5), dtype=torch.long)
        pos_step = torch.tensor([4], dtype=torch.long)

        decode_out = mha(x_step, attention_mask=mask_step, position_ids=pos_step)
        self.assertEqual(tuple(decode_out.shape), (2, 1, self.config["embed_dim"]))

    def test_training_mode_bypasses_and_clears_kv_cache(self):
        mha = MultiHeadAttention(config=self.config, is_causal=True)
        mha.eval()

        x = torch.randn(2, 4, self.config["embed_dim"])
        mask = torch.ones((2, 4), dtype=torch.long)
        pos = torch.arange(4).unsqueeze(0).expand(2, 4)
        _ = mha(x, attention_mask=mask, position_ids=pos)

        self.assertIsNotNone(mha.kv_cache)
        assert mha.kv_cache is not None
        self.assertGreater(mha.kv_cache.cache_pos, 0)

        mha.train(True)
        train_out = mha(x, attention_mask=mask, position_ids=pos)

        self.assertEqual(tuple(train_out.shape), (2, 4, self.config["embed_dim"]))
        assert mha.kv_cache is not None
        self.assertEqual(mha.kv_cache.cache_pos, 0)

    def test_qwen_style_bias_flags_create_qkv_bias_without_output_bias(self):
        config = dict(self.config)
        config["qkv_bias"] = True
        config["o_proj_bias"] = False
        mha = MultiHeadAttention(config=config)

        self.assertIsNotNone(mha.q_proj.bias)
        self.assertIsNotNone(mha.k_proj.bias)
        self.assertIsNotNone(mha.v_proj.bias)
        self.assertIsNone(mha.o_proj.bias)

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

        with torch.no_grad():
            mha.q_proj.weight.zero_()
            mha.k_proj.weight.zero_()
            mha.v_proj.weight.copy_(torch.eye(2))
            mha.o_proj.weight.copy_(torch.eye(2))

        x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]]])
        mask = torch.ones((1, 3), dtype=torch.long)
        pos = torch.zeros((1, 3), dtype=torch.long)

        out = mha(x, attention_mask=mask, position_ids=pos)

        torch.testing.assert_close(out[0, 0], x[0, 0], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
