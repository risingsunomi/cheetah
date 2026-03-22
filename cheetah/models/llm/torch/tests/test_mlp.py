import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from cheetah.models.llm.torch.mlp import MLP
    from cheetah.models.llm.torch.moe import MOEExperts, MOEMLP, MOERouter


@unittest.skipIf(torch is None, "torch is not installed")
class TestMLP(unittest.TestCase):
    def setUp(self):
        self.mlp = MLP(embed_dim=128, hidden_dim=256, activation="relu")

    def test_forward(self):
        input_tensor = torch.randn(1, 128)
        output = self.mlp(input_tensor)
        self.assertEqual(tuple(output.shape), (1, 128))


@unittest.skipIf(torch is None, "torch is not installed")
class TestMOEMLP(unittest.TestCase):
    def test_decode_mxfp4_without_lookup_table(self):
        experts = MOEExperts(num_experts=1, embed_dim=32, hidden_dim=32)
        blocks = torch.tensor([[[0x98, 0xF2]]], dtype=torch.uint8)
        scales = torch.tensor([[127]], dtype=torch.uint8)

        decoded = experts._decode_mxfp4(blocks, scales)
        expected = torch.tensor([[-0.0, -0.5, 1.0, -6.0]], dtype=torch.float32)
        torch.testing.assert_close(decoded, expected)

    def test_router_returns_sparse_top_k_scores(self):
        router = MOERouter(embed_dim=4, num_experts=4, top_k=2)
        with torch.no_grad():
            router.weight.zero_()
            router.bias.copy_(torch.tensor([5.0, 4.0, -2.0, -3.0], dtype=router.bias.dtype))

        scores, indices = router(torch.randn(2, 3, 4))
        self.assertEqual(tuple(scores.shape), (6, 4))
        self.assertEqual(tuple(indices.shape), (6, 2))
        self.assertTrue(torch.all(indices.eq(torch.tensor([0, 1])).all(dim=1)).item())
        self.assertTrue(torch.allclose(scores.sum(dim=-1), torch.ones(6, dtype=scores.dtype)))
        self.assertTrue(torch.all(scores[:, 2:] == 0).item())

    def test_gpt_oss_swiglu_uses_interleaved_gate_up(self):
        config = {
            "embed_dim": 32,
            "intermediate_dim": 32,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "swiglu_limit": 7.0,
            "torch_dtype": torch.float32,
        }
        mlp = MOEMLP(config)
        gate_up = torch.tensor([[10.0, 1.0, 4.0, 2.0]], dtype=torch.float32)
        out = mlp.experts._gpt_oss_swiglu(gate_up, limit=7.0)

        gate = torch.tensor([[7.0, 4.0]], dtype=torch.float32)  # clamped from [10, 4]
        linear = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        expected = gate * torch.sigmoid(1.702 * gate) * (linear + 1.0)
        torch.testing.assert_close(out, expected)

    def test_forward_shape(self):
        config = {
            "embed_dim": 32,
            "intermediate_dim": 32,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "swiglu_limit": 7.0,
            "torch_dtype": torch.float32,
        }
        mlp = MOEMLP(config)

        with torch.no_grad():
            # Route mostly to expert 0 and 1.
            mlp.router.weight.zero_()
            mlp.router.bias.copy_(torch.tensor([5.0, 4.0, -2.0, -3.0], dtype=mlp.router.bias.dtype))

            # Fill packed weights with FP4 code 0x2 (value +1.0) and unit exponent.
            mlp.experts.gate_up_proj_blocks.fill_(0x22)
            mlp.experts.gate_up_proj_scales.fill_(127)
            mlp.experts.gate_up_proj_bias.zero_()
            mlp.experts.down_proj_blocks.fill_(0x22)
            mlp.experts.down_proj_scales.fill_(127)
            mlp.experts.down_proj_bias.zero_()

        x = torch.randn(2, 3, 32)
        y = mlp(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(y).all().item())


if __name__ == "__main__":
    unittest.main()
