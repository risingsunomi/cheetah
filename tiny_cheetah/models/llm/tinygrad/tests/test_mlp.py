import unittest

import numpy as np
import tinygrad as tg

from tiny_cheetah.models.llm.tinygrad.mlp import MLP
from tiny_cheetah.models.llm.tinygrad.moe import MOEExperts, MOEMLP, MOERouter


class TestMLP(unittest.TestCase):
    def setUp(self):
        self.mlp = MLP(embed_dim=128, hidden_dim=256, activation="relu")

    def test_forward(self):
        input_tensor = tg.Tensor.randn(1, 128)
        output = self.mlp(input_tensor)
        self.assertEqual(output.shape, (1, 128))


class TestMOEMLP(unittest.TestCase):
    def test_decode_mxfp4_without_lookup_table(self):
        experts = MOEExperts(num_experts=1, embed_dim=32, hidden_dim=32)
        blocks = tg.Tensor(np.array([[[0x98, 0xF2]]], dtype=np.uint8))
        scales = tg.Tensor(np.array([[127]], dtype=np.uint8))

        decoded = experts._decode_mxfp4(blocks, scales).numpy()
        expected = np.array([[-0.0, -0.5, 1.0, -6.0]], dtype=np.float32)
        np.testing.assert_allclose(decoded, expected)

    def test_router_returns_sparse_top_k_scores(self):
        router = MOERouter(embed_dim=4, num_experts=4, top_k=2)
        router.weight = tg.Tensor(np.zeros((4, 4), dtype=np.float32))
        router.bias = tg.Tensor(np.array([5.0, 4.0, -2.0, -3.0], dtype=np.float32))

        scores, indices = router(tg.Tensor.randn(2, 3, 4))
        self.assertEqual(scores.shape, (6, 4))
        self.assertEqual(indices.shape, (6, 2))
        np.testing.assert_allclose(scores.numpy().sum(axis=-1), np.ones(6, dtype=np.float32))
        self.assertTrue(np.all(indices.numpy() == np.array([0, 1], dtype=np.int32)))

    def test_forward_shape(self):
        config = {
            "embed_dim": 32,
            "intermediate_dim": 32,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "swiglu_limit": 7.0,
        }
        mlp = MOEMLP(config)

        mlp.router.weight = tg.Tensor(np.zeros((4, 32), dtype=np.float32))
        mlp.router.bias = tg.Tensor(np.array([5.0, 4.0, -2.0, -3.0], dtype=np.float32))
        mlp.experts.gate_up_proj_blocks = tg.Tensor(np.full((4, 64, 1, 16), 0x22, dtype=np.uint8))
        mlp.experts.gate_up_proj_scales = tg.Tensor(np.full((4, 64, 1), 127, dtype=np.uint8))
        mlp.experts.gate_up_proj_bias = tg.Tensor(np.zeros((4, 64), dtype=np.float32))
        mlp.experts.down_proj_blocks = tg.Tensor(np.full((4, 32, 1, 16), 0x22, dtype=np.uint8))
        mlp.experts.down_proj_scales = tg.Tensor(np.full((4, 32, 1), 127, dtype=np.uint8))
        mlp.experts.down_proj_bias = tg.Tensor(np.zeros((4, 32), dtype=np.float32))

        x = tg.Tensor.randn(2, 3, 32)
        y = mlp(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.isfinite(y.numpy()).all())


if __name__ == "__main__":
    unittest.main()
