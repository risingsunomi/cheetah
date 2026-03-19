import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from tiny_cheetah.models.llm.torch.rope import RotaryPositionalEmbedding


@unittest.skipIf(torch is None, "torch is not installed")
class TestRotaryPositionalEmbedding(unittest.TestCase):
    def setUp(self):
        self.embedding = RotaryPositionalEmbedding(dim=8, max_seq_len=64)

    def test_forward_shapes(self):
        q = torch.randn(2, 6, 4, 8)
        k = torch.randn(2, 6, 2, 8)
        position_ids = torch.arange(6)
        q_out, k_out = self.embedding(q, k, position_ids)

        self.assertEqual(tuple(q_out.shape), tuple(q.shape))
        self.assertEqual(tuple(k_out.shape), tuple(k.shape))

    def test_forward_with_batched_position_ids(self):
        q = torch.randn(2, 5, 4, 8)
        k = torch.randn(2, 5, 2, 8)
        position_ids = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        q_out, k_out = self.embedding(q, k, position_ids)

        self.assertEqual(tuple(q_out.shape), tuple(q.shape))
        self.assertEqual(tuple(k_out.shape), tuple(k.shape))

    def test_cache(self):
        self.embedding.build_rope_cache(torch.device("cpu"))
        self.assertIsNotNone(self.embedding.cache)


if __name__ == "__main__":
    unittest.main()
