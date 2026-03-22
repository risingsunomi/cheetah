import unittest

import tinygrad
from cheetah.models.llm.tinygrad.rope import RotaryPositionalEmbedding

class TestRotaryPositionalEmbedding(unittest.TestCase):

    def setUp(self):
        self.embedding = RotaryPositionalEmbedding(
            dim=64, max_seq_len=512)

    def test_forward(self):
        x = tinygrad.Tensor.randn(1, 512, 64)
        y = self.embedding(x)
        print(f"Output shape: {y.shape}")
        print(f"y: {y}")
        self.assertEqual(y.shape, (1, 512, 64))

    def test_cache(self):
        self.embedding.build_rope_cache()
        self.assertIsNotNone(self.embedding.cache)
