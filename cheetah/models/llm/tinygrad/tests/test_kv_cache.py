import unittest

import tinygrad as tg

from cheetah.models.llm.tinygrad.kv_cache import KVCache


class TestKVCache(unittest.TestCase):
    def setUp(self):
        # small, deterministic-ish dimensions
        self.B = 2
        self.Kv = 2
        self.D = 8
        self.T_MAX = 16
        # construct cache explicitly on CPU to avoid device surprises in CI
        self.cache = KVCache(
            self.T_MAX,
            self.B,
            self.Kv,
            self.T_MAX,
            self.D,
            dtype=tg.dtypes.float32,
            device="METAL",
        )

    def test_prefill_and_step_shapes(self):
        S = 4
        k = tg.Tensor.randn(self.B, S, self.Kv, self.D)
        v = tg.Tensor.randn(self.B, S, self.Kv, self.D)

        self.cache.update(k, v)
        keys, values = self.cache.get()

        self.assertEqual(self.cache.cache_pos, S)
        self.assertEqual(keys.shape, (self.B, S, self.Kv, self.D))
        self.assertEqual(values.shape, (self.B, S, self.Kv, self.D))

        # one decode step
        k2 = tg.Tensor.randn(self.B, 1, self.Kv, self.D)
        v2 = tg.Tensor.randn(self.B, 1, self.Kv, self.D)
        self.cache.update(k2, v2)

        keys2, values2 = self.cache.get()
        self.assertEqual(self.cache.cache_pos, S + 1)
        self.assertEqual(keys2.shape, (self.B, S + 1, self.Kv, self.D))
        self.assertEqual(values2.shape, (self.B, S + 1, self.Kv, self.D))

    def test_gqa_replication_shapes(self):
        # prefill some entries
        S = 3
        k = tg.Tensor.randn(self.B, S, self.Kv, self.D)
        v = tg.Tensor.randn(self.B, S, self.Kv, self.D)
        self.cache.update(k, v)

        keys, _ = self.cache.get()  # [B, T, Kv, D]
        B, T, Kv, D = keys.shape
        q_per_kv = 3

        # GQA replication to match H = Kv * q_per_kv
        keys_rep = keys.transpose(1, 2).reshape(B, Kv, 1, T, D).expand((B, Kv, q_per_kv, T, D)).flatten(1, 2)
        self.assertEqual(keys_rep.shape, (B, Kv * q_per_kv, T, D))


if __name__ == "__main__":
    unittest.main()

