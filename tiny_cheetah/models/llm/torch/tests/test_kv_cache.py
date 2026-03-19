import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from tiny_cheetah.models.llm.torch.kv_cache import KVCache


@unittest.skipIf(torch is None, "torch is not installed")
class TestKVCache(unittest.TestCase):
    def setUp(self):
        self.batch = 2
        self.num_kv_heads = 2
        self.head_dim = 8
        self.max_len = 16
        self.cache = KVCache(
            self.max_len,
            self.batch,
            self.num_kv_heads,
            self.max_len,
            self.head_dim,
            dtype=torch.float32,
            device="cpu",
        )

    def test_prefill_and_step_shapes(self):
        seq_len = 4
        k = torch.randn(self.batch, seq_len, self.num_kv_heads, self.head_dim)
        v = torch.randn(self.batch, seq_len, self.num_kv_heads, self.head_dim)

        self.cache.update(k, v)
        keys, values = self.cache.get()

        self.assertEqual(self.cache.cache_pos, seq_len)
        self.assertEqual(tuple(keys.shape), (self.batch, seq_len, self.num_kv_heads, self.head_dim))
        self.assertEqual(tuple(values.shape), (self.batch, seq_len, self.num_kv_heads, self.head_dim))

        k2 = torch.randn(self.batch, 1, self.num_kv_heads, self.head_dim)
        v2 = torch.randn(self.batch, 1, self.num_kv_heads, self.head_dim)
        self.cache.update(k2, v2)

        keys2, values2 = self.cache.get()
        self.assertEqual(self.cache.cache_pos, seq_len + 1)
        self.assertEqual(tuple(keys2.shape), (self.batch, seq_len + 1, self.num_kv_heads, self.head_dim))
        self.assertEqual(tuple(values2.shape), (self.batch, seq_len + 1, self.num_kv_heads, self.head_dim))

    def test_gqa_replication_shapes(self):
        seq_len = 3
        k = torch.randn(self.batch, seq_len, self.num_kv_heads, self.head_dim)
        v = torch.randn(self.batch, seq_len, self.num_kv_heads, self.head_dim)
        self.cache.update(k, v)

        keys, _ = self.cache.get()
        batch, total, kv_heads, dim = keys.shape
        q_per_kv = 3

        keys_rep = (
            keys.transpose(1, 2)
            .reshape(batch, kv_heads, 1, total, dim)
            .expand(batch, kv_heads, q_per_kv, total, dim)
            .reshape(batch, kv_heads * q_per_kv, total, dim)
        )
        self.assertEqual(tuple(keys_rep.shape), (batch, kv_heads * q_per_kv, total, dim))


if __name__ == "__main__":
    unittest.main()
