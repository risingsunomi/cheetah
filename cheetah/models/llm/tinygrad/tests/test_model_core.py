import unittest
from types import SimpleNamespace

from cheetah.models.llm.tinygrad.model import Model


class _DummyCache:
    def __init__(self):
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1


class _DummyJit:
    def __init__(self):
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class TestModelResetPaths(unittest.TestCase):
    def test_reset_kv_cache_does_not_invalidate_jit(self):
        model = Model.__new__(Model)
        cache = _DummyCache()
        model.layers = [SimpleNamespace(self_attn=SimpleNamespace(kv_cache=cache))]
        model._decode_token_jit = _DummyJit()
        model._decode_hidden_jit = _DummyJit()

        Model.reset_kv_cache(model)

        self.assertEqual(cache.clear_calls, 1)
        self.assertEqual(model._decode_token_jit.reset_calls, 0)
        self.assertEqual(model._decode_hidden_jit.reset_calls, 0)

    def test_reset_decode_jit_only_resets_jit_runners(self):
        model = Model.__new__(Model)
        cache = _DummyCache()
        model.layers = [SimpleNamespace(self_attn=SimpleNamespace(kv_cache=cache))]
        model._decode_token_jit = _DummyJit()
        model._decode_hidden_jit = _DummyJit()

        Model.reset_decode_jit(model)

        self.assertEqual(cache.clear_calls, 0)
        self.assertEqual(model._decode_token_jit.reset_calls, 1)
        self.assertEqual(model._decode_hidden_jit.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
