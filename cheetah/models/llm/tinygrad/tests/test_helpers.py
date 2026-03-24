import unittest

import tinygrad as tg

from cheetah.models.llm.tinygrad.helpers import generate


class _DummyGenerateModel:
    def __init__(self):
        self.position_ids_seen: list[tg.Tensor] = []
        self.reset_calls = 0

    def reset_kv_cache(self) -> None:
        self.reset_calls += 1

    def __call__(
        self,
        x: tg.Tensor,
        attention_mask: tg.Tensor | None = None,
        position_ids: tg.Tensor | None = None,
        hidden_state: tg.Tensor | None = None,
    ) -> tg.Tensor:
        del attention_mask, hidden_state
        if position_ids is None:
            raise AssertionError("position_ids should be provided")
        self.position_ids_seen.append(position_ids)
        batch_size, seq_len = x.shape
        logits = tg.Tensor.zeros((batch_size, seq_len, 4), device=x.device)
        logits = logits + tg.Tensor([[[0.0, 1.0, 0.0, 0.0]]], device=x.device)
        return logits


class _DummyTokenizer:
    eos_token_id = 1


class _DummyOptimizedGenerateModel(_DummyGenerateModel):
    def __init__(self):
        super().__init__()
        self.decode_start_pos_seen: list[int] = []

    def __call__(
        self,
        x: tg.Tensor,
        attention_mask: tg.Tensor | None = None,
        position_ids: tg.Tensor | None = None,
        hidden_state: tg.Tensor | None = None,
    ) -> tg.Tensor:
        del hidden_state
        if position_ids is None:
            raise AssertionError("position_ids should be provided")
        self.position_ids_seen.append(position_ids)
        if attention_mask is None:
            raise AssertionError("prefill should keep the original attention mask")
        batch_size, seq_len = x.shape
        logits = tg.Tensor.zeros((batch_size, seq_len, 4), device=x.device)
        logits = logits + tg.Tensor([[[0.0, 0.0, 1.0, 0.0]]], device=x.device)
        return logits

    def decode_token(
        self,
        x: tg.Tensor,
        position_ids: tg.Tensor | None = None,
        *,
        start_pos: int | tg.UOp | None = None,
    ) -> tg.Tensor:
        del position_ids
        if isinstance(start_pos, tg.UOp):
            raise AssertionError("helpers should pass a concrete start_pos to the model wrapper")
        self.decode_start_pos_seen.append(int(start_pos))
        batch_size, seq_len = x.shape
        logits = tg.Tensor.zeros((batch_size, seq_len, 4), device=x.device)
        logits = logits + tg.Tensor([[[0.0, 1.0, 0.0, 0.0]]], device=x.device)
        return logits


class TestHelpersGenerate(unittest.TestCase):
    def test_generate_handles_eos_without_tensor_bool(self):
        model = _DummyGenerateModel()
        tokenizer = _DummyTokenizer()

        out = generate(
            model,
            input_ids=tg.Tensor([[10, 11, 12]]),
            attention_mask=tg.Tensor([[1, 1, 1]]),
            tokenizer=tokenizer,
            max_new_tokens=4,
            temp=0.0,
            top_k=0,
            top_p=1.0,
        )

        self.assertEqual(out, [1])
        self.assertEqual(model.reset_calls, 1)
        self.assertEqual(len(model.position_ids_seen), 1)
        self.assertEqual(model.position_ids_seen[0].tolist(), [[0, 1, 2]])

    def test_generate_uses_decode_token_fast_path_for_single_token_decode(self):
        model = _DummyOptimizedGenerateModel()
        tokenizer = _DummyTokenizer()

        out = generate(
            model,
            input_ids=tg.Tensor([[10, 11, 12]]),
            attention_mask=tg.Tensor([[1, 1, 1]]),
            tokenizer=tokenizer,
            max_new_tokens=4,
            temp=0.0,
            top_k=0,
            top_p=1.0,
        )

        self.assertEqual(out, [2, 1])
        self.assertEqual(model.reset_calls, 1)
        self.assertEqual(len(model.position_ids_seen), 1)
        self.assertEqual(model.position_ids_seen[0].tolist(), [[0, 1, 2]])
        self.assertEqual(model.decode_start_pos_seen, [3])


if __name__ == "__main__":
    unittest.main()
