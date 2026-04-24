from pathlib import Path
import tempfile
import unittest

import numpy as np
import tinygrad as tg
from safetensors.numpy import save_file

from cheetah.models.llm.tinygrad.helpers import generate, load_safetensors
from cheetah.models.shard import Shard


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


class _DummyShardedLayerModel:
    def __init__(self):
        self.layers = [tg.nn.Linear(2, 2, bias=False)]
        self.shard = Shard("demo", start_layer=1, end_layer=2, total_layers=3)


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

    def test_load_safetensors_maps_shard_local_layer_index_to_global_weight_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            global_zero = np.zeros((2, 2), dtype=np.float32)
            global_one = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

            save_file(
                {
                    "model.layers.0.weight": global_zero,
                    "model.layers.1.weight": global_one,
                },
                str(model_dir / weight_file),
            )
            (model_dir / "model.safetensors.index.json").write_text(
                '{"metadata":{"total_size":0},"weight_map":{"model.layers.0.weight":"model.safetensors","model.layers.1.weight":"model.safetensors"}}'
            )

            model = _DummyShardedLayerModel()
            load_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1},
                weight_device="CPU",
                use_tied=False,
            )

            self.assertEqual(model.layers[0].weight.numpy().tolist(), global_one.tolist())


if __name__ == "__main__":
    unittest.main()
