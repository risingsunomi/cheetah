import asyncio
from pathlib import Path
import unittest
from types import SimpleNamespace

from cheetah.orchestration.model_engine import ModelEngine, _decode_tensor, _encode_tensor
from cheetah.orchestration.cdevice import CDevice
from cheetah.models.shard import Shard
from cheetah.models.llm.backend import backend_helpers_module, backend_model_class, get_backend_device
from cheetah.repos import RepoCustom

try:
    import tinygrad as tg
except Exception:
    tg = None
try:
    import torch
except Exception:
    torch = None


class TestModelEngine(unittest.TestCase):
    def test_plan_shards_assigns_and_sets_peer_shard(self):
        peers = []
        for idx, ram in enumerate([8, 4, 2]):
            p = CDevice(f"p{idx+1}", "0.0.0.0", 0)
            p.cpu_ram = str(ram)
            p.gpu_vram = ""
            peers.append(p)
        shards = ModelEngine.plan_shards(peers, "demo", total_layers=12)
        self.assertEqual(len(shards), 3)
        self.assertEqual(shards[0].start_layer, 0)
        self.assertEqual(shards[-1].end_layer, 11)
        self.assertEqual(shards[-1].total_layers, 12)
        for peer in peers:
            self.assertIsNotNone(peer.shard)

    def test_get_tokens_prefill_runs_requested_shard_without_sampling(self):
        if tg is None:
            self.skipTest("tinygrad is required for this test.")

        class FakeShardModel:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.calls: list[dict[str, object]] = []

            def reset_kv_cache(self) -> None:
                self.reset_calls += 1

            def run_shard(
                self,
                x,
                *,
                attention_mask=None,
                position_ids=None,
                hidden_state=None,
                shard=None,
                start_pos=None,
            ):
                self.calls.append(
                    {
                        "x": x,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "hidden_state": hidden_state,
                        "shard": shard,
                        "start_pos": start_pos,
                    }
                )
                return tg.Tensor.ones((1, attention_mask.shape[1], 4))

        engine = ModelEngine(shard=Shard("demo", start_layer=0, end_layer=2, total_layers=5))
        model = FakeShardModel()
        input_ids = tg.Tensor([[1, 2, 3]], dtype=tg.dtypes.int32)
        attention_mask = tg.Tensor([[1, 1, 1]], dtype=tg.dtypes.int32)

        payload = engine.get_tokens(
            model,
            input_ids,
            attention_mask,
            SimpleNamespace(eos_token_id=99),
            prefill=True,
        )

        self.assertEqual(model.reset_calls, 1)
        self.assertIn("hidden_state", payload)
        self.assertFalse(payload["end_token"])
        self.assertEqual(len(model.calls), 1)
        call = model.calls[0]
        assert isinstance(call["position_ids"], tg.Tensor)
        self.assertEqual(tuple(call["position_ids"].shape), (1, 3))
        self.assertIsNone(call["start_pos"])
        shard = call["shard"]
        self.assertIsNotNone(shard)
        assert shard is not None
        self.assertEqual(shard.start_layer, 0)
        self.assertEqual(shard.end_layer, 2)

    def test_encode_tensor_upcasts_torch_bfloat16_for_network(self):
        if torch is None:
            self.skipTest("torch is required for this test.")

        tensor = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3).to(dtype=torch.bfloat16)

        payload = _encode_tensor(tensor)
        self.assertEqual(payload["dtype"], "float32")

        decoded = _decode_tensor(payload, backend="torch")
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertIsInstance(decoded, torch.Tensor)
        self.assertEqual(decoded.dtype, torch.float32)
        self.assertTrue(torch.allclose(decoded, tensor.float()))

    def test_model_engine_loads_llama_3_2_1b(self):
        model_name = "unsloth/Llama-3.2-1B-Instruct"
        repo = RepoCustom(model_name)
        config_path = repo.base_dir / "config.json"
        weight_files = list(repo.base_dir.glob("*.safetensors"))
        if config_path.exists() and weight_files:
            repo._load_configs()
            model_path = repo.base_dir
        else:
            model_path, _, _ = asyncio.run(repo.download())
            
        if not repo.model_config.config:
            self.skipTest("Model config missing for Llama 3.2 1B.")

        model_path = Path(model_path)
        if not list(model_path.glob("*.safetensors")):
            self.skipTest("Model weights missing for Llama 3.2 1B.")

        try:
            import transformers
        except Exception:
            self.skipTest("transformers is required for this test.")
        try:
            import tinygrad
        except Exception:
            self.skipTest("tinygrad is required for this test.")

        Model = backend_model_class("tinygrad")
        load_safetensors = backend_helpers_module("tinygrad").load_safetensors

        config = repo.model_config.config
        shard = Shard(
            model_name,
            start_layer=0,
            end_layer=config["num_layers"],
            total_layers=config["num_layers"] + 1,
        )
        model = Model(config, shard, use_tied=config.get("tie_word_embeddings", False))
        load_safetensors(
            model,
            model_path,
            config,
            weight_device=get_backend_device("tinygrad", default="CPU") or "CPU",
            use_tied=config.get("tie_word_embeddings", False),
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        enc = tokenizer("Hello", return_tensors="np")
        input_ids = tinygrad.Tensor(enc["input_ids"])
        attention_mask = tinygrad.Tensor(enc["attention_mask"])

        engine = ModelEngine(shard=shard)
        payload = engine.get_tokens(
            model,
            input_ids,
            attention_mask,
            tokenizer,
            temp=0.6,
            top_k=0,
            top_p=0.8,
        )

        self.assertEqual(payload["shard"]["model_name"], model_name)
        self.assertIn("token", payload)
        self.assertIn("tensor", payload)
        self.assertIsInstance(payload["end_token"], bool)
