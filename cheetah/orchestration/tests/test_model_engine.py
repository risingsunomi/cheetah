import asyncio
from pathlib import Path
import unittest

from cheetah.orchestration.model_engine import ModelEngine
from cheetah.orchestration.cdevice import CDevice
from cheetah.models.shard import Shard
from cheetah.models.llm.backend import backend_helpers_module, backend_model_class, get_backend_device
from cheetah.repos import RepoCustom


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
        self.assertEqual(shards[-1].end_layer, 12)
        for peer in peers:
            self.assertIsNotNone(peer.shard)

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
