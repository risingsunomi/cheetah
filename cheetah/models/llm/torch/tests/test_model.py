import asyncio
import os
from pathlib import Path
import unittest

from transformers import AutoTokenizer

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from cheetah.models.llm.torch.model import Model
    from cheetah.models.llm.torch.helpers import generate, load_safetensors

from cheetah.models.llm.backend import get_backend_device
from cheetah.models.shard import Shard
from cheetah.repos import RepoCustom

TOP_K = 20
TEMP = 0.7
TOP_P = 0.8
ALPHA_F = 0.0
ALPHA_P = 0.0
MAX_NEW_TOKENS = int(os.getenv("TC_TEST_MAX_NEW_TOKENS", "16"))

DEVICE = get_backend_device("torch", default="cpu") or "cpu"

@unittest.skipIf(torch is None, "torch is not installed")
class TestModel(unittest.TestCase):
    def setUp(self):
        # self.test_model = "unsloth/Llama-3.2-1B-Instruct"
        # self.test_model = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
        # self.test_model = "Qwen/Qwen2.5-7B-Instruct"
        self.test_model = "Qwen/Qwen2.5-3B-Instruct"
        repo = RepoCustom(self.test_model, backend="torch")

        config_path = repo.base_dir / "config.json"
        weight_files = list(repo.base_dir.glob("*.safetensors"))

        try:
            if config_path.exists() and weight_files:
                repo._load_configs()
                self.model_path = repo.base_dir
                self.model_config = repo.model_config.config
            else:
                self.model_path, self.model_config, _ = asyncio.run(repo.download())
                self.model_path = Path(self.model_path)
        except Exception as exc:
            self.skipTest(f"Unable to prepare model '{self.test_model}': {exc}")

        if not self.model_config:
            self.skipTest("Model config unavailable")

        shard = Shard(
            self.test_model,
            start_layer=0,
            end_layer=self.model_config["num_layers"],
            total_layers=self.model_config["num_layers"] + 1,
        )

        self.model = Model(self.model_config, shard)
        self.model.eval()

        try:
            load_safetensors(
                self.model,
                self.model_path,
                self.model_config,
                weight_device=DEVICE,
                use_tied=self.model_config.get("tie_word_embeddings", False),
            )
        except Exception as exc:
            self.skipTest(f"Unable to load safetensors for '{self.test_model}': {exc}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )

    def test_model_generate(self):
        user_prompt = "Tell me about the allegory of the cave"
        messages = [{"role": "user", "content": user_prompt}]
        temp_chat = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = self.tokenizer(temp_chat, return_tensors="np")

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)

        temp = TEMP if self.model_config.get("temperature") is None else self.model_config["temperature"]
        top_k = TOP_K if self.model_config.get("top_k") is None else self.model_config["top_k"]
        top_p = TOP_P if self.model_config.get("top_p") is None else self.model_config["top_p"]
        repetition_penalty = float(self.model_config.get("repetition_penalty") or 1.0)
        max_new_tokens = min(int(self.model_config.get("max_seq_len", MAX_NEW_TOKENS)), MAX_NEW_TOKENS)

        out_tokens = generate(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            alpha_f=ALPHA_F,
            alpha_p=ALPHA_P,
            repetition_penalty=repetition_penalty,
        )
        print(f"[User]: {user_prompt}")
        print(f"[Model]: {self.tokenizer.decode(out_tokens, skip_special_tokens=True)}")
        self.assertGreater(len(out_tokens), 0)


if __name__ == "__main__":
    unittest.main()
