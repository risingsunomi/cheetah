import asyncio
from pathlib import Path
import time
import unittest

from transformers import AutoTokenizer

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from tiny_cheetah.models.llm.torch.model import Model
    from tiny_cheetah.models.llm.torch.helpers import sample
    from tiny_cheetah.models.llm.torch.quantize import (
        is_quantized_model_config,
        load_quantized_safetensors,
    )

from tiny_cheetah.models.llm.backend import get_backend_device
from tiny_cheetah.models.shard import Shard
from tiny_cheetah.repos import RepoCustom

TOP_K = 0
TEMP = 0.1
TOP_P = 0.9
ALPHA_F = 0.0
ALPHA_P = 0.0

DEVICE = get_backend_device("torch", default="mps") or "mps"

@unittest.skipIf(torch is None, "torch is not installed")
class TestQuantizedLlamaModel(unittest.TestCase):
    def setUp(self):
        self.test_model = "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit"

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
            self.skipTest(f"Unable to prepare quantized model '{self.test_model}': {exc}")

        if not self.model_config:
            self.skipTest("Quantized model config unavailable")

        self.assertTrue(
            is_quantized_model_config(self.model_config),
            f"Expected quantized config for {self.test_model}",
        )

        shard = Shard(
            self.test_model,
            start_layer=0,
            end_layer=self.model_config["num_layers"],
            total_layers=self.model_config["num_layers"] + 1,
        )

        self.model = Model(self.model_config, shard)
        self.model.eval()

        try:
            load_quantized_safetensors(
                self.model,
                self.model_path,
                self.model_config,
                weight_device=DEVICE,
                use_tied=self.model_config.get("tie_word_embeddings", False),
            )
        except Exception as exc:
            self.skipTest(f"Unable to load quantized safetensors for '{self.test_model}': {exc}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )

    def test_model_generate(self):
        user_prompt = "Tell me a funny short story"
        messages = [{"role": "user", "content": user_prompt}]
        temp_chat = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = self.tokenizer(temp_chat, return_tensors="np")

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)

        temp = TEMP
        top_k = TOP_K if self.model_config.get("top_k") is None else self.model_config["top_k"]
        top_p = TOP_P if self.model_config.get("top_p") is None else self.model_config["top_p"]
        alpha_f = ALPHA_F
        alpha_p = ALPHA_P
        max_new_tokens = self.model_config.get("max_seq_len", 64)

        self.model.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        curr_pos = int(input_ids.shape[1] - 1)

        out_tokens: list[int] = []
        print(f"[User]: {user_prompt}")
        print("[Model]:\n")

        position_ids = ((attention_mask.cumsum(dim=1) - 1) * attention_mask).to(
            device=DEVICE,
            dtype=torch.long,
        )

        logits = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        next_logit = logits[:, -1, :].flatten()
        tok = sample(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p)
        out_token = int(tok.item())
        out_tokens.append(out_token)

        t0 = time.time()
        generated = 1
        curr_pos += 1
        eos_hit = False

        print(self.tokenizer.decode(out_token, skip_special_tokens=True), end="", flush=True)

        limit = max_new_tokens - 1 if max_new_tokens > 0 else None

        while True:
            if out_token == self.tokenizer.eos_token_id:
                elapsed = time.time() - t0
                tok_s = generated / elapsed if elapsed > 0 else float("inf")
                print(f"\n[decode] {generated} tokens in {elapsed:.3f}s  ->  {tok_s:.2f} tok/s")
                eos_hit = True
                break

            if limit is not None and generated >= limit:
                break

            generated += 1
            next_tok = torch.tensor([[out_token]], device=DEVICE, dtype=torch.long)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=DEVICE, dtype=attention_mask.dtype),
                ),
                dim=1,
            )
            position_ids = torch.tensor([curr_pos], device=DEVICE, dtype=torch.long)

            logits = self.model(
                next_tok,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            next_logit = logits[:, -1, :].flatten()
            tok = sample(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p)
            out_token = int(tok.item())
            out_tokens.append(out_token)
            curr_pos += 1

            print(self.tokenizer.decode(out_token, skip_special_tokens=True), end="", flush=True)

        if not eos_hit:
            elapsed = time.time() - t0
            tok_s = generated / elapsed if elapsed > 0 else float("inf")
            print(f"\n[decode] {generated} tokens in {elapsed:.3f}s  ->  {tok_s:.2f} tok/s (no EOS)")

        self.assertGreater(len(out_tokens), 0)


if __name__ == "__main__":
    unittest.main()
