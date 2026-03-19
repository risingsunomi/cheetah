import asyncio
from pathlib import Path
import unittest
import time

from transformers import AutoTokenizer
import tinygrad as tg

from tiny_cheetah.models.llm.backend import get_backend_device
from tiny_cheetah.models.llm.tinygrad.model import Model
from tiny_cheetah.models.llm.tinygrad.helpers import sample
from tiny_cheetah.models.llm.tinygrad.quantize import is_quantized_model_config, load_quantized_safetensors
from tiny_cheetah.models.shard import Shard
from tiny_cheetah.repos import RepoCustom

TOP_K = 0
TEMP = 0.1
TOP_P = 0.9
ALPHA_F = 0.0
ALPHA_P = 0.0

DEVICE = get_backend_device("tinygrad", default="CPU") or "CPU"


class TestQuantizedLlamaModel(unittest.TestCase):
    def setUp(self):
        self.test_model = "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit"

        repo = RepoCustom(self.test_model)
        config_path = repo.base_dir / "config.json"
        weight_files = list(repo.base_dir.glob("*.safetensors"))
        if config_path.exists() and weight_files:
            repo._load_configs()
            self.model_path = repo.base_dir
            self.model_config = repo.model_config.config
        else:
            self.model_path, self.model_config, _ = asyncio.run(repo.download())
            self.model_path = Path(self.model_path)

        self.assertTrue(
            is_quantized_model_config(self.model_config),
            "Expected quantized config for unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit",
        )

        shard = Shard(
            self.test_model,
            start_layer=0,
            end_layer=self.model_config["num_layers"],
            total_layers=self.model_config["num_layers"] + 1,
        )
        self.model = Model(self.model_config, shard)
        load_quantized_safetensors(
            self.model,
            self.model_path,
            self.model_config,
            weight_device=DEVICE,
            use_tied=self.model_config["tie_word_embeddings"],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True
        )

    def test_model_generate(self):
        user_prompt = "Tell me a funny short story"
        messages = [{"role": "user", "content": user_prompt}]
        temp_chat = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self.tokenizer(temp_chat, return_tensors="np")

        input_ids = tg.Tensor(enc["input_ids"]).to(DEVICE)
        attention_mask = tg.Tensor(enc["attention_mask"]).to(DEVICE)
        temp = TEMP #if self.model_config["temperature"] is None else self.model_config["temperature"]
        top_k = TOP_K if self.model_config["top_k"] is None else self.model_config["top_k"]
        top_p = TOP_P if self.model_config["top_p"] is None else self.model_config["top_p"]
        alpha_f = ALPHA_F
        alpha_p = ALPHA_P
        max_new_tokens = 256
        curr_pos = input_ids.shape[1] - 1

        print(f"[User]: {user_prompt}")
        print(f"[Model]:\n")
        # run generation (prints tok/s when EOS or max tokens hit)
        
        # select device
        configured_device = get_backend_device("tinygrad", default=None)
        if configured_device is None:
            available_devices = tg.Device.DEFAULT.split(",")    
            if "METAL" in available_devices:
                device = "METAL"
            elif "AMD" in available_devices:
                device = "AMD"
            elif "CUDA" in available_devices:
                device = "CUDA"
            else:
                print(f"Using default CPU device")
                device = "CPU"
        else:
            device = configured_device

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        position_ids = ((attention_mask.cumsum(axis=1) - 1) * attention_mask).to(device) # [B, S]

        # get logits
        logits = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        ) # [B, S, V]

        # prefill
        next_logit = logits[:, -1, :].flatten() # [B, V]
        tok = sample(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p)
        out_token = tok.item()
        # toks/sec timer
        t0 = time.time()
        tok_s = 0
        generated = 1
        curr_pos += 1

        eos_hit = False

        out_token_dec = self.tokenizer.decode(out_token, skip_special_tokens=True)
        print(f"{out_token_dec}", end="", flush=True)

        # first token sampled; appended to out_tokens above

        limit = max_new_tokens - 1 if max_new_tokens > 0 else None

        while True:
            if out_token == self.tokenizer.eos_token_id:
                elapsed = time.time() - t0
                tok_s = generated / elapsed if elapsed > 0 else float("inf")
                eos_hit = True
                break

            if limit is not None and generated >= limit:
                break

            generated += 1

            next_tok = tg.Tensor([[out_token]], device=device)  # [B, 1]
            # grow attention mask and use absolute position for the new token
            attention_mask = attention_mask.cat(
                tg.Tensor.ones((attention_mask.shape[0], 1), device=device), dim=1
            )
            position_ids = tg.Tensor([curr_pos], device=device)

            logits = self.model(
                next_tok,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            next_logit = logits[:, -1, :].flatten()  # [B, V]
            tok = sample(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p)
            out_token = tok.item()
            curr_pos += 1

            out_token_dec = self.tokenizer.decode(out_token, skip_special_tokens=True)
            print(f"{out_token_dec}", end="", flush=True)

        if not eos_hit:
            elapsed = time.time() - t0
            tok_s = generated / elapsed if elapsed > 0 else float("inf")

        print(f"[decode] {generated} tokens in {elapsed:.3f}s  ->  {tok_s:.2f} tok/s")
