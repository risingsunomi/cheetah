import asyncio
import os
from pathlib import Path
import time
import unittest

from transformers import AutoTokenizer

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from cheetah.models.llm.torch.helpers import load_safetensors, sample
    from cheetah.models.llm.torch.model import Model

from cheetah.models.shard import Shard
from cheetah.repos import RepoCustom

TOP_K = 20
TEMP = 1.0
TOP_P = 0.0
ALPHA_F = 0.0
ALPHA_P = 0.0
MAX_NEW_TOKENS = 128

DEVICE = "mps"
# DEVICE = "cpu"

HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"
HARMONY_STOP_MARKERS = ("<|end|>", "<|return|>", "<|call|>", "<|channel|>")


def _extract_harmony_final(raw_text: str) -> str | None:
    marker_idx = raw_text.rfind(HARMONY_FINAL_MARKER)
    if marker_idx < 0:
        return None

    content = raw_text[marker_idx + len(HARMONY_FINAL_MARKER) :]
    stop_idx = len(content)
    for marker in HARMONY_STOP_MARKERS:
        idx = content.find(marker)
        if idx >= 0:
            stop_idx = min(stop_idx, idx)
    return content[:stop_idx].strip() or None


@unittest.skipIf(torch is None, "torch is not installed")
class TestGptOssModel(unittest.TestCase):
    def setUp(self):
        self.test_model = "openai/gpt-oss-20b"
        repo = RepoCustom(self.test_model, backend="torch")

        config_path = repo.base_dir / "config.json"
        chat_template_path = repo.base_dir / "chat_template.jinja"
        weight_files = list(repo.base_dir.glob("*.safetensors"))

        try:
            if config_path.exists() and weight_files and chat_template_path.exists():
                repo._load_configs()
                self.model_path = repo.base_dir
                self.model_config = repo.model_config.config
            else:
                self.model_path, self.model_config, _ = asyncio.run(
                    repo.download(extra_files=["chat_template.jinja"])
                )
                self.model_path = Path(self.model_path)
        except Exception as exc:
            self.skipTest(f"Unable to prepare model '{self.test_model}': {exc}")

        if not self.model_config:
            self.skipTest("Model config unavailable")
        if str(self.model_config.get("model_type", "")).lower() != "gpt_oss":
            self.skipTest(f"Expected gpt_oss model_type, got: {self.model_config.get('model_type')}")
        self.assertTrue(self.model_config.get("moe"), "Expected torch GPT-OSS config to enable MoE")

        shard = Shard(
            self.test_model,
            start_layer=0,
            end_layer=self.model_config["num_layers"],
            total_layers=self.model_config["num_layers"] + 1,
        )
        use_tied = bool(self.model_config.get("tie_word_embeddings", False))

        self.model = Model(self.model_config, shard, use_tied=use_tied)
        self.model.eval()

        try:
            load_safetensors(
                self.model,
                self.model_path,
                self.model_config,
                weight_device=DEVICE,
                use_tied=use_tied,
            )
        except Exception as exc:
            self.skipTest(f"Unable to load safetensors for '{self.test_model}': {exc}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
        except Exception as exc:
            self.skipTest(f"Unable to load tokenizer for '{self.test_model}': {exc}")

    def test_model_generate(self):
        user_prompt = "What is the capital of the USA?"
        messages = [{"role": "user", "content": user_prompt}]

        try:
            temp_chat = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                reasoning_effort="low",
            )
        except Exception as exc:
            self.skipTest(f"GPT-OSS tokenizer chat template unavailable: {exc}")

        enc = self.tokenizer(temp_chat, return_tensors="np")

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)

        temp = TEMP if self.model_config.get("temperature") is None else self.model_config["temperature"]
        top_k = TOP_K if self.model_config.get("top_k") is None else self.model_config["top_k"]
        top_p = TOP_P if self.model_config.get("top_p") is None else self.model_config["top_p"]
        alpha_f = ALPHA_F
        alpha_p = ALPHA_P
        max_new_tokens = MAX_NEW_TOKENS

        print(f"Generating with temperature={temp}, top_k={top_k}, top_p={top_p}, max_new_tokens={max_new_tokens}")

        try:
            self.model.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
        except Exception as exc:
            self.skipTest(f"Unable to move tensors/model to device '{DEVICE}': {exc}")

        curr_pos = int(input_ids.shape[1] - 1)

        out_tokens: list[int] = []
        raw_token_text: list[str] = []
        print(f"[User]: {user_prompt}")
        print("[Model]:\n")

        position_ids = ((attention_mask.cumsum(dim=1) - 1) * attention_mask).to(
            device=DEVICE,
            dtype=torch.long,
        )

        with torch.inference_mode():
            try:
                logits = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or "mps backend out of memory" in msg:
                    self.skipTest(f"Skipped due to memory pressure during prefill: {exc}")
                raise

            next_logit = logits[:, -1, :].flatten()
            tok = sample(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p)
            out_token = int(tok.item())
            out_tokens.append(out_token)
            raw_token_text.append(self.tokenizer.decode([out_token], skip_special_tokens=False))

            t0 = time.time()
            generated = 1
            curr_pos += 1
            eos_hit = False

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

                try:
                    logits = self.model(
                        next_tok,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "out of memory" in msg or "mps backend out of memory" in msg:
                        self.skipTest(f"Skipped due to memory pressure during decode: {exc}")
                    raise

                next_logit = logits[:, -1, :].flatten()
                tok = sample(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p)
                out_token = int(tok.item())
                out_tokens.append(out_token)
                raw_token_text.append(self.tokenizer.decode([out_token], skip_special_tokens=False))
                curr_pos += 1

        if not eos_hit:
            elapsed = time.time() - t0
            tok_s = generated / elapsed if elapsed > 0 else float("inf")
            print(f"[decode] {generated} tokens in {elapsed:.3f}s  ->  {tok_s:.2f} tok/s (no EOS)")

        raw_text = "".join(raw_token_text)
        final_text = _extract_harmony_final(raw_text)
        if final_text is not None:
            print(final_text)
        else:
            print(self.tokenizer.decode(out_tokens, skip_special_tokens=True))
            print(f"[raw-harmony] {raw_text}")

        self.assertGreater(len(out_tokens), 0)


if __name__ == "__main__":
    unittest.main()
