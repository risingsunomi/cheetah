import unittest

import transformers as hf_transformers
import tinygrad as tg

from cheetah.models.llm.backend import get_backend_device
from cheetah.models.llm.tinygrad.model import Model
from cheetah.models.llm.tinygrad.helpers import load_safetensors, generate
from cheetah.models.shard import Shard
from cheetah.repos import RepoHuggingFace

TOP_K = 0
TEMP = 0.65
TOP_P = 0.0
ALPHA_F = 0.0
ALPHA_P = 0.0

DEVICE = get_backend_device("tinygrad", default="CPU") or "CPU"

class TestModel(unittest.TestCase):
    def setUp(self):
        # self.test_model = "Qwen/Qwen3-4B-Instruct-2507"
        self.test_model = "unsloth/Qwen3-0.6B"
        
        repo = RepoHuggingFace(self.test_model)
        self.model_path, self.model_config = repo.download()
        shard = Shard(
            self.test_model,
            start_layer=0,
            end_layer=self.model_config["num_layers"],
            total_layers=self.model_config["num_layers"]+1,
        )
        self.model = Model(self.model_config, shard)
        load_safetensors(
            self.model,
            self.model_path,
            self.model_config,
            weight_device=DEVICE,   # or "CUDA"/"CPU"
            use_tied=self.model_config["tie_word_embeddings"]
        )

        self.tokenizer = hf_transformers.AutoTokenizer.from_pretrained(
            self.test_model, local_files_only=True
        )

    def test_model_generate(self):
        user_prompt = "Tell me a funny short story"
        messages = [{"role": "user", "content": user_prompt}]
        temp_chat = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self.tokenizer(temp_chat, return_tensors="np")

        input_ids = tg.Tensor(enc["input_ids"]).to(DEVICE)
        attention_mask = tg.Tensor(enc["attention_mask"]).to(DEVICE)

        # run generation (prints tok/s when EOS or max tokens hit)
        max_new = 50
        out_tokens = generate(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new,
            temp=TEMP if self.model_config["temperature"] is None else self.model_config["temperature"],
            top_k=TOP_K if self.model_config["top_k"] is None else self.model_config["top_k"],
            top_p=TOP_P if self.model_config["top_p"] is None else self.model_config["top_p"],
            alpha_f=ALPHA_F,
            alpha_p=ALPHA_P,
        )

        model_reply = self.tokenizer.decode(out_tokens, skip_special_tokens=True)
        print(f"[User]: {user_prompt}")
        print(f"[Model]: {model_reply}")
