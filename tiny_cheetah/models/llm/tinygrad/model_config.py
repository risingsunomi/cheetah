from __future__ import annotations

import json
import os
from pathlib import Path

from tinygrad import dtypes

from tiny_cheetah.logging_utils import get_logger

logger = get_logger(__name__)


class ModelConfig:
    def __init__(self):
        self.config: dict = {}
        self.qk_norm_models = ["qwen3", "qwen3_moe"]

    def load(self, config_file: Path):
        with open(config_file, "r") as handle:
            base_config = json.loads(handle.read())

        hf_precision_str_to_dtype = {
            "float16": dtypes.float16,
            "bfloat16": dtypes.bfloat16,
            "float32": dtypes.float32,
            "float64": dtypes.float64,
            "int8": dtypes.int8,
            "int16": dtypes.int16,
            "int32": dtypes.int32,
            "int64": dtypes.int64,
            "uint8": dtypes.uint8,
            "uint16": dtypes.uint16,
            "uint32": dtypes.uint32,
            "uint64": dtypes.uint64,
            "bool": dtypes.bool,
        }

        model_type = str(base_config.get("model_type", "")).lower()
        default_attn_bias = base_config.get("attention_bias", base_config.get("attn_bias", False))
        qkv_bias = bool(base_config.get("qkv_bias", default_attn_bias))
        o_proj_bias = bool(base_config.get("o_proj_bias", base_config.get("attention_output_bias", qkv_bias)))

        if model_type in {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe"}:
            qkv_bias = True
            o_proj_bias = False

        self.config = {
            "architectures": base_config.get("architectures", []),
            "embed_dim": base_config["hidden_size"],
            "num_heads": base_config["num_attention_heads"],
            "head_dim": base_config.get(
                "head_dim",
                base_config["hidden_size"] // base_config["num_attention_heads"],
            ),
            "num_kv_heads": base_config["num_key_value_heads"],
            "max_seq_len": base_config["max_position_embeddings"],
            "intermediate_dim": base_config["intermediate_size"],
            "attn_dropout": base_config.get("attention_dropout", 0.0),
            "norm_eps": base_config.get("rms_norm_eps", base_config.get("norm_eps", 1e-6)),
            "rope_scaling": base_config.get("rope_scaling", None),
            "rope_theta": base_config.get("rope_theta", 100000.0),
            "layer_types": list(base_config.get("layer_types", [])),
            "sliding_window": int(base_config.get("sliding_window", 0) or 0),
            "vocab_size": base_config.get("vocab_size", 0),
            "num_layers": base_config.get("num_hidden_layers", 0),
            "attn_bias": qkv_bias,
            "qkv_bias": qkv_bias,
            "o_proj_bias": o_proj_bias,
            "mlp_bias": base_config.get("mlp_bias", False),
            "lm_head_bias": base_config.get("lm_head_bias", base_config.get("output_bias", False)),
            "hidden_act": base_config.get("hidden_act", "silu"),
            "tinygrad_dtype": hf_precision_str_to_dtype.get(
                base_config.get("torch_dtype", "bfloat16"),
                dtypes.bfloat16,
            ),
            "tie_word_embeddings": base_config.get("tie_word_embeddings", False),
            "model_type": model_type,
            "num_local_experts": int(base_config.get("num_local_experts", base_config.get("num_experts", 0)) or 0),
            "experts_per_token": int(
                base_config.get("experts_per_token", base_config.get("num_experts_per_tok", 0)) or 0
            ),
            "num_experts_per_tok": int(
                base_config.get("num_experts_per_tok", base_config.get("experts_per_token", 0)) or 0
            ),
            "swiglu_limit": float(base_config.get("swiglu_limit", 7.0)),
            "quantization_config": base_config.get("quantization_config"),
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "repetition_penalty": None,
            "eos_token_id": base_config.get("eos_token_id"),
            "pad_token_id": base_config.get("pad_token_id"),
            "bos_token_id": base_config.get("bos_token_id"),
        }

        self.config["rope_scaling_factor"] = None
        self.config["rope_low_freq_factor"] = 0.0
        self.config["rope_high_freq_factor"] = 0.0
        self.config["rope_original_max_pos_embeddings"] = 0
        self.config["rope_type"] = "default"

        rope_scaling = self.config.get("rope_scaling")
        if rope_scaling is not None:
            self.config["rope_scaling_factor"] = rope_scaling.get("factor", rope_scaling.get("rope_factor", 0))
            self.config["rope_low_freq_factor"] = rope_scaling.get(
                "low_freq_factor",
                rope_scaling.get("beta_fast", 0),
            )
            self.config["rope_high_freq_factor"] = rope_scaling.get(
                "high_freq_factor",
                rope_scaling.get("beta_slow", 0),
            )
            self.config["rope_original_max_pos_embeddings"] = rope_scaling.get(
                "original_max_position_embeddings",
                rope_scaling.get("original_max_pos", 0),
            )
            self.config["rope_type"] = rope_scaling.get("rope_type", "llama3")

        attn_qk_norm = base_config.get("attn_qk_norm")
        if isinstance(attn_qk_norm, bool):
            self.config["qk_norm"] = attn_qk_norm
        elif isinstance(attn_qk_norm, str):
            self.config["qk_norm"] = len(attn_qk_norm) > 0
        elif any(self.config.get("model_type", "").startswith(prefix) for prefix in self.qk_norm_models):
            self.config["qk_norm"] = True
        else:
            self.config["qk_norm"] = False

        self.config["moe"] = bool(
            self.config.get("num_local_experts", 0) > 0
            and self.config.get("num_experts_per_tok", 0) > 0
        )

        custom_seq = os.getenv("TC_MAX_SEQ_LEN")
        if custom_seq is not None:
            self.config["max_seq_len"] = int(custom_seq)

        logger.info("Loaded model config: %s", self.config)

    def load_generation_config(self, gen_config_file: Path):
        with open(gen_config_file, "r") as handle:
            gen_config = json.loads(handle.read())

        if os.getenv("TC_TEMP") is not None:
            self.config["temperature"] = float(os.getenv("TC_TEMP"))
        elif "temperature" in gen_config:
            temp = gen_config["temperature"]
            self.config["temperature"] = None if temp is None else float(temp)

        if os.getenv("TC_TOP_K") is not None:
            self.config["top_k"] = int(os.getenv("TC_TOP_K"))
        elif "top_k" in gen_config:
            top_k = gen_config["top_k"]
            self.config["top_k"] = None if top_k is None else int(top_k)

        if os.getenv("TC_TOP_P") is not None:
            self.config["top_p"] = float(os.getenv("TC_TOP_P"))
        elif "top_p" in gen_config:
            top_p = gen_config["top_p"]
            self.config["top_p"] = None if top_p is None else float(top_p)

        if os.getenv("TC_REPETITION_PENALTY") is not None:
            self.config["repetition_penalty"] = float(os.getenv("TC_REPETITION_PENALTY"))
        elif "repetition_penalty" in gen_config:
            repetition_penalty = gen_config["repetition_penalty"]
            self.config["repetition_penalty"] = (
                None if repetition_penalty is None else float(repetition_penalty)
            )

        self.config["eos_token_id"] = gen_config.get("eos_token_id")
        self.config["pad_token_id"] = gen_config.get("pad_token_id")
        self.config["bos_token_id"] = gen_config.get("bos_token_id")

    def __str__(self):
        return str(self.config)
