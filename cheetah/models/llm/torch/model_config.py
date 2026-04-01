from __future__ import annotations

from pathlib import Path
import json
import os

import torch

from cheetah.logging_utils import get_logger

logger = get_logger(__name__)


class ModelConfig:
    def __init__(self):
        self.config: dict = {}
        self.qk_norm_models = ["qwen3", "qwen3_moe"]

    @staticmethod
    def _decode_hybrid_layers(pattern: str, num_layers: int) -> list[str]:
        mapping = {"M": "mamba", "*": "attention", "-": "mlp"}
        if not pattern:
            return []
        decoded = [mapping.get(ch, "mlp") for ch in pattern]
        if num_layers > 0 and len(decoded) != num_layers:
            logger.warning(
                "Hybrid override pattern length %d does not match num_hidden_layers %d",
                len(decoded),
                num_layers,
            )
        return decoded

    def load(self, config_file: Path) -> None:
        with open(config_file, "r") as handle:
            base_config = json.loads(handle.read())

        hf_precision_str_to_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }

        model_type = str(base_config.get("model_type", "")).lower()
        default_attn_bias = base_config.get("attention_bias", base_config.get("attn_bias", False))
        qkv_bias = bool(base_config.get("qkv_bias", default_attn_bias))
        o_proj_bias = bool(base_config.get("o_proj_bias", base_config.get("attention_output_bias", qkv_bias)))

        if model_type in {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe"}:
            qkv_bias = True
            o_proj_bias = False

        num_hidden_layers = int(base_config.get("num_hidden_layers", 0) or 0)
        hybrid_override_pattern = str(base_config.get("hybrid_override_pattern", ""))
        layers_block_type = base_config.get("layers_block_type")
        if not isinstance(layers_block_type, list):
            layers_block_type = self._decode_hybrid_layers(hybrid_override_pattern, num_hidden_layers)

        self.config = {
            "architectures": base_config.get("architectures", []),
            "embed_dim": base_config["hidden_size"],
            "num_heads": base_config["num_attention_heads"],
            "head_dim": base_config.get(
                "head_dim",
                base_config["hidden_size"] // base_config["num_attention_heads"],
            ),
            "attention_head_dim": int(
                base_config.get(
                    "attention_head_dim",
                    base_config.get(
                        "head_dim",
                        base_config["hidden_size"] // base_config["num_attention_heads"],
                    ),
                )
            ),
            "num_kv_heads": base_config["num_key_value_heads"],
            "max_seq_len": base_config["max_position_embeddings"],
            "intermediate_dim": base_config["intermediate_size"],
            "attn_dropout": base_config.get("attention_dropout", 0.0),
            "norm_eps": base_config.get("rms_norm_eps", base_config.get("norm_eps", 1e-6)),
            "layer_norm_epsilon": base_config.get(
                "layer_norm_epsilon",
                base_config.get("rms_norm_eps", base_config.get("norm_eps", 1e-6)),
            ),
            "rope_scaling": base_config.get("rope_scaling", None),
            "rope_theta": base_config.get("rope_theta", 100000.0),
            "layer_types": list(base_config.get("layer_types", [])),
            "layers_block_type": list(layers_block_type),
            "hybrid_override_pattern": hybrid_override_pattern,
            "sliding_window": int(base_config.get("sliding_window", 0) or 0),
            "vocab_size": base_config.get("vocab_size", 0),
            "num_layers": num_hidden_layers,
            "attn_bias": qkv_bias,
            "qkv_bias": qkv_bias,
            "o_proj_bias": o_proj_bias,
            "mlp_bias": base_config.get("mlp_bias", False),
            "lm_head_bias": base_config.get("lm_head_bias", base_config.get("output_bias", False)),
            "hidden_act": base_config.get("hidden_act", "silu"),
            "mlp_hidden_act": base_config.get("mlp_hidden_act", base_config.get("hidden_act", "silu")),
            "initializer_range": float(base_config.get("initializer_range", 0.02)),
            "torch_dtype": hf_precision_str_to_dtype.get(
                base_config.get("torch_dtype", "bfloat16"),
                torch.bfloat16,
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
            "initial_context_length": int(base_config.get("initial_context_length", 0) or 0),
            "output_router_logits": bool(base_config.get("output_router_logits", False)),
            "router_aux_loss_coef": float(base_config.get("router_aux_loss_coef", 0.0)),
            "use_cache": bool(base_config.get("use_cache", True)),
            "transformers_version": str(base_config.get("transformers_version", "")),
            "quantization_config": base_config.get("quantization_config"),
            "use_bias": bool(base_config.get("use_bias", False)),
            "residual_in_fp32": bool(base_config.get("residual_in_fp32", False)),
            "rescale_prenorm_residual": bool(base_config.get("rescale_prenorm_residual", False)),
            "use_mamba_kernels": bool(base_config.get("use_mamba_kernels", False)),
            "ssm_state_size": int(base_config.get("ssm_state_size", 0) or 0),
            "mamba_num_heads": int(base_config.get("mamba_num_heads", 0) or 0),
            "n_groups": int(base_config.get("n_groups", 0) or 0),
            "mamba_head_dim": int(base_config.get("mamba_head_dim", 0) or 0),
            "conv_kernel": int(base_config.get("conv_kernel", 0) or 0),
            "expand": float(base_config.get("expand", 2.0) or 2.0),
            "mamba_hidden_act": str(base_config.get("mamba_hidden_act", base_config.get("hidden_act", "silu"))),
            "time_step_min": float(base_config.get("time_step_min", 0.001)),
            "time_step_max": float(base_config.get("time_step_max", 0.1)),
            "time_step_floor": float(base_config.get("time_step_floor", 1e-4)),
            "time_step_limit": tuple(base_config.get("time_step_limit", (0.0, float("inf")))),
            "use_conv_bias": bool(base_config.get("use_conv_bias", True)),
            "mamba_proj_bias": bool(base_config.get("mamba_proj_bias", False)),
            "chunk_size": int(base_config.get("chunk_size", 256) or 256),
            "temperature": None,
            "max_new_tokens": None,
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
        self.config["rope_truncate"] = True

        rope_scaling = self.config.get("rope_scaling")
        if rope_scaling is not None:
            self.config["rope_scaling_factor"] = rope_scaling.get("factor", rope_scaling.get("rope_factor", 0))
            self.config["rope_low_freq_factor"] = rope_scaling.get("low_freq_factor", rope_scaling.get("beta_fast", 0))
            self.config["rope_high_freq_factor"] = rope_scaling.get("high_freq_factor", rope_scaling.get("beta_slow", 0))
            self.config["rope_original_max_pos_embeddings"] = rope_scaling.get(
                "original_max_position_embeddings",
                rope_scaling.get("original_max_pos", 0),
            )
            self.config["rope_type"] = rope_scaling.get("rope_type", "llama3")
            self.config["rope_truncate"] = bool(rope_scaling.get("truncate", True))

        attn_qk_norm = base_config.get("attn_qk_norm")
        if isinstance(attn_qk_norm, bool):
            self.config["qk_norm"] = attn_qk_norm
        elif isinstance(attn_qk_norm, str):
            self.config["qk_norm"] = len(attn_qk_norm) > 0
        elif any(self.config.get("model_type", "").startswith(prefix) for prefix in self.qk_norm_models):
            self.config["qk_norm"] = True
        else:
            self.config["qk_norm"] = False

        if self.config.get("model_type") == "gpt_oss":
            # GPT-OSS exposes separate attention and expert bias tensors.
            self.config["attn_bias"] = True
            self.config["qkv_bias"] = True
            self.config["o_proj_bias"] = True
            self.config["mlp_bias"] = True

        self.config["moe"] = bool(
            self.config.get("num_local_experts", 0) > 0
            and self.config.get("num_experts_per_tok", 0) > 0
        )
        self.config["mamba"] = self.config.get("model_type") == "nemotron_h"

        custom_seq = os.getenv("TC_MAX_SEQ_LEN")
        if custom_seq is not None:
            self.config["max_seq_len"] = int(custom_seq)

        logger.info("Loaded model config: %s", self.config)

    def load_generation_config(self, gen_config_file: Path) -> None:
        with open(gen_config_file, "r") as handle:
            gen_config = json.loads(handle.read())

        if os.getenv("TC_TEMP") is not None:
            self.config["temperature"] = float(os.getenv("TC_TEMP"))
        elif "temperature" in gen_config:
            temp = gen_config["temperature"]
            self.config["temperature"] = None if temp is None else float(temp)

        if "max_new_tokens" in gen_config:
            max_new_tokens = gen_config["max_new_tokens"]
            self.config["max_new_tokens"] = None if max_new_tokens is None else int(max_new_tokens)

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

    def __str__(self) -> str:
        return str(self.config)
