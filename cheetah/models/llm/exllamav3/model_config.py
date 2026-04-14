from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _read_nested(config: dict[str, Any], key: str) -> Any:
    value: Any = config
    for part in key.split("->"):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _read_first(config: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = _read_nested(config, key)
        if value is not None:
            return value
    return default


class ModelConfig:
    """Minimal HF config loader for exllamav3-backed chat inference."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}

    def load(self, config_file: Path) -> None:
        with config_file.open("r", encoding="utf-8") as handle:
            base_config = json.load(handle)

        max_seq_len = _read_first(
            base_config,
            [
                "max_position_embeddings",
                "text_config->max_position_embeddings",
                "max_seq_len",
            ],
            2048,
        )
        try:
            max_seq_len = int(max_seq_len)
        except (TypeError, ValueError):
            max_seq_len = 2048

        custom_seq = os.getenv("TC_MAX_SEQ_LEN")
        if custom_seq is not None:
            max_seq_len = int(custom_seq)

        self.config = {
            "architectures": list(base_config.get("architectures", [])),
            "model_type": str(
                _read_first(
                    base_config,
                    [
                        "model_type",
                        "text_config->model_type",
                    ],
                    "",
                )
                or ""
            ).lower(),
            "max_seq_len": max(256, max_seq_len),
            "num_layers": int(
                _read_first(
                    base_config,
                    [
                        "num_hidden_layers",
                        "text_config->num_hidden_layers",
                    ],
                    0,
                )
                or 0
            ),
            "quantization_config": _read_first(
                base_config,
                [
                    "quantization_config",
                    "text_config->quantization_config",
                ],
            ),
            "temperature": None,
            "max_new_tokens": None,
            "top_k": None,
            "top_p": None,
            "repetition_penalty": None,
            "eos_token_id": _read_first(
                base_config,
                [
                    "eos_token_id",
                    "text_config->eos_token_id",
                ],
            ),
            "pad_token_id": _read_first(
                base_config,
                [
                    "pad_token_id",
                    "text_config->pad_token_id",
                ],
            ),
            "bos_token_id": _read_first(
                base_config,
                [
                    "bos_token_id",
                    "text_config->bos_token_id",
                ],
            ),
        }

    def load_generation_config(self, gen_config_file: Path) -> None:
        with gen_config_file.open("r", encoding="utf-8") as handle:
            gen_config = json.load(handle)

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

        if "eos_token_id" in gen_config:
            self.config["eos_token_id"] = gen_config.get("eos_token_id")
        if "pad_token_id" in gen_config:
            self.config["pad_token_id"] = gen_config.get("pad_token_id")
        if "bos_token_id" in gen_config:
            self.config["bos_token_id"] = gen_config.get("bos_token_id")

    def __str__(self) -> str:
        return str(self.config)
