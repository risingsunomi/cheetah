from __future__ import annotations

from typing import Any, Mapping


def is_quantized_model_config(model_config: Mapping[str, Any]) -> bool:
    quantization_config = model_config.get("quantization_config")
    if not isinstance(quantization_config, Mapping):
        return False

    quant_method = str(quantization_config.get("quant_method", "")).lower()
    if quant_method in {"exl3", "bitsandbytes", "gptq", "awq", "hqq", "quanto"}:
        return True

    return bool(quantization_config)
