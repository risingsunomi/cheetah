"""
Quantized checkpoint utilities for tinygrad models.

Current support:
- bitsandbytes 4-bit NF4 tensors saved in safetensors format
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np
import safetensors
import tinygrad as tg

from cheetah.logging_utils import get_logger

logger = get_logger(__name__)


def is_quantized_model_config(model_config: Mapping[str, Any]) -> bool:
    quantization_config = model_config.get("quantization_config")
    if not isinstance(quantization_config, Mapping):
        return False

    quant_method = str(quantization_config.get("quant_method", "")).lower()
    if quant_method in {"bitsandbytes", "gptq", "awq", "hqq", "quanto"}:
        return True

    return any(
        bool(quantization_config.get(flag))
        for flag in ("load_in_4bit", "_load_in_4bit", "load_in_8bit", "_load_in_8bit")
    )


def _permute(v: tg.Tensor, n_heads: int) -> tg.Tensor:
    return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1] if len(v.shape) > 1 else 1).transpose(1, 2).reshape(*v.shape[:2])


def _load_weight_map(model_path: Path) -> dict[str, str]:
    model_files = list(model_path.glob("*.safetensors"))
    if len(model_files) == 0:
        logger.error("No safetensor files found in model path %s", model_path)
        raise FileNotFoundError(f"No safetensor files found in model path {model_path}")

    weight_map_json = model_path / "model.safetensors.index.json"
    if weight_map_json.exists():
        with weight_map_json.open("r") as handle:
            safe_index = json.load(handle)
            return dict(safe_index["weight_map"])

    # Single-file checkpoint fallback.
    weight_map: dict[str, str] = {}
    with safetensors.safe_open(str(model_files[0]), framework="numpy") as weight_data:
        for key in weight_data.keys():
            weight_map[key] = model_files[0].name
    return weight_map

def _load_quant_state(raw_state: np.ndarray) -> dict[str, Any]:
    payload = raw_state.reshape(-1).astype(np.uint8, copy=False).tobytes()
    return json.loads(payload.decode("utf-8"))

def _dequantize_bnb_nf4(
    packed_weight: np.ndarray,
    packed_absmax: np.ndarray,
    quant_map: np.ndarray,
    nested_absmax: np.ndarray,
    nested_quant_map: np.ndarray,
    quant_state: Mapping[str, Any],
) -> np.ndarray:
    shape = tuple(int(x) for x in quant_state.get("shape", []))
    blocksize = int(quant_state.get("blocksize", 64))
    nested_blocksize = int(quant_state.get("nested_blocksize", 256))
    nested_offset = float(quant_state.get("nested_offset", 0.0))

    if blocksize <= 0 or nested_blocksize <= 0:
        raise ValueError("Invalid quantization blocksize in quant_state.")

    packed_weight = packed_weight.reshape(-1).astype(np.uint8, copy=False)
    packed_absmax = packed_absmax.reshape(-1).astype(np.uint8, copy=False)
    quant_map = quant_map.reshape(-1).astype(np.float32, copy=False)
    nested_absmax = nested_absmax.reshape(-1).astype(np.float32, copy=False)
    nested_quant_map = nested_quant_map.reshape(-1).astype(np.float32, copy=False)

    if quant_map.size != 16:
        raise ValueError(f"Expected NF4 quant_map with 16 entries, got {quant_map.size}.")

    # Unpack two 4-bit values per byte.
    unpacked = np.empty(packed_weight.size * 2, dtype=np.uint8)
    # bitsandbytes packs first value in high nibble, second value in low nibble
    unpacked[0::2] = (packed_weight >> 4) & 0x0F
    unpacked[1::2] = packed_weight & 0x0F

    total_values = unpacked.size
    if total_values != packed_absmax.size * blocksize:
        raise ValueError(
            "Packed weight and absmax sizes are inconsistent: "
            f"{total_values} values vs {packed_absmax.size}*{blocksize}."
        )

    nested_block_index = np.arange(packed_absmax.size, dtype=np.int64) // nested_blocksize
    if nested_block_index.max(initial=0) >= nested_absmax.size:
        raise ValueError("Nested absmax buffer is too small for packed absmax values.")

    absmax = nested_quant_map[packed_absmax.astype(np.int64)] * nested_absmax[nested_block_index] + nested_offset

    value_index = unpacked.astype(np.int64)
    block_index = np.arange(total_values, dtype=np.int64) // blocksize
    values = quant_map[value_index] * absmax[block_index]

    if shape:
        values = values.reshape(shape)
    return values.astype(np.float32, copy=False)


def _dequantize_bnb_nf4_simple(
    packed_weight: np.ndarray,
    absmax: np.ndarray,
    quant_map: np.ndarray,
    quant_state: Mapping[str, Any],
) -> np.ndarray:
    shape = tuple(int(x) for x in quant_state.get("shape", []))
    blocksize = int(quant_state.get("blocksize", 64))
    if blocksize <= 0:
        raise ValueError("Invalid quantization blocksize in quant_state.")

    packed_weight = packed_weight.reshape(-1).astype(np.uint8, copy=False)
    absmax = absmax.reshape(-1).astype(np.float32, copy=False)
    quant_map = quant_map.reshape(-1).astype(np.float32, copy=False)
    if quant_map.size != 16:
        raise ValueError(f"Expected NF4 quant_map with 16 entries, got {quant_map.size}.")

    unpacked = np.empty(packed_weight.size * 2, dtype=np.uint8)
    # bitsandbytes packs first value in high nibble, second value in low nibble
    unpacked[0::2] = (packed_weight >> 4) & 0x0F
    unpacked[1::2] = packed_weight & 0x0F

    total_values = unpacked.size
    if total_values != absmax.size * blocksize:
        raise ValueError(
            "Packed weight and absmax sizes are inconsistent: "
            f"{total_values} values vs {absmax.size}*{blocksize}."
        )

    block_index = np.arange(total_values, dtype=np.int64) // blocksize
    values = quant_map[unpacked.astype(np.int64)] * absmax[block_index]
    if shape:
        values = values.reshape(shape)
    return values.astype(np.float32, copy=False)


def _load_weight_numpy(weight_file: Path, model_weight_key: str) -> np.ndarray | None:
    with safetensors.safe_open(str(weight_file), framework="numpy") as weights:
        if model_weight_key not in weights.keys():
            raise KeyError(f"Missing tensor '{model_weight_key}' in {weight_file}")

        state_key = f"{model_weight_key}.quant_state.bitsandbytes__nf4"
        if state_key not in weights.keys():
            return None

        raw_weight = weights.get_tensor(model_weight_key)

        absmax_key = f"{model_weight_key}.absmax"
        qmap_key = f"{model_weight_key}.quant_map"
        quant_state = _load_quant_state(weights.get_tensor(state_key))
        quant_type = str(quant_state.get("quant_type", ""))
        if quant_type.lower() != "nf4":
            raise ValueError(f"Unsupported bitsandbytes quant_type '{quant_type}' for {model_weight_key}")

        nested_absmax_key = f"{model_weight_key}.nested_absmax"
        nested_qmap_key = f"{model_weight_key}.nested_quant_map"
        if nested_absmax_key in weights.keys() and nested_qmap_key in weights.keys():
            return _dequantize_bnb_nf4(
                packed_weight=raw_weight,
                packed_absmax=weights.get_tensor(absmax_key),
                quant_map=weights.get_tensor(qmap_key),
                nested_absmax=weights.get_tensor(nested_absmax_key),
                nested_quant_map=weights.get_tensor(nested_qmap_key),
                quant_state=quant_state,
            )

        return _dequantize_bnb_nf4_simple(
            packed_weight=raw_weight,
            absmax=weights.get_tensor(absmax_key),
            quant_map=weights.get_tensor(qmap_key),
            quant_state=quant_state,
        )


def _apply_weight(
    model_weight_key: str,
    key: str,
    weight_map: Mapping[str, str],
    model_path: Path,
    weight_device: str,
    model_state_dict: dict[str, tg.Tensor],
    model_config: Mapping[str, Any],
) -> None:
    weight_file = model_path / weight_map[model_weight_key]
    weight_np = _load_weight_numpy(weight_file, model_weight_key)
    if weight_np is None:
        weight = tg.nn.state.safe_load(str(weight_file)).get(model_weight_key)
        if weight is None:
            raise KeyError(f"Missing tensor '{model_weight_key}' in {weight_file}")
        weight = weight.to(weight_device)
    else:
        weight = tg.Tensor(weight_np).to(weight_device)

    if tg.dtypes.bfloat16:
        # Keep behavior aligned with existing unquantized loader.
        weight = weight.cast(tg.dtypes.float32).cast(tg.dtypes.float16)

    if "q_proj" in key:
        weight = _permute(weight, model_config["num_heads"])
    elif "k_proj" in key:
        weight = _permute(weight, model_config["num_kv_heads"])

    model_state_dict[key] = weight


def load_quantized_safetensors(
    model: Any,
    model_path: Path,
    model_config: Mapping[str, Any],
    weight_device: str = "CPU",
    use_tied: bool = False,
) -> None:
    weight_map = _load_weight_map(model_path)
    model_state_dict = tg.nn.state.get_state_dict(model)

    first_weight_key = next(iter(weight_map.keys()))
    prefix_check = first_weight_key.split(".")[0]
    if prefix_check in {"model", "base_model", "transformer", "gpt_neox", "blk"}:
        prefix = prefix_check + "."
    else:
        prefix = ""

    for key in model_state_dict.keys():
        model_weight_key = prefix + key
        if model_weight_key not in weight_map:
            if use_tied and key == "output.weight":
                embed_weight_key = "model.embed_tokens.weight"
                _apply_weight(
                    embed_weight_key,
                    key,
                    weight_map,
                    model_path,
                    weight_device,
                    model_state_dict,
                    model_config,
                )
                continue

            if key == "output.weight" and "lm_head.weight" in weight_map:
                _apply_weight(
                    "lm_head.weight",
                    key,
                    weight_map,
                    model_path,
                    weight_device,
                    model_state_dict,
                    model_config,
                )
            continue

        _apply_weight(
            model_weight_key,
            key,
            weight_map,
            model_path,
            weight_device,
            model_state_dict,
            model_config,
        )

    tg.nn.state.load_state_dict(model, model_state_dict)
