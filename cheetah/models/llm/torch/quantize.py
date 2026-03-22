"""
Quantized checkpoint utilities for torch models.

Current support:
- bitsandbytes 4-bit NF4 tensors saved in safetensors format
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np
import safetensors
import torch

from cheetah.logging_utils import get_logger

from .load_progress import WeightLoadProgress

logger = get_logger(__name__)

_FLOAT8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)


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


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        normalized = device.strip().lower()
        if normalized in {"cuda", "gpu"} and torch.cuda.is_available():
            return torch.device("cuda")
        if normalized in {"mps", "metal"} and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _permute(v: torch.Tensor, n_heads: int) -> torch.Tensor:
    return (
        v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1] if len(v.shape) > 1 else 1)
        .transpose(1, 2)
        .reshape(*v.shape[:2])
    )


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

    weight_map: dict[str, str] = {}
    for model_file in model_files:
        with safetensors.safe_open(str(model_file), framework="numpy") as weight_data:
            for key in weight_data.keys():
                weight_map[key] = model_file.name
    return weight_map


def _infer_prefix(weight_map: Mapping[str, str]) -> str:
    for candidate in ("model", "base_model", "transformer", "gpt_neox", "blk", "backbone"):
        prefix = f"{candidate}."
        if any(key.startswith(prefix) for key in weight_map):
            return prefix
    return ""


def _resolve_weight_key(key: str, weight_map: Mapping[str, str], prefix: str) -> str | None:
    if key in weight_map:
        return key
    prefixed = f"{prefix}{key}" if prefix else key
    if prefixed in weight_map:
        return prefixed
    return None


def _needs_qk_permute(model_config: Mapping[str, Any]) -> bool:
    model_type = str(model_config.get("model_type", "")).lower()
    return model_type not in {"gpt_oss", "nemotron_h"}


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

    unpacked = np.empty(packed_weight.size * 2, dtype=np.uint8)
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


def _load_weight_tensor(weight_file: Path, model_weight_key: str) -> torch.Tensor:
    with safetensors.safe_open(str(weight_file), framework="pt", device="cpu") as weights:
        if model_weight_key not in weights.keys():
            raise KeyError(f"Missing tensor '{model_weight_key}' in {weight_file}")
        weight = weights.get_tensor(model_weight_key)
        if weight.dtype in _FLOAT8_DTYPES:
            scale_key = f"{model_weight_key}_scale"
            if scale_key in weights.keys():
                scale = weights.get_tensor(scale_key).to(torch.float32)
                return weight.to(torch.float32) * scale
            return weight.to(torch.float32)
        return weight


def _apply_weight(
    model_weight_key: str,
    key: str,
    weight_map: Mapping[str, str],
    model_path: Path,
    weight_device: str | torch.device,
    model_state_dict: dict[str, torch.Tensor],
    model_config: Mapping[str, Any],
) -> None:
    weight_file = model_path / weight_map[model_weight_key]
    weight_np = _load_weight_numpy(weight_file, model_weight_key)
    if weight_np is None:
        weight = _load_weight_tensor(weight_file, model_weight_key)
    else:
        weight = torch.from_numpy(weight_np)

    if _needs_qk_permute(model_config) and "q_proj" in key:
        weight = _permute(weight, model_config["num_heads"])
    elif _needs_qk_permute(model_config) and "k_proj" in key:
        weight = _permute(weight, model_config["num_kv_heads"])

    target = model_state_dict.get(key)
    if target is not None:
        weight = weight.to(device=target.device, dtype=target.dtype)
    else:
        weight = weight.to(device=_resolve_device(weight_device))

    model_state_dict[key] = weight


def load_quantized_safetensors(
    model: Any,
    model_path: Path,
    model_config: Mapping[str, Any],
    weight_device: str | torch.device = "cpu",
    use_tied: bool = False,
) -> None:
    weight_map = _load_weight_map(model_path)
    model_state_dict = model.state_dict()

    prefix = _infer_prefix(weight_map)
    progress = WeightLoadProgress(total=len(model_state_dict), label="torch-qload")

    for idx, key in enumerate(model_state_dict.keys(), start=1):
        model_weight_key = _resolve_weight_key(key, weight_map, prefix)
        if model_weight_key not in weight_map:
            if use_tied and key == "output.weight":
                embed_weight_key = "model.embed_tokens.weight"
                progress.update(
                    idx,
                    key,
                    embed_weight_key,
                    weight_map.get(embed_weight_key),
                )
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
                progress.update(
                    idx,
                    key,
                    "lm_head.weight",
                    weight_map.get("lm_head.weight"),
                )
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

        progress.update(
            idx,
            key,
            model_weight_key,
            weight_map.get(model_weight_key),
        )
        _apply_weight(
            model_weight_key,
            key,
            weight_map,
            model_path,
            weight_device,
            model_state_dict,
            model_config,
        )

    incompatible = model.load_state_dict(model_state_dict, strict=False)
    progress.done()
    if incompatible.missing_keys:
        logger.warning("Missing quantized torch checkpoint keys: %s", incompatible.missing_keys)
    if incompatible.unexpected_keys:
        logger.warning("Unexpected quantized torch checkpoint keys: %s", incompatible.unexpected_keys)
