"""
Helpers for torch-based LLM models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional
import json
import time

import safetensors
import torch
from transformers import AutoTokenizer

from tiny_cheetah.logging_utils import get_logger
from tiny_cheetah.models.llm.backend import get_backend_device
from tiny_cheetah.models.shard import Shard

from .load_progress import WeightLoadProgress
from .model import Model
from .mamba_model import MambaModel
from .model_config import ModelConfig
from .quantize import is_quantized_model_config, load_quantized_safetensors

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

# From tinygrad helper loader, adapted for torch.
def permute(v: torch.Tensor, n_heads: int) -> torch.Tensor:
    return (
        v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1] if len(v.shape) > 1 else 1)
        .transpose(1, 2)
        .reshape(*v.shape[:2])
    )

def _is_float8_tensor(weight: torch.Tensor) -> bool:
    return weight.dtype in _FLOAT8_DTYPES

def _maybe_dequantize_fp8_weight(
    weights: safetensors.safe_open,
    model_weight_key: str,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not _is_float8_tensor(weight):
        return weight

    scale_key = f"{model_weight_key}_scale"
    if scale_key not in weights.keys():
        return weight.to(torch.float32)

    scale = weights.get_tensor(scale_key).to(torch.float32)
    return weight.to(torch.float32) * scale

def _load_weight_tensor(weight_file: Path, model_weight_key: str) -> torch.Tensor | None:
    with safetensors.safe_open(str(weight_file), framework="pt", device="cpu") as weights:
        if model_weight_key not in weights.keys():
            return None
        weight = weights.get_tensor(model_weight_key)
        return _maybe_dequantize_fp8_weight(weights, model_weight_key, weight)


def _infer_prefix(weight_map: dict[str, str]) -> str:
    for candidate in ("model", "base_model", "transformer", "gpt_neox", "blk", "backbone"):
        prefix = f"{candidate}."
        if any(key.startswith(prefix) for key in weight_map):
            return prefix
    return ""

def _resolve_weight_key(key: str, weight_map: dict[str, str], prefix: str) -> str | None:
    if key in weight_map:
        return key
    prefixed = f"{prefix}{key}" if prefix else key
    if prefixed in weight_map:
        return prefixed
    return None

def _needs_qk_permute(model_config: dict) -> bool:
    model_type = str(model_config.get("model_type", "")).lower()
    return model_type not in {"gpt_oss", "nemotron_h"}

def build_model(config: dict, shard: Shard, use_tied: bool = False) -> Model | MambaModel:
    if bool(config.get("mamba")) or str(config.get("model_type", "")).lower() == "nemotron_h":
        return MambaModel(config, shard, use_tied=use_tied)
    return Model(config, shard, use_tied=use_tied)

def apply_weight(
    model_weight_key: str,
    key: str,
    weight_map: dict[str, str],
    model_path: Path,
    weight_device: str | torch.device,
    model_state_dict: dict[str, torch.Tensor],
    model_config: dict,
) -> None:
    weight_file = model_path / weight_map[model_weight_key]
    weight = _load_weight_tensor(weight_file, model_weight_key)

    if weight is None:
        return

    if _needs_qk_permute(model_config) and "q_proj" in key:
        weight = permute(weight, model_config["num_heads"])
    elif _needs_qk_permute(model_config) and "k_proj" in key:
        weight = permute(weight, model_config["num_kv_heads"])

    target = model_state_dict.get(key)
    if target is not None:
        weight = weight.to(device=target.device, dtype=target.dtype)
    else:
        weight = weight.to(weight_device)

    model_state_dict[key] = weight

def load_safetensors(
    model: Any,
    model_path: Path,
    model_config: dict,
    weight_device: str | torch.device = "cpu",
    use_tied: bool = False,
) -> None:
    model_files = list(model_path.glob("*.safetensors"))
    if len(model_files) == 0:
        logger.error("No safetensor files found in model path %s", model_path)
        raise FileNotFoundError(f"No safetensor files found in model path {model_path}")

    weight_map: dict[str, str] = {}

    weight_map_json = model_path / "model.safetensors.index.json"
    if weight_map_json.exists():
        with weight_map_json.open("r") as handle:
            safe_index = json.load(handle)
            weight_map = dict(safe_index["weight_map"])
    else:
        # Fallback for repos that only contain shard files without an index json.
        for model_file in model_files:
            with safetensors.safe_open(str(model_file), framework="numpy") as weight_data:
                for key in weight_data.keys():
                    weight_map[key] = model_file.name

    model_state_dict = model.state_dict()
    prefix = _infer_prefix(weight_map)
    progress = WeightLoadProgress(total=len(model_state_dict), label="torch-load")

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
                apply_weight(
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
                apply_weight(
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
        apply_weight(
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
        logger.warning("Missing torch checkpoint keys: %s", incompatible.missing_keys)
    if incompatible.unexpected_keys:
        logger.warning("Unexpected torch checkpoint keys: %s", incompatible.unexpected_keys)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    seen_tokens: list[int] | tuple[int, ...] | None,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty <= 0.0 or abs(repetition_penalty - 1.0) < 1e-6 or not seen_tokens:
        return logits

    seen = torch.tensor(seen_tokens, device=logits.device, dtype=torch.long).unique()
    seen = seen[(seen >= 0) & (seen < logits.numel())]
    if seen.numel() == 0:
        return logits

    penalized = logits.index_select(0, seen)
    penalized = torch.where(
        penalized < 0,
        penalized * float(repetition_penalty),
        penalized / float(repetition_penalty),
    )
    return logits.clone().scatter_(0, seen, penalized)

# Standard openai-style sampling adapted for torch.
def sample(
    logits: torch.Tensor,
    temp: float = 1.0,
    k: Optional[int] = 0,
    p: Optional[float] = 0.8,
    af: Optional[float] = 0.0,
    ap: Optional[float] = 0.0,
    seen_tokens: list[int] | tuple[int, ...] | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    if logits.ndim != 1:
        raise ValueError("only works on 1d tensors")
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1")
    if not (0 <= k <= logits.numel()):
        raise ValueError("k must be between 0 and numel")

    logits = _apply_repetition_penalty(logits, seen_tokens, float(repetition_penalty))

    if temp < 1e-6:
        return torch.argmax(logits)

    logits = logits.to(device=logits.device)

    if af or ap:
        alpha_counter = getattr(sample, "alpha_counter", None)
        if (
            alpha_counter is None
            or alpha_counter.shape != logits.shape
            or alpha_counter.device != logits.device
        ):
            alpha_counter = torch.zeros_like(logits, dtype=torch.int32)
        logits = logits - (
            alpha_counter.to(dtype=logits.dtype) * float(af)
            + (alpha_counter > 0).to(dtype=logits.dtype) * float(ap)
        )
        sample.alpha_counter = alpha_counter

    logits = torch.nan_to_num(logits, nan=-float("inf"))
    probs = torch.softmax(logits / temp, dim=-1)

    if k and k > 0:
        probs, indices = torch.topk(probs, int(k), dim=-1)
    else:
        indices = torch.arange(probs.numel(), device=probs.device)

    if p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > float(p)
        if cutoff.any():
            cutoff[0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)

    prob_sum = probs.sum()
    if prob_sum <= 0:
        output_pos = torch.argmax(probs)
    else:
        probs = probs / prob_sum
        output_pos = torch.multinomial(probs, num_samples=1).squeeze(0)

    output_token = indices[output_pos]

    if af or ap:
        counter = torch.arange(sample.alpha_counter.numel(), device=sample.alpha_counter.device)
        sample.alpha_counter = torch.where(
            counter == output_token,
            sample.alpha_counter + 1,
            sample.alpha_counter,
        )

    return output_token

def generate(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int = 2048,
    temp: float = 1.0,
    top_k: Optional[int] = 0,
    top_p: Optional[float] = 0.8,
    alpha_f: Optional[float] = 0.0,
    alpha_p: Optional[float] = 0.0,
    repetition_penalty: float = 1.0,
    curr_pos: int = 0,
    verbose: bool = False,
    on_token: Callable[[int], None] | None = None,
) -> list[int]:
    device = get_backend_device("torch", default="cpu")
    assert device is not None

    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
    if hasattr(sample, "alpha_counter"):
        delattr(sample, "alpha_counter")

    input_ids = input_ids.to(device=device, dtype=torch.long)
    attention_mask = attention_mask.to(device=device)
    out_tokens: list[int] = []
    seen_tokens = input_ids.flatten().tolist()

    position_ids = ((attention_mask.cumsum(dim=1) - 1) * attention_mask).to(
        device=device,
        dtype=torch.long,
    )

    print("Generating")
    print("_________________________________\n\n")
    if verbose:
        print(
            f"input_ids: {input_ids.tolist()}\n"
            f"position_ids: {position_ids.tolist()}\n"
            f"attention_mask: {attention_mask.tolist()}"
        )

    with torch.inference_mode():
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        next_logit = logits[:, -1, :].flatten()
        tok_sample = sample(
            next_logit,
            temp=temp,
            k=top_k,
            p=top_p,
            af=alpha_f,
            ap=alpha_p,
            seen_tokens=seen_tokens,
            repetition_penalty=repetition_penalty,
        )
        tok = int(tok_sample.item())
        out_tokens.append(tok)
        seen_tokens.append(tok)
        if on_token is not None:
            on_token(tok)

        t0 = time.time()
        generated = 1
        if curr_pos <= 0:
            curr_pos = int(position_ids[:, -1].max().item()) + 1
        eos_hit = False

        while True:
            if tok == tokenizer.eos_token_id:
                elapsed = time.time() - t0
                tok_s = generated / elapsed if elapsed > 0 else float("inf")
                print(f"[decode] {generated} tokens in {elapsed:.3f}s  ->  {tok_s:.2f} tok/s")
                eos_hit = True
                break

            if max_new_tokens > 0 and generated >= max_new_tokens:
                break

            next_tok = torch.tensor([[tok]], device=device, dtype=torch.long)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device),
                ),
                dim=1,
            )
            position_ids = torch.tensor([curr_pos], device=device, dtype=torch.long)

            if verbose:
                print(
                    f"next_tok: {[int(next_tok.item())]}\n"
                    f" position_ids: {position_ids.tolist()}\n"
                    f" attention_mask_len: {attention_mask.shape[1]}"
                )

            logits = model(
                next_tok,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            next_logit = logits[:, -1, :].flatten()
            tok_sample = sample(
                next_logit,
                temp=temp,
                k=top_k,
                p=top_p,
                af=alpha_f,
                ap=alpha_p,
                seen_tokens=seen_tokens,
                repetition_penalty=repetition_penalty,
            )
            tok = int(tok_sample.item())
            out_tokens.append(tok)
            seen_tokens.append(tok)
            if on_token is not None:
                on_token(tok)
            generated += 1
            curr_pos += 1

    if not eos_hit:
        elapsed = time.time() - t0
        tok_s = generated / elapsed if elapsed > 0 else float("inf")
        if verbose:
            print(f"[decode] {generated} tokens in {elapsed:.3f}s  ->  {tok_s:.2f} tok/s (no EOS)")

    return out_tokens

def load_model_config(model_path: Path) -> Dict[str, Any]:
    model_config = ModelConfig()
    model_config_file = model_path / "config.json"
    if not model_config_file.exists():
        logger.error("Model config file not found at %s", model_config_file)
        raise FileNotFoundError(f"Model config file not found at {model_config_file}")

    model_config.load(model_config_file)
    gen_config = model_path / "generation_config.json"
    if gen_config.exists():
        model_config.load_generation_config(gen_config)

    return model_config.config

async def load_model(
    model_id: str,
    shard: Shard = None,
    weight_device: str | torch.device | None = None,
    offline_mode: bool = False,
    progress_callback: Callable[[str], Awaitable[None] | None] | None = None,
) -> tuple[Any, dict, AutoTokenizer, Path]:
    from tiny_cheetah.repos import RepoCustom

    sanitized = model_id.replace("/", "__")
    cache_path = (Path.home() / ".cache" / "tiny_cheetah_models") / sanitized
    candidate_path = Path(model_id).expanduser()

    resolved_path = None
    if candidate_path.exists():
        resolved_path = candidate_path
    elif cache_path.exists():
        resolved_path = cache_path

    model_repo = RepoCustom(model_id, backend="torch")
    if resolved_path is not None and any(resolved_path.glob("*.*")):
        model_config = load_model_config(resolved_path)
        model_path = resolved_path
    elif resolved_path is None and not offline_mode:
        logger.info(
            "No path resolved to model %s, creating a new path %s and downloading model",
            model_id,
            cache_path,
        )
        model_path = cache_path
        model_path, model_config, _ = await model_repo.download(progress_callback=progress_callback)
    elif offline_mode:
        logger.error("Model %s not found in offline mode", model_id)
        raise FileNotFoundError(f"Model {model_id} not found in offline mode")

    if shard is None:
        shard = Shard(
            model_name=model_id,
            start_layer=0,
            end_layer=model_config["num_layers"],
            total_layers=model_config["num_layers"] + 1,
        )

    if weight_device is None:
        weight_device = get_backend_device("torch", default="cpu")
        assert weight_device is not None

    model = build_model(
        model_config,
        shard,
        use_tied=model_config.get("tie_word_embeddings", False),
    )
    model.to(get_backend_device("torch", default="cpu"))
    if is_quantized_model_config(model_config):
        logger.info("Detected quantized model for %s, loading with torch NF4 loader.", model_id)
        load_quantized_safetensors(
            model,
            model_path,
            model_config,
            weight_device=weight_device,
            use_tied=model_config["tie_word_embeddings"],
        )
    else:
        load_safetensors(
            model,
            model_path,
            model_config,
            weight_device=weight_device,
            use_tied=model_config["tie_word_embeddings"],
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=offline_mode,
    )

    return model, model_config, tokenizer, model_path
