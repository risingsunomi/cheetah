"""
Helpers for LLM models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional, Any
import json
import os, time
import numpy as np

import safetensors
import tinygrad as tg
from transformers import AutoTokenizer

from cheetah.models.llm.backend import get_backend_device
from .model import Model
from .model_config import ModelConfig
from .quantize import is_quantized_model_config, load_quantized_safetensors
from cheetah.models.shard import Shard
from cheetah.logging_utils import get_logger

logger = get_logger(__name__)


# From https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py#L204C3-L205C138
# permute for weights using huggingface
def permute(v: tg.Tensor, n_heads: int):
    return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1] if len(v.shape) > 1 else 1).transpose(1, 2).reshape(*v.shape[:2])


def _infer_prefix(weight_map: dict[str, str]) -> str:
    for candidate in ("model", "base_model", "transformer", "gpt_neox", "blk", "backbone"):
        prefix = f"{candidate}."
        if any(key.startswith(prefix) for key in weight_map):
            return prefix
    return ""


def _global_layer_key(key: str, shard: Shard | None) -> str:
    if shard is None:
        return key
    try:
        start_layer = int(getattr(shard, "start_layer", 0) or 0)
    except (TypeError, ValueError):
        start_layer = 0
    if start_layer <= 0:
        return key

    parts = key.split(".")
    if len(parts) < 2 or parts[0] != "layers":
        return key
    try:
        local_index = int(parts[1])
    except (TypeError, ValueError):
        return key
    parts[1] = str(local_index + start_layer)
    return ".".join(parts)


def _resolve_weight_key(key: str, weight_map: dict[str, str], prefix: str, shard: Shard | None = None) -> str | None:
    candidates = [key]
    global_key = _global_layer_key(key, shard)
    if global_key != key:
        candidates.append(global_key)

    for candidate in candidates:
        if candidate in weight_map:
            return candidate
        prefixed = f"{prefix}{candidate}" if prefix else candidate
        if prefixed in weight_map:
            return prefixed
    return None

def apply_weight(
    model_weight_key: str,
    key: str,
    weight_map,
    model_path,
    weight_device,
    model_state_dict,
    model_config
):
    
    weight_file = model_path / weight_map[model_weight_key]
    weights = tg.nn.state.safe_load(str(weight_file))
    weight = weights.get(model_weight_key)

    if weight is None:
        # print(f"!!! WARNING: {model_weight_key} not found in {weight_file}")
        return

    weight = weight.to(weight_device)

    if tg.dtypes.bfloat16:
        # bfloat16 fix for tinygrad. Need to research reasoning
        # From https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py#L251
        weight = weight.cast(tg.dtypes.float32).cast(tg.dtypes.float16)

    if "q_proj" in key:
        weight = permute(weight, model_config["num_heads"])
    elif "k_proj" in key:
        weight = permute(weight, model_config["num_kv_heads"])
    
    model_state_dict[key] = weight
    
# Load safetensor weights into model
def load_safetensors(
        model: Model,
        model_path: Path,
        model_config: dict,
        weight_device: str = "CPU",
        use_tied: bool = False
    ) -> None:
    """
    Loading weights into model
    """
    model_files = list(model_path.glob("*.safetensors"))

    if len(model_files) == 0:
        logger.error("No safetensor files found in model path %s", model_path)
        raise FileNotFoundError(f"No safetensor files found in model path {model_path}")
    
    weight_map = {}

    # get weight map if there is one
    weight_map_json = model_path / "model.safetensors.index.json"
    if weight_map_json.exists():
        with weight_map_json.open("r") as f:
            safe_index = json.load(f)
            weight_map = safe_index["weight_map"]
    else:
        # get weightmap from safetensor
        # this is usually when the model only has one weight
        with safetensors.safe_open(str(model_files[0]), framework="numpy") as weight_data:
            for key in weight_data.keys():
                weight_map[key] = model_files[0].name

    model_state_dict = tg.nn.state.get_state_dict(model)
    prefix = _infer_prefix(weight_map)
    shard = getattr(model, "shard", None)

    for key in model_state_dict.keys():
        model_weight_key = _resolve_weight_key(key, weight_map, prefix, shard)
        if model_weight_key not in weight_map.keys():
            if use_tied and key == "output.weight":
                embed_weight_key = "model.embed_tokens.weight"
                apply_weight(
                    embed_weight_key,
                    key,
                    weight_map,
                    model_path,
                    weight_device,
                    model_state_dict,
                    model_config
                )

                continue
            elif "lm_head.weight" in weight_map.keys() and key == "output.weight":
                apply_weight(
                    "lm_head.weight",
                    key,
                    weight_map,
                    model_path,
                    weight_device,
                    model_state_dict,
                    model_config
                )

                continue

            continue

        apply_weight(
            model_weight_key,
            key,
            weight_map,
            model_path,
            weight_device,
            model_state_dict,
            model_config
        )
    
    tg.nn.state.load_state_dict(model, model_state_dict)
"""
LLM sampling
"""


def _apply_repetition_penalty(
    logits: tg.Tensor,
    seen_tokens: list[int] | tuple[int, ...] | None,
    repetition_penalty: float,
) -> tg.Tensor:
    if repetition_penalty <= 0.0 or abs(repetition_penalty - 1.0) < 1e-6 or not seen_tokens:
        return logits

    seen = np.unique(np.asarray(seen_tokens, dtype=np.int64))
    seen = seen[(seen >= 0) & (seen < logits.numel())]
    if seen.size == 0:
        return logits

    logits_np = logits.numpy().astype(np.float32, copy=True)
    penalized = logits_np[seen]
    logits_np[seen] = np.where(
        penalized < 0,
        penalized * float(repetition_penalty),
        penalized / float(repetition_penalty),
    )
    return tg.Tensor(logits_np, device=logits.device, dtype=logits.dtype)

# From https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py#L119
# standard openai sampling
def sample(
    logits: tg.Tensor,
    temp: float=1.0,
    k: Optional[int]=0,
    p: Optional[float]=0.8,
    af: Optional[float] = 0.0,
    ap: Optional[float] = 0.0,
    seen_tokens: list[int] | tuple[int, ...] | None = None,
    repetition_penalty: float = 1.0,
):
    assert logits.ndim == 1, "only works on 1d tensors"
    assert 0 <= p <= 1, "p must be between 0 and 1"
    assert 0 <= k <= logits.numel(), "k must be between 0 and numel"

    logits = _apply_repetition_penalty(logits, seen_tokens, float(repetition_penalty))

    # if temperature is very low just use argmax
    if temp < 1e-6: return logits.argmax()

    logits = logits.to(tg.Device.DEFAULT)

    # alpha sampling
    if af or ap:
        if not hasattr(sample, "alpha_counter"):
            setattr(sample, "alpha_counter", tg.Tensor.zeros_like(logits, dtype=tg.dtypes.int32).contiguous())
        logits = logits - (sample.alpha_counter * af + (sample.alpha_counter > 0) * ap)

    # replace NaNs with -inf
    logits = (logits != logits).where(-float("inf"), logits)

    # softmax
    t = (logits / temp).softmax()

    counter, counter2 = tg.Tensor.arange(t.numel(), device=logits.device).contiguous(), tg.Tensor.arange(t.numel() - 1, -1, -1, device=logits.device).contiguous()
    # top k
    if k:
        output, output_indices = tg.Tensor.zeros(k, device=logits.device).contiguous(), tg.Tensor.zeros(k, device=logits.device, dtype=tg.dtypes.int32).contiguous()
        for i in range(k):
            t_argmax = (t.numel() - ((t == (t_max := t.max())) * counter2).max() - 1).cast(tg.dtypes.default_int)
            output = output + t_max.unsqueeze(0).pad(((i, k - i - 1),))
            output_indices = output_indices + t_argmax.unsqueeze(0).pad(((i, k - i - 1),))
            t = (counter == t_argmax).where(0, t)

        # approximate top p
        # because we are already limited to top k elements we can do top p "without sorting"
        output_cumsum = output[::-1].cumsum()[::-1] + t.sum()
        output = (output_cumsum >= (1 - p)) * output
        output_indices = (output_cumsum >= (1 - p)) * output_indices

        # sample
        output_idx = output.multinomial()
        output_token = output_indices[output_idx]
    else:
        output_token = t.multinomial()

    # increase alpha counter
    if af or ap:
        sample.alpha_counter = (counter == output_token).where(sample.alpha_counter + 1, sample.alpha_counter)

    return output_token

def generate(
    model: Model,
    input_ids: tg.Tensor, # [B, S]
    attention_mask: tg.Tensor,
    tokenizer: Any,
    max_new_tokens: int = 2048,
    temp: float=1.0,
    top_k: Optional[int] = 0,
    top_p: Optional[float] = 0.8,
    alpha_f: Optional[float] = 0.0,
    alpha_p: Optional[float] = 0.0,
    repetition_penalty: float = 1.0,
    curr_pos: int = 0,
    verbose: bool = False
) -> list:
    decode_token = getattr(model, "decode_token", None)
    use_decode_fast_path = callable(decode_token)

    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
    if hasattr(sample, "alpha_counter"):
        delattr(sample, "alpha_counter")

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
    out_tokens = []
    seen_tokens = [int(v) for row in input_ids.tolist() for v in row]

    position_ids = ((attention_mask.cumsum(axis=1) - 1) * attention_mask).to(device) # [B, S]

    print("Generating")
    print("_________________________________\n\n")
    if verbose:
        print(f"input_ids: {input_ids.tolist()}\nposition_ids: {position_ids.tolist()}\nattention_mask: {attention_mask.tolist()}")

    # get logits
    logits = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids
    ) # [B, S, V]

    # prefill
    next_logit = logits[:, -1, :].flatten() # [B, V]
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
    tok = tok_sample.item()
    out_tokens.append(tok)
    seen_tokens.append(int(tok))
    # toks/sec timer
    t0 = time.time()
    generated = 1
    if curr_pos <= 0:
        curr_pos = int(position_ids[:, -1].max().item()) + 1

    eos_hit = False

    # first token sampled; appended to out_tokens above

    while True:
        if tok == tokenizer.eos_token_id:
            elapsed = time.time() - t0
            tok_s = generated / elapsed if elapsed > 0 else float("inf")
            print(f"[decode] {generated} tokens in {elapsed:.3f}s  ->  {tok_s:.2f} tok/s")
            eos_hit = True
            break

        if max_new_tokens > 0 and generated >= max_new_tokens:
            break

        next_tok = tg.Tensor([[tok]], device=device)  # [B, 1]

        if verbose:
            next_mask_len = curr_pos + 1
            print(
                f"next_tok: {[next_tok.item()]}\n start_pos: {curr_pos}\n attention_mask_len: {next_mask_len}"
            )

        if use_decode_fast_path:
            logits = decode_token(next_tok, start_pos=curr_pos)
        else:
            # Non-JIT fallback keeps the original dynamic attention-mask path.
            attention_mask = attention_mask.cat(
                tg.Tensor.ones((attention_mask.shape[0], 1), device=device), dim=1
            )
            position_ids = tg.Tensor([curr_pos], device=device, dtype=tg.dtypes.int32)
            logits = model(
                next_tok,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        next_logit = logits[:, -1, :].flatten()  # [B, V]
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
        tok = tok_sample.item()
        out_tokens.append(tok)
        seen_tokens.append(int(tok))
        generated += 1
        curr_pos += 1

    if not eos_hit:
        elapsed = time.time() - t0
        tok_s = generated / elapsed if elapsed > 0 else float("inf")
        if verbose:
            print(f"[decode] {generated} tokens in {elapsed:.3f}s  ->  {tok_s:.2f} tok/s (no EOS)")

    return out_tokens

# async model load
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
    weight_device: str | None = None,
    offline_mode: bool = False,
    progress_callback: Callable[[str], Awaitable[None] | None] | None = None,
) -> tuple[Model, dict, AutoTokenizer, Path]:
    from cheetah.repos import RepoCustom

    sanitized = model_id.replace("/", "__")
    cache_path = (Path.home() / ".cache" / "cheetah_models") / sanitized
    candidate_path = Path(model_id).expanduser()

    resolved_path = None
    if candidate_path.exists():
        resolved_path = candidate_path
    elif cache_path.exists():
        resolved_path = cache_path
    
    model_repo = RepoCustom(model_id)
    if resolved_path is not None and any(resolved_path.glob("*.*")):
        model_config = load_model_config(resolved_path)
        model_path = resolved_path
    elif resolved_path is None and not offline_mode:
        logger.info("No path resolved to model %s, creating a new path %s and downloading model", model_id, cache_path)
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
            total_layers=model_config["num_layers"]+1
        )

    if weight_device is None:
        weight_device = get_backend_device("tinygrad", default="CPU")
        assert weight_device is not None

    model = Model(model_config, shard)
    if is_quantized_model_config(model_config):
        logger.info("Detected quantized model for %s, loading with tinygrad NF4 loader.", model_id)
        load_quantized_safetensors(
            model,
            model_path,
            model_config,
            weight_device=weight_device,
            use_tied=model_config["tie_word_embeddings"]
        )
    else:
        load_safetensors(
            model,
            model_path,
            model_config,
            weight_device=weight_device,
            use_tied=model_config["tie_word_embeddings"]
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=offline_mode
    )

    return model, model_config, tokenizer, model_path
