from __future__ import annotations
import gc
import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import tinygrad as tg
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]
try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
from transformers import AutoTokenizer

from textual.widgets import RichLog

from tiny_cheetah.models.llm.backend import (
    backend_helpers_module,
    detect_quantization_mode as detect_quantization_mode_backend,
    get_backend_device,
)
from tiny_cheetah.models.shard import Shard
from tiny_cheetah.orchestration.model_engine import ModelEngine
from tiny_cheetah.logging_utils import get_logger
logger = get_logger(__name__)
TokenCallback = Callable[[int], None]
AbortCheck = Callable[[], str | None]
THINK_OPEN_TOKEN = "<think>"
THINK_CLOSE_TOKEN = "</think>"


class MemoryPressureError(RuntimeError):
    """Raised when memory guard thresholds are exceeded during long-running loops."""


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _bytes_to_gib(value: float) -> float:
    return value / (1024.0 ** 3)


def memory_abort_reason(context: str = "") -> str | None:
    """
    Return a human-readable reason if memory pressure is too high, otherwise None.

    Tunables:
    - TC_MEM_MAX_PERCENT (default 92)
    - TC_MEM_MIN_AVAILABLE_GB (default 0.75)
    """
    if psutil is None:
        return None

    mem = psutil.virtual_memory()
    max_mem_percent = _env_float("TC_MEM_MAX_PERCENT", 92.0)
    min_available_gb = _env_float("TC_MEM_MIN_AVAILABLE_GB", 0.75)
    available_gb = _bytes_to_gib(float(mem.available))
    ram_high = float(mem.percent) >= max_mem_percent
    available_low = available_gb <= min_available_gb

    reasons: list[str] = []
    if ram_high:
        reasons.append(f"RAM usage {float(mem.percent):.1f}% >= {max_mem_percent:.1f}%")
    if available_low:
        reasons.append(f"Available RAM {available_gb:.2f} GiB <= {min_available_gb:.2f} GiB")

    if not reasons:
        return None

    label = f" ({context})" if context else ""
    return (
        f"Memory guard triggered{label}: {'; '.join(reasons)} "
        f"[RAM {float(mem.percent):.1f}% used, available {available_gb:.2f} GiB]"
    )


def relieve_memory_pressure(model: Any | None = None) -> None:
    """Best-effort cache cleanup before giving up on a long-running generation loop."""
    if model is not None and hasattr(model, "reset_kv_cache"):
        try:
            model.reset_kv_cache()
        except Exception:
            logger.debug("Failed to reset KV cache during memory relief", exc_info=True)

    if torch is not None:
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Failed to empty torch CUDA cache", exc_info=True)
        try:
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()
        except Exception:
            logger.debug("Failed to empty torch MPS cache", exc_info=True)

    try:
        device_manager = getattr(tg, "Device", None)
        opened_devices = list(getattr(device_manager, "_opened_devices", [])) if device_manager is not None else []
        for device_name in opened_devices:
            allocator = getattr(device_manager[device_name], "allocator", None)
            if allocator is not None and hasattr(allocator, "free_cache"):
                allocator.free_cache()
    except Exception:
        logger.debug("Failed to clear tinygrad allocator caches", exc_info=True)

    gc.collect()


def _raise_if_abort(abort_check: AbortCheck | None) -> None:
    if abort_check is None:
        return
    reason = abort_check()
    if reason:
        raise MemoryPressureError(reason)


def detect_quantization_mode(model_config: Any) -> tuple[bool, str]:
    return detect_quantization_mode_backend(model_config)


def _env_flag(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def model_supports_thinking(
    *,
    model_path: str | Path | None = None,
    tokenizer: Any = None,
) -> bool:
    config_path: Path | None = None
    if model_path is not None:
        candidate = Path(model_path) / "tokenizer_config.json"
        if candidate.exists():
            config_path = candidate

    if config_path is None and tokenizer is not None:
        name_or_path = getattr(tokenizer, "name_or_path", None)
        if isinstance(name_or_path, str) and name_or_path.strip():
            candidate = Path(name_or_path).expanduser() / "tokenizer_config.json"
            if candidate.exists():
                config_path = candidate

    if config_path is None:
        return False

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    decoder = payload.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        return False

    found_open = False
    found_close = False
    for value in decoder.values():
        content = value.get("content") if isinstance(value, dict) else None
        if content == "<think>":
            found_open = True
        elif content == "</think>":
            found_close = True
        if found_open and found_close:
            return True
    return False


def default_enable_thinking(
    *,
    model_path: str | Path | None = None,
    tokenizer: Any = None,
) -> bool:
    env_value = _env_flag("TC_ENABLE_THINKING")
    if env_value is not None:
        return env_value
    return model_supports_thinking(model_path=model_path, tokenizer=tokenizer)


def apply_chat_template_with_thinking(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    tokenize: bool,
    model_path: str | Path | None = None,
    enable_thinking: bool | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": tokenize,
    }
    supports_thinking = model_supports_thinking(
        model_path=model_path,
        tokenizer=tokenizer,
    )
    if supports_thinking:
        thinking_value = default_enable_thinking(
            model_path=model_path,
            tokenizer=tokenizer,
        ) if enable_thinking is None else bool(enable_thinking)

        try:
            signature = inspect.signature(tokenizer.apply_chat_template)
        except (TypeError, ValueError):
            signature = None

        accepts_thinking = signature is None or "enable_thinking" in signature.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        if accepts_thinking:
            kwargs["enable_thinking"] = thinking_value

    return tokenizer.apply_chat_template(messages, **kwargs)


def _cleanup_special_tokens(text: str, tokenizer: Any = None) -> str:
    cleaned = text
    special_tokens = getattr(tokenizer, "all_special_tokens", []) or []
    for token in special_tokens:
        if not isinstance(token, str):
            continue
        if token in {THINK_OPEN_TOKEN, THINK_CLOSE_TOKEN}:
            continue
        if token:
            cleaned = cleaned.replace(token, "")

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)

    lines: list[str] = []
    blank_pending = False
    for raw_line in cleaned.split("\n"):
        line = re.sub(r"[ \t]{2,}", " ", raw_line).strip()
        if not line:
            if lines:
                blank_pending = True
            continue
        if blank_pending:
            lines.append("")
            blank_pending = False
        lines.append(line)
    return "\n".join(lines).strip()


def split_thinking_response(raw_text: str, tokenizer: Any = None) -> tuple[str, str]:
    if not raw_text:
        return "", ""

    thinking_parts: list[str] = []
    final_parts: list[str] = []
    cursor = 0
    text_len = len(raw_text)

    while cursor < text_len:
        open_idx = raw_text.find(THINK_OPEN_TOKEN, cursor)
        if open_idx == -1:
            final_parts.append(raw_text[cursor:])
            break

        final_parts.append(raw_text[cursor:open_idx])
        think_start = open_idx + len(THINK_OPEN_TOKEN)
        close_idx = raw_text.find(THINK_CLOSE_TOKEN, think_start)
        if close_idx == -1:
            thinking_parts.append(raw_text[think_start:])
            cursor = text_len
            break

        thinking_parts.append(raw_text[think_start:close_idx])
        cursor = close_idx + len(THINK_CLOSE_TOKEN)

    thinking_text = _cleanup_special_tokens("".join(thinking_parts), tokenizer=tokenizer)
    final_text = _cleanup_special_tokens("".join(final_parts), tokenizer=tokenizer)
    if thinking_text or final_text:
        return thinking_text, final_text

    return "", _cleanup_special_tokens(
        raw_text.replace(THINK_OPEN_TOKEN, "").replace(THINK_CLOSE_TOKEN, ""),
        tokenizer=tokenizer,
    )


def _sample_with_backend(*args: Any, **kwargs: Any):
    if args:
        logits = args[0]
        if torch is not None and isinstance(logits, torch.Tensor):
            return backend_helpers_module("torch").sample(*args, **kwargs)
        if isinstance(logits, tg.Tensor):
            return backend_helpers_module("tinygrad").sample(*args, **kwargs)

    # Fallback to configured backend, then tinygrad.
    try:
        return backend_helpers_module().sample(*args, **kwargs)
    except Exception:
        return backend_helpers_module("tinygrad").sample(*args, **kwargs)


def streaming_generate(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
    temp: float = 1.0,
    top_k: int = 35,
    top_p: float = 0.8,
    alpha_f: float = 0.0,
    alpha_p: float = 0.0,
    repetition_penalty: float = 1.0,
    verbose: bool = False,
    on_token: TokenCallback | None = None,
    abort_check: AbortCheck | None = None,
) -> tuple[list[int], float]:
    if torch is not None and isinstance(input_ids, torch.Tensor):
        return _streaming_generate_torch(
            model,
            input_ids,
            attention_mask,
            tokenizer,
            max_new_tokens=max_new_tokens,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            alpha_f=alpha_f,
            alpha_p=alpha_p,
            repetition_penalty=repetition_penalty,
            verbose=verbose,
            on_token=on_token,
            abort_check=abort_check,
        )

    _raise_if_abort(abort_check)
    device = input_ids.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    out_tokens: list[int] = []
    curr_pos = attention_mask.shape[1] - 1
    generated = 0
    start_time = time.time()
    seen_tokens = [int(v) for row in input_ids.tolist() for v in row]

    # initial prefill
    position_ids = ((attention_mask.cumsum(axis=1) - 1) * attention_mask).to(device)
    logits = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    next_logit = logits[:, -1, :].flatten()
    tok = _sample_with_backend(
        next_logit,
        temp=temp,
        k=top_k,
        p=top_p,
        af=alpha_f,
        ap=alpha_p,
        repetition_penalty=repetition_penalty,
        seen_tokens=seen_tokens,
    ).item()
    out_tokens.append(tok)
    seen_tokens.append(int(tok))
    if on_token is not None:
        on_token(int(tok))
    generated += 1
    curr_pos += 1

    eos_hit = tok == tokenizer.eos_token_id

    while not eos_hit:
        _raise_if_abort(abort_check)
        if max_new_tokens > 0 and generated >= max_new_tokens:
            break

        next_tok = tg.Tensor([[tok]], device=device)
        attention_mask = attention_mask.cat(
            tg.Tensor.ones((attention_mask.shape[0], 1), device=device), dim=1
        )
        position_ids = tg.Tensor([curr_pos], device=device)

        logits = model(
            next_tok,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        next_logit = logits[:, -1, :].flatten()
        tok = _sample_with_backend(
            next_logit,
            temp=temp,
            k=top_k,
            p=top_p,
            af=alpha_f,
            ap=alpha_p,
            repetition_penalty=repetition_penalty,
            seen_tokens=seen_tokens,
        ).item()
        out_tokens.append(tok)
        seen_tokens.append(int(tok))
        if on_token is not None:
            on_token(int(tok))
        generated += 1
        curr_pos += 1

        if tok == tokenizer.eos_token_id:
            eos_hit = True

    elapsed = time.time() - start_time

    if verbose:
        tok_s = generated / elapsed if elapsed > 0 else float("inf")
        print(f"[stream] {generated} tokens in {elapsed:.3f}s -> {tok_s:.2f} tok/s")

    return out_tokens, elapsed


def _streaming_generate_torch(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.8,
    alpha_f: float = 0.0,
    alpha_p: float = 0.0,
    repetition_penalty: float = 1.0,
    verbose: bool = False,
    on_token: TokenCallback | None = None,
    abort_check: AbortCheck | None = None,
) -> tuple[list[int], float]:
    if torch is None:
        raise RuntimeError("Torch tensor generation requested but torch is unavailable")

    _raise_if_abort(abort_check)
    device = get_backend_device("torch", default="cpu")
    assert device is not None
    input_ids = input_ids.to(device=device, dtype=torch.long)
    attention_mask = attention_mask.to(device=device, dtype=torch.long)

    out_tokens: list[int] = []
    curr_pos = int(input_ids.shape[1] - 1)
    seen_tokens = input_ids.flatten().tolist()
    position_ids = ((attention_mask.cumsum(dim=1) - 1) * attention_mask).to(
        device=device,
        dtype=torch.long,
    )

    logits = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    next_logit = logits[:, -1, :].flatten()
    tok_sample = _sample_with_backend(
        next_logit,
        temp=temp,
        k=top_k,
        p=top_p,
        af=alpha_f,
        ap=alpha_p,
        repetition_penalty=repetition_penalty,
        seen_tokens=seen_tokens,
    )
    tok = int(tok_sample.item())
    out_tokens.append(tok)
    seen_tokens.append(tok)
    if on_token is not None:
        on_token(tok)

    start_time = time.time()
    generated = 1
    curr_pos += 1
    eos_hit = tok == tokenizer.eos_token_id
    while not eos_hit:
        _raise_if_abort(abort_check)
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
        logits = model(
            next_tok,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        next_logit = logits[:, -1, :].flatten()
        tok_sample = _sample_with_backend(
            next_logit,
            temp=temp,
            k=top_k,
            p=top_p,
            af=alpha_f,
            ap=alpha_p,
            repetition_penalty=repetition_penalty,
            seen_tokens=seen_tokens,
        )
        tok = int(tok_sample.item())
        out_tokens.append(tok)
        seen_tokens.append(tok)
        if on_token is not None:
            on_token(tok)
        generated += 1
        curr_pos += 1
        eos_hit = tok == tokenizer.eos_token_id

    elapsed = time.time() - start_time
    if verbose:
        tok_s = generated / elapsed if elapsed > 0 else float("inf")
        print(f"[stream] {generated} tokens in {elapsed:.3f}s -> {tok_s:.2f} tok/s")
    return out_tokens, elapsed


def streaming_generate_with_peers(
    peer_client: Any,
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.8,
    alpha_f: float = 0.0,
    alpha_p: float = 0.0,
    repetition_penalty: float = 1.0,
    verbose: bool = False,
    on_token: TokenCallback | None = None,
    abort_check: AbortCheck | None = None,
) -> tuple[list[int], float]:
    peers = _normalize_peer_entries(peer_client.get_peers(include_self=True)) if peer_client is not None else []
    backend = "torch" if torch is not None and isinstance(input_ids, torch.Tensor) else "tinygrad"

    if not peers or len(peers) <= 1:
        return streaming_generate(
            model,
            input_ids,
            attention_mask,
            tokenizer,
            max_new_tokens,
            temp,
            top_k,
            top_p,
            alpha_f,
            alpha_p,
            repetition_penalty,
            verbose,
            on_token,
            abort_check,
        )

    local_peer_id = str(getattr(peer_client, "peer_client_id", "") or "")
    remote_peers = [
        peer for peer in peers
        if str(getattr(peer, "peer_client_id", "") or "") != local_peer_id
    ]
    if not remote_peers:
        return streaming_generate(
            model,
            input_ids,
            attention_mask,
            tokenizer,
            max_new_tokens,
            temp,
            top_k,
            top_p,
            alpha_f,
            alpha_p,
            repetition_penalty,
            verbose,
            on_token,
            abort_check,
        )

    engine = _local_engine(model)
    _plan_peer_shards(engine, peer_client, model)

    input_list = _tensor_to_list(input_ids)
    mask_list = _tensor_to_list(attention_mask)
    position_list = [[0]]
    hidden_state_list = []
    seen_tokens = [int(v) for row in input_list for v in row]

    out_tokens: list[int] = []
    start_time = time.time()

    for step in range(max_new_tokens):
        _raise_if_abort(abort_check)
        logger.debug(f"[step: {step} Starting distributed generation with {len(peers)} peers for up to {max_new_tokens} tokens")
        logger.debug(f"Initial input_ids: {input_list}, attention_mask: {mask_list}")
        input_ids = _list_to_tensor(input_list, like=input_ids)
        attention_mask = _list_to_tensor(mask_list, like=attention_mask)
        otoken_data = engine.get_tokens(
            model,
            input_ids,
            attention_mask,
            tokenizer,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            alpha_f=alpha_f,
            alpha_p=alpha_p,
            repetition_penalty=repetition_penalty,
            seen_tokens=seen_tokens,
        )

        prev_len = len(out_tokens)
        mask_list, position_list, hidden_state_list, end_token = _apply_token_data(
            engine,
            tokenizer,
            otoken_data,
            input_list,
            mask_list,
            position_list,
            hidden_state_list,
            out_tokens,
            backend=backend,
            on_token=on_token,
        )
        if len(out_tokens) > prev_len:
            seen_tokens.extend(int(tok) for tok in out_tokens[prev_len:])

        if end_token:
            break

        for peer in remote_peers:
            payload = {
                "command": "generate_token",
                "payload": {
                    "input_ids": input_list,
                    "attention_mask": mask_list,
                    "position_ids": position_list,
                    "hidden_state": hidden_state_list, 
                    "temp": temp,
                    "top_k": top_k,
                    "top_p": top_p,
                    "alpha_f": alpha_f,
                    "alpha_p": alpha_p,
                    "repetition_penalty": repetition_penalty,
                    "seen_tokens": seen_tokens,
                    "shard": _peer_shard_payload(peer),
                },
            }
            try:
                logger.debug(
                    "Sending payload to peer %s",
                    _peer_identifier(peer),
                )
                host = str(getattr(peer, "ip_address", "") or getattr(peer, "address", "0.0.0.0"))
                port = int(getattr(peer, "port", os.getenv("TC_TENSOR_PORT", 1045)))
                address = (host, port)
                resp = peer_client.send_payload(
                    payload,
                    expect_reply=True,
                    address=address,
                )
                otoken_data = resp
                logger.debug("Received token response from peer %s", _peer_identifier(peer))
            except Exception as err:
                logger.error(f"Error communicating with peer {_peer_identifier(peer)}: {err}")
                break

            prev_len = len(out_tokens)
            mask_list, position_list, hidden_state_list, end_token = _apply_token_data(
                engine,
                tokenizer,
                otoken_data,
                input_list,
                mask_list,
                position_list,
                hidden_state_list,
                out_tokens,
                backend=backend,
                on_token=on_token,
            )
            if len(out_tokens) > prev_len:
                seen_tokens.extend(int(tok) for tok in out_tokens[prev_len:])
            
            if end_token:
                break

        if end_token:
            break

    elapsed = time.time() - start_time
    if verbose and out_tokens:
        tok_s = len(out_tokens) / elapsed if elapsed > 0 else float("inf")
        print(f"[stream-peers] {len(out_tokens)} tokens in {elapsed:.3f}s -> {tok_s:.2f} tok/s")
    return out_tokens, elapsed


def _tensor_to_list(tensor: Any) -> list[list[Any]]:
    if tensor is None:
        return [[]]
    if isinstance(tensor, list):
        if not tensor:
            return [[]]
        if isinstance(tensor[0], list):
            return tensor
        return [tensor]
    if torch is not None and isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy().tolist()
    elif hasattr(tensor, "numpy"):
        data = tensor.numpy().tolist()
    elif hasattr(tensor, "tolist"):
        data = tensor.tolist()
    else:
        data = []
    if not data:
        return [[]]
    if isinstance(data[0], list):
        return data
    return [data]


def _list_to_tensor(data: list[list[Any]], like: Any) -> Any:
    if torch is not None and isinstance(like, torch.Tensor):
        return torch.tensor(data, dtype=like.dtype, device=like.device)
    device = getattr(like, "device", None)
    return tg.Tensor(data, device=device)


def _normalize_peer_entries(peers: Any) -> list[Any]:
    normalized: list[Any] = []
    for peer in peers or []:
        if isinstance(peer, tuple) and len(peer) == 2:
            normalized.append(peer[1])
        else:
            normalized.append(peer)
    return normalized


def _peer_identifier(peer: Any) -> str:
    return str(getattr(peer, "peer_client_id", "") or getattr(peer, "ip_address", "") or "unknown")


def _plan_peer_shards(model_engine: ModelEngine, peer_client: Any, model: Any) -> None:
    if peer_client is None:
        return
    peers = _normalize_peer_entries(peer_client.get_peers(include_self=True))
    if len(peers) <= 1:
        return
    total_layers = _infer_total_layers(model)
    if total_layers <= 0:
        return
    model_name = str(getattr(model, "model_name", "") or getattr(model, "name", "") or "model")
    model_engine.plan_shards(peers, model_name, total_layers)


def _apply_token_data(
    engine: ModelEngine,
    tokenizer: AutoTokenizer,
    otoken_data: dict,
    input_list: list[list[int]],
    mask_list: list[list[int]],
    position_list: list[list[int]],
    hidden_state_list: list[list[Any]],
    out_tokens: list[int],
    backend: str | None = None,
    on_token: TokenCallback | None = None,
) -> tuple[list[list[int]], list[list[int]], list[list[Any]], bool]:
    msg = engine.recv_tokens(otoken_data, tokenizer, backend=backend)
    attention_mask = msg.get("attention_mask")
    position_ids = msg.get("position_ids")
    hidden_state = msg.get("hidden_state")
    token = msg.get("token")

    if attention_mask is not None:
        mask_list = _tensor_to_list(attention_mask)
    if position_ids is not None:
        position_list = _tensor_to_list(position_ids)

    if token is not None:
        tok = int(token)
        out_tokens.append(tok)
        if on_token is not None:
            on_token(tok)
        input_list[0].append(tok)
        if mask_list:
            mask_list[0].append(1)
        else:
            mask_list = [[1]]
        hidden_state_list = []
    elif hidden_state is not None:
        hidden_state_list = _tensor_to_list(hidden_state)

    end_token = bool(msg.get("end_token", False))
    return mask_list, position_list, hidden_state_list, end_token


def _infer_total_layers(model: Any) -> int:
    config = getattr(model, "config", {}) or {}
    num_layers = config.get("num_layers")
    if num_layers is None:
        try:
            num_layers = len(getattr(model, "layers", []))
        except Exception:
            num_layers = 0
    try:
        num_layers_int = int(num_layers) if num_layers else 0
    except (TypeError, ValueError):
        num_layers_int = 0
    if num_layers_int <= 0:
        return 0
    return num_layers_int + 1


def _local_engine(model: Any) -> ModelEngine:
    shard = getattr(model, "shard", None)
    if shard is not None:
        return ModelEngine(shard=shard)
    total_layers = _infer_total_layers(model)
    if total_layers <= 0:
        return ModelEngine()
    return ModelEngine(shard=Shard("local", 0, total_layers - 1, total_layers))


def _peer_shard_payload(peer: Any) -> dict[str, int | str]:
    shard = getattr(peer, "shard", None)
    if shard is None:
        return {}
    try:
        return {
            "model_name": shard.model_name,
            "start_layer": int(shard.start_layer),
            "end_layer": int(shard.end_layer),
            "total_layers": int(shard.total_layers),
        }
    except Exception:
        return {}
