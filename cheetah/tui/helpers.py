from __future__ import annotations
import gc
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any

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

from cheetah.models.llm.backend import (
    detect_quantization_mode as detect_quantization_mode_backend,
)
from cheetah.logging_utils import get_logger
logger = get_logger(__name__)
THINK_OPEN_TOKEN = "<think>"
THINK_CLOSE_TOKEN = "</think>"


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
