from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

LLM_BACKEND_ENV = "TC_LLM_BACKEND"
TORCH_DEVICE_ENV = "TC_TORCH_DEVICE"
TINYGRAD_DEVICE_ENV = "TC_TINYGRAD_DEVICE"
DEFAULT_LLM_BACKEND = "tinygrad"
SUPPORTED_LLM_BACKENDS = {"tinygrad", "torch"}
_UNSET = object()
_BACKEND_DEVICE_ENVS = {
    "tinygrad": TINYGRAD_DEVICE_ENV,
    "torch": TORCH_DEVICE_ENV,
}
_BACKEND_DEFAULT_DEVICES = {
    "tinygrad": "CPU",
    "torch": "cpu",
}
_TOKENIZER_FINGERPRINT_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
)
RUNTIME_FINGERPRINT_PROTOCOL = 2
_MODEL_CONFIG_RUNTIME_ONLY_KEYS = {
    "temperature",
    "max_new_tokens",
    "top_k",
    "top_p",
    "repetition_penalty",
}


def normalize_llm_backend(value: str | None) -> str:
    if value is None:
        return DEFAULT_LLM_BACKEND
    normalized = value.strip().lower()
    if normalized in SUPPORTED_LLM_BACKENDS:
        return normalized
    return DEFAULT_LLM_BACKEND


def get_llm_backend() -> str:
    return normalize_llm_backend(os.getenv(LLM_BACKEND_ENV))


def backend_device_env(backend: str | None = None) -> str:
    selected = normalize_llm_backend(backend or os.getenv(LLM_BACKEND_ENV))
    return _BACKEND_DEVICE_ENVS[selected]


def default_backend_device(backend: str | None = None) -> str:
    selected = normalize_llm_backend(backend or os.getenv(LLM_BACKEND_ENV))
    return _BACKEND_DEFAULT_DEVICES[selected]


def normalize_backend_device(value: str | None, backend: str | None = None) -> str:
    selected = normalize_llm_backend(backend or os.getenv(LLM_BACKEND_ENV))
    raw = str(value or "").strip()
    if not raw:
        return default_backend_device(selected)

    if selected == "torch":
        normalized = raw.lower()
        return "mps" if normalized == "metal" else normalized

    normalized = raw.upper()
    return "METAL" if normalized == "MPS" else normalized


def get_backend_device(
    backend: str | None = None,
    *,
    default: str | None | object = _UNSET
) -> str | None:
    selected = normalize_llm_backend(backend or os.getenv(LLM_BACKEND_ENV))
    env_name = backend_device_env(selected)
    configured = os.getenv(env_name)
    if configured is not None and configured.strip():
        return normalize_backend_device(configured, selected)

    if default is _UNSET:
        return default_backend_device(selected)
    if default is None:
        return None
    return normalize_backend_device(str(default), selected)

def set_backend_device(
    value: str | None,
    backend: str | None = None
) -> str:
    selected = normalize_llm_backend(backend or os.getenv(LLM_BACKEND_ENV))
    device = normalize_backend_device(value, selected)
    os.environ[backend_device_env(selected)] = device
    return device


def set_llm_backend(value: str | None) -> str:
    backend = normalize_llm_backend(value)
    os.environ[LLM_BACKEND_ENV] = backend
    return backend


def _backend_module(module_name: str, backend: str | None = None):
    selected = normalize_llm_backend(backend or os.getenv(LLM_BACKEND_ENV))
    module_path = f"cheetah.models.llm.{selected}.{module_name}"
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if selected == "torch" and exc.name == "torch":
            raise RuntimeError(
                "Torch backend selected but PyTorch is not installed. "
                "Install torch or switch TC_LLM_BACKEND=tinygrad."
            ) from exc
        if selected == "tinygrad" and exc.name == "tinygrad":
            raise RuntimeError(
                "tinygrad backend selected but tinygrad is not installed. "
                "Install tinygrad or switch TC_LLM_BACKEND=torch."
            ) from exc
        raise


def backend_helpers_module(backend: str | None = None):
    return _backend_module("helpers", backend=backend)


def backend_quantize_module(backend: str | None = None):
    return _backend_module("quantize", backend=backend)


def backend_model_module(backend: str | None = None):
    return _backend_module("model", backend=backend)


def backend_model_class(backend: str | None = None):
    return backend_model_module(backend=backend).Model


def backend_model_config_module(backend: str | None = None):
    return _backend_module("model_config", backend=backend)


def backend_model_config_class(backend: str | None = None):
    return backend_model_config_module(backend=backend).ModelConfig


def model_config_fingerprint(model_config: Any) -> str:
    normalized = _json_safe_value(_stable_model_config_view(model_config))
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_config_file_fingerprint(model_path: str | Path | None) -> str:
    return _asset_fingerprint(model_path, ("config.json",))


def tokenizer_assets_fingerprint(model_path: str | Path | None) -> str:
    return _asset_fingerprint(model_path, _TOKENIZER_FINGERPRINT_FILES)


def runtime_asset_fingerprints(
    *,
    model_config: Any,
    model_path: str | Path | None,
) -> dict[str, str]:
    config_fingerprint = model_config_file_fingerprint(model_path)
    if not config_fingerprint:
        config_fingerprint = model_config_fingerprint(model_config)
    return {
        "config_fingerprint": config_fingerprint,
        "tokenizer_fingerprint": tokenizer_assets_fingerprint(model_path),
    }


async def resolve_model_assets_for_backend(
    model_id: str,
    *,
    offline_mode: bool = False,
    backend: str | None = None,
    progress_callback: Callable[[str], Awaitable[None] | None] | None = None,
) -> tuple[dict[str, Any], Path]:
    from cheetah.repos import RepoCustom

    selected_backend = normalize_llm_backend(backend or os.getenv(LLM_BACKEND_ENV))
    sanitized = model_id.replace("/", "__")
    cache_path = (Path.home() / ".cache" / "cheetah_models") / sanitized
    candidate_path = Path(model_id).expanduser()

    resolved_path: Path | None = None
    if candidate_path.exists():
        resolved_path = candidate_path
    elif cache_path.exists():
        resolved_path = cache_path

    helpers = backend_helpers_module(backend=selected_backend)
    if resolved_path is not None and any(resolved_path.glob("*.*")):
        model_config = helpers.load_model_config(resolved_path)
        return model_config, resolved_path

    if offline_mode:
        raise FileNotFoundError(f"Model {model_id} not found in offline mode")

    model_repo = RepoCustom(model_id, backend=selected_backend)
    model_path, model_config, _ = await model_repo.download(progress_callback=progress_callback)
    return model_config, model_path


async def load_model_for_backend(
    model_id: str,
    shard: Any = None,
    weight_device: str | None = None,
    offline_mode: bool = False,
    backend: str | None = None,
    progress_callback: Callable[[str], Awaitable[None] | None] | None = None,
):
    helpers = backend_helpers_module(backend=backend)
    return await helpers.load_model(
        model_id=model_id,
        shard=shard,
        weight_device=weight_device,
        offline_mode=offline_mode,
        progress_callback=progress_callback,
    )


def detect_quantization_mode(model_config: Any, backend: str | None = None) -> tuple[bool, str]:
    if not isinstance(model_config, dict):
        return False, "standard"

    quantize = backend_quantize_module(backend=backend)
    if not quantize.is_quantized_model_config(model_config):
        return False, "standard"

    quantization_config = model_config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        return True, "quantized"

    quant_method = str(quantization_config.get("quant_method", "quantized")).lower()
    quant_bits = "4-bit" if quantization_config.get("load_in_4bit") or quantization_config.get("_load_in_4bit") else (
        "8-bit" if quantization_config.get("load_in_8bit") or quantization_config.get("_load_in_8bit") else ""
    )
    quant_type = str(quantization_config.get("bnb_4bit_quant_type", "")).lower()

    parts = [quant_method]
    if quant_bits:
        parts.append(quant_bits)
    if quant_type:
        parts.append(quant_type)
    return True, " ".join(parts)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _asset_fingerprint(model_path: str | Path | None, filenames: tuple[str, ...]) -> str:
    if model_path is None:
        return ""
    root = Path(model_path).expanduser()
    if not root.exists():
        return ""

    digest = hashlib.sha256()
    found = False
    for filename in filenames:
        candidate = root / filename
        if not candidate.exists() or not candidate.is_file():
            continue
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        found = True
    return digest.hexdigest() if found else ""


def _stable_model_config_view(model_config: Any) -> Any:
    if not isinstance(model_config, dict):
        return model_config
    return {
        key: value
        for key, value in model_config.items()
        if str(key) not in _MODEL_CONFIG_RUNTIME_ONLY_KEYS
    }
