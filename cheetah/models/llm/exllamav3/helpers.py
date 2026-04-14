from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from cheetah.models.llm.backend import get_backend_device
from cheetah.logging_utils import get_logger

from .model import Model as ExLlamaV3Model
from .model_config import ModelConfig

logger = get_logger(__name__)


def sample(*args: Any, **kwargs: Any):
    try:
        from cheetah.models.llm.torch.helpers import sample as torch_sample
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "exllamav3 backend requires PyTorch. Install torch and exllamav3, "
                "or switch TC_LLM_BACKEND to tinygrad."
            ) from exc
        raise

    return torch_sample(*args, **kwargs)


def load_model_config(model_path: Path) -> dict[str, Any]:
    model_config = ModelConfig()
    model_config_file = model_path / "config.json"
    if not model_config_file.exists():
        raise FileNotFoundError(f"Model config file not found at {model_config_file}")

    model_config.load(model_config_file)
    gen_config = model_path / "generation_config.json"
    if gen_config.exists():
        model_config.load_generation_config(gen_config)

    return model_config.config


def _cache_size_tokens(model_config: dict[str, Any]) -> int:
    configured = model_config.get("max_seq_len", 2048)
    try:
        seq_len = int(configured)
    except (TypeError, ValueError):
        seq_len = 2048
    seq_len = max(256, seq_len)
    page_size = 256
    return ((seq_len + page_size - 1) // page_size) * page_size


async def _emit_progress(
    progress_callback: Callable[[str], Awaitable[None] | None] | None,
    message: str,
) -> None:
    if progress_callback is None:
        return
    result = progress_callback(message)
    if inspect.isawaitable(result):
        await result


def _stop_conditions(model: ExLlamaV3Model, tokenizer: Any | None = None) -> list[int]:
    stop_conditions: list[int] = []

    eos_token_ids = getattr(model.exllama_config, "eos_token_id_list", None)
    if eos_token_ids:
        stop_conditions.extend(int(token_id) for token_id in eos_token_ids if token_id is not None)

    if not stop_conditions:
        eos_token_id = getattr(model.exllama_tokenizer, "eos_token_id", None)
        if eos_token_id is None and tokenizer is not None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            stop_conditions.append(int(eos_token_id))

    return stop_conditions


def _build_sampler(
    *,
    temp: float,
    top_k: int,
    top_p: float,
    alpha_f: float,
    alpha_p: float,
    repetition_penalty: float,
):
    try:
        from exllamav3 import ComboSampler
    except ModuleNotFoundError as exc:
        if exc.name in {"exllamav3", "torch"}:
            raise RuntimeError(
                "exllamav3 backend requires PyTorch and exllamav3. "
                "Install both or switch TC_LLM_BACKEND to torch or tinygrad."
            ) from exc
        raise

    normalized_top_p = float(top_p)
    if normalized_top_p <= 0.0 or normalized_top_p > 1.0:
        normalized_top_p = 1.0

    return ComboSampler(
        rep_p=float(repetition_penalty),
        freq_p=float(alpha_f),
        pres_p=float(alpha_p),
        temperature=float(temp),
        top_k=max(0, int(top_k)),
        top_p=normalized_top_p,
    )


def stream_generate(
    model: ExLlamaV3Model,
    input_ids: Any,
    tokenizer: Any | None = None,
    *,
    max_new_tokens: int = 512,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.8,
    alpha_f: float = 0.0,
    alpha_p: float = 0.0,
    repetition_penalty: float = 1.0,
    on_token: Callable[[int], None] | None = None,
    abort_check: Callable[[], str | None] | None = None,
) -> tuple[list[int], float]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "exllamav3 backend requires PyTorch and exllamav3. "
                "Install both or switch TC_LLM_BACKEND to torch or tinygrad."
            ) from exc
        raise

    try:
        from exllamav3 import Job
    except ModuleNotFoundError as exc:
        if exc.name in {"exllamav3", "torch"}:
            raise RuntimeError(
                "exllamav3 backend requires PyTorch and exllamav3. "
                "Install both or switch TC_LLM_BACKEND to torch or tinygrad."
            ) from exc
        raise

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.device.type != "cpu":
        input_ids = input_ids.to(device="cpu")
    input_ids = input_ids.to(dtype=torch.long)

    model.reset_kv_cache()

    job = Job(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_conditions=_stop_conditions(model, tokenizer),
        sampler=_build_sampler(
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            alpha_f=alpha_f,
            alpha_p=alpha_p,
            repetition_penalty=repetition_penalty,
        ),
    )

    generator = model.generator
    generator.enqueue(job)

    out_tokens: list[int] = []
    start_time = time.time()
    try:
        while generator.num_remaining_jobs():
            if abort_check is not None:
                reason = abort_check()
                if reason:
                    from cheetah.tui.helpers import MemoryPressureError

                    raise MemoryPressureError(reason)

            results = generator.iterate()
            for result in results:
                if result.get("stage") != "streaming":
                    continue

                token_ids = result.get("token_ids")
                if token_ids is not None:
                    if isinstance(token_ids, torch.Tensor):
                        chunk_tokens = [int(token) for token in token_ids.reshape(-1).tolist()]
                    else:
                        chunk_tokens = [int(token) for token in token_ids]
                    out_tokens.extend(chunk_tokens)
                    if on_token is not None:
                        for token in chunk_tokens:
                            on_token(token)

                if bool(result.get("eos")):
                    break
    finally:
        if generator.num_remaining_jobs():
            generator.clear_queue()

    return out_tokens, time.time() - start_time


async def load_model(
    model_id: str,
    shard: Any = None,
    weight_device: str | None = None,
    offline_mode: bool = False,
    progress_callback: Callable[[str], Awaitable[None] | None] | None = None,
) -> tuple[ExLlamaV3Model, dict[str, Any], AutoTokenizer, Path]:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        if exc.name == "transformers":
            raise RuntimeError(
                "exllamav3 backend requires transformers, PyTorch, and exllamav3. "
                "Install the missing dependencies or switch TC_LLM_BACKEND."
            ) from exc
        raise

    try:
        from exllamav3 import Cache, Config, Generator, Model, Tokenizer
    except ModuleNotFoundError as exc:
        if exc.name in {"exllamav3", "torch"}:
            raise RuntimeError(
                "exllamav3 backend requires PyTorch and exllamav3. "
                "Install both or switch TC_LLM_BACKEND to torch or tinygrad."
            ) from exc
        raise

    del shard, weight_device

    sanitized = model_id.replace("/", "__")
    cache_path = (Path.home() / ".cache" / "cheetah_models") / sanitized
    candidate_path = Path(model_id).expanduser()

    resolved_path = None
    if candidate_path.exists():
        resolved_path = candidate_path
    elif cache_path.exists():
        resolved_path = cache_path

    if resolved_path is not None and any(resolved_path.glob("*.*")):
        model_config = load_model_config(resolved_path)
        model_path = resolved_path
    elif resolved_path is None and not offline_mode:
        from cheetah.repos import RepoCustom

        model_repo = RepoCustom(model_id, backend="exllamav3")
        await _emit_progress(progress_callback, "Preparing exllamav3 model download...")
        model_path, model_config, _ = await model_repo.download(progress_callback=progress_callback)
    elif offline_mode:
        raise FileNotFoundError(f"Model {model_id} not found in offline mode")
    else:
        raise FileNotFoundError(f"Unable to resolve model path for {model_id}")

    device = get_backend_device("exllamav3", default="cuda")
    assert device is not None

    exllama_config = Config.from_directory(str(model_path))
    runtime_model = Model.from_config(exllama_config)
    cache_size_tokens = _cache_size_tokens(model_config)
    cache = Cache(runtime_model, max_num_tokens=cache_size_tokens)
    exllama_tokenizer = Tokenizer.from_config(exllama_config)

    await _emit_progress(progress_callback, f"Loading exllamav3 model on {device}...")
    runtime_model.load(device=device, progressbar=False)

    generator = Generator(
        model=runtime_model,
        cache=cache,
        tokenizer=exllama_tokenizer,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=offline_mode,
    )

    model = ExLlamaV3Model(
        runtime_model=runtime_model,
        cache=cache,
        generator=generator,
        exllama_tokenizer=exllama_tokenizer,
        exllama_config=exllama_config,
        model_path=model_path,
        device=str(device),
        cache_size_tokens=cache_size_tokens,
    )
    logger.info(
        "Loaded exllamav3 model %s on %s with cache_size_tokens=%d",
        model_id,
        device,
        cache_size_tokens,
    )
    return model, model_config, tokenizer, model_path
