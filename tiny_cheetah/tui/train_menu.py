from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import copy
import asyncio
import time
import traceback
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import numpy as np
from textual.app import ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Checkbox, Footer, Header, Input, Label, Log, Static
from transformers import AutoTokenizer

from tiny_cheetah.orchestration.peer_client import PeerClient
from tiny_cheetah.models.llm.backend import (
    backend_helpers_module,
    backend_model_class,
    backend_model_config_class,
    backend_device_env,
    get_backend_device,
    get_llm_backend,
    normalize_backend_device,
    normalize_llm_backend,
)
from tiny_cheetah.models.shard import Shard
from tiny_cheetah.repos import RepoHuggingFace
from tiny_cheetah.tui.orchestration_screen import OrchestrationScreen
from tiny_cheetah.tui.help_screen import HelpScreen

from tiny_cheetah.tui.training_path_types import TrainingNode, NODE_STATUS_STYLES, NODE_STATUS_SYMBOLS
from tiny_cheetah.tui.training_path_screen import TrainingPathScreen
from tiny_cheetah.tui.helpers import memory_abort_reason, relieve_memory_pressure
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is optional at runtime
    psutil = None

try:
    import tinygrad as tg
    from tinygrad.nn.state import get_parameters as tg_get_parameters
    from tinygrad.nn.state import get_state_dict as tg_get_state_dict
    from tinygrad.nn.state import safe_save as tg_safe_save
except Exception:
    tg = None
    tg_get_parameters = None
    tg_get_state_dict = None
    tg_safe_save = None

try:
    import torch
    import torch.nn.functional as torch_F
except Exception:
    torch = None
    torch_F = None

try:
    from safetensors.torch import save_file as torch_save_file
except Exception:
    torch_save_file = None


@dataclass
class TrainingStats:
    status: str = "Idle"
    step: int = 0
    epoch: int = 0
    total_epochs: Optional[int] = None
    loss: Optional[float] = None
    mean_loss: Optional[float] = None
    tokens: int = 0
    tok_rate: Optional[float] = None


@dataclass
class Batch:
    input_ids: Any
    labels: Any
    attention_mask: Optional[Any]
    position_ids: Any


class TrainingCancelled(Exception):
    pass


class _QueueLogStream(io.TextIOBase):
    def __init__(self, outbox: "Queue[Optional[str]]") -> None:
        self._outbox = outbox
        self._buffer = ""

    def write(self, data: str) -> int:
        text = str(data or "")
        if not text:
            return 0
        merged = (self._buffer + text).replace("\r", "\n")
        parts = merged.split("\n")
        self._buffer = parts.pop() if parts else ""
        for part in parts:
            self._outbox.put(part)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._outbox.put(self._buffer)
            self._buffer = ""


SPIN_LOOP = ["|", "/", "-", "\\"]


def default_training_settings() -> Dict[str, object]:
    backend = get_llm_backend()
    from_scratch = False
    from_scratch_text = (os.getenv("TC_TRAINING_FROM_SCRATCH") or "").strip()
    finetune_text = (os.getenv("TC_TRAINING_FINETUNE") or "").strip()
    offline_text = (os.getenv("TC_TRAINING_OFFLINE") or "").strip().lower()
    offline_mode_text = (os.getenv("TC_OFFLINE_MODE") or "").strip().lower()
    offline_default = offline_mode_text in {"1", "true", "yes", "on", "y"}
    offline = (
        offline_text in {"1", "true", "yes", "on", "y"}
        if offline_text
        else offline_default
    )
    if from_scratch_text:
        from_scratch = from_scratch_text.lower() in {"1", "true", "yes", "on", "y"}
    elif finetune_text:
        from_scratch = finetune_text.lower() not in {"1", "true", "yes", "on", "y"}

    return {
        "model-id": (os.getenv("TC_TRAINING_MODEL_ID") or "").strip(),
        "data-path": (os.getenv("TC_TRAINING_DATA_PATH") or "").strip(),
        "dataset-id": (os.getenv("TC_TRAINING_DATASET_ID") or "").strip(),
        "max-dataset-entries": (os.getenv("TC_TRAINING_MAX_DATASET_ENTRIES") or "").strip(),
        "seq-length": (os.getenv("TC_TRAINING_SEQ_LENGTH") or "").strip() or "256",
        "batch-size": (os.getenv("TC_TRAINING_BATCH_SIZE") or "").strip() or "2",
        "epochs": (os.getenv("TC_TRAINING_EPOCHS") or "").strip() or "1",
        "lr": (os.getenv("TC_TRAINING_LR") or "").strip() or 1e-4,
        "device": (os.getenv("TC_TRAINING_DEVICE") or "").strip() or (get_backend_device(backend, default="CPU") or "CPU"),
        "gradient-accumulation": (os.getenv("TC_TRAINING_GRADIENT_ACCUMULATION") or "").strip() or "1",
        "save-dir": (os.getenv("TC_TRAINING_SAVE_DIR") or "").strip(),
        "offline": offline,
        "from-scratch": from_scratch,
    }


class TrainingProcess:
    """In-process training worker."""

    def __init__(self, settings: Dict[str, object]) -> None:
        self.settings = dict(settings)
        self._queue: "Queue[Optional[str]]" = Queue()
        self._thread: Optional[Thread] = None
        self._stop = Event()
        self._failed = False
        self._runtime_state: Dict[str, Any] = {}

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Training process already started.")

        def worker() -> None:
            stream = _QueueLogStream(self._queue)
            try:
                with redirect_stdout(stream), redirect_stderr(stream):
                    _run_training_job(self.settings, self._stop, self._runtime_state)
            except TrainingCancelled:
                if self._stop.is_set():
                    stream.write("[info] Training loop acknowledged stop request.\n")
                else:
                    stream.write("[info] Training cancelled.\n")
            except Exception:
                self._failed = True
                stream.write(traceback.format_exc())
            finally:
                _release_training_runtime(
                    self._runtime_state,
                    announce=lambda message: stream.write(f"{message}\n"),
                )
                stream.flush()
                self._queue.put(None)

        self._thread = Thread(target=worker, name="train-runner", daemon=True)
        self._thread.start()

    def terminate(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._queue.put("[info] Stop requested. Waiting for training loop to reach a safe cleanup point...")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def failed(self) -> bool:
        return self._failed

    def drain(self) -> Iterable[Optional[str]]:
        while True:
            try:
                yield self._queue.get_nowait()
            except Empty:
                break


def _cross_entropy_loss(logits: Any, targets: Any, *, backend: str) -> Any:
    if backend == "torch":
        if torch_F is None:
            raise RuntimeError("PyTorch is required for torch training.")
        return torch_F.cross_entropy(logits.float(), targets.long())

    if tg is None:
        raise RuntimeError("tinygrad is required for tinygrad training.")
    log_probs = logits.log_softmax(axis=-1)
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (-gathered).mean()


def _build_training_namespace(settings: Dict[str, object]) -> Namespace:
    backend = normalize_llm_backend(str(settings.get("backend") or get_llm_backend()))
    device = normalize_backend_device(
        str(settings.get("device") or get_backend_device(backend, default="")),
        backend,
    )

    def _text(key: str) -> str:
        value = settings.get(key, "")
        if isinstance(value, bool):
            return ""
        return str(value).strip()

    def _path(key: str) -> Path | None:
        raw = _text(key)
        return Path(raw) if raw else None

    def _int(key: str, default: int) -> int:
        raw = _text(key)
        return int(raw) if raw else default

    def _optional_int(key: str) -> int | None:
        raw = _text(key)
        return int(raw) if raw else None

    def _float(key: str, default: float) -> float:
        raw = _text(key)
        return float(raw) if raw else default

    model_id = _text("model-id") or None
    config_path = _path("config-path")
    if model_id and config_path is not None:
        config_path = None

    return Namespace(
        backend=backend,
        model_id=model_id,
        custom_model_id=None,
        tokenizer_id=None,
        tokenizer_file=None,
        config_path=config_path,
        generation_config_path=None,
        weights_dir=_path("weights-dir"),
        data_path=_path("data-path"),
        dataset_id=_text("dataset-id") or None,
        dataset_cache_dir=None,
        max_dataset_entries=_optional_int("max-dataset-entries"),
        max_sequences_per_epoch=None,
        offline=bool(settings.get("offline", False)),
        seq_length=_int("seq-length", 256),
        batch_size=_int("batch-size", 2),
        epochs=_int("epochs", 1),
        lr=_float("lr", 1e-4),
        device=device,
        torch_device=None,
        tinygrad_device=None,
        gradient_accumulation=_int("gradient-accumulation", 1),
        low_mem=False,
        from_scratch=bool(settings.get("from-scratch", False)),
        save_dir=_path("save-dir"),
        finetune=False,
    )


def _finetune_requested(args: Namespace) -> bool:
    if bool(args.finetune) and bool(args.from_scratch):
        raise ValueError("Use either finetune or from_scratch, not both.")
    return bool(args.finetune) or not bool(args.from_scratch)


def _ensure_required_keys(config_obj: Any) -> None:
    if hasattr(config_obj, "model_config"):
        backing = config_obj.model_config
    elif hasattr(config_obj, "config"):
        backing = config_obj.config
    else:
        backing = config_obj

    if not isinstance(backing, dict):
        raise TypeError(f"Unsupported config backing type: {type(backing)!r}")

    backing.setdefault("attn_scale", None)
    backing.setdefault("mlp_scale", None)
    backing.setdefault("temperature", None)
    backing.setdefault("top_k", None)
    backing.setdefault("top_p", None)


def _parse_remote_identifier(identifier: Path, default_filename: str) -> tuple[str, str]:
    raw = identifier.as_posix().strip().lstrip("./")
    if not raw or raw.startswith("/"):
        raise FileNotFoundError(f"Invalid remote identifier: {identifier}")

    if raw.endswith(".json"):
        repo_id, _, filename = raw.rpartition("/")
        if not repo_id:
            raise FileNotFoundError(
                f"Remote identifier '{raw}' must include repo id before the filename."
            )
        return repo_id, filename

    return raw, default_filename


def _fetch_model_file(
    repo_id: str,
    filename: str,
    *,
    backend: str,
    cache_dir: Optional[Path] = None,
    local_only: bool = False,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download remote config files. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    kwargs: Dict[str, object] = {"repo_id": repo_id, "filename": filename, "repo_type": "model"}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if local_only:
        kwargs["local_files_only"] = True

    try:
        return Path(hf_hub_download(**kwargs))
    except Exception as err:
        if local_only:
            raise FileNotFoundError(
                f"File '{filename}' not found in local cache for repo '{repo_id}'."
            ) from err
        print(
            f"[config] Direct download failed for {repo_id}/{filename}: {err}. "
            "Falling back to RepoHuggingFace snapshot download."
        )
        snapshot_path, _ = RepoHuggingFace(repo_id, backend=backend).download()
        candidate = snapshot_path / filename
        if candidate.exists():
            return candidate
        matches = list(snapshot_path.rglob(filename))
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"Unable to locate {filename} in repo {repo_id} after snapshot download."
        ) from err


def _download_dataset(dataset_id: str, cache_dir: Optional[Path] = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download datasets. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    kwargs: Dict[str, object] = {"repo_id": dataset_id, "repo_type": "dataset"}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return Path(snapshot_download(**kwargs))


def _conversation_to_text(record: dict) -> Optional[str]:
    candidates = record.get("conversations") or record.get("messages") or record.get("turns")
    if not candidates:
        return None

    turns: List[str] = []
    for turn in candidates:
        role = turn.get("from") or turn.get("role") or ""
        content = turn.get("value") or turn.get("content") or ""
        content = content.strip()
        if not content:
            continue
        prefix = f"{role.strip()}: " if role else ""
        turns.append(prefix + content)

    if not turns:
        return None
    return "\n".join(turns)


def _extract_dataset(dataset_root: Path, limit: Optional[int] = None) -> Iterator[str]:
    jsonl_files = sorted(dataset_root.rglob("*.jsonl"))
    json_files = sorted(
        path for path in dataset_root.rglob("*.json")
        if path.name not in {"dataset_info.json", "info.json"}
    )
    zst_files = sorted(dataset_root.rglob("*.zst"))

    emitted = 0

    def maybe_yield(text_value: Optional[str]) -> bool:
        nonlocal emitted
        if text_value and (limit is None or emitted < limit):
            emitted += 1
            return True
        return False

    if jsonl_files:
        for path in jsonl_files:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = _conversation_to_text(record)
                    if maybe_yield(text):
                        yield text
                        if limit is not None and emitted >= limit:
                            return

        for path in json_files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                maybe_data = data.get("data") or data.get("examples") or data.get("conversations")
                if isinstance(maybe_data, list):
                    data = maybe_data
            if not isinstance(data, list):
                continue
            for record in data:
                if not isinstance(record, dict):
                    continue
                text = _conversation_to_text(record)
                if maybe_yield(text):
                    yield text
                    if limit is not None and emitted >= limit:
                        return
        return

    if zst_files:
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError(
                "zstandard is required to read .zst dataset files. Install it with `pip install zstandard`."
            ) from exc
        dctx = zstd.ZstdDecompressor()
        for path in zst_files:
            with path.open("rb") as compressed:
                with dctx.stream_reader(compressed) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    for line in text_stream:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = _conversation_to_text(record)
                        if maybe_yield(text):
                            yield text
                            if limit is not None and emitted >= limit:
                                return
        return

    raise RuntimeError(f"No supported dataset (jsonl, .zst) files found under {dataset_root}.")


def _prepare_dataset_corpus(
    dataset_root: Path,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Path:
    if output_dir is None:
        output_dir = dataset_root / "processed_" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "all" if limit is None else f"limit_{limit}"
    corpus_path = output_dir / f"dataset_corpus_{suffix}.txt"
    if corpus_path.exists():
        print(f"[dataset] Reusing prepared corpus at {corpus_path}")
        return corpus_path

    print(
        f"[dataset] Preparing corpus from {dataset_root} "
        f"({'all entries' if limit is None else f'up to {limit} entries'})"
    )
    count = 0
    last_report = time.time()
    with corpus_path.open("w", encoding="utf-8") as out:
        for text in _extract_dataset(dataset_root, limit=limit):
            out.write(text)
            out.write("\n\n")
            count += 1
            now = time.time()
            if count == 1 or count % 25 == 0 or now - last_report >= 1.0:
                cap = "" if limit is None else f"/{limit}"
                print(f"[dataset] corpus prep: {count}{cap} conversations")
                last_report = now

    if count == 0:
        raise RuntimeError(f"No conversations found when processing dataset under {dataset_root}.")

    if limit is not None and count >= limit:
        print(f"[dataset] Reached entry cap at {count} conversations.")
    print(f"[dataset] Prepared {count} dataset conversations at {corpus_path}")
    return corpus_path


def _resolve_tokenizer_asset(tokenizer_id: str, *, backend: str, local_only: bool) -> Optional[Path]:
    if not tokenizer_id or "/" not in tokenizer_id:
        return None

    for name in ["tokenizer.json", "tokenizer.model", "tokenizer.sp.model", "tokenizer.spm"]:
        try:
            return _fetch_model_file(tokenizer_id, name, backend=backend, local_only=local_only)
        except FileNotFoundError:
            continue
    return None


def _stream_corpus_batches(
    tokenizer: AutoTokenizer,
    data_path: Path,
    *,
    seq_length: int,
    batch_size: int,
    backend: str,
    device: str,
    max_sequences: Optional[int] = None,
    stop_event: Optional[Event] = None,
) -> Iterator[Batch]:
    if not data_path.exists():
        raise FileNotFoundError(f"Corpus file {data_path} not found")
    total_bytes = max(int(data_path.stat().st_size), 1)
    processed_bytes = 0
    next_progress_mark = 10
    print(
        f"[data] tokenizing corpus from {data_path} "
        f"with seq_length={seq_length}, batch_size={batch_size}"
    )
    print("[data] tokenizing progress: 0%")

    token_buffer: List[int] = []
    seq_inputs: List[List[int]] = []
    seq_targets: List[List[int]] = []
    pending_lines: List[str] = []
    pending_chars = 0
    total_sequences = 0
    hit_sequence_cap = False
    total_lines = 0
    last_heartbeat = time.time()

    def report_progress(*, force: bool = False, done: bool = False, heartbeat: bool = False) -> None:
        nonlocal next_progress_mark
        percent = min(100, int((processed_bytes * 100) / total_bytes))
        if not force and not heartbeat and percent < next_progress_mark:
            return
        while percent >= next_progress_mark:
            next_progress_mark += 10
        suffix = f" ({total_sequences} sequences ready, {total_lines} lines read)"
        if done and hit_sequence_cap and percent < 100:
            suffix = f" ({total_sequences} sequences ready, {total_lines} lines read, stopped at cap)"
        print(f"[data] tokenizing progress: {percent}%{suffix}")

    def flush_batch() -> Batch:
        nonlocal seq_inputs, seq_targets
        input_arr = np.asarray(seq_inputs, dtype=np.int32)
        target_arr = np.asarray(seq_targets, dtype=np.int32)
        position_arr = np.tile(np.arange(seq_length, dtype=np.int32), (input_arr.shape[0], 1))
        if backend == "torch":
            if torch is None:
                raise RuntimeError("PyTorch is required for torch training.")
            device_name = str(device or "cpu").strip().lower() or "cpu"
            batch = Batch(
                input_ids=torch.tensor(input_arr, device=device_name, dtype=torch.long),
                labels=torch.tensor(target_arr, device=device_name, dtype=torch.long),
                attention_mask=None,
                position_ids=torch.tensor(position_arr, device=device_name, dtype=torch.long),
            )
        else:
            if tg is None:
                raise RuntimeError("tinygrad is required for tinygrad training.")
            batch = Batch(
                input_ids=tg.Tensor(input_arr, device=device, dtype=tg.dtypes.int32),
                labels=tg.Tensor(target_arr, device=device, dtype=tg.dtypes.int32),
                attention_mask=None,
                position_ids=tg.Tensor(position_arr, device=device, dtype=tg.dtypes.int32),
            )
        seq_inputs, seq_targets = [], []
        return batch

    def tokenize_lines(lines: List[str]) -> List[List[int]]:
        if not lines:
            return []
        try:
            encoded = tokenizer(lines, add_special_tokens=False)
            if isinstance(encoded, dict):
                input_ids = encoded.get("input_ids")
            else:
                input_ids = getattr(encoded, "input_ids", None)
            if isinstance(input_ids, list):
                return [list(tokens) for tokens in input_ids]
        except Exception:
            pass
        return [tokenizer.encode(text, add_special_tokens=False) for text in lines]

    def emit_sequences(tokens: List[int]) -> Iterator[Batch]:
        nonlocal total_sequences, hit_sequence_cap, token_buffer
        if not tokens:
            return
        token_buffer.extend(tokens)
        while len(token_buffer) >= seq_length + 1:
            if stop_event is not None and stop_event.is_set():
                raise TrainingCancelled()
            window = token_buffer[:seq_length + 1]
            seq_inputs.append(window[:-1])
            seq_targets.append(window[1:])
            total_sequences += 1
            del token_buffer[:seq_length]

            if len(seq_inputs) == batch_size:
                yield flush_batch()
            if max_sequences is not None and total_sequences >= max_sequences:
                hit_sequence_cap = True
                return

    def flush_pending_lines(*, force_log: bool = False) -> Iterator[Batch]:
        nonlocal pending_lines, pending_chars, last_heartbeat
        if not pending_lines:
            return
        now = time.time()
        if force_log or now - last_heartbeat >= 0.75:
            report_progress(heartbeat=True)
            last_heartbeat = now
        for tokens in tokenize_lines(pending_lines):
            yield from emit_sequences(tokens)
            if hit_sequence_cap:
                break
        pending_lines = []
        pending_chars = 0

    with data_path.open("r", encoding="utf-8") as corpus:
        for line in corpus:
            if stop_event is not None and stop_event.is_set():
                raise TrainingCancelled()
            total_lines += 1
            processed_bytes += len(line.encode("utf-8", errors="ignore"))
            pending_lines.append(line)
            pending_chars += len(line)
            report_progress()

            if len(pending_lines) >= 64 or pending_chars >= 32768:
                yield from flush_pending_lines(force_log=True)

            if max_sequences is not None and total_sequences >= max_sequences:
                break

        if pending_lines and not hit_sequence_cap:
            yield from flush_pending_lines(force_log=True)

    if seq_inputs:
        yield flush_batch()
    if not hit_sequence_cap:
        processed_bytes = total_bytes
    report_progress(force=True, done=True)


def _save_checkpoint(model: Any, save_dir: Path, step: int, *, backend: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"model_step_{step}.safetensors"
    if backend == "torch":
        if torch_save_file is None:
            raise RuntimeError("safetensors.torch is required to save torch checkpoints.")
        state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
        torch_save_file(state, str(checkpoint_path))
    else:
        if tg_get_state_dict is None or tg_safe_save is None:
            raise RuntimeError("tinygrad is required to save tinygrad checkpoints.")
        tg_safe_save(tg_get_state_dict(model), str(checkpoint_path))
    print(f"[checkpoint] saved to {checkpoint_path}")


def _release_training_runtime(
    runtime_state: Dict[str, Any],
    *,
    announce: Optional[Callable[[str], None]] = None,
) -> None:
    model = runtime_state.get("model")
    optimizer = runtime_state.get("optimizer")

    if announce is not None:
        announce("[info] Releasing training runtime...")

    if optimizer is not None:
        try:
            if hasattr(optimizer, "zero_grad"):
                zero_grad = optimizer.zero_grad
                try:
                    zero_grad(set_to_none=True)
                except TypeError:
                    zero_grad()
        except Exception:
            pass

    relieve_memory_pressure(model)
    runtime_state.clear()

    if announce is not None:
        announce("[info] Training runtime cleared.")


def _train_epoch(
    model: Any,
    optimizer: Any,
    batches: Iterable[Batch],
    grad_accum: int,
    *,
    backend: str,
    stop_event: Optional[Event] = None,
) -> float:
    model_loss = 0.0
    step_loss = 0.0
    steps = 0
    accum_window = max(1, grad_accum)
    start_time = time.time()
    last_display = start_time
    total_tokens = 0

    if backend == "torch":
        if torch is None:
            raise RuntimeError("PyTorch is required for torch training.")
        optimizer.zero_grad(set_to_none=True)
        original_mode = model.training
        model.train(True)
        if hasattr(model, "reset_kv_cache"):
            model.reset_kv_cache()
        try:
            for step, batch in enumerate(batches, start=1):
                if stop_event is not None and stop_event.is_set():
                    raise TrainingCancelled()
                logits = model(batch.input_ids, attention_mask=batch.attention_mask, position_ids=batch.position_ids)
                vocab_size = logits.shape[-1]
                logits = logits.reshape(-1, vocab_size)
                labels = batch.labels.reshape(-1).long()
                loss = _cross_entropy_loss(logits, labels, backend=backend)
                cost = (loss / accum_window)
                cost.backward()
                loss_value = float(loss.item())
                step_loss += loss_value

                seq_tokens = int(batch.input_ids.shape[0] * batch.input_ids.shape[1])
                total_tokens += seq_tokens
                now = time.time()
                if now - last_display >= 0.5:
                    spinner = SPIN_LOOP[step % len(SPIN_LOOP)]
                    elapsed = max(now - start_time, 1e-6)
                    tok_rate = total_tokens / elapsed
                    sys.stdout.write(
                        f"\r[train] {spinner} step={step} loss={loss_value:.4f} total_tok={total_tokens} tok/s={tok_rate:.1f}"
                    )
                    sys.stdout.flush()
                    last_display = now

                if step % accum_window == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                steps += 1
                model_loss += loss_value

                if step % 10 == 0:
                    avg = step_loss / min(step, 10)
                    elapsed = max(time.time() - start_time, 1e-6)
                    tok_rate = total_tokens / elapsed
                    print(f"\r[train] ✓ step={step} loss={avg:.4f} total_tok={total_tokens} tok/s={tok_rate:.1f}           ")
                    step_loss = 0.0

            if steps == 0:
                return math.nan
            if steps % accum_window != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            elapsed = max(time.time() - start_time, 1e-6)
            tok_rate = total_tokens / elapsed
            print(f"\r[train] done steps={steps} mean_loss={model_loss / steps:.4f} total_tok={total_tokens} tok/s={tok_rate:.1f}          ")
            return model_loss / steps
        finally:
            model.train(original_mode)

    if tg is None:
        raise RuntimeError("tinygrad is required for tinygrad training.")

    optimizer.zero_grad()
    original_training = tg.Tensor.training
    tg.Tensor.training = True
    try:
        for step, batch in enumerate(batches, start=1):
            if stop_event is not None and stop_event.is_set():
                raise TrainingCancelled()
            logits = model(batch.input_ids, attention_mask=batch.attention_mask, position_ids=batch.position_ids)
            vocab_size = logits.shape[-1]
            logits = logits.reshape(-1, vocab_size)
            labels = batch.labels.reshape(-1).cast(tg.dtypes.default_int)
            loss = _cross_entropy_loss(logits, labels, backend=backend)
            (loss / accum_window).backward()
            loss_value = float(loss.item())
            step_loss += loss_value

            seq_tokens = int(batch.input_ids.shape[0] * batch.input_ids.shape[1])
            total_tokens += seq_tokens
            now = time.time()
            if now - last_display >= 0.5:
                spinner = SPIN_LOOP[step % len(SPIN_LOOP)]
                elapsed = max(now - start_time, 1e-6)
                tok_rate = total_tokens / elapsed
                sys.stdout.write(
                    f"\r[train] {spinner} step={step} loss={loss_value:.4f} total_tok={total_tokens} tok/s={tok_rate:.1f}"
                )
                sys.stdout.flush()
                last_display = now

            if step % accum_window == 0:
                missing_grads = [p for p in optimizer.params if p.grad is None]
                for tensor in missing_grads:
                    tensor.grad = tg.Tensor.zeros_like(tensor)
                optimizer.step()
                optimizer.zero_grad()

            steps += 1
            model_loss += loss_value

            if step % 10 == 0:
                avg = step_loss / min(step, 10)
                elapsed = max(time.time() - start_time, 1e-6)
                tok_rate = total_tokens / elapsed
                print(f"\r[train] ✓ step={step} loss={avg:.4f} total_tok={total_tokens} tok/s={tok_rate:.1f}           ")
                step_loss = 0.0

        if steps == 0:
            return math.nan
        if steps % accum_window != 0:
            missing_grads = [p for p in optimizer.params if p.grad is None]
            for tensor in missing_grads:
                tensor.grad = tg.Tensor.zeros_like(tensor)
            optimizer.step()
            optimizer.zero_grad()

        elapsed = max(time.time() - start_time, 1e-6)
        tok_rate = total_tokens / elapsed
        print(f"\r[train] done steps={steps} mean_loss={model_loss / steps:.4f} total_tok={total_tokens} tok/s={tok_rate:.1f}          ")
        return model_loss / steps
    finally:
        tg.Tensor.training = original_training


def _run_training_job(
    settings: Dict[str, object],
    stop_event: Event,
    runtime_state: Optional[Dict[str, Any]] = None,
) -> None:
    args = _build_training_namespace(settings)
    backend = normalize_llm_backend(str(args.backend or get_llm_backend()))
    device = normalize_backend_device(str(args.device or get_backend_device(backend, default="")), backend)
    finetune = _finetune_requested(args)
    if runtime_state is None:
        runtime_state = {}

    print(f"[backend] using {backend} on {device}")
    if backend == "torch" and torch is None:
        raise RuntimeError("PyTorch is required for torch training.")
    if backend == "tinygrad" and tg is None:
        raise RuntimeError("tinygrad is required for tinygrad training.")

    Model = backend_model_class(backend)
    ModelConfig = backend_model_config_class(backend)
    load_safetensors = backend_helpers_module(backend).load_safetensors

    if args.gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be at least 1")
    if args.config_path is None and args.model_id is None:
        raise ValueError("Provide either a model id or a config path.")

    model_path: Optional[Path] = None
    remote_repo_hint: Optional[str] = None

    if args.config_path is not None:
        config_loader = ModelConfig()
        config_candidate = args.config_path
        if config_candidate.exists():
            config_loader.load(config_candidate)
        else:
            repo_id, filename = _parse_remote_identifier(config_candidate, "config.json")
            config_file = _fetch_model_file(repo_id, filename, backend=backend, local_only=args.offline)
            config_loader.load(config_file)
            remote_repo_hint = repo_id
        runtime_state["config"] = config_loader
        model_path = args.weights_dir
        if model_path is None:
            print("[warn] No weights directory supplied; switching to training new model.")
            finetune = False
    else:
        snapshot_path, config_loader = RepoHuggingFace(args.model_id, backend=backend).download()
        runtime_state["config"] = config_loader
        model_path = args.weights_dir or snapshot_path

    _ensure_required_keys(config_loader)
    if hasattr(config_loader, "model_config"):
        config_dict = config_loader.model_config
    elif hasattr(config_loader, "config"):
        config_dict = config_loader.config
    else:
        config_dict = config_loader

    model_name = args.model_id or remote_repo_hint or config_dict.get("model_type") or "custom"
    shard = Shard(
        model_name,
        start_layer=0,
        end_layer=config_dict["num_layers"],
        total_layers=config_dict["num_layers"] + 1,
    )

    model = Model(config_dict, shard, use_tied=config_dict.get("tie_word_embeddings", False))
    runtime_state["model"] = model
    if backend == "torch":
        model.to(device)

    if finetune:
        if model_path is None:
            raise ValueError("Weights directory must be provided when not training from scratch.")
        print("Loading pretrained weights...")
        load_safetensors(
            model,
            model_path,
            config_dict,
            weight_device=device,
            use_tied=config_dict.get("tie_word_embeddings", False),
        )

    tokenizer_id = args.model_id or remote_repo_hint
    tokenizer_asset = None
    if isinstance(tokenizer_id, str) and "/" in tokenizer_id:
        tokenizer_asset = _resolve_tokenizer_asset(tokenizer_id, backend=backend, local_only=args.offline)
    if tokenizer_id is None:
        raise ValueError("Tokenizer could not be resolved. Provide a model id for training.")

    tokenizer_kwargs: dict[str, object] = {"local_files_only": args.offline}
    if tokenizer_asset is not None:
        suffix = tokenizer_asset.suffix.lower()
        path_str = str(tokenizer_asset)
        if suffix == ".json":
            tokenizer_kwargs["tokenizer_file"] = path_str
            tokenizer_kwargs.setdefault("use_fast", True)
            tokenizer_kwargs.setdefault("legacy", False)
        else:
            tokenizer_kwargs["vocab_file"] = path_str
            tokenizer_kwargs.setdefault("legacy", True)

    print(f"[tokenizer] using '{tokenizer_id}'")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **tokenizer_kwargs)
    runtime_state["tokenizer"] = tokenizer

    data_path = args.data_path
    if data_path is None:
        if args.dataset_id is None:
            raise ValueError("Provide a local data path or a dataset id.")
        dataset_snapshot = _download_dataset(args.dataset_id)
        processed_dir = Path.cwd() / "datasets" / args.dataset_id.replace("/", "__")
        data_path = _prepare_dataset_corpus(
            dataset_snapshot,
            processed_dir,
            limit=args.max_dataset_entries,
        )
        print(f"[dataset] Using processed corpus at {data_path}")
    runtime_state["data_path"] = data_path

    if backend == "torch":
        optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=args.lr)
    else:
        assert tg is not None and tg_get_parameters is not None
        parameters = [param.requires_grad_(True) for param in tg_get_parameters(model)]
        optimizer = tg.nn.optim.Adam(parameters, lr=args.lr)
    runtime_state["optimizer"] = optimizer

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if stop_event.is_set():
            raise TrainingCancelled()
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        batches = _stream_corpus_batches(
            tokenizer,
            data_path,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            backend=backend,
            device=device,
            max_sequences=None,
            stop_event=stop_event,
        )
        avg_loss = _train_epoch(
            model,
            optimizer,
            batches,
            args.gradient_accumulation,
            backend=backend,
            stop_event=stop_event,
        )
        if math.isnan(avg_loss):
            raise RuntimeError("No training batches were produced. Check dataset size, sequence length, or batch size.")
        print(f"[epoch] {epoch} mean loss = {avg_loss:.4f}")
        if args.save_dir is not None:
            global_step += 1
            _save_checkpoint(model, args.save_dir, global_step, backend=backend)


class TrainScreen(Screen[None]):
    """Textual screen that presents training progress and controls."""

    CSS_PATH = Path(__file__).with_name("train_menu.tcss")

    BINDINGS = [
        ("escape", "pop_screen", "Back"),
        ("s", "start_training", "Start training"),
        ("x", "stop_training", "Stop training"),
        ("n", "open_network", "Network"),
        ("h", "open_help", "Help"),
    ]

    @staticmethod
    def _normalize_bool(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "y"}
        return bool(value)

    def __init__(self, peer_client: PeerClient) -> None:
        super().__init__()
        self._stats = TrainingStats()
        self._training: Optional[TrainingProcess] = None
        self._poll_timer = None
        self._resource_timer = None
        self._stat_labels: Dict[str, Label] = {}
        self._resource_labels: Dict[str, Label] = {}
        self._runtime_summary_label: Optional[Label] = None
        self._settings_summary: Dict[str, Label] = {}
        self._log: Optional[Log] = None
        self._settings: Dict[str, object] = default_training_settings()
        self._node_steps: List[TrainingNode] = [TrainingNode("Base Training")]
        self._current_node_index: Optional[int] = None
        self._path_output_root: Optional[Path] = None
        self._node_output_dirs: Dict[int, Path] = {}
        self._path_summary_label: Optional[Label] = None
        self._auto_training_runs = 1
        self._stopped_by_user = False
        self._stopped_for_memory = False
        self._peer_client = peer_client
        self._peer_label: Optional[Label] = None
        self._llm_backend = get_llm_backend()
        self._settings["device"] = get_backend_device(self._llm_backend, default="CPU") or "CPU"
        self._sync_base_node_name()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="train-root"):
            with Container(id="train-left"):
                yield Log(id="train-log", highlight=False, auto_scroll=True)
            with Container(id="train-right"):
                with VerticalScroll(id="train-right-scroll"):
                    with Container(id="train-right-content"):
                        with Static(id="stats-panel"):
                            yield Label("Status: Idle", id="stat-status")
                            yield Label("Step: 0", id="stat-step")
                            yield Label("Epoch: 0", id="stat-epoch")
                            yield Label("Loss: --", id="stat-loss")
                            yield Label("Tokens: 0", id="stat-tokens")
                            yield Label("Tok/s: --", id="stat-tok-rate")
                        with Static(id="settings-panel"):
                            yield Label("Settings", id="settings-title")
                            yield Label("Model: --", id="settings-model")
                            yield Label("Dataset: --", id="settings-data")
                            runtime_summary = Label("Runtime: --", id="training-runtime")
                            self._runtime_summary_label = runtime_summary
                            yield runtime_summary
                            yield Button("Edit Settings", id="open-settings")
                            path_summary = Label("Training Path: Base only", id="training-path-summary")
                            self._path_summary_label = path_summary
                            yield path_summary
                            yield Button("Training Path", id="open-training-path")
                        with Static(id="resource-panel"):
                            yield Label("CPU: --", id="resource-cpu")
                            yield Label("Memory: --", id="resource-ram")
                            yield Label("GPU: --", id="resource-gpu")
                            yield Label("Nodes: --", id="resource-peers")
        yield Footer()

    def apply_default_settings(self, defaults: Dict[str, object]) -> None:
        for key, value in defaults.items():
            if key not in self._settings:
                continue
            current = self._settings[key]
            if isinstance(current, bool):
                self._settings[key] = self._normalize_bool(value)
            else:
                if value is None:
                    self._settings[key] = ""
                else:
                    self._settings[key] = str(value).strip()
        if getattr(self, "_settings_summary", None):
            self._update_settings_summary()

    async def on_mount(self) -> None:
        self._log = self.query_one("#train-log", Log)
        if self._log is not None:
            self._log.clear()
        self._stat_labels = {
            "status": self.query_one("#stat-status", Label),
            "step": self.query_one("#stat-step", Label),
            "epoch": self.query_one("#stat-epoch", Label),
            "loss": self.query_one("#stat-loss", Label),
            "tokens": self.query_one("#stat-tokens", Label),
            "tok_rate": self.query_one("#stat-tok-rate", Label),
        }
        self._resource_labels = {
            "cpu": self.query_one("#resource-cpu", Label),
            "ram": self.query_one("#resource-ram", Label),
            "gpu": self.query_one("#resource-gpu", Label),
            "peers": self.query_one("#resource-peers", Label),
        }
        if self._runtime_summary_label is None:
            self._runtime_summary_label = self.query_one("#training-runtime", Label)
        self._settings_summary = {
            "model": self.query_one("#settings-model", Label),
            "data": self.query_one("#settings-data", Label),
        }
        if self._path_summary_label is None:
            self._path_summary_label = self.query_one("#training-path-summary", Label)
        self._update_settings_summary()
        await asyncio.to_thread(self._get_peer_count)
        self.set_interval(5.0, self._get_peer_count)

        self._poll_timer = self.set_interval(0.25, self._poll_training_output, pause=True)
        self._resource_timer = self.set_interval(1.0, self._update_resource_usage)

        # Prime CPU stats if psutil is available to avoid the initial 0.0 reading.
        if psutil is not None:
            _ = psutil.cpu_percent(interval=None)
        self._update_path_summary()

    def on_unmount(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None
        if self._resource_timer is not None:
            self._resource_timer.stop()
            self._resource_timer = None
        if self._training is not None and self._training.is_running():
            self._training.terminate()
    
    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_start_training(self) -> None:
        self._start_training()

    def action_stop_training(self) -> None:
        self._stop_training()

    def action_open_network(self) -> None:
        self.app.push_screen(OrchestrationScreen(self._peer_client))

    def action_open_help(self) -> None:
        self.app.push_screen(HelpScreen("Train Help", self._help_text()))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "open-settings":
            self._open_settings()
        elif button_id == "open-training-path":
            self._open_training_path()

    def _open_settings(self) -> None:
        screen = TrainSettingsScreen(dict(self._settings))
        self.app.push_screen(screen, self._on_settings_result)

    def _on_settings_result(self, result: Optional[Dict[str, object]]) -> None:
        if not result:
            return
        for key, value in result.items():
            if key not in self._settings:
                continue
            if key in {"offline", "from-scratch"}:
                self._settings[key] = self._normalize_bool(value)
            else:
                self._settings[key] = "" if value is None else str(value).strip()
        self._update_settings_summary()

    def _open_training_path(self) -> None:
        if self._training is not None and self._training.is_running():
            self._append_log("[warn] Stop training before editing the training path.")
            return
        screen = TrainingPathScreen(copy.deepcopy(self._node_steps))
        self.app.push_screen(screen, self._on_training_path_result)

    def _on_training_path_result(self, result: Optional[List[TrainingNode]]) -> None:
        if result is None:
            return
        if not result:
            result = [TrainingNode("Base Training")]
        self._node_steps = result
        self._current_node_index = None
        self._path_output_root = None
        self._node_output_dirs.clear()
        self._auto_training_runs = 1
        for node in self._node_steps:
            if node.status not in NODE_STATUS_STYLES:
                node.status = "pending"
        self._sync_base_node_name()
        self._append_log(f"[info] Training path updated ({len(self._node_steps)} step{'s' if len(self._node_steps) != 1 else ''}).")

    def _prepare_node_for_run(self) -> Optional[int]:
        self._stopped_by_user = False
        index = self._find_next_node()
        if index is None:
            self._auto_training_runs += 1
            name = f"Training Pass {self._auto_training_runs}"
            self._node_steps.append(TrainingNode(name, settings={"from-scratch": False}, repeated=True))
            self._append_log(f"[info] Added repeat training step '{name}'.")
            index = len(self._node_steps) - 1
        if index == 0:
            self._path_output_root = None
            self._node_output_dirs.clear()
        self._current_node_index = index
        self._set_node_status(index, "running")
        return index

    def _find_next_node(self) -> Optional[int]:
        for idx, node in enumerate(self._node_steps):
            if node.status != "complete":
                return idx
        return None

    def _set_node_status(self, index: int, status: str) -> None:
        if index < 0 or index >= len(self._node_steps):
            return
        self._node_steps[index].status = status
        self._update_path_summary()

    def _update_path_summary(self) -> None:
        if self._path_summary_label is None:
            return
        total = len(self._node_steps)
        if total == 0:
            summary = "No steps defined"
        else:
            completed = sum(1 for node in self._node_steps if node.status == "complete")
            running_node = next((node for node in self._node_steps if node.status == "running"), None)
            pending_node = next((node for node in self._node_steps if node.status not in {"complete"}), None)
            summary = f"{total} step{'s' if total != 1 else ''}"
            if completed:
                summary += f", {completed} done"
            if running_node is not None:
                summary += f", running '{running_node.name}'"
            elif pending_node is not None:
                summary += f", next '{pending_node.name}'"
        self._path_summary_label.update(f"Training Path: {summary}")

    @staticmethod
    def _slugify_step_name(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
        return slug or "step"

    def _resolve_step_output_root(self) -> Path:
        if self._path_output_root is not None:
            return self._path_output_root
        configured = str(self._settings.get("save-dir") or "").strip()
        if configured:
            root = Path(configured)
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            root = Path.cwd() / "checkpoints" / "training_path" / stamp
        self._path_output_root = root
        return root

    def _step_output_dir(self, index: int) -> Path:
        if index in self._node_output_dirs:
            return self._node_output_dirs[index]
        root = self._resolve_step_output_root()
        node = self._node_steps[index]
        path = root / f"step_{index + 1:02d}_{self._slugify_step_name(node.name)}"
        self._node_output_dirs[index] = path
        return path

    def _derive_base_node_label(self) -> str:
        model = str(
            self._settings.get("model-id")
            or self._settings.get("custom-model-id")
            or "Base Training"
        ).strip()
        if not model:
            model = "Base Training"
        max_len = 40
        if len(model) > max_len:
            model = model[: max_len - 3] + "..."
        return f"Base: {model}"

    def _sync_base_node_name(self) -> None:
        if not self._node_steps:
            return
        self._node_steps[0].name = self._derive_base_node_label()
        self._update_path_summary()

    def _start_training(self) -> None:
        if self._training is not None and self._training.is_running():
            self._append_log("[warn] Training is already running.")
            return

        guard_reason = memory_abort_reason("training startup")
        if guard_reason:
            self._append_log(f"[error] {guard_reason}")
            self._append_log("[error] Training start blocked by memory guard.")
            return

        node_index = self._prepare_node_for_run()
        if node_index is None:
            self._append_log("[error] No available training step.")
            return

        training_settings = self._build_training_settings(node_index=node_index)
        if training_settings is None:
            self._set_node_status(node_index, "pending")
            self._current_node_index = None
            return

        self._training = TrainingProcess(training_settings)
        self._stopped_for_memory = False
        try:
            self._training.start()
        except Exception as exc:  # pragma: no cover - defensive
            self._append_log(f"[error] Failed to start training: {exc}")
            self._set_node_status(node_index, "pending")
            self._current_node_index = None
            self._training = None
            return

        self._stats = TrainingStats(status="Running")
        self._update_stats_display()
        self._set_buttons(running=True)
        if self._poll_timer is not None:
            self._poll_timer.resume()
        self._append_log("[info] Launched training.")
        self._append_log(
            f"[info] Training backend: {self._llm_backend} "
            f"({backend_device_env(self._llm_backend)}={training_settings.get('device', '')})"
        )
        if int(training_settings.get("path-node-index", 0)) > 0:
            self._append_log(
                f"[info] Step mode: {'from scratch' if training_settings.get('from-scratch', False) else 'fine-tune previous step'}."
            )

    def _stop_training(self) -> None:
        if self._training is None or not self._training.is_running():
            self._append_log("[warn] No active training process.")
            return
        self._stopped_by_user = True
        self._training.terminate()

    def _build_training_settings(self, *, node_index: Optional[int] = None) -> Optional[Dict[str, object]]:
        settings = self._settings

        def get_str(key: str) -> str:
            value = settings.get(key, "")
            if isinstance(value, bool):
                return ""
            return str(value).strip()

        def get_step_override(node: TrainingNode, key: str) -> Optional[str]:
            if key not in node.settings:
                return None
            value = node.settings.get(key, "")
            if isinstance(value, bool):
                return ""
            return str(value).strip()

        node: Optional[TrainingNode] = None
        if node_index is not None and 0 <= node_index < len(self._node_steps):
            node = self._node_steps[node_index]

        def get_effective_str(key: str) -> str:
            if node is not None and node_index is not None and node_index > 0:
                override = get_step_override(node, key)
                if override is not None:
                    return override
            return get_str(key)

        model_id = get_str("model-id")
        if not model_id:
            self._append_log("[error] Provide a model id before starting training.")
            return None

        self._llm_backend = get_llm_backend()
        device = normalize_backend_device(
            get_str("device") or get_backend_device(self._llm_backend, default=""),
            self._llm_backend,
        )

        data_path = get_effective_str("data-path")
        dataset_id = get_effective_str("dataset-id")

        if not data_path and not dataset_id:
            self._append_log("[warn] No dataset provided; training will likely fail.")

        from_scratch = self._normalize_bool(settings.get("from-scratch", False))
        weights_dir = ""
        if node is not None and node_index is not None:
            if node_index > 0 and "from-scratch" in node.settings:
                from_scratch = self._normalize_bool(node.settings.get("from-scratch", False))
            if node_index == 0:
                step_mode = "scratch" if from_scratch else "fine-tune model"
            elif from_scratch:
                step_mode = "scratch"
            else:
                previous_output = self._node_output_dirs.get(node_index - 1)
                if previous_output is None:
                    previous_output = self._step_output_dir(node_index - 1)
                weights_dir = str(previous_output)
                step_mode = "fine-tune previous"
                self._append_log(
                    f"[info] Step '{node.name}' will fine-tune from {previous_output}."
                )
        else:
            step_mode = "scratch" if from_scratch else "fine-tune model"

        save_dir = str(self._settings.get("save-dir") or "").strip()
        if node_index is not None:
            save_dir = str(self._step_output_dir(node_index))

        return {
            "backend": self._llm_backend,
            "model-id": model_id,
            "weights-dir": weights_dir,
            "data-path": data_path,
            "dataset-id": dataset_id,
            "max-dataset-entries": get_effective_str("max-dataset-entries"),
            "seq-length": get_effective_str("seq-length") or "256",
            "batch-size": get_effective_str("batch-size") or "2",
            "epochs": get_effective_str("epochs") or "1",
            "lr": get_effective_str("lr") or "1e-4",
            "device": device,
            "gradient-accumulation": get_effective_str("gradient-accumulation") or "1",
            "save-dir": save_dir,
            "offline": self._normalize_bool(settings.get("offline", False)),
            "from-scratch": from_scratch,
            "path-node-index": node_index if node_index is not None else 0,
            "path-step-mode": step_mode,
        }

    def _poll_training_output(self) -> None:
        if self._training is None:
            if self._poll_timer is not None:
                self._poll_timer.pause()
            return

        if self._training.is_running():
            guard_reason = memory_abort_reason("training loop")
            if guard_reason:
                self._stopped_for_memory = True
                self._training.terminate()
                self._append_log(f"[error] {guard_reason}")
                self._append_log("[error] Terminating training process to avoid out-of-memory crash.")

        for line in self._training.drain():
            if line is None:
                self._handle_training_complete()
                return
            self._append_log(line)
            self._parse_training_line(line)

    def _handle_training_complete(self) -> None:
        failed = self._training.failed() if self._training is not None else False
        if failed:
            self._append_log("[error] Training ended with an error.")
        elif self._stopped_by_user:
            self._append_log("[info] Training stopped by user.")
        elif self._stopped_for_memory:
            self._append_log("[warn] Training stopped by memory guard.")
        else:
            self._append_log("[info] Training finished.")
        self._set_buttons(running=False)
        self._stats.status = "Idle"
        self._update_stats_display()
        if self._poll_timer is not None:
            self._poll_timer.pause()
        self._training = None
        if self._current_node_index is not None:
            if self._stopped_by_user:
                self._set_node_status(self._current_node_index, "stopped")
            elif self._stopped_for_memory:
                self._set_node_status(self._current_node_index, "stopped")
            else:
                self._set_node_status(self._current_node_index, "complete")
            self._current_node_index = None
        self._stopped_by_user = False
        self._stopped_for_memory = False

    def _append_log(self, line: str) -> None:
        if self._log is None:
            return
        fragments = line.replace("\r", "\n").splitlines() or [""]
        for fragment in fragments:
            self._log.write_line(fragment)

    def _update_settings_summary(self) -> None:
        model = str(self._settings.get("model-id") or "--")
        data = str(self._settings.get("dataset-id") or self._settings.get("data-path") or "--")
        self._llm_backend = get_llm_backend()
        if self._runtime_summary_label is not None:
            mode = "scratch" if self._normalize_bool(self._settings.get("from-scratch", False)) else "finetune"
            device = str(self._settings.get("device") or get_backend_device(self._llm_backend, default="")).strip() or "--"
            self._runtime_summary_label.update(
                f"Runtime: {self._llm_backend} on {device} ({mode})"
            )
        if self._settings_summary:
            self._settings_summary["model"].update(f"Model: {model}")
            self._settings_summary["data"].update(f"Dataset: {data}")
        if self._node_steps:
            self._node_steps[0].settings["from-scratch"] = self._normalize_bool(
                self._settings.get("from-scratch", False)
            )
        self._sync_base_node_name()

    def _set_buttons(self, *, running: bool) -> None:
        try:
            start_btn = self.query_one("#start-training", Button)
        except Exception:
            start_btn = None
        try:
            stop_btn = self.query_one("#stop-training", Button)
        except Exception:
            stop_btn = None
        if start_btn is not None:
            start_btn.disabled = running
        if stop_btn is not None:
            stop_btn.disabled = not running

    def _parse_training_line(self, line: str) -> None:
        step_match = re.search(r"step=(\d+)", line)
        loss_match = re.search(r"loss=([0-9]*\.?[0-9]+)", line)
        token_match = re.search(r"total_tok=(\d+)", line)
        tok_rate_match = re.search(r"tok/s=([0-9]*\.?[0-9]+)", line)
        epoch_header = re.search(r"=== Epoch (\d+)/(\d+) ===", line)
        epoch_summary = re.search(r"\[epoch\]\s+(\d+)\s+mean loss = ([0-9]*\.?[0-9]+)", line)
        done_summary = re.search(r"done steps=(\d+)\s+mean_loss=([0-9]*\.?[0-9]+)", line)

        if epoch_header:
            self._stats.epoch = int(epoch_header.group(1))
            self._stats.total_epochs = int(epoch_header.group(2))
            self._stats.status = "Epoch Running"
        if epoch_summary:
            self._stats.epoch = int(epoch_summary.group(1))
            self._stats.mean_loss = float(epoch_summary.group(2))
        if step_match:
            self._stats.step = int(step_match.group(1))
        if loss_match:
            self._stats.loss = float(loss_match.group(1))
        if token_match:
            self._stats.tokens = int(token_match.group(1))
        if tok_rate_match:
            self._stats.tok_rate = float(tok_rate_match.group(1))
        if done_summary:
            self._stats.step = int(done_summary.group(1))
            self._stats.mean_loss = float(done_summary.group(2))
            self._stats.status = "Completed"

        self._update_stats_display()

    def _update_stats_display(self) -> None:
        status = self._stats.status
        epoch_text = (
            f"{self._stats.epoch}/{self._stats.total_epochs}"
            if self._stats.total_epochs
            else str(self._stats.epoch)
        )
        loss_text = f"{self._stats.loss:.4f}" if self._stats.loss is not None else "--"
        tok_rate = f"{self._stats.tok_rate:.1f}" if self._stats.tok_rate is not None else "--"
        mean_loss_text = (
            f"{self._stats.mean_loss:.4f}"
            if self._stats.mean_loss is not None
            else "--"
        )

        self._stat_labels["status"].update(f"Status: {status}")
        self._stat_labels["step"].update(f"Step: {self._stats.step}")
        self._stat_labels["epoch"].update(f"Epoch: {epoch_text}")
        self._stat_labels["loss"].update(f"Loss: {loss_text} (mean {mean_loss_text})")
        self._stat_labels["tokens"].update(f"Tokens: {self._stats.tokens}")
        self._stat_labels["tok_rate"].update(f"Tok/s: {tok_rate}")


    def _update_resource_usage(self) -> None:
        if psutil is not None:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            self._resource_labels["cpu"].update(f"CPU: {cpu:.1f}%")
            mem_used_gb = mem.used / (1024 ** 3)
            mem_total_gb = mem.total / (1024 ** 3)
            self._resource_labels["ram"].update(
                f"Memory: {mem.percent:.1f}% ({mem_used_gb:.1f} / {mem_total_gb:.1f} GiB)"
            )
        else:
            load_avg = os.getloadavg()
            self._resource_labels["cpu"].update(
                f"Load Avg: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}"
            )
            self._resource_labels["ram"].update("Memory: --")

        # GPU metrics placeholder (extend in future when integrations are available)
        self._resource_labels["gpu"].update("GPU: N/A")
        devices = ", ".join(self._aggregate_devices()) or "local only"
        self._resource_labels["peers"].update(f"Nodes: {self._peer_client.peer_count()} ({devices})")

    def _aggregate_devices(self) -> List[str]:
        labels: List[str] = []
        seen: set[str] = set()
        for peer in self._peer_client.get_peers(include_self=True):
            label = self._device_label(peer)
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)
        return labels

    @staticmethod
    def _device_label(peer: object) -> str:
        gpu_model = str(getattr(peer, "gpu_model", "") or "").strip()
        if gpu_model:
            return gpu_model
        tg_device = str(getattr(peer, "tg_device", "") or "").strip()
        if tg_device:
            return tg_device
        cpu_model = str(getattr(peer, "cpu_model", "") or "").strip()
        if cpu_model:
            return cpu_model
        return ""

    def _get_peer_count(self) -> None:
        if self.app is None:
            return
        count = self._peer_client.peer_count()
        self.app.title = f"[Nodes: {count}]"

    @staticmethod
    def _help_text() -> str:
        return "\n".join(
            [
                "Train Screen",
                "- s: Start training",
                "- x: Stop training",
                "- n: Open network screen",
                "- h: Open this help screen",
                "- b / Esc: Back",
                "",
                "Buttons",
                "- Edit Settings: configure training inside the app",
                "- Training Path: edit sequential training steps",
            ]
        )


class TrainSettingsScreen(ModalScreen[Dict[str, object]]):
    """Modal for editing training invocation settings."""

    CSS_PATH = Path(__file__).with_name("train_menu.tcss")

    BINDINGS = [("escape", "pop_screen", "Back")]

    def __init__(self, values: Dict[str, object]) -> None:
        super().__init__(id="train-settings")
        self._initial = dict(values)
        self._inputs: Dict[str, Input | Checkbox] = {}

    def compose(self) -> ComposeResult:
        text_fields = [
            ("model-id", "Model ID", "Hugging Face repo"),
            ("data-path", "Data Path", "Local UTF-8 training corpus"),
            ("dataset-id", "Dataset ID", "Optional HF dataset identifier"),
            ("max-dataset-entries", "Max Entries", "Optional dataset entry cap"),
            ("seq-length", "Seq Length", "Default 256"),
            ("batch-size", "Batch Size", "Default 2"),
            ("epochs", "Epochs", "Default 1"),
            ("lr", "Learning Rate", "Default 1e-4"),
            ("gradient-accumulation", "Grad Accum", "Default 1"),
            ("save-dir", "Save Dir", "Optional checkpoint directory"),
        ]
        with Container(id="settings-modal-container"):
            yield Static("Training Settings", id="settings-modal-title")
            with VerticalScroll(id="settings-scroll"):
                for name, label, placeholder in text_fields:
                    yield Label(label, classes="settings-field-label")
                    widget = Input(id=f"settings-{name}", placeholder=placeholder)
                    widget.add_class("settings-field-input")
                    self._inputs[name] = widget
                    yield widget

                for name, label in [("offline", "Offline Mode"), ("from-scratch", "From Scratch")]:
                    widget = Checkbox(label, id=f"settings-{name}")
                    widget.add_class("settings-field-input")
                    self._inputs[name] = widget
                    yield widget
            with Container(id="settings-modal-buttons"):
                yield Button("Cancel", id="settings-cancel")
                yield Button("Apply", id="settings-apply", variant="primary")

    def on_mount(self) -> None:
        for name, widget in self._inputs.items():
            value = self._initial.get(name)
            if isinstance(widget, Checkbox):
                widget.value = bool(value)
            elif value is not None:
                widget.value = str(value)
        first = self._inputs.get("model-id")
        if isinstance(first, Input):
            self.call_after_refresh(first.focus)
    
    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-cancel":
            self.dismiss(None)
        elif event.button.id == "settings-apply":
            self.dismiss(self._gather_values())

    def _gather_values(self) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for name, widget in self._inputs.items():
            if isinstance(widget, Checkbox):
                result[name] = widget.value
            else:
                result[name] = widget.value.strip()
        return result
