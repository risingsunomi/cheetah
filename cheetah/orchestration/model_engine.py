from __future__ import annotations

import json
import base64
import struct
from typing import Any, Dict, List, Sequence

import numpy as np
import tinygrad as tg
try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

from cheetah.models.llm.backend import backend_helpers_module, get_backend_device
from cheetah.models.shard import Shard


class ModelEngine:
    """Lightweight token generation and shard planning."""
    def __init__(
        self,
        shard: Shard | None = None,
    ) -> None:
        self.shard = shard or Shard("local", 0, 0, 0)

    def get_tokens(
        self,
        model: Any,
        input_ids: Any,
        attention_mask: Any,
        tokenizer: Any,
        hidden_state: Any | None = None,
        position_ids: Any | None = None,
        *,
        prefill: bool = False,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.8,
        alpha_f: float = 0.0,
        alpha_p: float = 0.0,
        repetition_penalty: float = 1.0,
        seen_tokens: Sequence[int] | None = None,
        trace: bool = False,
    ) -> Dict[str, Any]:
        """Generate the next token and return a JSON-serializable payload."""
        phase = "prefill" if prefill else "decode"
        if prefill:
            reset_kv_cache = getattr(model, "reset_kv_cache", None)
            if callable(reset_kv_cache):
                reset_kv_cache()

            if position_ids is None:
                position_ids = _full_position_ids_tensor(
                    attention_mask,
                    like=input_ids if hidden_state is None else hidden_state,
                )

            model_output = _run_model_shard(
                model,
                input_ids if hidden_state is None else None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                hidden_state=hidden_state,
                shard=self.shard,
                start_pos=None,
            )
        else:
            if hidden_state is not None:
                curr_pos = int(attention_mask.shape[1] - 1)
                _ensure_decode_cache_ready(model, curr_pos)
                if position_ids is None:
                    position_ids = _position_ids_tensor(curr_pos, hidden_state)
                model_output = _run_model_shard(
                    model,
                    None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    hidden_state=hidden_state,
                    shard=self.shard,
                    start_pos=curr_pos,
                )
            else:
                curr_pos = int(attention_mask.shape[1])
                _ensure_decode_cache_ready(model, curr_pos)
                prev_token = _scalar_int(input_ids[:, -1], default=0)
                next_tok = _next_token_tensor(prev_token, input_ids)
                attention_mask = _append_attention_mask(attention_mask)
                if position_ids is None:
                    position_ids = _position_ids_tensor(curr_pos, input_ids)
                model_output = _run_model_shard(
                    model,
                    next_tok,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    hidden_state=None,
                    shard=self.shard,
                    start_pos=curr_pos,
                )

        is_final = _is_final_shard(self.shard)
        if not is_final:
            resp_data = {
                "hidden_state": _encode_tensor(model_output),
                "attention_mask": _encode_tensor(attention_mask),
                "position_ids": _encode_tensor(position_ids),
                "shard": _shard_payload(self.shard),
                "end_token": False,
            }
            if trace:
                resp_data["diagnostics"] = _diagnostics_payload(
                    model,
                    phase=phase,
                    prefill=prefill,
                    shard=self.shard,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    hidden_state=hidden_state,
                    output=model_output,
                )
            return resp_data

        next_logit = model_output[:, -1, :].flatten()
        tok = _sample_with_backend(
            next_logit,
            temp=temp,
            k=top_k,
            p=top_p,
            af=alpha_f,
            ap=alpha_p,
            repetition_penalty=repetition_penalty,
            seen_tokens=list(seen_tokens or []),
        ).item()
        tok = int(tok)
        end_token = bool(getattr(tokenizer, "eos_token_id", None) == tok)

        resp_data = {
            "token": _encode_token_tensor(tok),
            "tensor": _encode_token_tensor(tok),
            "attention_mask": _encode_tensor(attention_mask),
            "position_ids": _encode_tensor(position_ids),
            "shard": _shard_payload(self.shard),
            "end_token": end_token,
        }
        if trace:
            resp_data["diagnostics"] = _diagnostics_payload(
                model,
                phase=phase,
                prefill=prefill,
                shard=self.shard,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                hidden_state=hidden_state,
                output=model_output,
                token=tok,
            )
        return resp_data

    def recv_tokens(
        self,
        payload: Any,
        tokenizer: Any | None = None,
        backend: str | None = None,
    ) -> Dict[str, Any]:
        """Normalize a token payload for chat/training consumers."""
        if isinstance(payload, (bytes, bytearray)):
            try:
                payload = json.loads(payload.decode("utf-8"))
            except Exception:
                return {}
        elif isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                return {}

        if not isinstance(payload, dict):
            return {}

        msg = payload.get("payload", payload)
        if not isinstance(msg, dict):
            return {}

        token = msg.get("token")
        if isinstance(token, dict):
            token = _decode_tensor(token, backend=backend)
        if token is None:
            token = _decode_tensor(msg.get("tensor"), backend=backend)
        token_int = _scalar_int(token)
        if token_int is not None:
            msg["token"] = token_int

        hidden_state = _decode_tensor(msg.get("hidden_state"), backend=backend)
        if hidden_state is not None:
            msg["hidden_state"] = hidden_state

        attention_mask = _decode_tensor(msg.get("attention_mask"), backend=backend)
        if attention_mask is not None:
            msg["attention_mask"] = attention_mask

        position_ids = _decode_tensor(msg.get("position_ids"), backend=backend)
        if position_ids is not None:
            msg["position_ids"] = position_ids

        if "shard" not in msg:
            msg["shard"] = _shard_payload(self.shard)

        if tokenizer is not None and token_int is not None:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            msg["end_token"] = bool(msg.get("end_token", False) or token_int == eos_id)
        else:
            msg.setdefault("end_token", False)

        return msg

    @staticmethod
    def plan_shards(peers: Sequence[Any], model_name: str, total_layers: int) -> List[Shard]:
        if total_layers <= 1:
            return []

        transformer_layers = max(int(total_layers) - 1, 1)
        usable_peers = list(peers[: min(len(peers), transformer_layers)])
        if not usable_peers:
            return []

        base_span, remainder = divmod(transformer_layers, len(usable_peers))
        shards: List[Shard] = []
        start = 0
        for index, peer in enumerate(usable_peers):
            span = base_span + (1 if index < remainder else 0)
            end = min(start + span, transformer_layers)
            shards.append(Shard(model_name=model_name, start_layer=start, end_layer=end, total_layers=total_layers))
            try:
                peer.shard = shards[-1]
            except Exception:
                pass
            start = end
        return shards


def _sample_with_backend(*args: Any, **kwargs: Any):
    # Distributed generation is tinygrad-based today, but resolve through backend utility
    # so we don't hardcode tinygrad module paths in callers.
    if args:
        logits = args[0]
        if torch is not None and isinstance(logits, torch.Tensor):
            return backend_helpers_module("torch").sample(*args, **kwargs)
        if isinstance(logits, tg.Tensor):
            return backend_helpers_module("tinygrad").sample(*args, **kwargs)
    try:
        return backend_helpers_module().sample(*args, **kwargs)
    except Exception:
        return backend_helpers_module("tinygrad").sample(*args, **kwargs)


def _run_model_shard(
    model: Any,
    x: Any,
    *,
    attention_mask: Any,
    position_ids: Any,
    hidden_state: Any | None,
    shard: Shard,
    start_pos: int | None,
) -> Any:
    run_shard = getattr(model, "run_shard", None)
    if callable(run_shard):
        return run_shard(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            hidden_state=hidden_state,
            shard=shard,
            start_pos=start_pos,
        )

    decode_hidden = getattr(model, "decode_hidden", None)
    if hidden_state is not None and start_pos is not None and callable(decode_hidden):
        return decode_hidden(hidden_state, position_ids=position_ids, start_pos=start_pos)

    decode_token = getattr(model, "decode_token", None)
    if hidden_state is None and start_pos is not None and callable(decode_token):
        return decode_token(x, position_ids=position_ids, start_pos=start_pos)

    return model(
        x,
        attention_mask=attention_mask,
        position_ids=position_ids,
        hidden_state=hidden_state,
    )


def _encode_token_tensor(token: int) -> Dict[str, Any]:
    buf = struct.pack("<i", int(token))
    return {
        "buffer": base64.b64encode(buf).decode("ascii"),
        "shape": [1, 1],
        "dtype": "int32",
    }

def _encode_tensor(tensor: Any) -> Dict[str, Any]:
    if torch is not None and isinstance(tensor, torch.Tensor):
        detached = tensor.detach().cpu().contiguous()
        if detached.dtype == torch.bfloat16:
            raw_view = detached.view(torch.float16).numpy()
            # print(f"raw_view: {raw_view}")
            return {
                "buffer": base64.b64encode(raw_view.tobytes()).decode("ascii"),
                "shape": list(detached.shape),
                "dtype": "bfloat16",
            }
        try:
            arr = detached.numpy()
        except TypeError:
            if detached.is_floating_point():
                arr = detached.to(dtype=torch.float32).numpy()
            else:
                raise
    elif hasattr(tensor, "numpy"):
        arr = tensor.numpy()
    else:
        arr = np.asarray(tensor)
    return {
        "buffer": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def _decode_tensor(tensor_payload: Any, backend: str | None = None) -> Any | None:
    if not isinstance(tensor_payload, dict):
        return None
    buf = tensor_payload.get("buffer")
    if not buf:
        return None
    try:
        raw = base64.b64decode(buf)
        dtype = _normalize_dtype(str(tensor_payload.get("dtype", "float32")))
        shape = tensor_payload.get("shape")
        selected_backend = str(backend or "").strip().lower()

        if dtype == "bfloat16":
            if selected_backend == "torch" and torch is not None:
                arr = _bfloat16_buffer_to_float32(raw, shape)
                tensor = torch.from_numpy(arr).to(dtype=torch.bfloat16)
                target_device = _torch_target_device()
                try:
                    tensor = tensor.to(device=target_device)
                except Exception:
                    pass
                return tensor

            arr = _bfloat16_buffer_to_float32(raw, shape)
        else:
            arr = np.frombuffer(raw, dtype=np.dtype(_numpy_dtype(dtype)))
            if shape:
                arr = arr.reshape(shape)
            arr = np.array(arr, copy=True)

        if selected_backend == "torch" and torch is not None:
            try:
                tensor = torch.from_numpy(arr)
            except Exception:
                tensor = torch.from_numpy(arr.astype(np.float32))
            target_device = _torch_target_device()
            try:
                tensor = tensor.to(device=target_device)
            except Exception:
                pass
            return tensor
        target_device = get_backend_device("tinygrad", default="CPU")
        return tg.Tensor(arr, device=target_device)
    except Exception:
        return None


def _scalar_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return int(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, tg.Tensor):
        try:
            return int(value.item())
        except Exception:
            return default
    try:
        return int(value)
    except Exception:
        return default


def _next_token_tensor(token: int, like: Any) -> Any:
    if torch is not None and isinstance(like, torch.Tensor):
        return torch.tensor([[int(token)]], device=like.device, dtype=torch.long)
    return tg.Tensor([[int(token)]], device=getattr(like, "device", None))


def _append_attention_mask(attention_mask: Any) -> Any:
    if torch is not None and isinstance(attention_mask, torch.Tensor):
        return torch.cat(
            (
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ),
            dim=1,
        )
    return attention_mask.cat(
        tg.Tensor.ones((attention_mask.shape[0], 1), device=attention_mask.device),
        dim=1,
    )


def _full_position_ids_tensor(attention_mask: Any, *, like: Any) -> Any:
    if attention_mask is None:
        seq_len = _sequence_length(like)
        if torch is not None and isinstance(like, torch.Tensor):
            return torch.arange(seq_len, device=like.device, dtype=torch.long).unsqueeze(0)
        device = getattr(like, "device", None)
        return tg.Tensor.arange(seq_len, device=device).reshape(1, seq_len).cast(tg.dtypes.int32)

    if torch is not None and isinstance(attention_mask, torch.Tensor):
        mask = attention_mask.long()
        return (mask.cumsum(dim=1) - 1) * mask

    mask = attention_mask.cast(tg.dtypes.int32)
    return (mask.cumsum(axis=1) - 1) * mask


def _position_ids_tensor(position: int, like: Any) -> Any:
    batch = int(getattr(like, "shape", [1])[0]) if getattr(like, "shape", None) else 1

    if torch is not None and isinstance(like, torch.Tensor):
        return torch.full(
            (batch, 1),
            int(position),
            device=like.device,
            dtype=torch.long,
        )

    return tg.Tensor(
        [[int(position)] for _ in range(batch)],
        device=getattr(like, "device", None),
        dtype=tg.dtypes.int32,
    )


def _sequence_length(value: Any) -> int:
    try:
        shape = tuple(getattr(value, "shape", ()) or ())
    except Exception:
        shape = ()
    if len(shape) >= 2:
        return int(shape[1])
    if len(shape) == 1:
        return int(shape[0])
    return 1


def _is_final_shard(shard: Shard) -> bool:
    try:
        return int(shard.end_layer) >= int(shard.total_layers) - 1
    except Exception:
        return False


def _torch_target_device() -> str:
    configured = str(get_backend_device("torch", default="cpu") or "cpu").strip().lower()
    if configured in {"metal", "mps"}:
        return "mps"
    if configured.startswith("cuda"):
        return configured
    return "cpu"


def _normalize_dtype(dtype: str) -> str:
    lower = dtype.lower()
    for candidate in (
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint8",
    ):
        if candidate in lower:
            return candidate
    return "float32"


def _numpy_dtype(dtype: str) -> str:
    normalized = _normalize_dtype(dtype)
    if normalized == "bfloat16":
        return "float32"
    return normalized


def _bfloat16_buffer_to_float32(raw: bytes, shape: Any) -> np.ndarray:
    raw_values = np.frombuffer(raw, dtype=np.uint16)
    if shape:
        raw_values = raw_values.reshape(shape)
    widened = raw_values.astype(np.uint32) << 16
    return np.array(widened.view(np.float32), copy=True)


def _diagnostics_payload(
    model: Any,
    *,
    phase: str,
    prefill: bool,
    shard: Shard,
    input_ids: Any,
    attention_mask: Any,
    position_ids: Any,
    hidden_state: Any,
    output: Any,
    token: int | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "phase": str(phase),
        "prefill": bool(prefill),
        "shard": _shard_payload(shard),
        "input_ids": _tensor_meta(input_ids),
        "attention_mask": _tensor_meta(attention_mask),
        "position_ids": _tensor_meta(position_ids),
        "hidden_state": _tensor_meta(hidden_state),
        "output": _tensor_meta(output),
        "kv_cache": _kv_cache_meta(model),
    }
    if token is not None:
        payload["token"] = int(token)
    return payload


def _tensor_meta(value: Any) -> Dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        shape = value.get("shape")
        dtype = value.get("dtype")
        buf = value.get("buffer")
        return {
            "shape": list(shape) if isinstance(shape, list) else shape,
            "dtype": str(dtype or ""),
            "bytes": len(buf) if isinstance(buf, str) else 0,
            "encoded": True,
        }
    try:
        shape = tuple(getattr(value, "shape", ()) or ())
    except Exception:
        shape = ()
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if shape:
        return {
            "shape": list(shape),
            "dtype": str(dtype or ""),
            "device": str(device or ""),
        }
    if isinstance(value, (list, tuple)):
        return {
            "shape": _nested_shape(value),
            "dtype": type(value).__name__,
        }
    return {
        "shape": [],
        "dtype": type(value).__name__,
    }


def _nested_shape(value: Any) -> list[int]:
    shape: list[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape


def _kv_cache_meta(model: Any) -> Dict[str, Any]:
    layers = list(getattr(model, "layers", []) or [])
    positions: list[int | None] = []
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        cache = getattr(attn, "kv_cache", None) if attn is not None else None
        cache_pos = getattr(cache, "cache_pos", None) if cache is not None else None
        try:
            positions.append(int(cache_pos) if cache_pos is not None else None)
        except (TypeError, ValueError):
            positions.append(None)

    populated = [pos for pos in positions if pos is not None]
    return {
        "layer_count": len(layers),
        "cache_pos": positions,
        "min_cache_pos": min(populated) if populated else None,
        "max_cache_pos": max(populated) if populated else None,
    }


def _ensure_decode_cache_ready(model: Any, curr_pos: int) -> None:
    if curr_pos <= 0:
        return
    meta = _kv_cache_meta(model)
    if int(meta.get("layer_count", 0) or 0) <= 0:
        return
    positions = meta.get("cache_pos", [])
    if not isinstance(positions, list) or not positions:
        return
    bad_positions = [pos for pos in positions if pos != curr_pos]
    if bad_positions:
        raise RuntimeError(
            "Shard KV cache is not primed for decode: "
            f"expected cache_pos={curr_pos}, actual={positions}. "
            "The shard likely missed the prefill request or received a mismatched decode step."
        )


def _shard_payload(shard: Shard) -> Dict[str, Any]:
    return {
        "model_name": shard.model_name,
        "start_layer": shard.start_layer,
        "end_layer": shard.end_layer,
        "total_layers": shard.total_layers,
    }
