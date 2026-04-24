from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import tinygrad as tg
try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
from transformers import AutoTokenizer

from cheetah.models.llm.backend import (
    RUNTIME_FINGERPRINT_PROTOCOL,
    backend_helpers_module,
    get_backend_device,
    runtime_asset_fingerprints,
)
from cheetah.models.shard import Shard
from cheetah.orchestration.model_engine import ModelEngine
from cheetah.logging_utils import get_logger

logger = get_logger(__name__)
TokenCallback = Callable[[int], None]
AbortCheck = Callable[[], str | None]


class MemoryPressureError(RuntimeError):
    """Raised when memory guard thresholds are exceeded during long-running loops."""


def _raise_if_abort(abort_check: AbortCheck | None) -> None:
    if abort_check is None:
        return
    reason = abort_check()
    if reason:
        raise MemoryPressureError(reason)


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
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
    device = input_ids.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    decode_token = getattr(model, "decode_token", None)
    use_decode_fast_path = callable(decode_token)

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

        if use_decode_fast_path:
            logits = decode_token(next_tok, start_pos=curr_pos)
        else:
            attention_mask = attention_mask.cat(
                tg.Tensor.ones((attention_mask.shape[0], 1), device=device), dim=1
            )
            position_ids = tg.Tensor([curr_pos], device=device, dtype=tg.dtypes.int32)
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
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
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

    if _model_has_full_local_shard(model):
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
    position_list = [[]]
    hidden_state_list = []
    seen_tokens = [int(v) for row in input_list for v in row]

    out_tokens: list[int] = []
    start_time = time.time()

    for step in range(max_new_tokens):
        _raise_if_abort(abort_check)
        prefill = step == 0
        logger.debug(f"[step: {step} Starting distributed generation with {len(peers)} peers for up to {max_new_tokens} tokens")
        logger.debug(f"Initial input_ids: {input_list}, attention_mask: {mask_list}")
        input_ids = _list_to_tensor(input_list, like=input_ids)
        attention_mask = _list_to_tensor(mask_list, like=attention_mask)
        otoken_data = engine.get_tokens(
            model,
            input_ids,
            attention_mask,
            tokenizer,
            position_ids=None,
            prefill=prefill,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            alpha_f=alpha_f,
            alpha_p=alpha_p,
            repetition_penalty=repetition_penalty,
            seen_tokens=seen_tokens,
        )
        transport_attention_mask = otoken_data.get("attention_mask", mask_list)
        transport_position_ids = otoken_data.get("position_ids", position_list)
        transport_hidden_state = otoken_data.get("hidden_state", hidden_state_list)

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
            peer_id = _peer_identifier(peer)
            request_timeout = _peer_generation_timeout_seconds(
                prefill=prefill,
                token_count=_flow_token_count(
                    {
                        "input_ids": input_list,
                        "attention_mask": transport_attention_mask,
                        "position_ids": transport_position_ids,
                        "hidden_state": transport_hidden_state,
                    }
                ),
            )
            payload = {
                "command": "generate_token",
                "payload": {
                    "sender_peer_id": local_peer_id,
                    "input_ids": input_list,
                    "attention_mask": transport_attention_mask,
                    "position_ids": transport_position_ids,
                    "hidden_state": transport_hidden_state,
                    "prefill": prefill,
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
                request_tokens = _flow_token_count(payload)
                _record_peer_flow(peer_client, local_peer_id or "local", peer_id, request_tokens, phase="request")
                logger.debug(
                    "Sending payload to peer %s",
                    peer_id,
                )
                host = str(getattr(peer, "ip_address", "") or getattr(peer, "address", "0.0.0.0"))
                port = int(getattr(peer, "port", os.getenv("TC_TENSOR_PORT", 1045)))
                address = (host, port)
                resp = peer_client.send_payload(
                    payload,
                    expect_reply=True,
                    address=address,
                    timeout=request_timeout,
                )
                if isinstance(resp, dict) and resp.get("error"):
                    raise RuntimeError(str(resp.get("error")))
                _mark_peer_seen(peer_client, peer_id)
                response_tokens = _flow_token_count(resp)
                _record_peer_flow(peer_client, peer_id, local_peer_id or "local", response_tokens, phase="response")
                otoken_data = resp
                transport_attention_mask = resp.get("attention_mask", transport_attention_mask)
                transport_position_ids = resp.get("position_ids", transport_position_ids)
                transport_hidden_state = resp.get("hidden_state", transport_hidden_state)
                logger.debug("Received token response from peer %s", peer_id)
            except TimeoutError as err:
                phase = "prefill" if prefill else "decode"
                logger.error(
                    "Timed out waiting for peer %s during %s after %.1fs",
                    peer_id,
                    phase,
                    request_timeout,
                )
                raise TimeoutError(
                    f"peer {peer_id} timed out during {phase} after {request_timeout:.1f}s"
                ) from err
            except Exception as err:
                logger.error(f"Error communicating with peer {peer_id}: {err}")
                raise

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


def distributed_shard_log_messages(
    peer_client: Any,
    *,
    model_name: str,
    total_layers: int,
) -> list[str]:
    get_peers = getattr(peer_client, "get_peers", None)
    if not callable(get_peers):
        return []
    local_peer_id = str(getattr(peer_client, "peer_client_id", "") or "")
    return distributed_shard_plan_messages(
        get_peers(include_self=True),
        local_peer_id=local_peer_id,
        model_name=model_name,
        total_layers=total_layers,
    )


def distributed_shard_plan_messages(
    peers: Any,
    *,
    local_peer_id: str,
    model_name: str,
    total_layers: int,
) -> list[str]:
    normalized = planned_peer_shards(
        peers,
        model_name=model_name,
        total_layers=total_layers,
    )
    if len(normalized) <= 1 or int(total_layers or 0) <= 1:
        return []

    lines = [f"Using {len(normalized)} nodes for shard-aware execution."]
    for index, peer in enumerate(normalized):
        peer_id = str(_peer_value(peer, "peer_client_id", "") or f"peer-{index + 1}")
        label = _peer_log_label(peer, fallback=peer_id)
        shard = _peer_value(peer, "shard", None)
        if shard is None:
            continue
        scope = "local shard" if str(peer_id) == str(local_peer_id) else "shard on peer"
        lines.append(f"Loading {scope} {label}: {format_shard_span(shard)}.")
    return lines


def total_layers_from_model_config(model_config: Any) -> int:
    if not isinstance(model_config, dict):
        return 0
    try:
        num_layers = int(model_config.get("num_layers", 0) or 0)
    except (TypeError, ValueError):
        return 0
    return num_layers + 1 if num_layers > 0 else 0


def planned_peer_shards(
    peers: Any,
    *,
    model_name: str,
    total_layers: int,
) -> list[Any]:
    normalized = _normalize_peer_entries(peers)
    total_layers_int = int(total_layers or 0)
    transformer_layers = max(total_layers_int - 1, 0)
    if len(normalized) <= 1 or transformer_layers <= 1:
        return normalized[:1] if normalized else []

    usable_count = min(len(normalized), transformer_layers)
    normalized = normalized[:usable_count]

    planning_peers: list[Any] = []
    for index, peer in enumerate(normalized):
        peer_id = str(_peer_value(peer, "peer_client_id", "") or f"peer-{index + 1}")
        planning_peer = peer
        if isinstance(peer, dict) or not hasattr(peer, "gpu_vram") or not hasattr(peer, "cpu_ram") or not hasattr(peer, "gpu_flops"):
            planning_peer = SimpleNamespace(
                peer_client_id=peer_id,
                gpu_vram=_peer_value(peer, "gpu_vram", 0.0),
                cpu_ram=_peer_value(peer, "cpu_ram", 0.0),
                gpu_flops=_peer_value(peer, "gpu_flops", 0.0),
            )
            if hasattr(peer, "__dict__"):
                for attr, value in vars(peer).items():
                    setattr(planning_peer, attr, value)
            elif isinstance(peer, dict):
                for attr, value in peer.items():
                    setattr(planning_peer, attr, value)
        planning_peers.append(planning_peer)

    ModelEngine.plan_shards(planning_peers, model_name or "model", total_layers_int)
    return planning_peers


def build_peer_load_plan(
    peer_client: Any,
    *,
    model_name: str,
    total_layers: int,
) -> dict[str, Any]:
    get_peers = getattr(peer_client, "get_peers", None)
    if not callable(get_peers):
        return {
            "distributed": False,
            "peers": [],
            "remote_peers": [],
            "local_peer": None,
            "local_shard": None,
        }

    peers = planned_peer_shards(
        get_peers(include_self=True),
        model_name=model_name,
        total_layers=total_layers,
    )
    local_peer_id = str(getattr(peer_client, "peer_client_id", "") or "")
    local_peer = next(
        (
            peer for peer in peers
            if str(_peer_value(peer, "peer_client_id", "") or "") == local_peer_id
        ),
        peers[0] if peers else None,
    )
    remote_peers = [
        peer for peer in peers
        if str(_peer_value(peer, "peer_client_id", "") or "") != local_peer_id
    ]
    local_shard = _peer_value(local_peer, "shard", None) if local_peer is not None else None
    return {
        "distributed": len(peers) > 1 and int(total_layers or 0) > 1,
        "peers": peers,
        "remote_peers": remote_peers,
        "local_peer": local_peer,
        "local_shard": local_shard,
    }


def load_model_shards_on_peers(
    peer_client: Any,
    *,
    model_id: str,
    backend: str,
    offline_mode: bool,
    total_layers: int,
    peers: Any | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    if peers is None:
        plan = build_peer_load_plan(peer_client, model_name=model_id, total_layers=total_layers)
    else:
        planned = planned_peer_shards(peers, model_name=model_id, total_layers=total_layers)
        plan = {
            "distributed": len(planned) > 1 and int(total_layers or 0) > 1,
            "peers": planned,
        }
    if "remote_peers" not in plan:
        planned = plan.get("peers", [])
        local_peer_id = str(getattr(peer_client, "peer_client_id", "") or "")
        plan["remote_peers"] = [
            peer for peer in planned
            if str(_peer_value(peer, "peer_client_id", "") or "") != local_peer_id
        ]
        local_peer = next(
            (
                peer for peer in planned
                if str(_peer_value(peer, "peer_client_id", "") or "") == local_peer_id
            ),
            planned[0] if planned else None,
        )
        plan["local_peer"] = local_peer
        plan["local_shard"] = _peer_value(local_peer, "shard", None) if local_peer is not None else None

    if not plan.get("distributed"):
        return plan

    load_timeout = _peer_model_load_timeout_seconds() if timeout is None else timeout
    remote_results: list[dict[str, Any]] = []
    for peer in plan["remote_peers"]:
        peer_id = _peer_identifier(peer)
        host = str(_peer_value(peer, "ip_address", "") or _peer_value(peer, "address", "0.0.0.0"))
        port = int(_peer_value(peer, "port", os.getenv("TC_TENSOR_PORT", 1045)))
        response = peer_client.send_payload(
            {
                "command": "load_model",
                "payload": {
                    "model_id": model_id,
                    "backend": backend,
                    "offline_mode": bool(offline_mode),
                    "shard": _peer_shard_payload(peer),
                },
            },
            expect_reply=True,
            address=(host, port),
            timeout=load_timeout,
        )
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(f"{_peer_log_label(peer, fallback=peer_id)}: {response.get('error')}")
        _mark_peer_seen(peer_client, peer_id)
        remote_results.append(
            {
                "peer": peer,
                "response": response,
            }
        )

    plan["remote_results"] = remote_results
    return plan


def clear_model_shards_on_peers(
    peer_client: Any,
    *,
    peers: Any | None = None,
    model_id: str | None = None,
    timeout: float | None = None,
) -> None:
    get_peers = getattr(peer_client, "get_peers", None)
    if peers is None:
        if not callable(get_peers):
            return
        peers = get_peers(include_self=False)

    remote_peers = _normalize_peer_entries(peers)
    if not remote_peers:
        return

    clear_timeout = _peer_model_load_timeout_seconds() if timeout is None else timeout
    for peer in remote_peers:
        host = str(_peer_value(peer, "ip_address", "") or _peer_value(peer, "address", "0.0.0.0"))
        port = int(_peer_value(peer, "port", os.getenv("TC_TENSOR_PORT", 1045)))
        try:
            peer_client.send_payload(
                {
                    "command": "clear_model",
                    "payload": {
                        "model_id": str(model_id or ""),
                    },
                },
                expect_reply=True,
                address=(host, port),
                timeout=clear_timeout,
            )
            _mark_peer_seen(peer_client, _peer_identifier(peer))
        except Exception:
            logger.debug("Failed to clear peer model runtime", exc_info=True)


def format_shard_span(shard: Any) -> str:
    start_layer = int(getattr(shard, "start_layer", 0) or 0)
    end_layer = int(getattr(shard, "end_layer", 0) or 0)
    total_layers = int(getattr(shard, "total_layers", end_layer + 1) or (end_layer + 1))
    transformer_layers = max(total_layers - 1, 1)
    return f"transformer layers {start_layer}:{end_layer} of {transformer_layers}"


def local_runtime_fingerprints(*, model_config: Any, model_path: str | Path | None) -> dict[str, str]:
    return runtime_asset_fingerprints(
        model_config=model_config,
        model_path=model_path,
    )


def validate_peer_runtime_fingerprints(
    remote_results: list[dict[str, Any]],
    *,
    local_model_config: Any,
    local_model_path: str | Path | None,
) -> list[str]:
    local = local_runtime_fingerprints(
        model_config=local_model_config,
        model_path=local_model_path,
    )
    mismatches: list[str] = []
    for entry in remote_results:
        peer = entry.get("peer")
        response = entry.get("response", {}) if isinstance(entry.get("response"), dict) else {}
        label = _peer_log_label(peer, fallback=_peer_identifier(peer))
        remote_protocol = response.get("fingerprint_protocol")
        remote_config_fp = str(response.get("config_fingerprint", "") or "")
        remote_tokenizer_fp = str(response.get("tokenizer_fingerprint", "") or "")
        if remote_protocol is None:
            mismatches.append(
                f"{label} is running an older peer fingerprint protocol; update and restart that node"
            )
            continue
        try:
            remote_protocol_int = int(remote_protocol)
        except (TypeError, ValueError):
            mismatches.append(f"{label} reported an invalid peer fingerprint protocol")
            continue
        if remote_protocol_int != RUNTIME_FINGERPRINT_PROTOCOL:
            mismatches.append(
                f"{label} peer fingerprint protocol {remote_protocol_int} != {RUNTIME_FINGERPRINT_PROTOCOL}"
            )
            continue
        if not remote_config_fp:
            mismatches.append(f"{label} did not report a model config fingerprint")
            continue
        if remote_config_fp and remote_config_fp != local["config_fingerprint"]:
            mismatches.append(f"{label} model config fingerprint mismatch")
        if local["tokenizer_fingerprint"] and not remote_tokenizer_fp:
            mismatches.append(f"{label} did not report a tokenizer fingerprint")
        elif local["tokenizer_fingerprint"] and remote_tokenizer_fp != local["tokenizer_fingerprint"]:
            mismatches.append(f"{label} tokenizer fingerprint mismatch")
    return mismatches


def _peer_model_load_timeout_seconds() -> float:
    raw = (os.getenv("TC_PEER_MODEL_LOAD_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return 180.0
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 180.0


def _peer_generation_timeout_seconds(*, prefill: bool, token_count: int = 0) -> float:
    env_name = "TC_PEER_PREFILL_TIMEOUT_SECONDS" if prefill else "TC_PEER_DECODE_TIMEOUT_SECONDS"
    default_timeout = 60.0 if prefill else 30.0
    raw = (os.getenv(env_name) or "").strip()
    if raw:
        try:
            return max(float(raw), 0.1)
        except ValueError:
            pass

    if prefill and token_count > 0:
        # Give larger prompts more time on slower peers without forcing every
        # decode step to wait the full prefill ceiling.
        return max(default_timeout, 15.0 + (float(token_count) / 8.0))
    return default_timeout


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
        detached = tensor.detach().cpu()
        try:
            data = detached.numpy().tolist()
        except TypeError:
            if detached.is_floating_point():
                data = detached.to(dtype=torch.float32).numpy().tolist()
            else:
                raise
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


def _peer_value(peer: Any, attr: str, default: Any = None) -> Any:
    if isinstance(peer, dict):
        return peer.get(attr, default)
    return getattr(peer, attr, default)


def _peer_identifier(peer: Any) -> str:
    return str(_peer_value(peer, "peer_client_id", "") or _peer_value(peer, "ip_address", "") or "unknown")


def _peer_log_label(peer: Any, *, fallback: str) -> str:
    peer_id = str(_peer_value(peer, "peer_client_id", "") or "").strip() or fallback
    host = str(_peer_value(peer, "ip_address", "") or _peer_value(peer, "address", "")).strip()
    if host in {"", "0.0.0.0"} or host == peer_id:
        return peer_id
    return f"{peer_id} ({host})"


def _flow_token_count(message: Any) -> int:
    payload = message.get("payload", message) if isinstance(message, dict) else message
    if not isinstance(payload, dict):
        return 0

    for key in ("attention_mask", "position_ids", "input_ids", "hidden_state"):
        count = _token_count_from_value(payload.get(key))
        if count > 0:
            return count

    if payload.get("token") is not None or payload.get("tensor") is not None:
        return 1
    return 0


def _token_count_from_value(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        if not value:
            return 0
        if isinstance(value[0], list):
            return max((len(row) for row in value if isinstance(row, list)), default=0)
        return len(value)
    if isinstance(value, dict):
        shape = value.get("shape")
        if isinstance(shape, list) and shape:
            index = 1 if len(shape) >= 2 else 0
            try:
                return max(int(shape[index]), 0)
            except (TypeError, ValueError):
                return 0
    try:
        shape = tuple(getattr(value, "shape", ()) or ())
    except Exception:
        shape = ()
    if len(shape) >= 2:
        try:
            return max(int(shape[1]), 0)
        except (TypeError, ValueError):
            return 0
    if len(shape) == 1:
        try:
            return max(int(shape[0]), 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _record_peer_flow(peer_client: Any, source: str, target: str, tokens: int, *, phase: str) -> None:
    recorder = getattr(peer_client, "record_flow", None)
    if callable(recorder):
        recorder(source, target, tokens, phase=phase)


def _mark_peer_seen(peer_client: Any, peer_client_id: str) -> None:
    marker = getattr(peer_client, "mark_peer_seen", None)
    if callable(marker):
        marker(peer_client_id)


def _shard_matches_total(shard: Any, total_layers: int) -> bool:
    try:
        start_layer = int(getattr(shard, "start_layer", 0) or 0)
        end_layer = int(getattr(shard, "end_layer", 0) or 0)
        shard_total = int(getattr(shard, "total_layers", 0) or 0)
    except (TypeError, ValueError):
        return False
    return start_layer < end_layer and shard_total == int(total_layers)


def _plan_peer_shards(model_engine: ModelEngine, peer_client: Any, model: Any) -> None:
    if peer_client is None:
        return
    total_layers = _infer_total_layers(model)
    if total_layers <= 0:
        return

    loaded_local_shard = getattr(model, "shard", None)
    if _shard_matches_total(loaded_local_shard, total_layers):
        model_engine.shard = loaded_local_shard
        try:
            peer_client.shard = loaded_local_shard
        except Exception:
            pass
        peer_device = getattr(peer_client, "peer_device", None)
        if peer_device is not None and not _shard_matches_total(getattr(peer_device, "shard", None), total_layers):
            try:
                peer_device.shard = loaded_local_shard
            except Exception:
                pass
        return

    model_name = str(getattr(model, "model_name", "") or getattr(model, "name", "") or "model")
    peers = planned_peer_shards(
        peer_client.get_peers(include_self=True),
        model_name=model_name,
        total_layers=total_layers,
    )
    if len(peers) <= 1:
        return
    local_shard = getattr(getattr(peer_client, "peer_device", None), "shard", None)
    if local_shard is not None:
        model_engine.shard = local_shard
        try:
            peer_client.shard = local_shard
        except Exception:
            pass


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


def _model_has_full_local_shard(model: Any) -> bool:
    total_layers = _infer_total_layers(model)
    if total_layers <= 1:
        return False
    shard = getattr(model, "shard", None)
    if shard is None:
        return False
    try:
        return int(getattr(shard, "start_layer", 0) or 0) == 0 and int(getattr(shard, "end_layer", 0) or 0) >= total_layers - 1
    except (TypeError, ValueError):
        return False


def _local_engine(model: Any) -> ModelEngine:
    shard = getattr(model, "shard", None)
    if shard is not None:
        return ModelEngine(shard=shard)
    total_layers = _infer_total_layers(model)
    if total_layers <= 0:
        return ModelEngine()
    return ModelEngine(shard=Shard("local", 0, total_layers - 1, total_layers))


def _peer_shard_payload(peer: Any) -> dict[str, int | str]:
    shard = _peer_value(peer, "shard", None)
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
