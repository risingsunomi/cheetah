from __future__ import annotations

import argparse
import asyncio
import gc
import os
import signal
import sys
import time
from typing import Any

import numpy as np
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]

from cheetah.models.llm.backend import (
    get_backend_device,
    load_model_for_backend,
    resolve_model_assets_for_backend,
    set_backend_device,
    set_llm_backend,
)
from cheetah.models.shard import Shard
from cheetah.orchestration.distributed_inference import (
    build_peer_load_plan,
    distributed_shard_log_messages,
    format_shard_span,
    load_model_shards_on_peers,
    streaming_generate,
    streaming_generate_with_peers,
    total_layers_from_model_config,
    validate_peer_runtime_fingerprints,
)
from cheetah.orchestration.model_engine import ModelEngine
from cheetah.orchestration.peer_client import PeerClient


def main(argv: list[str] | None = None) -> int:
    if load_dotenv is not None:
        load_dotenv()

    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_runtime(args)

    if args.command == "serve":
        return _serve(args)
    if args.command == "generate":
        return _generate(args)
    if args.command == "compare-local":
        return _compare_local(args)

    parser.error("missing command")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a distributed inference probe without starting the TUI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="start a peer RPC server")
    _add_runtime_args(serve, default_port=8765)
    serve.add_argument("--model", default="", help="optional model to preload on this server")
    serve.add_argument("--shard", default="", help="optional preload shard as start:end:total_layers")
    serve.add_argument("--offline", action="store_true", help="only use locally cached model files")

    generate = subparsers.add_parser("generate", help="load shards and run a prompt")
    _add_runtime_args(generate, default_port=8766)
    generate.add_argument("--model", required=True, help="Hugging Face model id or local model path")
    generate.add_argument(
        "--peer",
        action="append",
        default=[],
        help="remote peer as host:port or peer_id@host:port; repeat for more peers",
    )
    generate.add_argument("--prompt", required=True, help="prompt or user message to generate from")
    generate.add_argument("--system", default="", help="optional system message when using the chat template")
    generate.add_argument("--no-chat-template", action="store_true", help="tokenize --prompt literally")
    generate.add_argument("--offline", action="store_true", help="only use locally cached model files")
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--repetition-penalty", type=float, default=1.0)
    generate.add_argument("--stream", action="store_true", help="stream decoded text as tokens arrive")
    generate.add_argument(
        "--trace-tensors",
        action="store_true",
        default=_env_flag("TC_TRACE_TENSOR_TRANSFERS"),
        help="print distributed tensor transfer shapes, sizes, and remote KV cache positions",
    )

    compare = subparsers.add_parser(
        "compare-local",
        help="compare full-model generation against a local split-shard pipeline",
    )
    _add_runtime_args(compare, default_port=8766)
    compare.add_argument("--model", required=True, help="Hugging Face model id or local model path")
    compare.add_argument("--prompt", required=True, help="prompt or user message to generate from")
    compare.add_argument("--system", default="", help="optional system message when using the chat template")
    compare.add_argument("--no-chat-template", action="store_true", help="tokenize --prompt literally")
    compare.add_argument("--offline", action="store_true", help="only use locally cached model files")
    compare.add_argument("--parts", type=int, default=2, help="number of local shards to simulate")
    compare.add_argument("--max-new-tokens", type=int, default=8)
    compare.add_argument("--temperature", type=float, default=0.0)
    compare.add_argument("--top-k", type=int, default=0)
    compare.add_argument("--top-p", type=float, default=1.0)
    compare.add_argument("--repetition-penalty", type=float, default=1.0)
    return parser


def _add_runtime_args(parser: argparse.ArgumentParser, *, default_port: int) -> None:
    parser.add_argument("--backend", choices=("tinygrad", "torch"), default=os.getenv("TC_LLM_BACKEND", "tinygrad"))
    parser.add_argument("--device", default="", help="backend device override, e.g. CPU, METAL, mps, cuda")
    parser.add_argument("--bind", default=os.getenv("TC_BIND_ADDRESS", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TC_PORT", str(default_port))))
    parser.add_argument("--peer-id", default=os.getenv("TC_PEER_ID", ""))


def _configure_runtime(args: argparse.Namespace) -> None:
    set_llm_backend(args.backend)
    if args.device:
        set_backend_device(args.device, args.backend)
    os.environ["TC_BIND_ADDRESS"] = str(args.bind)
    os.environ["TC_PORT"] = str(int(args.port))
    if args.peer_id:
        os.environ["TC_PEER_ID"] = str(args.peer_id)


def _serve(args: argparse.Namespace) -> int:
    client = PeerClient()
    print(
        f"peer ready id={client.peer_client_id} address={client.address}:{client.port} "
        f"backend={args.backend} device={get_backend_device(args.backend)}",
        flush=True,
    )

    if args.model:
        if not args.shard:
            raise SystemExit("--shard is required when preloading --model in serve mode")
        shard = _parse_shard(args.shard, model_name=args.model)
        model, model_config, tokenizer, model_path = asyncio.run(
            load_model_for_backend(
                model_id=args.model,
                shard=shard,
                weight_device=None,
                offline_mode=bool(args.offline),
                backend=args.backend,
            )
        )
        client.register_generation_runtime(
            model=model,
            tokenizer=tokenizer,
            backend=args.backend,
            model_id=args.model,
            model_config=model_config,
            model_path=str(model_path),
            shard=getattr(model, "shard", None),
        )
        print(f"preloaded {args.model}: {format_shard_span(getattr(model, 'shard', shard))}", flush=True)

    stop = False

    def _stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    while not stop:
        time.sleep(0.5)

    client.stop_tensor_recv = True
    client.stop_udp_discovery = True
    client.stop_udp_broadcast = True
    print("peer stopped", flush=True)
    return 0


def _generate(args: argparse.Namespace) -> int:
    client = PeerClient()
    for peer_spec in args.peer:
        _add_manual_peer(client, peer_spec, backend=args.backend)

    model_config, _ = asyncio.run(
        resolve_model_assets_for_backend(
            args.model,
            offline_mode=bool(args.offline),
            backend=args.backend,
        )
    )
    total_layers = total_layers_from_model_config(model_config)
    peer_plan = build_peer_load_plan(
        client,
        model_name=args.model,
        total_layers=total_layers,
    )

    for line in distributed_shard_log_messages(client, model_name=args.model, total_layers=total_layers):
        print(line, flush=True)

    local_shard = peer_plan.get("local_shard") if peer_plan.get("distributed") else None
    model, loaded_config, tokenizer, model_path = asyncio.run(
        load_model_for_backend(
            model_id=args.model,
            shard=local_shard,
            weight_device=None,
            offline_mode=bool(args.offline),
            backend=args.backend,
        )
    )
    client.register_generation_runtime(
        model=model,
        tokenizer=tokenizer,
        backend=args.backend,
        model_id=args.model,
        model_config=loaded_config,
        model_path=str(model_path),
        shard=getattr(model, "shard", None),
    )
    if local_shard is not None:
        print(f"local ready: {format_shard_span(local_shard)}", flush=True)
    else:
        print("local ready: full model", flush=True)

    if peer_plan.get("distributed"):
        peer_load_plan = load_model_shards_on_peers(
            client,
            model_id=args.model,
            backend=args.backend,
            offline_mode=bool(args.offline),
            total_layers=total_layers,
            peers=peer_plan.get("peers"),
        )
        mismatches = validate_peer_runtime_fingerprints(
            peer_load_plan.get("remote_results", []),
            local_model_config=loaded_config,
            local_model_path=model_path,
        )
        if mismatches:
            raise RuntimeError("; ".join(mismatches))
        for entry in peer_load_plan.get("remote_results", []):
            peer = entry.get("peer")
            response = entry.get("response", {}) if isinstance(entry.get("response"), dict) else {}
            status = "already loaded" if response.get("already_loaded") else "loaded"
            print(f"remote {status}: {_peer_label(peer)} {format_shard_span(getattr(peer, 'shard', None))}", flush=True)

    input_ids, attention_mask = _tokenize_prompt(
        tokenizer,
        backend=args.backend,
        prompt=args.prompt,
        system=args.system,
        use_chat_template=not bool(args.no_chat_template),
    )

    def _on_token(token_id: int) -> None:
        if not args.stream:
            return
        piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        print(piece, end="", flush=True)

    out_tokens, elapsed = streaming_generate_with_peers(
        client,
        model,
        input_ids,
        attention_mask,
        tokenizer,
        max_new_tokens=int(args.max_new_tokens),
        temp=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        on_token=_on_token,
        trace_transfers=bool(args.trace_tensors),
        on_transfer=_print_transfer_event if args.trace_tensors else None,
    )

    if args.stream:
        print("", flush=True)

    text = tokenizer.decode(out_tokens, skip_special_tokens=False)
    print(f"\ntokens: {out_tokens}", flush=True)
    print(f"elapsed: {elapsed:.3f}s ({(len(out_tokens) / elapsed) if elapsed > 0 else float('inf'):.2f} tok/s)", flush=True)
    print("output:")
    print(text)
    return 0


def _compare_local(args: argparse.Namespace) -> int:
    full_model, loaded_config, tokenizer, _ = asyncio.run(
        load_model_for_backend(
            model_id=args.model,
            shard=None,
            weight_device=None,
            offline_mode=bool(args.offline),
            backend=args.backend,
        )
    )
    total_layers = total_layers_from_model_config(loaded_config)
    if total_layers <= 1:
        raise RuntimeError(f"model config does not expose transformer layers: total_layers={total_layers}")

    input_ids, attention_mask = _tokenize_prompt(
        tokenizer,
        backend=args.backend,
        prompt=args.prompt,
        system=args.system,
        use_chat_template=not bool(args.no_chat_template),
    )

    print(
        f"full model ready: backend={args.backend} device={get_backend_device(args.backend)} "
        f"layers={total_layers - 1}",
        flush=True,
    )
    full_tokens, full_elapsed = streaming_generate(
        full_model,
        input_ids,
        attention_mask,
        tokenizer,
        max_new_tokens=int(args.max_new_tokens),
        temp=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
    )
    print(f"full tokens:  {full_tokens}", flush=True)
    print(f"full text:    {tokenizer.decode(full_tokens, skip_special_tokens=False)!r}", flush=True)
    print(f"full elapsed: {full_elapsed:.3f}s", flush=True)

    del full_model
    gc.collect()

    shards = _build_local_compare_shards(
        model_name=args.model,
        total_layers=total_layers,
        parts=int(args.parts),
    )
    split_models: list[Any] = []
    for shard in shards:
        model, _, _, _ = asyncio.run(
            load_model_for_backend(
                model_id=args.model,
                shard=shard,
                weight_device=None,
                offline_mode=bool(args.offline),
                backend=args.backend,
            )
        )
        split_models.append(model)
        print(f"split ready:  {format_shard_span(shard)}", flush=True)

    split_input_ids, split_attention_mask = _tokenize_prompt(
        tokenizer,
        backend=args.backend,
        prompt=args.prompt,
        system=args.system,
        use_chat_template=not bool(args.no_chat_template),
    )
    split_tokens, split_elapsed = _generate_local_split(
        split_models,
        split_input_ids,
        split_attention_mask,
        tokenizer,
        max_new_tokens=int(args.max_new_tokens),
        temp=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        backend=args.backend,
    )
    print(f"split tokens: {split_tokens}", flush=True)
    print(f"split text:   {tokenizer.decode(split_tokens, skip_special_tokens=False)!r}", flush=True)
    print(f"split elapsed: {split_elapsed:.3f}s", flush=True)

    if full_tokens == split_tokens:
        print("compare-local: PASS full and split token streams match", flush=True)
        return 0

    diff_at = next(
        (idx for idx, pair in enumerate(zip(full_tokens, split_tokens)) if pair[0] != pair[1]),
        min(len(full_tokens), len(split_tokens)),
    )
    full_at = full_tokens[diff_at] if diff_at < len(full_tokens) else None
    split_at = split_tokens[diff_at] if diff_at < len(split_tokens) else None
    print(
        "compare-local: FAIL token streams diverged "
        f"at generated index {diff_at}: full={full_at} split={split_at}",
        flush=True,
    )
    return 1


def _build_local_compare_shards(*, model_name: str, total_layers: int, parts: int) -> list[Shard]:
    transformer_layers = max(int(total_layers) - 1, 0)
    if transformer_layers <= 0:
        return []
    part_count = max(1, min(int(parts or 1), transformer_layers))
    base_span, remainder = divmod(transformer_layers, part_count)
    shards: list[Shard] = []
    start = 0
    for index in range(part_count):
        span = base_span + (1 if index < remainder else 0)
        end = min(start + span, transformer_layers)
        shards.append(Shard(model_name, start, end, int(total_layers)))
        start = end
    return shards


def _generate_local_split(
    models: list[Any],
    input_ids: Any,
    attention_mask: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    temp: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    backend: str,
) -> tuple[list[int], float]:
    if not models:
        return [], 0.0

    engines = [ModelEngine(shard=getattr(model, "shard", None)) for model in models]
    input_list = _tensor_to_nested_list(input_ids)
    mask_list = _tensor_to_nested_list(attention_mask)
    seen_tokens = [int(v) for row in input_list for v in row]
    out_tokens: list[int] = []

    start_time = time.time()
    for step in range(max_new_tokens):
        prefill = step == 0
        msg: dict[str, Any] | None = None
        step_input_ids = _nested_list_to_tensor(input_list, like=input_ids, integer=True, backend=backend)
        step_attention_mask = _nested_list_to_tensor(mask_list, like=attention_mask, integer=True, backend=backend)

        for index, (engine, model) in enumerate(zip(engines, models)):
            if index == 0:
                msg = engine.get_tokens(
                    model,
                    step_input_ids,
                    step_attention_mask,
                    tokenizer,
                    prefill=prefill,
                    temp=temp,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    seen_tokens=seen_tokens,
                )
                continue

            assert msg is not None
            decoded = engine.recv_tokens(msg, tokenizer, backend=backend)
            hidden_state = decoded.get("hidden_state")
            next_attention_mask = decoded.get("attention_mask", step_attention_mask)
            position_ids = decoded.get("position_ids")
            msg = engine.get_tokens(
                model,
                step_input_ids,
                next_attention_mask,
                tokenizer,
                hidden_state=hidden_state,
                position_ids=position_ids,
                prefill=prefill,
                temp=temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seen_tokens=seen_tokens,
            )

        assert msg is not None
        final_msg = engines[-1].recv_tokens(msg, tokenizer, backend=backend)
        attention_mask_out = final_msg.get("attention_mask")
        if attention_mask_out is not None:
            mask_list = _tensor_to_nested_list(attention_mask_out)

        token = final_msg.get("token")
        if token is None:
            raise RuntimeError(f"local split final shard did not return a token at step {step}")

        tok = int(token)
        out_tokens.append(tok)
        seen_tokens.append(tok)
        input_list[0].append(tok)
        if final_msg.get("end_token", False):
            break

    return out_tokens, time.time() - start_time


def _tokenize_prompt(
    tokenizer: Any,
    *,
    backend: str,
    prompt: str,
    system: str,
    use_chat_template: bool,
) -> tuple[Any, Any]:
    rendered = prompt
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    enc = tokenizer(rendered, return_tensors="np")
    ids = np.asarray(enc["input_ids"], dtype=np.int64)
    mask = np.asarray(enc["attention_mask"], dtype=np.int64)

    if backend == "torch":
        import torch

        device = _torch_device_name()
        return (
            torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(mask, dtype=torch.long, device=device),
        )

    import tinygrad as tg

    device = get_backend_device("tinygrad", default="CPU")
    return (
        tg.Tensor(ids.astype(np.int32), device=device),
        tg.Tensor(mask.astype(np.int32), device=device),
    )


def _tensor_to_nested_list(tensor: Any) -> list[list[Any]]:
    if tensor is None:
        return [[]]
    if isinstance(tensor, list):
        if not tensor:
            return [[]]
        if isinstance(tensor[0], list):
            return tensor
        return [tensor]
    try:
        import torch

        if isinstance(tensor, torch.Tensor):
            data = tensor.detach().cpu().tolist()
        elif hasattr(tensor, "numpy"):
            data = tensor.numpy().tolist()
        elif hasattr(tensor, "tolist"):
            data = tensor.tolist()
        else:
            data = []
    except Exception:
        if hasattr(tensor, "numpy"):
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


def _nested_list_to_tensor(
    data: list[list[Any]],
    *,
    like: Any,
    integer: bool,
    backend: str,
) -> Any:
    if backend == "torch":
        import torch

        device = _torch_device_name()
        dtype = torch.long if integer else getattr(like, "dtype", torch.float32)
        return torch.tensor(data, dtype=dtype, device=device)

    import tinygrad as tg

    device = get_backend_device("tinygrad", default="CPU")
    tensor = tg.Tensor(data, device=device)
    if integer:
        tensor = tensor.cast(tg.dtypes.int32)
    return tensor


def _torch_device_name() -> str:
    device = str(get_backend_device("torch", default="cpu") or "cpu").strip().lower()
    if device in {"metal", "mps"}:
        return "mps"
    return device


def _print_transfer_event(event: dict[str, Any]) -> None:
    name = str(event.get("event", "transfer"))
    phase = str(event.get("phase", ""))
    step = event.get("step", "")
    peer = str(event.get("peer", "local"))
    shard = _format_shard_payload(event.get("shard"))
    byte_count = int(event.get("bytes", 0) or 0)
    prefix = f"[trace] step={step} phase={phase} event={name} peer={peer}"
    if shard:
        prefix += f" shard={shard}"
    if byte_count:
        prefix += f" bytes={byte_count}"
    print(prefix, flush=True)

    tensors = event.get("tensors")
    if isinstance(tensors, dict) and tensors:
        _print_tensor_group("tensors", tensors)
    for group_name in ("request", "response"):
        group = event.get(group_name)
        if isinstance(group, dict) and group:
            _print_tensor_group(group_name, group)

    diagnostics = event.get("diagnostics")
    if isinstance(diagnostics, dict):
        cache = diagnostics.get("kv_cache")
        if isinstance(cache, dict):
            print(
                "[trace]   kv_cache: "
                f"layers={cache.get('layer_count')} "
                f"min={cache.get('min_cache_pos')} "
                f"max={cache.get('max_cache_pos')} "
                f"positions={cache.get('cache_pos')}",
                flush=True,
            )
        output = diagnostics.get("output")
        if isinstance(output, dict):
            print(f"[trace]   output{_format_tensor_summary(output)}", flush=True)


def _print_tensor_group(name: str, tensors: dict[str, Any]) -> None:
    parts = [f"{key}{_format_tensor_summary(value)}" for key, value in tensors.items()]
    print(f"[trace]   {name}: {', '.join(parts)}", flush=True)


def _format_tensor_summary(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    shape = value.get("shape")
    dtype = value.get("dtype", "")
    device = value.get("device", "")
    bytes_count = value.get("bytes")
    extra = []
    if dtype:
        extra.append(str(dtype))
    if device:
        extra.append(str(device))
    if bytes_count:
        extra.append(f"{bytes_count}B64")
    if value.get("numel") is not None:
        extra.append(f"n={value.get('numel')}")
    if value.get("nan"):
        extra.append(f"nan={value.get('nan')}")
    if value.get("inf"):
        extra.append(f"inf={value.get('inf')}")
    for stat_name in ("min", "max", "mean"):
        stat_value = value.get(stat_name)
        if isinstance(stat_value, (int, float)):
            extra.append(f"{stat_name}={float(stat_value):.4g}")
    suffix = f" ({', '.join(extra)})" if extra else ""
    return f" shape={shape}{suffix}"


def _format_shard_payload(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    return f"{value.get('start_layer')}:{value.get('end_layer')}/{value.get('total_layers')}"


def _env_flag(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _add_manual_peer(client: PeerClient, peer_spec: str, *, backend: str) -> None:
    peer_id, host, port = _parse_peer(peer_spec)
    client.add_peer(
        {
            "peer_client_id": peer_id,
            "address": host,
            "ip_address": host,
            "port": port,
            "peer_device": {
                "peer_client_id": peer_id,
                "ip_address": host,
                "port": port,
                "tg_device": get_backend_device(backend, default="CPU") or "CPU",
                "cpu_ram": "1",
                "gpu_vram": "",
                "gpu_flops": 0.0,
            },
        },
        source_address=host,
    )


def _parse_peer(value: str) -> tuple[str, str, int]:
    raw = value.strip()
    if not raw:
        raise argparse.ArgumentTypeError("peer cannot be empty")
    peer_id = ""
    target = raw
    if "@" in raw:
        peer_id, target = raw.split("@", 1)
    host, port_text = target.rsplit(":", 1)
    port = int(port_text)
    if not peer_id:
        peer_id = f"{host}:{port}"
    return peer_id, host, port


def _parse_shard(value: str, *, model_name: str) -> Shard:
    parts = value.replace(",", ":").split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shard must be start:end:total_layers")
    start_layer, end_layer, total_layers = (int(part) for part in parts)
    return Shard(model_name, start_layer, end_layer, total_layers)


def _peer_label(peer: Any) -> str:
    peer_id = str(getattr(peer, "peer_client_id", "") or "unknown")
    host = str(getattr(peer, "ip_address", "") or getattr(peer, "address", ""))
    port = str(getattr(peer, "port", ""))
    if host:
        return f"{peer_id} ({host}:{port})"
    return peer_id


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
