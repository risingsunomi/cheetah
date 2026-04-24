from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from typing import Any

import numpy as np

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
    streaming_generate_with_peers,
    total_layers_from_model_config,
    validate_peer_runtime_fingerprints,
)
from cheetah.orchestration.peer_client import PeerClient


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_runtime(args)

    if args.command == "serve":
        return _serve(args)
    if args.command == "generate":
        return _generate(args)

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
    )

    if args.stream:
        print("", flush=True)

    text = tokenizer.decode(out_tokens, skip_special_tokens=False)
    print(f"\ntokens: {out_tokens}", flush=True)
    print(f"elapsed: {elapsed:.3f}s ({(len(out_tokens) / elapsed) if elapsed > 0 else float('inf'):.2f} tok/s)", flush=True)
    print("output:")
    print(text)
    return 0


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


def _torch_device_name() -> str:
    device = str(get_backend_device("torch", default="cpu") or "cpu").strip().lower()
    if device in {"metal", "mps"}:
        return "mps"
    return device


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
