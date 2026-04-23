from __future__ import annotations

import base64
from collections import deque
import gc
import json
import os
import queue
import socket
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from contextlib import closing
import asyncio

from cheetah.logging_utils import get_logger
from cheetah.models.llm.backend import (
    RUNTIME_FINGERPRINT_PROTOCOL,
    get_backend_device,
    get_llm_backend,
    load_model_for_backend,
    runtime_asset_fingerprints,
)
from cheetah.models.shard import Shard
from cheetah.orchestration.model_engine import ModelEngine, _decode_tensor
from cheetah.orchestration.cdevice import CDevice

logger = get_logger(__name__)


def _is_unspecified_address(value: str | None) -> bool:
    text = str(value or "").strip()
    return text in {"", "0.0.0.0", "::", "::0"}


def _is_loopback_address(value: str | None) -> bool:
    text = str(value or "").strip()
    return text.startswith("127.") or text == "::1"


def _peer_stale_after_seconds() -> float:
    raw = (os.getenv("TC_PEER_STALE_SECONDS") or "").strip()
    if not raw:
        return 8.0
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 8.0


def _resolve_advertise_address(bind_address: str) -> str:
    override = os.getenv("TC_ADVERTISE_ADDRESS", "").strip()
    if override:
        return override

    bind_address = bind_address.strip() or "0.0.0.0"
    if not _is_unspecified_address(bind_address):
        return bind_address

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            candidate = str(sock.getsockname()[0]).strip()
            if candidate and not _is_unspecified_address(candidate) and not _is_loopback_address(candidate):
                return candidate
    except OSError:
        pass

    try:
        candidate = str(socket.gethostbyname(socket.gethostname())).strip()
        if candidate and not _is_unspecified_address(candidate) and not _is_loopback_address(candidate):
            return candidate
    except OSError:
        pass

    return bind_address


def _peer_host_from_payload(peer_data: dict[str, Any], source_address: str | None = None) -> str:
    peer_device = peer_data.get("peer_device") if isinstance(peer_data.get("peer_device"), dict) else {}
    advertised = str(
        peer_device.get(
            "ip_address",
            peer_data.get("ip_address", peer_data.get("address", "")),
        )
    ).strip()
    if _is_unspecified_address(advertised) and source_address:
        return source_address
    if advertised:
        return advertised
    return str(source_address or "0.0.0.0")


class PeerClient:
    """Network client for peer discovery and tensor exchange."""

    def __init__(self) -> None:
        self.peer_client_id = os.getenv("TC_PEER_ID") or f"cheetah-{uuid.uuid4().hex[:6]}"
        self.bind_address = os.getenv("TC_BIND_ADDRESS", "0.0.0.0").strip() or "0.0.0.0"
        self.address = _resolve_advertise_address(self.bind_address)
        self.port = int(os.getenv("TC_PORT", "8765"))
        peer_device = get_backend_device(get_llm_backend(), default="CPU") or "CPU"
        self.peer_device = CDevice(
            self.peer_client_id,
            self.address,
            self.port,
            peer_device,
        )
        self.shard = Shard("", 0, 0, 0)
        self.in_use = False
        self.stop_ping: bool = False
        self.stop_udp_discovery = False
        self.stop_udp_broadcast = False
        self.stop_tensor_recv = False
        self._generate_handler: Optional[Callable[[dict], dict]] = None
        self._generation_model: Any | None = None
        self._generation_tokenizer: Any | None = None
        self._generation_backend: str = get_llm_backend()
        self._generation_model_id: str = ""
        self._generation_model_config: Any | None = None
        self._generation_model_path: str = ""
        self._generation_shard: Shard | None = None
        self._peers: Dict[str, CDevice] = {}
        self._lock = threading.RLock()
        self._peer_last_seen: Dict[str, float] = {}
        self._peer_stale_after = _peer_stale_after_seconds()
        self._flow_events: deque[dict[str, Any]] = deque(maxlen=256)
        self._thread_ping: Optional[threading.Thread] = None        
        self._thread_udp_discovery: Optional[threading.Thread] = None        
        self._thread_udp_brodcast: Optional[threading.Thread] = None
        self._thread_tensor_recv: Optional[threading.Thread] = None
        self._tensor_inbox: queue.Queue[bytes] = queue.Queue()
        

        self._run_udp_response()
        self._run_udp_discover()
        self._run_tensor_receiver()
        self.set_generate_handler(self._handle_generate_token_request)

    # Networking ---------------------------------------------------------
    def recv_tensor_bytes(
        self,
        timeout: float = 5.0,
        bind_address: Tuple[str, int] | None = None,
    ) -> bytes:
        """Blocking receive helper; expects peer to initiate a send."""
        if self._thread_tensor_recv and self._thread_tensor_recv.is_alive():
            try:
                return self._tensor_inbox.get(timeout=timeout)
            except queue.Empty:
                return b""

        if self.in_use:
            logger.warning("recv_tensor_bytes called '%s' already in use", self.peer_client_id)
            logger.info("Try again later")
            return b""

        self.in_use = True
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind(bind_address or (self.bind_address, 1045))
                sock.settimeout(timeout)
                data, _ = sock.recvfrom(65536)
                msg = json.loads(data.decode("utf-8"))
            buf = msg.get("payload", {}).get("buffer", "")
            return base64.b64decode(buf)
        except Exception:
            logger.exception("Error receiving tensor bytes")
            return b""
        finally:
            self.in_use = False

    def _tensor_recv_loop(self) -> None:
        bind = (self.bind_address, self.port)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(bind)
            sock.listen(5)
            sock.settimeout(0.5)
        except Exception as err:
            logger.error("Failed payload receiver: %s", err)
            return

        with closing(sock):
            while not self.stop_tensor_recv:
                try:
                    conn, addr = sock.accept()
                except socket.timeout:
                    continue
                except Exception:
                    logger.exception("Error accepting payload connection")
                    continue
                with conn:
                    try:
                        chunks: list[bytes] = []
                        while True:
                            data = conn.recv(65536)
                            if not data:
                                break
                            chunks.append(data)
                        raw = b"".join(chunks)
                        if not raw:
                            continue
                        msg = json.loads(raw.decode("utf-8"))
                    except Exception:
                        logger.exception("Error decoding payload")
                        continue

                    try:
                        command = msg.get("command")
                        if command == "tensor_bytes":
                            buf = msg.get("payload", {}).get("buffer", "")
                            if buf:
                                tensor_bytes = base64.b64decode(buf)
                                self._tensor_inbox.put(tensor_bytes)
                                logger.info("Received tensor bytes from %s", addr)
                            self._send_reply(conn, {"ok": True})
                        elif command == "generate_token":
                            if self._generate_handler is None:
                                self._send_reply(conn, {"error": "generate handler not set"})
                            else:
                                response = self._generate_handler(msg)
                                self._send_reply(conn, response)
                        elif command == "load_model":
                            response = self._handle_load_model_request(msg)
                            self._send_reply(conn, response)
                        elif command == "clear_model":
                            response = self._handle_clear_model_request(msg)
                            self._send_reply(conn, response)
                        else:
                            self._send_reply(conn, {"error": f"unknown command {command}"})
                    except Exception:
                        logger.exception("Error handling payload")
                        continue

    def _run_tensor_receiver(self) -> None:
        if self._thread_tensor_recv and self._thread_tensor_recv.is_alive():
            return
        self._thread_tensor_recv = threading.Thread(
            target=self._tensor_recv_loop,
            name="tensor-receiver",
            daemon=True,
        )
        self._thread_tensor_recv.start()

    def _send_reply(self, conn: socket.socket, payload: dict) -> None:
        try:
            data = json.dumps(payload).encode("utf-8")
            conn.sendall(data)
        except Exception:
            logger.debug("Failed to send reply payload")

    def send_payload(
        self,
        message: dict,
        *,
        expect_reply: bool = True,
        address: Tuple[str, int] | None = None,
        timeout: float | None = None,
    ) -> Dict[str, Any]:
        return self._send(message, expect_reply=expect_reply, address=address, timeout=timeout)

    def set_generate_handler(self, handler: Callable[[dict], dict]) -> None:
        self._generate_handler = handler

    def register_generation_runtime(
        self,
        *,
        model: Any,
        tokenizer: Any,
        backend: str,
        model_id: str = "",
        model_config: Any | None = None,
        model_path: str | None = None,
        shard: Shard | None = None,
    ) -> None:
        self._generation_model = model
        self._generation_tokenizer = tokenizer
        self._generation_backend = str(backend or get_llm_backend() or "tinygrad")
        self._generation_model_id = str(model_id or "")
        self._generation_model_config = model_config
        self._generation_model_path = str(model_path or "")
        self._generation_shard = shard or getattr(model, "shard", None)
        if self._generation_shard is not None:
            self.shard = self._generation_shard
        self.set_generate_handler(self._handle_generate_token_request)

    def clear_generation_runtime(
        self,
        *,
        model: Any | None = None,
        model_id: str | None = None,
    ) -> None:
        if model is not None and model is not self._generation_model:
            return
        current_model_id = str(getattr(self, "_generation_model_id", "") or "")
        if model_id is not None and current_model_id and str(model_id) != current_model_id:
            return
        self._generation_model = None
        self._generation_tokenizer = None
        self._generation_model_id = ""
        self._generation_model_config = None
        self._generation_model_path = ""
        self._generation_shard = None
        self.shard = Shard("", 0, 0, 0)
        gc.collect()

    def _handle_clear_model_request(self, message: dict) -> dict:
        payload = message.get("payload", {})
        model_id = ""
        if isinstance(payload, dict):
            model_id = str(payload.get("model_id", "") or "").strip()
        self.clear_generation_runtime(model_id=model_id or None)
        return {
            "ok": True,
            "cleared": True,
            "model_id": model_id,
        }

    def _handle_load_model_request(self, message: dict) -> dict:
        payload = message.get("payload", {})
        if not isinstance(payload, dict):
            return {"error": "invalid payload"}

        model_id = str(payload.get("model_id", "") or "").strip()
        if not model_id:
            return {"error": "missing model id"}

        backend = str(payload.get("backend", "") or get_llm_backend() or "tinygrad")
        offline_mode = bool(payload.get("offline_mode", False))
        shard = _shard_from_payload(payload.get("shard"), fallback_model_name=model_id)

        current_model = getattr(self, "_generation_model", None)
        current_tokenizer = getattr(self, "_generation_tokenizer", None)
        current_model_id = str(getattr(self, "_generation_model_id", "") or "")
        current_backend = str(getattr(self, "_generation_backend", "") or get_llm_backend() or "tinygrad")
        current_shard = getattr(self, "_generation_shard", None)

        if (
            current_model is not None
            and current_tokenizer is not None
            and current_model_id == model_id
            and current_backend == backend
            and _shards_equal(current_shard, shard)
        ):
            if current_shard is not None:
                self.shard = current_shard
            fingerprints = runtime_asset_fingerprints(
                model_config=getattr(self, "_generation_model_config", None),
                model_path=getattr(self, "_generation_model_path", ""),
            )
            return {
                "ok": True,
                "already_loaded": True,
                "model_id": model_id,
                "backend": backend,
                "fingerprint_protocol": RUNTIME_FINGERPRINT_PROTOCOL,
                "shard": _shard_payload(current_shard),
                **fingerprints,
                "elapsed": 0.0,
            }

        try:
            self.clear_generation_runtime()
            start = time.time()
            model, model_config, tokenizer, model_path = asyncio.run(
                load_model_for_backend(
                    model_id=model_id,
                    shard=shard,
                    weight_device=None,
                    offline_mode=offline_mode,
                    backend=backend,
                )
            )
            elapsed = time.time() - start
            self.register_generation_runtime(
                model=model,
                tokenizer=tokenizer,
                backend=backend,
                model_id=model_id,
                model_config=model_config,
                model_path=str(model_path),
                shard=getattr(model, "shard", None) or shard,
            )
            fingerprints = runtime_asset_fingerprints(
                model_config=model_config,
                model_path=model_path,
            )
            return {
                "ok": True,
                "already_loaded": False,
                "model_id": model_id,
                "backend": backend,
                "fingerprint_protocol": RUNTIME_FINGERPRINT_PROTOCOL,
                "model_path": str(model_path),
                "shard": _shard_payload(getattr(self, "_generation_shard", None)),
                **fingerprints,
                "elapsed": elapsed,
            }
        except Exception as exc:
            logger.exception("Peer model load failed: %s", exc)
            self.clear_generation_runtime()
            return {
                "error": str(exc),
                "model_id": model_id,
                "backend": backend,
                "shard": _shard_payload(shard),
            }

    def _handle_generate_token_request(self, message: dict) -> dict:
        payload = message.get("payload", {})
        if not isinstance(payload, dict):
            return {"error": "invalid payload"}

        model = self._generation_model
        tokenizer = self._generation_tokenizer
        if model is None or tokenizer is None:
            return {"error": "model not loaded"}

        backend = str(self._generation_backend or get_llm_backend() or "tinygrad")
        sender_peer_id = str(payload.get("sender_peer_id", "") or "").strip()
        request_tokens = self._flow_token_count(payload)
        if sender_peer_id:
            self.record_flow(sender_peer_id, self.peer_client_id, request_tokens, phase="request")
            self.mark_peer_seen(sender_peer_id)

        try:
            input_ids = self._payload_to_tensor(
                payload.get("input_ids"),
                backend=backend,
                integer=True,
            )
            attention_mask = self._payload_to_tensor(
                payload.get("attention_mask"),
                backend=backend,
                integer=True,
            )
            position_ids = self._payload_to_tensor(
                payload.get("position_ids"),
                backend=backend,
                integer=True,
            )
            hidden_state = self._payload_to_tensor(
                payload.get("hidden_state"),
                backend=backend,
                integer=False,
            )

            shard = getattr(model, "shard", None)
            shard_payload = payload.get("shard")
            if isinstance(shard_payload, dict):
                model_name = str(
                    shard_payload.get("model_name")
                    or getattr(shard, "model_name", "")
                    or self._generation_model_id
                    or "model"
                )
                start_layer = int(shard_payload.get("start_layer", getattr(shard, "start_layer", 0)) or 0)
                end_layer = int(shard_payload.get("end_layer", getattr(shard, "end_layer", 0)) or 0)
                total_layers = int(
                    shard_payload.get("total_layers", getattr(shard, "total_layers", end_layer)) or end_layer
                )
                shard = Shard(model_name, start_layer, end_layer, total_layers)

            engine = ModelEngine(shard=shard) if shard is not None else ModelEngine()
            response = engine.get_tokens(
                model,
                input_ids,
                attention_mask,
                tokenizer,
                hidden_state=hidden_state,
                position_ids=position_ids,
                prefill=bool(payload.get("prefill", False)),
                temp=float(payload.get("temp", 1.0) or 1.0),
                top_k=int(payload.get("top_k", 0) or 0),
                top_p=float(payload.get("top_p", 0.8) or 0.8),
                alpha_f=float(payload.get("alpha_f", 0.0) or 0.0),
                alpha_p=float(payload.get("alpha_p", 0.0) or 0.0),
                repetition_penalty=float(payload.get("repetition_penalty", 1.0) or 1.0),
                seen_tokens=[int(tok) for tok in payload.get("seen_tokens", []) or []],
            )
            if sender_peer_id:
                response_tokens = self._flow_token_count(response)
                self.record_flow(self.peer_client_id, sender_peer_id, response_tokens, phase="response")
            return response
        except Exception as exc:
            logger.exception("Peer token generation failed: %s", exc)
            return {"error": str(exc)}

    def _send(
        self,
        message: dict,
        expect_reply: bool = True,
        address: Tuple[str, int] | None = None,
        timeout: float | None = None,
    ) -> Dict[str, Any]:
        self.in_use = True
        try:
            data = json.dumps(message).encode("utf-8")
            if address is not None:
                host, port = address
            else:
                host, port = self.address, self.port
                logger.warning(f"Using default address {host}:{port}")

            socket_timeout = 3.0 if timeout is None else max(float(timeout), 0.1)
            with socket.create_connection((host, port), timeout=socket_timeout) as sock:
                sock.settimeout(socket_timeout)
                sock.sendall(data)
                if not expect_reply:
                    return {}
                sock.shutdown(socket.SHUT_WR)
                chunks: list[bytes] = []
                while True:
                    response = sock.recv(65536)
                    if not response:
                        break
                    chunks.append(response)
            
                self.in_use = False
                raw = b"".join(chunks)
                json_response = json.loads(raw.decode("utf-8"))
                logger.debug(f"Received response from {host}:{port} - {json_response}")
                return json_response
        except Exception as err:
            self.in_use = False
            logger.error("Invalid response from peer %s: %s", self.peer_client_id, err)
            raise

    def _resolve_address(self, address: Tuple[str, int] | None) -> Tuple[str, int]:
        if address is not None:
            return address
        return self.address, self.port

    # Connections ---------------------------------------------------------
    def get_peers(self, include_self: bool = False) -> List[CDevice]:
        self._prune_stale_peers()
        with self._lock:
            peers = [
                peer
                for peer_id, peer in self._peers.items()
                if peer_id != self.peer_client_id
            ]
        if include_self:
            return [self.peer_device, *peers]
        return peers

    def peer_count(self) -> int:
        return len(self.get_peers(include_self=True))

    @staticmethod
    def _payload_to_nested_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            if not value:
                return []
            if isinstance(value[0], list):
                return value
            return [value]
        return []

    @staticmethod
    def _flow_token_count(message: Any) -> int:
        payload = message.get("payload", message) if isinstance(message, dict) else message
        if not isinstance(payload, dict):
            return 0
        for key in ("attention_mask", "position_ids", "input_ids", "hidden_state"):
            count = PeerClient._token_count_from_payload_value(payload.get(key))
            if count > 0:
                return count
        if payload.get("token") is not None or payload.get("tensor") is not None:
            return 1
        return 0

    @staticmethod
    def _token_count_from_payload_value(value: Any) -> int:
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

    def _payload_to_tensor(self, value: Any, *, backend: str, integer: bool) -> Any:
        if value is None:
            return None

        selected_backend = str(backend or get_llm_backend() or "tinygrad").strip().lower()

        if isinstance(value, dict):
            tensor = _decode_tensor(value, backend=backend)
            if tensor is None:
                return None
            if selected_backend == "torch":
                import torch

                dtype = torch.long if integer else self._torch_tensor_transport_dtype(tensor.dtype)
                return tensor.to(device=self._torch_runtime_device(), dtype=dtype)
            if not integer:
                return tensor
            import tinygrad as tg

            return tensor.cast(tg.dtypes.int32)

        data = self._payload_to_nested_list(value)
        if data in ([], [[]]):
            return None

        if selected_backend == "torch":
            import torch

            device = self._torch_runtime_device()
            dtype = torch.long if integer else torch.float32
            return torch.tensor(data, dtype=dtype, device=device)

        import tinygrad as tg

        dtype = tg.dtypes.int32 if integer else None
        device = get_backend_device("tinygrad", default="CPU")
        assert device is not None
        tensor = tg.Tensor(data, device=device)
        if dtype is not None:
            tensor = tensor.cast(dtype)
        return tensor

    @staticmethod
    def _torch_runtime_device() -> str:
        device = str(get_backend_device("torch", default="cpu") or "cpu").strip().lower()
        if device in {"metal", "mps"}:
            return "mps"
        if device.startswith("cuda"):
            return device
        return "cpu"

    def _torch_tensor_transport_dtype(self, dtype: Any):
        import torch

        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            return dtype

        device = self._torch_runtime_device()
        if device == "mps" and dtype == torch.bfloat16:
            return torch.float16
        if device == "cpu" and dtype == torch.float16:
            return torch.float32
        return dtype

    def mark_peer_seen(self, peer_client_id: str | None) -> None:
        peer_id = str(peer_client_id or "").strip()
        if not peer_id or peer_id == self.peer_client_id:
            return
        with self._lock:
            self._peer_last_seen[peer_id] = time.time()

    def peer_last_seen(self, peer_client_id: str | None) -> float | None:
        peer_id = str(peer_client_id or "").strip()
        if not peer_id:
            return None
        if peer_id == self.peer_client_id:
            return time.time()
        with self._lock:
            last_seen = self._peer_last_seen.get(peer_id)
        return None if last_seen is None else float(last_seen)

    def peer_is_active(self, peer_client_id: str | None) -> bool:
        peer_id = str(peer_client_id or "").strip()
        if not peer_id:
            return False
        if peer_id == self.peer_client_id:
            return True
        last_seen = self.peer_last_seen(peer_id)
        if last_seen is None:
            return False
        stale_after = float(getattr(self, "_peer_stale_after", 8.0) or 8.0)
        return last_seen >= (time.time() - max(stale_after, 1.0))

    def record_flow(
        self,
        source: str | None,
        target: str | None,
        tokens: int | None,
        *,
        phase: str = "transfer",
    ) -> None:
        source_id = str(source or "").strip() or "unknown"
        target_id = str(target or "").strip() or "unknown"
        token_count = max(int(tokens or 0), 0)
        event = {
            "source": source_id,
            "target": target_id,
            "tokens": token_count,
            "phase": str(phase or "transfer"),
            "timestamp": time.time(),
        }
        with self._lock:
            self._flow_events.append(event)

    def recent_flows(self, *, max_age: float = 60.0, limit: int = 8) -> list[dict[str, Any]]:
        cutoff = time.time() - max(max_age, 0.0)
        with self._lock:
            events = [dict(event) for event in self._flow_events if float(event.get("timestamp", 0.0) or 0.0) >= cutoff]

        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for event in events:
            source = str(event.get("source", "") or "unknown")
            target = str(event.get("target", "") or "unknown")
            key = (source, target)
            if key not in grouped:
                grouped[key] = {
                    "source": source,
                    "target": target,
                    "tokens": 0,
                    "count": 0,
                    "phase": str(event.get("phase", "transfer") or "transfer"),
                    "last_seen": 0.0,
                }
            bucket = grouped[key]
            bucket["tokens"] += max(int(event.get("tokens", 0) or 0), 0)
            bucket["count"] += 1
            bucket["phase"] = str(event.get("phase", bucket["phase"]) or bucket["phase"])
            bucket["last_seen"] = max(float(bucket["last_seen"]), float(event.get("timestamp", 0.0) or 0.0))

        ordered = sorted(grouped.values(), key=lambda item: float(item.get("last_seen", 0.0) or 0.0), reverse=True)
        return ordered[: max(int(limit or 0), 0)] if limit else ordered

    def _prune_stale_peers(self) -> None:
        stale_after = float(getattr(self, "_peer_stale_after", 8.0) or 8.0)
        cutoff = time.time() - max(stale_after, 1.0)
        dropped: list[str] = []
        with self._lock:
            peer_last_seen = getattr(self, "_peer_last_seen", {})
            peers = getattr(self, "_peers", {})
            for peer_id, last_seen in list(peer_last_seen.items()):
                if peer_id == self.peer_client_id:
                    continue
                if peer_id in peers and float(last_seen or 0.0) < cutoff:
                    peers.pop(peer_id, None)
                    peer_last_seen.pop(peer_id, None)
                    dropped.append(peer_id)
        for peer_id in dropped:
            logger.info("Peer %s timed out and was removed from the peer list", peer_id)

    # Internal ------------------------------------------------------------
    def _ordered_peers(self) -> List[CDevice]:
        peers = [p for p in self._peers.values() if p.peer_client_id != self.peer_client_id]

        def capacity(peer: CDevice) -> float:
            vram = _to_float(peer.gpu_vram)
            ram = _to_float(peer.cpu_ram)
            return max(vram, ram, 1.0)

        return sorted(peers, key=capacity, reverse=True)

    def _build_peer_from_info(self, info: dict) -> Optional[CDevice]:
        peer_client_id = info.get("peer_client_id")
        if not peer_client_id:
            return None

        shard_info = info.get("shard", {}) if isinstance(info.get("shard"), dict) else {}
        peer = CDevice(
            peer_client_id,
            info.get("ip_address", info.get("address", "")),
            int(info.get("port", self.port)),
            tg_device=str(info.get("tg_device", "CPU")),
        )
        peer.shard = Shard(
            shard_info.get("model_name", ""),
            int(shard_info.get("start_layer", 0) or 0),
            int(shard_info.get("end_layer", 0) or 0),
            int(shard_info.get("total_layers", shard_info.get("end_layer", 0)) or 0),
        )
        peer.cpu_model = str(info.get("cpu_model", ""))
        peer.gpu_model = str(info.get("gpu_model", ""))
        peer.gpu_vram = str(info.get("gpu_vram", ""))
        peer.gpu_flops = float(info.get("gpu_flops", 0.0) or 0.0)

        return peer

    # Ping handling
    # --------------------------------
    def _ping_peer(self, peer_client: PeerClient) -> dict:
        missed_pong = {}
        while not self.stop_ping:
            try:
                for _, peer_client in self._peers:
                    payload = json.dumps({"command": "ping"}).encode("utf-8")
                    logger.debug(f"Pinging peer {peer_client.peer_client_id} @ {peer_client.ip_address}:{peer_client.port}")
                    start = time.time()
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                        sock.settimeout(1.0)
                        sock.sendto(payload, (peer_client.ip_address, peer_client.port))
                        data, _ = sock.recvfrom(4096)
                    ping_ms = max((time.time() - start) * 1000.0, 0.0)

                    data_msg = json.loads(data.decode("utf-8"))
                    if data_msg.get("command") == "pong":
                        peer_client.ping_ms = ping_ms
                        logger.debug(f"Received pong from {peer_client.peer_client_id} - ping {ping_ms:.2f} ms")
                    else:
                        peer_client_id = peer_client.peer_client_id
                        if peer_client_id not in missed_pong.keys():
                            missed_pong[peer_client_id] = 0
                        else:
                            missed_pong[peer_client_id] += 1

                        logger.warning(f"Unexpected ping response from {peer_client.peer_client_id}: {data_msg}")

                        missed_pong_limit = os.getenv("TC_PONG_MISS_LIMIT", 5)
                        if missed_pong[peer_client_id] == missed_pong_limit:
                            logger.info(f"Missing pong from {peer_client_id} {missed_pong_limit} times, dropping from peer list")
                            self._peers = {key: value for key, value in self._peers.items() if key != peer_client_id}
                            missed_pong = {key: value for key, value in self._peers.items() if key != peer_client_id}

            except Exception:
                continue
    
    def _run_ping(self)-> None:
        if self._thread_ping and self._thread_ping.is_alive():
            return
        self._thread_ping = threading.Thread(target=self._ping_loop, name="ping-peer", daemon=True)
        self._thread_ping.start()


    # UDP handling
    #-------------------------------------------------------
    def _udp_discover(self):
        payload = json.dumps(
            {
                "command": "D001",
                "peer_client_id": self.peer_client_id
        }).encode("utf-8")
        unicast_targets = os.getenv("TC_PEER_UNICAST_TARGETS", "")
        targets: list[tuple[str, int]] = []
        if unicast_targets:
            for entry in unicast_targets.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if ":" in entry:
                    host, port_str = entry.rsplit(":", 1)
                    try:
                        port = int(port_str)
                    except ValueError:
                        logger.warning("Invalid unicast target port: %s", entry)
                        continue
                else:
                    host = entry
                    port = self.port
                targets.append((host, port))
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(1.0)
        except Exception as err:
            logger.error(f"Error discovering local UDP peers {err}")
            raise

        with closing(sock):
            while not self.stop_udp_discovery:
                logger.info(f"Broadcasting client {self.peer_client_id} @ {self.address}:{self.port}")
                try:
                    sock.sendto(payload, ("<broadcast>", self.port))
                    for host, port in targets:
                        sock.sendto(payload, (host, port))
                except Exception as err:
                    logger.error(f"Error broadcasting discovery packet: {err}")
                
                while True:
                    try:
                        data, addr = sock.recvfrom(4096)
                    except socket.timeout:
                        break
                    
                    try:
                        udp_peer_info = json.loads(data.decode("utf-8"))
                        if udp_peer_info["command"] == "D002":
                            udp_client_id = udp_peer_info.get("peer_client_id")
                            if udp_client_id is not None and udp_client_id != self.peer_client_id:
                                is_new_peer = udp_client_id not in self._peers
                                if is_new_peer:
                                    logger.info("UDP discovery response from %s: %s", addr, udp_peer_info)
                                    logger.info(f"Current peer list: {self._peers}")
                                    logger.info(f"New peer discovered @ {addr}.")
                                    logger.info(f"Adding peer {udp_client_id}")
                                self.add_peer(udp_peer_info, source_address=addr[0])
                    except Exception as err:
                        logger.error(f"Error processing UDP discovery response: {err}")

    def _run_udp_discover(self) -> None:
        if self._thread_udp_discovery and self._thread_udp_discovery.is_alive():
            return
        self._thread_udp_discovery = threading.Thread(target=self._udp_discover, name="udp-discover", daemon=True)
        self._thread_udp_discovery.start()

    def _udp_response(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.bind_address, self.port))
            sock.settimeout(1.0)
        except Exception as err:
            logger.error(f"Failed UDP response: {err}")
            raise

        with closing(sock):
            while not self.stop_udp_discovery:
                try:
                    data, addr = sock.recvfrom(4096)
                    msg = json.loads(data.decode("utf-8"))
                    if msg.get("command") == "D001" and msg.get("peer_client_id") != self.peer_client_id:
                        logger.info("UDP discovery request from %s: %s", addr, msg)
                        try:
                            response = self.as_dict()
                            response["command"] = "D002"
                            resp_data = json.dumps(response).encode("utf-8")
                            logger.info(f"Responding to D001 from {addr}")
                            sock.sendto(resp_data, addr)
                        except Exception as err:
                            logger.error(f"Error responding to request: {err}")
                except Exception as err:
                    logger.debug(f"UDP response error: {err}")

    def _run_udp_response(self) -> None:
        if self._thread_udp_brodcast and self._thread_udp_brodcast.is_alive():
            return
        self._thread_udp_brodcast = threading.Thread(target=self._udp_response, name="udp-responder", daemon=True)
        self._thread_udp_brodcast.start()

    def add_peer(self, peer_data: dict, source_address: str | None = None) -> CDevice | None:
        try:
            peer_device = peer_data.get("peer_device")
            peer_client_id = str(peer_data.get("peer_client_id", "") or "")
            if not peer_client_id:
                logger.warning("Skipped peer with missing peer_client_id: %s", peer_data)
                return None

            if peer_device is None:
                cdevice = CDevice(
                    peer_client_id,
                    _peer_host_from_payload(peer_data, source_address=source_address),
                    peer_data.get("port", self.port)
                )
            else:
                cdevice = CDevice(
                    peer_device.get("peer_client_id", peer_client_id),
                    _peer_host_from_payload(peer_data, source_address=source_address),
                    peer_device.get("port", peer_data.get("port", self.port)),
                    peer_device.get("tg_device", "CPU"),
                )

                cdevice.cpu_model = peer_device.get("cpu_model", "")
                cdevice.cpu_proc_speed = peer_device.get("cpu_proc_speed", "")
                cdevice.cpu_cores = peer_device.get("cpu_cores", 0)
                cdevice.cpu_ram = peer_device.get("cpu_ram", "")
                cdevice.gpu_model = peer_device.get("gpu_model", "")
                cdevice.gpu_vram = peer_device.get("gpu_vram", "")
                cdevice.gpu_flops = float(peer_device.get("gpu_flops", 0.0) or 0.0)

            shard_data = peer_data.get("shard", {}) if isinstance(peer_data.get("shard"), dict) else {}
            cdevice.shard = Shard(
                shard_data.get("model_name", ""),
                shard_data.get("start_layer", 0),
                shard_data.get("end_layer", 0),
                shard_data.get("total_layers", shard_data.get("end_layer", 0)),
            )
            cdevice.ip_address = _peer_host_from_payload(peer_data, source_address=source_address)
            cdevice.port = peer_data.get("port", cdevice.port)

            with self._lock:
                is_new_peer = peer_client_id not in self._peers
                self._peers[peer_client_id] = cdevice
                self._peer_last_seen[peer_client_id] = time.time()
            if is_new_peer:
                logger.info("Added peer %s to peer list", peer_client_id)
            else:
                logger.debug("Refreshed peer %s", peer_client_id)
            return cdevice
        except Exception as err:
            logger.error(f"Failed to add peer to list: {err}")
            return None

    def as_dict(self) -> dict:
        return {
            "peer_client_id": self.peer_client_id,
            "address": self.address,
            "port": self.port,
            "shard": {
                "model_name": self.shard.model_name,
                "start_layer": self.shard.start_layer,
                "end_layer": self.shard.end_layer,
                "total_layers": self.shard.total_layers,
            },
            "peer_device": self.peer_device.as_dict()
        }

def _to_float(val: Any) -> float:
    try:
        if isinstance(val, str):
            txt = val.lower().replace("gb", "").strip()
            return float(txt)
        return float(val)
    except Exception:
        return 0.0


def _shard_from_payload(payload: Any, *, fallback_model_name: str = "model") -> Shard | None:
    if not isinstance(payload, dict):
        return None
    try:
        model_name = str(payload.get("model_name", "") or fallback_model_name or "model")
        start_layer = int(payload.get("start_layer", 0) or 0)
        end_layer = int(payload.get("end_layer", 0) or 0)
        total_layers = int(payload.get("total_layers", end_layer + 1) or (end_layer + 1))
    except (TypeError, ValueError):
        return None
    return Shard(model_name, start_layer, end_layer, total_layers)


def _shard_payload(shard: Shard | None) -> dict[str, int | str]:
    if shard is None:
        return {}
    return {
        "model_name": str(getattr(shard, "model_name", "") or ""),
        "start_layer": int(getattr(shard, "start_layer", 0) or 0),
        "end_layer": int(getattr(shard, "end_layer", 0) or 0),
        "total_layers": int(getattr(shard, "total_layers", 0) or 0),
    }


def _shards_equal(lhs: Shard | None, rhs: Shard | None) -> bool:
    if lhs is rhs:
        return True
    if lhs is None or rhs is None:
        return False
    return _shard_payload(lhs) == _shard_payload(rhs)
