import asyncio
import base64
from collections import deque
import json
from pathlib import Path
import unittest
import os
import threading
from unittest import mock
from types import SimpleNamespace

try:
    import tinygrad as tg
except Exception:
    tg = None
try:
    import torch
except Exception:
    torch = None
from cheetah.orchestration.peer_client import (
    PeerClient,
    _peer_host_from_payload,
    _resolve_advertise_address,
)
from cheetah.models.shard import Shard

TEST_RECEIVER_BIND_HOST = "0.0.0.0"
TEST_RECEIVER_CONNECT_HOST = "127.0.0.1"
TEST_PORT = 6668
TEST_TIMEOUT = 30.0
TEST_TENSOR_PAYLOAD = tg.Tensor.randn(10, 10).numpy().tobytes() if tg is not None else b"peer-client-test"


def _stop_peer_client(client: PeerClient) -> None:
    client.stop_ping = True
    client.stop_udp_discovery = True
    client.stop_udp_broadcast = True
    client.stop_tensor_recv = True

    for attr in ("_thread_ping", "_thread_udp_discovery", "_thread_udp_brodcast", "_thread_tensor_recv"):
        thread = getattr(client, attr, None)
        if thread is not None:
            thread.join(timeout=1.0)


def _tensor_message(payload: bytes) -> dict:
    return {
        "command": "tensor_bytes",
        "payload": {
            "buffer": base64.b64encode(payload).decode("utf-8"),
        },
    }


class TestPeerClientSender(unittest.TestCase):
    def test_peer_client_sender(self):
        host = os.getenv("TEST_TARGET_HOST", None)
        if host is None:
            self.skipTest("TEST_TARGET_HOST not set")
        port = TEST_PORT
        
        client = None
        try:
            with mock.patch.dict(os.environ, {"TC_PORT": "0"}, clear=False):
                client = PeerClient()
                _stop_peer_client(client)
                client.send_payload(
                    _tensor_message(TEST_TENSOR_PAYLOAD),
                    expect_reply=False,
                    address=(host, port),
                )
        finally:
            if client is not None:
                _stop_peer_client(client)


class TestPeerClientReceiver(unittest.IsolatedAsyncioTestCase):
    async def test_peer_client_receiver(self):
        host = TEST_RECEIVER_BIND_HOST
        port = TEST_PORT
        timeout = TEST_TIMEOUT
        expected = TEST_TENSOR_PAYLOAD

        payload_queue: asyncio.Queue[bytes] = asyncio.Queue()

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            data = await reader.read(65536)
            writer.close()
            await writer.wait_closed()
            await payload_queue.put(data)

        server = await asyncio.start_server(handle, host, port)
        client = None
        payload = None
        try:
            with mock.patch.dict(os.environ, {"TC_PORT": "0"}, clear=False):
                client = PeerClient()
                _stop_peer_client(client)
                await asyncio.to_thread(
                    lambda: client.send_payload(
                        _tensor_message(expected),
                        expect_reply=False,
                        address=(TEST_RECEIVER_CONNECT_HOST, port),
                    )
                )

            raw = await asyncio.wait_for(payload_queue.get(), timeout=timeout)

            msg = json.loads(raw.decode("utf-8"))
            self.assertEqual(msg.get("command"), "tensor_bytes")
            buf = msg.get("payload", {}).get("buffer", "")
            payload = base64.b64decode(buf)
            self.assertIsInstance(payload, bytes)
        finally:
            if client is not None:
                _stop_peer_client(client)
            server.close()
            await server.wait_closed()

        self.assertEqual(payload, expected)


class TestPeerDiscoveryHelpers(unittest.TestCase):
    def test_resolve_advertise_address_prefers_explicit_override(self):
        with mock.patch.dict(os.environ, {"TC_ADVERTISE_ADDRESS": "192.168.50.12"}):
            self.assertEqual(_resolve_advertise_address("0.0.0.0"), "192.168.50.12")

    def test_peer_host_from_payload_prefers_response_source_for_unspecified_address(self):
        payload = {
            "peer_client_id": "peer-1",
            "address": "0.0.0.0",
            "port": TEST_PORT,
            "peer_device": {
                "peer_client_id": "peer-1",
                "ip_address": "0.0.0.0",
                "port": TEST_PORT,
                "tg_device": "CPU",
            },
        }
        self.assertEqual(
            _peer_host_from_payload(payload, source_address="192.168.0.42"),
            "192.168.0.42",
        )

    def test_add_peer_uses_response_source_when_payload_is_unspecified(self):
        client = PeerClient.__new__(PeerClient)
        client.port = TEST_PORT
        client._peers = {}
        client._peer_last_seen = {}
        client._lock = threading.RLock()

        peer = PeerClient.add_peer(
            client,
            {
                "peer_client_id": "peer-1",
                "address": "0.0.0.0",
                "port": TEST_PORT,
                "peer_device": {
                    "peer_client_id": "peer-1",
                    "ip_address": "0.0.0.0",
                    "port": TEST_PORT,
                    "tg_device": "CPU",
                    "cpu_model": "",
                    "cpu_proc_speed": "",
                    "cpu_cores": 0,
                    "cpu_ram": "",
                    "gpu_model": "",
                    "gpu_vram": "",
                    "gpu_flops": 0.0,
                },
                "shard": {},
            },
            source_address="192.168.0.42",
        )

        self.assertIsNotNone(peer)
        assert peer is not None
        self.assertEqual(peer.ip_address, "192.168.0.42")
        self.assertIn("peer-1", client._peers)

    def test_get_peers_prunes_stale_entries(self):
        client = PeerClient.__new__(PeerClient)
        client.peer_client_id = "self"
        client.peer_device = SimpleNamespace(peer_client_id="self")
        client._lock = threading.RLock()
        client._peers = {
            "peer-1": SimpleNamespace(peer_client_id="peer-1"),
        }
        client._peer_last_seen = {"peer-1": 10.0}
        client._peer_stale_after = 5.0

        with mock.patch("cheetah.orchestration.peer_client.time.time", return_value=20.0):
            peers = PeerClient.get_peers(client, include_self=False)

        self.assertEqual(peers, [])
        self.assertEqual(client._peers, {})

    def test_recent_flows_aggregates_transfers_by_route(self):
        client = PeerClient.__new__(PeerClient)
        client._lock = threading.RLock()
        client._flow_events = deque(maxlen=256)

        with mock.patch("cheetah.orchestration.peer_client.time.time", side_effect=[100.0, 101.0, 105.0]):
            PeerClient.record_flow(client, "self", "peer-1", 64, phase="request")
            PeerClient.record_flow(client, "self", "peer-1", 32, phase="request")
            flows = PeerClient.recent_flows(client, max_age=10.0, limit=8)

        self.assertEqual(len(flows), 1)
        self.assertEqual(flows[0]["source"], "self")
        self.assertEqual(flows[0]["target"], "peer-1")
        self.assertEqual(flows[0]["tokens"], 96)
        self.assertEqual(flows[0]["count"], 2)

    def test_peer_is_active_uses_last_seen_window(self):
        client = PeerClient.__new__(PeerClient)
        client.peer_client_id = "self"
        client._lock = threading.RLock()
        client._peer_last_seen = {"peer-1": 95.0}
        client._peer_stale_after = 10.0

        with mock.patch("cheetah.orchestration.peer_client.time.time", return_value=100.0):
            self.assertTrue(PeerClient.peer_is_active(client, "peer-1"))
            self.assertFalse(PeerClient.peer_is_active(client, "peer-2"))
            self.assertTrue(PeerClient.peer_is_active(client, "self"))

    def test_register_generation_runtime_sets_working_default_handler(self):
        if torch is None:
            self.skipTest("torch is required for this test.")

        class FakeModel:
            def __init__(self) -> None:
                self.shard = Shard("demo", 0, 1, 4)

            def run_shard(
                self,
                x,
                *,
                attention_mask=None,
                position_ids=None,
                hidden_state=None,
                shard=None,
                start_pos=None,
            ):
                return torch.ones((1, attention_mask.shape[1], 4), dtype=torch.float32)

        client = PeerClient.__new__(PeerClient)
        client.peer_client_id = "self"
        client._lock = threading.RLock()
        client._peer_last_seen = {}
        client._peer_stale_after = 10.0
        client._flow_events = deque(maxlen=256)
        client._generation_model = None
        client._generation_tokenizer = None
        client._generation_backend = "torch"
        client._generation_model_id = ""
        client._generate_handler = None

        model = FakeModel()
        tokenizer = SimpleNamespace(eos_token_id=999)
        PeerClient.register_generation_runtime(
            client,
            model=model,
            tokenizer=tokenizer,
            backend="torch",
            model_id="demo",
        )

        self.assertTrue(callable(client._generate_handler))
        response = client._generate_handler(
            {
                "payload": {
                    "sender_peer_id": "peer-1",
                    "input_ids": [[1, 2, 3]],
                    "attention_mask": [[1, 1, 1]],
                    "hidden_state": [[]],
                    "prefill": True,
                    "shard": {
                        "model_name": "demo",
                        "start_layer": 0,
                        "end_layer": 1,
                        "total_layers": 4,
                    },
                }
            }
        )

        self.assertIn("hidden_state", response)
        self.assertNotIn("error", response)
        self.assertEqual(client._peer_last_seen["peer-1"] > 0.0, True)

    def test_handle_load_model_request_loads_requested_shard(self):
        client = PeerClient.__new__(PeerClient)
        client.peer_client_id = "self"
        client.shard = Shard("", 0, 0, 0)
        client._lock = threading.RLock()
        client._peer_last_seen = {}
        client._peer_stale_after = 10.0
        client._flow_events = deque(maxlen=256)
        client._generation_model = None
        client._generation_tokenizer = None
        client._generation_backend = "torch"
        client._generation_model_id = ""
        client._generation_model_config = None
        client._generation_model_path = ""
        client._generation_shard = None
        client._generate_handler = None

        model = SimpleNamespace(shard=Shard("demo", 0, 2, 5))
        tokenizer = object()

        with mock.patch(
            "cheetah.orchestration.peer_client.load_model_for_backend",
            new=mock.AsyncMock(return_value=(model, {"num_layers": 4}, tokenizer, Path("/tmp/model"))),
        ) as load_model:
            response = PeerClient._handle_load_model_request(
                client,
                {
                    "payload": {
                        "model_id": "demo",
                        "backend": "torch",
                        "offline_mode": True,
                        "shard": {
                            "model_name": "demo",
                            "start_layer": 0,
                            "end_layer": 2,
                            "total_layers": 5,
                        },
                    }
                },
            )

        load_model.assert_awaited_once()
        self.assertTrue(response["ok"])
        self.assertFalse(response["already_loaded"])
        self.assertEqual(client._generation_model, model)
        self.assertEqual(client._generation_tokenizer, tokenizer)
        self.assertEqual(client._generation_model_id, "demo")
        self.assertEqual(client._generation_shard.start_layer, 0)
        self.assertEqual(client.shard.end_layer, 2)

    def test_handle_load_model_request_reuses_matching_runtime(self):
        client = PeerClient.__new__(PeerClient)
        client.peer_client_id = "self"
        client.shard = Shard("demo", 0, 2, 5)
        client._lock = threading.RLock()
        client._peer_last_seen = {}
        client._peer_stale_after = 10.0
        client._flow_events = deque(maxlen=256)
        client._generation_model = object()
        client._generation_tokenizer = object()
        client._generation_backend = "torch"
        client._generation_model_id = "demo"
        client._generation_model_config = {"num_layers": 4}
        client._generation_model_path = "/tmp/model"
        client._generation_shard = Shard("demo", 0, 2, 5)
        client._generate_handler = None

        with mock.patch("cheetah.orchestration.peer_client.load_model_for_backend", new=mock.AsyncMock()) as load_model:
            response = PeerClient._handle_load_model_request(
                client,
                {
                    "payload": {
                        "model_id": "demo",
                        "backend": "torch",
                        "shard": {
                            "model_name": "demo",
                            "start_layer": 0,
                            "end_layer": 2,
                            "total_layers": 5,
                        },
                    }
                },
            )

        load_model.assert_not_awaited()
        self.assertTrue(response["ok"])
        self.assertTrue(response["already_loaded"])
        self.assertEqual(response["shard"]["start_layer"], 0)
