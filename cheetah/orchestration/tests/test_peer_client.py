import asyncio
import base64
import json
import unittest
import os

import tinygrad as tg
from cheetah.orchestration.peer_client import PeerClient

TEST_RECEIVER_BIND_HOST = "0.0.0.0"
TEST_RECEIVER_CONNECT_HOST = "127.0.0.1"
TEST_PORT = 6668
TEST_TIMEOUT = 30.0
TEST_TENSOR_PAYLOAD = tg.Tensor.randn(10, 10).numpy().tobytes()


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
