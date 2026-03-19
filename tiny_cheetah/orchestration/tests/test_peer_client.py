import asyncio
import base64
import json
import time
import unittest
import os

import tinygrad as tg
from tiny_cheetah.orchestration.peer_client import PeerClient

TEST_RECIVER_HOST = "0.0.0.0"
TEST_PORT = 6668
TEST_TIMEOUT = 30.0
TEST_TENSOR_PAYLOAD = tg.Tensor.randn(10, 10).numpy().tobytes()


def _stop_peer_client(client: PeerClient) -> None:
    client._udp_stop.set()
    if client._udp_thread is not None:
        client._udp_thread.join(timeout=1.0)


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
            client.send_tensor_bytes(TEST_TENSOR_PAYLOAD, address=(host, port))
        finally:
            if client is not None:
                _stop_peer_client(client)


class TestPeerClientReceiver(unittest.IsolatedAsyncioTestCase):
    async def test_peer_client_receiver(self):
        host = TEST_RECIVER_HOST
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
        payload = None
        try:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.fail(f"Timed out waiting for sender on {host}:{port}")
                raw = await asyncio.wait_for(payload_queue.get(), timeout=remaining)

                try:
                    msg = json.loads(raw.decode("utf-8"))
                    print(f"Received message from {host}: {msg}")
                except Exception:
                    continue
                if msg.get("command") != "tensor_bytes":
                    continue
                buf = msg.get("payload", {}).get("buffer", "")
                try:
                    payload = base64.b64decode(buf)
                    self.assertIsInstance(payload, bytes)
                except Exception:
                    continue
                if payload == expected:
                    break
        finally:
            server.close()
            await server.wait_closed()

        self.assertEqual(payload, expected)
