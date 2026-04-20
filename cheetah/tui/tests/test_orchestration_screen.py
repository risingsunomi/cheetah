import unittest
from types import SimpleNamespace

from cheetah.tui.orchestration_screen import OrchestrationScreen


class _PeerClientStub:
    def __init__(self) -> None:
        self.peer_client_id = "self"
        self.peer_device = SimpleNamespace(
            peer_client_id="self",
            ip_address="192.168.0.2",
            cpu_model="CPU",
            cpu_proc_speed="3ghz",
            cpu_cores=8,
            cpu_ram="32",
            gpu_model="GPU",
            gpu_vram="16",
            gpu_flops=12.0,
            as_dict=lambda: {
                "cpu_model": "CPU",
                "cpu_proc_speed": "3ghz",
                "cpu_cores": 8,
                "cpu_ram": "32",
                "gpu_model": "GPU",
                "gpu_vram": "16",
                "gpu_flops": 12.0,
            },
        )
        self._peers = [
            SimpleNamespace(
                peer_client_id="peer-1",
                ip_address="192.168.0.3",
                cpu_model="CPU",
                cpu_cores=4,
                cpu_ram="16",
                gpu_model="GPU",
                gpu_vram="8",
                gpu_flops=6.0,
            )
        ]

    def get_peers(self, include_self: bool = False):
        if include_self:
            return [self.peer_device, *self._peers]
        return list(self._peers)

    def peer_count(self) -> int:
        return 1 + len(self._peers)

    def recent_flows(self, *, max_age: float = 60.0, limit: int = 8):
        del max_age, limit
        return [
            {
                "source": "self",
                "target": "peer-1",
                "tokens": 128,
                "count": 2,
                "last_seen": 0.0,
            }
        ]


class TestOrchestrationScreen(unittest.TestCase):
    def test_flow_text_renders_recent_node_to_node_tokens(self) -> None:
        screen = OrchestrationScreen(_PeerClientStub())

        text = screen._flow_text()

        self.assertIn("Recent Data Flow", text)
        self.assertIn("self (192.168.0.2) -> peer-1 (192.168.0.3)", text)
        self.assertIn("128 tok across 2 transfers", text)
