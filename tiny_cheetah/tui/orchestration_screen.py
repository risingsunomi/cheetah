from __future__ import annotations

import time
import asyncio
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, Static

from tiny_cheetah.orchestration.peer_client import PeerClient
from tiny_cheetah.tui.connect_peer_screen import ConnectPeerScreen
from tiny_cheetah.tui.help_screen import HelpScreen


class OrchestrationScreen(Screen[None]):
    """Host dashboard for remote compute orchestration."""

    CSS_PATH = Path(__file__).with_name("orchestration_screen.tcss")

    BINDINGS = [
        ("escape", "pop_screen", "Back"),
        ("b", "pop_screen", "Back"),
        ("r", "refresh_panels", "Refresh data"),
        ("a", "open_connect", "Add peer (IP)"),
        ("h", "open_help", "Help"),
    ]

    def __init__(self, peer_client: PeerClient) -> None:
        super().__init__()
        self._peer_client = peer_client
        self._summary_panel: Optional[Static] = None
        self._peer_summary_panel: Optional[Static] = None
        self._peer_panel: Optional[Static] = None
        self._hostmap_panel: Optional[Static] = None
        self._refresh_label: Optional[Label] = None
        self._last_refresh = time.time()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="orch-root"):
            with Container(id="orch-left"):
                local_panel = Static(self._local_node_text(), id="orch-local")
                self._summary_panel = local_panel
                yield local_panel
                peer_summary_panel = Static(self._peer_summary_text(), id="orch-peer-summary")
                self._peer_summary_panel = peer_summary_panel
                yield peer_summary_panel
            with Container(id="orch-right"):
                network_panel = Static(self._network_map_text(), id="orch-network")
                self._peer_panel = network_panel
                yield network_panel
        yield Footer()

    async def on_mount(self) -> None:
        await asyncio.to_thread(self._get_peer_count)
        self.set_interval(5.0, self._get_peer_count)
        self.set_interval(30.0, self._refresh_panels)

    # Textual actions ---------------------------------------------------------
    def action_refresh_panels(self) -> None:
        self._refresh_panels()

    def action_open_connect(self) -> None:
        self.app.push_screen(ConnectPeerScreen())

    def action_open_help(self) -> None:
        self.app.push_screen(HelpScreen("Network Help", self._help_text()))
    
    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _refresh_panels(self) -> None:
        self._last_refresh = time.time()
        if self._summary_panel is not None:
            self._summary_panel.update(self._local_node_text())
        if self._peer_summary_panel is not None:
            self._peer_summary_panel.update(self._peer_summary_text())
        if self._peer_panel is not None:
            self._peer_panel.update(self._network_map_text())
        if self._refresh_label is not None:
            self._refresh_label.update("")
        self._update_action_variants()

    def _get_peer_count(self) -> None:
        if self.app is None:
            return
        count = self._peer_client.peer_count()
        if count > 1:
            current = getattr(self.app, "sub_title", "")
            new_title = f"[Nodes: {count}]"
            self.app.title = new_title

    def _local_node_text(self) -> str:
        host_info = self._peer_client.peer_device.as_dict()
        peer_in_use = self._peer_client.in_use
        status = "[green]● Online[/]" if not peer_in_use else "[red]● Busy[/]"
        cpu_model = host_info.get("cpu_model", "Unknown CPU")
        cpu_proc_speed = host_info.get("cpu_proc_speed", "0hz")
        cpu_cores = host_info.get("cpu_cores", 0)
        cpu_ram = host_info.get("cpu_ram", "0B")
        gpu_model = host_info.get("gpu_model", "Unknown GPU")
        gpu_vram = host_info.get("gpu_vram", "0B")
        gpu_flops = host_info.get("gpu_flops", 0.0)
        return "\n".join(
            [
                "[b]Local Node Info[/]",
                "Identity:",
                f"- Node: {self._peer_client.peer_client_id}",
                f"- Status: {status}",
                "",
                "Hardware:",
                f"- CPU: {cpu_model} @ {cpu_proc_speed} ({cpu_cores} cores)",
                f"- RAM: {cpu_ram} GB",
                f"- GPUs: {gpu_model} ({gpu_vram} VRAM, {gpu_flops} FLOPS)",
            ]
        )

    def _network_map_text(self) -> str:
        peers = [self._peer_entry_to_obj(peer) for peer in self._peer_client.get_peers(include_self=True)]
        peers = [peer for peer in peers if peer is not None]
        def _status(peer) -> str:
            available = getattr(peer, "available", None)
            if available is not None:
                return "[green]●[/]" if available else "[red]●[/]"
            in_use = getattr(peer, "in_use", None)
            if in_use is not None:
                return "[red]●[/]" if in_use else "[green]●[/]"
            return "[green]●[/]"

        def _host_label(peer) -> str:
            hw = getattr(peer, "device_report", None)
            hostname = ""
            if isinstance(hw, dict):
                hostname = str(hw.get("hostname", "")).strip()
            host = hostname or str(getattr(peer, "ip_address", "") or getattr(peer, "address", "")).strip()
            if host in {"", "0.0.0.0"}:
                host = str(getattr(peer, "peer_client_id", "peer"))
            return host

        def _flops_value(peer) -> float:
            flops = getattr(peer, "gpu_flops", 0.0) or 0.0
            if not flops and hasattr(peer, "peer_device"):
                flops = getattr(peer.peer_device, "gpu_flops", 0.0) or 0.0
            hw = getattr(peer, "device_report", None)
            if not flops and isinstance(hw, dict):
                devices = hw.get("devices", []) or hw.get("gpus", [])
                for device in devices:
                    if not isinstance(device, dict):
                        continue
                    if device.get("kind") == "GPU" or device.get("device") == "GPU":
                        try:
                            flops = float(device.get("flops", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            flops = 0.0
                        break
            return flops

        ring_lines = ["[b]Peers[/]"]
        ordered = [p for p in peers if p.peer_client_id != self._peer_client.peer_client_id]
        ordered.insert(0, self._peer_client.peer_device)  # self first in ring

        for peer in ordered:
            flops = _flops_value(peer)
            flops_label = f"{flops:.1f} TFLOPS" if flops else "-- TFLOPS"
            ring_lines.append("🖥️")
            ring_lines.append(f"{_host_label(peer)} {flops_label} {_status(peer)}")
        return "\n".join(ring_lines)

    def _peer_summary_text(self) -> str:
        peers = [self._peer_entry_to_obj(peer) for peer in self._peer_client.get_peers(include_self=False)]
        peers = [peer for peer in peers if peer is not None]
        if not peers:
            return "\n".join(
                [
                    "[b]Remote Nodes[/]",
                    "No remote peers yet.",
                ]
            )

        lines = ["[b]Remote Nodes[/]"]
        for peer in peers:
            host = str(getattr(peer, "ip_address", "") or getattr(peer, "address", "")).strip()
            if host in {"", "0.0.0.0"}:
                host = str(getattr(peer, "peer_client_id", "peer"))
            flops = getattr(peer, "gpu_flops", 0.0) or 0.0
            flops_label = f"{flops:.1f} TFLOPS" if flops else "-- TFLOPS"
            cpu_model = str(getattr(peer, "cpu_model", "") or "Unknown CPU")
            cpu_cores = int(getattr(peer, "cpu_cores", 0) or 0)
            cpu_ram = str(getattr(peer, "cpu_ram", "") or "0")
            gpu_model = str(getattr(peer, "gpu_model", "") or "Unknown GPU")
            gpu_vram = str(getattr(peer, "gpu_vram", "") or "0")
            lines.append(f"{host} {flops_label}")
            lines.append(f"CPU: {cpu_model} ({cpu_cores} cores, {cpu_ram} GB)")
            lines.append(f"GPU: {gpu_model} ({gpu_vram} VRAM)")
        return "\n".join(lines)
    
    

    def _update_action_variants(self) -> None:
        return

    @staticmethod
    def _help_text() -> str:
        return "\n".join(
            [
                "Network / Orchestration Screen",
                "- r: Refresh host and peer panels",
                "- a: Add peer by IP",
                "- p: Open peer directory",
                "- h: Open this help screen",
                "- b / Esc: Back",
            ]
        )

    @staticmethod
    def _peer_entry_to_obj(peer: object) -> object | None:
        if isinstance(peer, tuple) and len(peer) == 2:
            return peer[1]
        return peer
