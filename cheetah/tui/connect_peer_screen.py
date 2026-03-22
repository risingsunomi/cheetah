from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Static

from cheetah.orchestration.peer_client import PeerClient
from cheetah.orchestration.cdevice import CDevice


class ConnectPeerScreen(Screen[None]):
    CSS_PATH = Path(__file__).with_name("connect_peer_screen.tcss")
    BINDINGS = [("escape", "pop_screen", "Back"), ("b", "pop_screen", "Back")]

    def __init__(self, peer_client: PeerClient) -> None:
        super().__init__()
        self._peer_client = peer_client
        self._host_input = Input(placeholder="host", id="connect-host")
        self._port_input = Input(placeholder="port", id="connect-port")
        self._status: Label | None = None
        self._host_info: Static | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="connect-root"):
            yield Label("Add Peer (IP:port)", id="connect-title")
            yield self._host_input
            yield self._port_input
            with Container(id="connect-actions"):
                yield Button("Add", id="connect-btn", variant="primary")
            status = Label("", id="connect-status")
            self._status = status
            yield status
            info = Static("Peer info will appear here after connecting.", id="connect-info")
            self._host_info = info
            yield info
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connect-btn":
            self._connect_peer()
        self._get_peer_count()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _connect_peer(self) -> None:
        host = self._host_input.value.strip()
        port_text = self._port_input.value.strip() or "8765"
        if not host:
            self._set_status("Specify host.")
            return
        try:
            port = int(port_text)
        except ValueError:
            self._set_status("Invalid port.")
            return
        try:
            peer = self._peer_client.add_peer(host, port)
        except Exception as exc:
            self._set_status(str(exc))
            return
        self._set_status(f"Added {peer.peer_client_id}")
        self._show_host_info(peer)

    def _set_status(self, message: str) -> None:
        if self._status is not None:
            self._status.update(message)

    def _show_host_info(self, peer: CDevice) -> None:
        lines = [
            f"Peer: [bold]{peer.peer_client_id}[/]",
            f"Devices: {', '.join(peer.devices) or 'unknown'}",
        ]
        if self._host_info is not None:
            self._host_info.update("\n".join(lines))

    async def on_mount(self) -> None:
        await asyncio.to_thread(self._get_peer_count)
        self.set_interval(5.0, self._get_peer_count)

    def _get_peer_count(self) -> None:
        if self.app is None:
            return
        count = self._peer_client.peer_count()
        if count > 1:
            new_title = f"[Nodes: {count}]"
            self.app.title = new_title
