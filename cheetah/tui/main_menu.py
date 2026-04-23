from pathlib import Path
from typing import Optional

import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Header, Button, Label
from textual.containers import Container

from .chat_menu import ChatScreen
from .agent_screen import AgentScreen
from .train_menu import TrainScreen
from .orchestration_screen import OrchestrationScreen
from .settings_screen import SettingsScreen
from .help_screen import HelpScreen
from .widget.static_footer import StaticBindingFooter
from cheetah.orchestration.peer_client import PeerClient


class MainMenu(App):
    CSS_PATH = Path(__file__).with_name("main_menu.tcss")
    BINDINGS = [
        ("c", "open_chat", "Chat"),
        ("a", "open_agent", "Agent"),
        ("t", "open_train", "Train"),
        ("n", "open_network", "Network"),
        ("s", "open_settings", "Settings"),
        ("h", "open_help", "Help"),
        ("q", "quit_app", "Quit"),
        ("escape", "quit_app", "Quit"),
    ]

    def __init__(
        self,
        training_defaults: Optional[dict] = None,
        chat_default: Optional[str] = None,
        offline_mode: bool = False,
    ) -> None:
        super().__init__()
        self.training_defaults = training_defaults or {}
        self.chat_default = chat_default
        self.offline_mode = offline_mode
        self._peer_client = PeerClient()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main-menu"):
            yield Label(r"""
░░      ░░░  ░░░░  ░░        ░░        ░░        ░░░      ░░░  ░░░░  ░
▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒
▓  ▓▓▓▓▓▓▓▓        ▓▓      ▓▓▓▓      ▓▓▓▓▓▓▓  ▓▓▓▓▓  ▓▓▓▓  ▓▓        ▓
█  ████  ██  ████  ██  ████████  ███████████  █████        ██  ████  █
██      ███  ████  ██        ██        █████  █████  ████  ██  ████  █
                """, id="title-txt")

        with Container(id="menu-btn-ctnr"):
            yield Button("Chat", id="chat-btn")
            yield Button("Agent", id="agent-btn")
            yield Button("Train", id="train-btn")
            yield Button("Network", id="network-btn")
            yield Button("Settings", id="settings-btn")
            yield Button("Quit", id="quit-btn")
        yield StaticBindingFooter(self.BINDINGS)

    async def on_mount(self) -> None:
        self.title="[Nodes: 1]"
        if self.offline_mode:
            self.title += " [offline]"
        await asyncio.to_thread(self._get_peer_count)
        self.set_interval(5.0, self._get_peer_count)
    
    def action_pop_screen(self) -> None:
        self.exit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "chat-btn":
            self._open_chat()
        elif button_id == "agent-btn":
            self._open_agent()
        elif button_id == "train-btn":
            self._open_train()
        elif button_id == "network-btn":
            self._open_network()
        elif button_id == "settings-btn":
            self._open_settings()
        elif button_id == "quit-btn":
            self.exit()

    def action_open_chat(self) -> None:
        self._open_chat()

    def action_open_agent(self) -> None:
        self._open_agent()

    def action_open_train(self) -> None:
        self._open_train()

    def action_open_network(self) -> None:
        self._open_network()

    def action_open_settings(self) -> None:
        self._open_settings()

    def action_open_help(self) -> None:
        self.push_screen(HelpScreen("Main Menu Help", self._help_text()))

    def action_quit_app(self) -> None:
        self.exit()

    def _default_model(self) -> Optional[str]:
        defaults = getattr(self, "training_defaults", {}) or {}
        return self.chat_default or defaults.get("model-id") or defaults.get("custom-model-id")

    def _open_chat(self) -> None:
        self.push_screen(
            ChatScreen(
                default_model=self._default_model(),
                offline=self.offline_mode,
                peer_client=self._peer_client,
            )
        )

    def _open_agent(self) -> None:
        self.push_screen(
            AgentScreen(
                default_model=self._default_model(),
                offline=self.offline_mode,
                peer_client=self._peer_client,
            )
        )

    def _open_train(self) -> None:
        screen = TrainScreen(peer_client=self._peer_client)
        defaults = getattr(self, "training_defaults", None)
        if defaults:
            screen.apply_default_settings(defaults)
        self.push_screen(screen)

    def _open_network(self) -> None:
        self.push_screen(OrchestrationScreen(self._peer_client))

    def _open_settings(self) -> None:
        self.push_screen(SettingsScreen())

    def set_training_defaults(self, settings: Optional[dict]) -> None:
        self.training_defaults = settings or {}

    def _get_peer_count(self) -> None:
        count = self._peer_client.peer_count()
        title = f"[Nodes: {count}]"
        if self.offline_mode:
            title += " [offline]"
        self.title = title

    @staticmethod
    def _help_text() -> str:
        return "\n".join(
            [
                "Navigation",
                "- c: Open Chat",
                "- a: Open Agent",
                "- t: Open Train",
                "- n: Open Network",
                "- s: Open Settings",
                "- h: Open this help screen",
                "- q / Esc: Quit",
            ]
        )
