from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, Static


class HelpScreen(Screen[None]):
    """Reusable help screen that renders menu-specific help text."""

    BINDINGS = [("escape", "pop_screen", "Back"), ("b", "pop_screen", "Back")]

    def __init__(self, title: str, help_text: str) -> None:
        super().__init__()
        self._title = title
        self._help_text = help_text

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="help-root"):
            yield Label(self._title, id="help-title")
            with VerticalScroll(id="help-scroll"):
                yield Static(self._help_text, id="help-content")
        yield Footer()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

