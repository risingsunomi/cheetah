from __future__ import annotations

from typing import Any, Iterable

from rich.text import Text

from textual.binding import Binding
from textual.widgets import Static


class StaticBindingFooter(Static):
    """Footer that renders a fixed binding list and ignores focus changes."""

    DEFAULT_CSS = """
    StaticBindingFooter {
        dock: bottom;
        height: 1;
        width: 1fr;
        background: $primary;
        color: $text;
        padding: 0 1;
    }
    """

    def __init__(self, bindings: Iterable[Binding | tuple[Any, ...]], *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self._bindings = list(bindings)

    def on_mount(self) -> None:
        self.update(self.render_bindings(self._bindings))

    @classmethod
    def render_bindings(cls, bindings: Iterable[Binding | tuple[Any, ...]]) -> Text:
        text = Text()
        first = True
        for binding in bindings:
            key, description = cls._binding_parts(binding)
            if not key or not description:
                continue
            if not first:
                text.append("  ")
            text.append(key, style="bold #f6c544")
            text.append(f" {description}", style="#f5f5f5")
            first = False
        return text

    @staticmethod
    def _binding_parts(binding: Binding | tuple[Any, ...]) -> tuple[str, str]:
        if isinstance(binding, Binding):
            if not getattr(binding, "show", True):
                return "", ""
            key = str(getattr(binding, "key_display", None) or binding.key or "").strip()
            description = str(binding.description or "").strip()
            return StaticBindingFooter._display_key(key), description

        if not isinstance(binding, tuple) or len(binding) < 3:
            return "", ""
        key = str(binding[0] or "").strip()
        description = str(binding[2] or "").strip()
        return StaticBindingFooter._display_key(key), description

    @staticmethod
    def _display_key(key: str) -> str:
        if key.lower() == "escape":
            return "esc"
        return key
