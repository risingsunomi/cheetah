from __future__ import annotations

import sys


class WeightLoadProgress:
    def __init__(self, total: int, label: str = "load") -> None:
        self.total = max(0, int(total))
        self.label = label
        self._frames = ("|", "/", "-", "\\")
        self._idx = 0
        self._isatty = sys.stdout.isatty()
        self._last_len = 0

    def update(self, current: int, key: str, source_key: str, source_file: str | None = None) -> None:
        if self.total <= 0:
            return

        frame = self._frames[self._idx % len(self._frames)]
        self._idx += 1
        source = source_key if source_file is None else f"{source_key} @ {source_file}"
        message = f"[{self.label} {current}/{self.total} {frame}] {key} <- {source}"
        if self._isatty:
            padding = " " * max(0, self._last_len - len(message))
            print(f"\r{message}{padding}", end="", flush=True)
            self._last_len = len(message)
        else:
            print(message, flush=True)

    def done(self) -> None:
        if self.total <= 0:
            return
        if self._isatty:
            self._last_len = 0
            print("", flush=True)
