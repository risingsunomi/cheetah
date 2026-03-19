# tiny-cheetah distributed AI
from __future__ import annotations

import argparse
import atexit
import os
import sys
from typing import Dict, Optional

from tiny_cheetah.tui import main_menu

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]

try:
    from tinygrad.device import Device
except Exception:  # pragma: no cover - tinygrad missing or import failure
    Device = None  # type: ignore[assignment]
else:
    def _cleanup_tinygrad_devices() -> None:
        if Device is None:
            return
        opened = getattr(Device, "_opened_devices", set())
        for dev_name in list(opened):
            try:
                device = Device[dev_name]
            except Exception:
                continue
            try:
                device.synchronize()
            except Exception:
                pass
            try:
                device.finalize()
            except Exception:
                pass

    atexit.register(_cleanup_tinygrad_devices)


class TinyCheetahApp(main_menu.MainMenu):
    """Main menu app with optional training and chat defaults."""

    def __init__(
        self,
        chat_default: Optional[str] = None,
        offline_mode: bool = False
    ) -> None:
        super().__init__(
            chat_default=chat_default,
            offline_mode=offline_mode
        )


def parse_cli_args(argv: list[str]) -> tuple[Dict[str, object], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="Tiny Cheetah TUI launcher",
        add_help=True
    )
    parser.add_argument(
        "--chat-model",
        dest="chat_model",
        type=str,
        default=None,
        help="Default chat model identifier passed to the chat screen."
    )
    parser.add_argument(
        "--offline-mode",
        dest="offline_mode",
        action="store_true",
        help="Force chat mode to operate offline (use cached models only)."
    )

    parsed = parser.parse_args(argv)

    return parsed.chat_model, parsed.offline_mode


def main():
    if load_dotenv is not None:
        load_dotenv()

    env_chat_default = os.getenv("TC_CHAT_MODEL")
    chat_default, offline_mode = parse_cli_args(sys.argv[1:])

    if chat_default is None:
        chat_default = env_chat_default

    if not offline_mode:
        offline_mode = os.getenv("TC_OFFLINE_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

    app = TinyCheetahApp(
        chat_default=chat_default,
        offline_mode=offline_mode,
    )
    app.run()


if __name__ == "__main__":
    main()
