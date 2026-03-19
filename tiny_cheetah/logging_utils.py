from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

_configured = False
_log_file: Path | None = None


def _log_dir() -> Path:
    base_env = os.getenv("TC_LOG_DIR", "").strip()
    base_dir = Path(base_env) if base_env else Path(__file__).resolve().parent.parent / "logs"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _log_path() -> Path:
    day_stamp = datetime.now().strftime("%Y%m%d")
    return _log_dir() / f"tiny_cheetah_{day_stamp}.log"


def _update_latest_log_indicator(log_file: Path) -> None:
    latest_link = log_file.parent / "tiny_cheetah_latest.log"
    marker_file = log_file.parent / "LATEST_LOG"

    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(log_file.name)
    except OSError:
        marker_file.write_text(f"{log_file.name}\n", encoding="utf-8")
    else:
        marker_file.write_text(f"{latest_link.name} -> {log_file.name}\n", encoding="utf-8")


def configure_logging() -> Path:
    """Configure a shared file handler for the process."""
    global _configured, _log_file
    if _configured and _log_file is not None:
        return _log_file
    _log_file = _log_path()
    handler = logging.FileHandler(_log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    _update_latest_log_indicator(_log_file)
    _configured = True
    return _log_file


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
