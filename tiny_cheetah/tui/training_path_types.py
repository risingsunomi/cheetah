from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class TrainingNode:
    name: str
    settings: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    repeated: bool = False
    x: int = 0
    y: int = 0


NODE_STATUS_STYLES: Dict[str, str] = {
    "pending": "#5c7080",
    "running": "#3bd6ee",
    "complete": "#00ff00",
    "stopped": "#ffaa00",
}

NODE_STATUS_SYMBOLS: Dict[str, str] = {
    "pending": "○",
    "running": "●",
    "complete": "✔",
    "stopped": "■",
}