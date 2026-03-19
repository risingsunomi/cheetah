from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AgentFunctionSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
