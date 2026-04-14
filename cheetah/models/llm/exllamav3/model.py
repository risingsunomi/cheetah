from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Model:
    """Loaded exllamav3 runtime objects bundled for Cheetah."""

    runtime_model: Any
    cache: Any
    generator: Any
    exllama_tokenizer: Any
    exllama_config: Any
    model_path: Path
    device: str
    cache_size_tokens: int

    def reset_kv_cache(self) -> None:
        try:
            self.generator.clear_queue()
        except Exception:
            pass

    def unload(self) -> None:
        try:
            self.generator.clear_queue()
        except Exception:
            pass
        unload = getattr(self.runtime_model, "unload", None)
        if callable(unload):
            unload()
