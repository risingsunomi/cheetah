from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


PROMPT_TEMPLATE_NAME = "cot_agent_system_prompt.j2"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_PROMPT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.j2$")


def _resolve_prompt_dir(prompt_dir: str | Path | None = None) -> Path:
    base = Path(prompt_dir) if prompt_dir is not None else PROMPTS_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def normalize_prompt_name(prompt_name: str | None) -> str:
    raw = (prompt_name or PROMPT_TEMPLATE_NAME).strip()
    if not raw:
        raw = PROMPT_TEMPLATE_NAME
    raw = Path(raw).name
    if not raw.endswith(".j2"):
        raw = f"{raw}.j2"
    if not _PROMPT_NAME_RE.fullmatch(raw):
        raise ValueError(f"Invalid prompt name: {prompt_name!r}")
    return raw


@lru_cache(maxsize=8)
def _prompt_environment(prompt_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(prompt_dir),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        auto_reload=True,
        cache_size=0,
    )


def list_agent_prompt_names(*, prompt_dir: str | Path | None = None) -> list[str]:
    base = _resolve_prompt_dir(prompt_dir)
    prompts = sorted(path.name for path in base.glob("*.j2") if path.is_file())
    if PROMPT_TEMPLATE_NAME in prompts:
        prompts.remove(PROMPT_TEMPLATE_NAME)
        prompts.insert(0, PROMPT_TEMPLATE_NAME)
    return prompts


def load_agent_prompt_template(
    prompt_name: str | None = None,
    *,
    prompt_dir: str | Path | None = None,
) -> str:
    base = _resolve_prompt_dir(prompt_dir)
    resolved_name = normalize_prompt_name(prompt_name)
    prompt_path = base / resolved_name
    return prompt_path.read_text(encoding="utf-8")


def save_agent_prompt_template(
    prompt_name: str,
    content: str,
    *,
    prompt_dir: str | Path | None = None,
) -> str:
    base = _resolve_prompt_dir(prompt_dir)
    resolved_name = normalize_prompt_name(prompt_name)
    prompt_path = base / resolved_name
    prompt_path.write_text(content, encoding="utf-8")
    return resolved_name


def render_agent_system_prompt(
    *,
    prompt_name: str | None = None,
    prompt_dir: str | Path | None = None,
    **context: Any,
) -> str:
    base = _resolve_prompt_dir(prompt_dir)
    template = _prompt_environment(str(base)).get_template(normalize_prompt_name(prompt_name))
    return template.render(**context).strip()
