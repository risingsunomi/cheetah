from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from cheetah.agent.model import AgentFunctionSpec


FUNCTION_SPECS_PATH = Path(__file__).with_name("functions.json")


class AgentFunctions:
    def __init__(self) -> None:
        self._handlers = {
            "list_dir": self._fn_list_dir,
            "read_file": self._fn_read_file,
            "write_file": self._fn_write_file,
            "edit_file": self._fn_edit_file,
            "run_shell": self._fn_run_shell,
            "get_env": self._fn_get_env,
            "end_run": self._fn_end_run,
            "web_search": self._fn_web_search,
        }
        self._functions = self._load_function_specs()

    def _load_function_specs(self) -> dict[str, AgentFunctionSpec]:
        payload = json.loads(FUNCTION_SPECS_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError(f"Invalid agent function spec file: {FUNCTION_SPECS_PATH}")

        specs: dict[str, AgentFunctionSpec] = {}
        for entry in payload:
            if not isinstance(entry, dict):
                raise RuntimeError("Agent function specs must be JSON objects")

            name = str(entry.get("name", "")).strip()
            description = str(entry.get("description", "")).strip()
            parameters = entry.get("parameters")
            if not name or not description or not isinstance(parameters, dict):
                raise RuntimeError(f"Invalid agent function spec entry: {entry!r}")

            handler = self._handlers.get(name)
            if handler is None:
                raise RuntimeError(f"Missing handler for agent function '{name}'")

            specs[name] = AgentFunctionSpec(
                name=name,
                description=description,
                parameters=parameters,
                handler=handler,
            )
        return specs

    @staticmethod
    def _resolve_path(path: str) -> Path:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = Path.cwd() / target
        return target.resolve()

    def _fn_list_dir(self, path: str = ".") -> dict[str, Any]:
        base = self._resolve_path(path)
        if not base.exists():
            return {"ok": False, "error": f"Path not found: {base}"}
        if not base.is_dir():
            return {"ok": False, "error": f"Not a directory: {base}"}
        items = sorted(p.name for p in base.iterdir())
        return {"ok": True, "path": str(base), "entries": items}

    def _fn_read_file(self, path: str, max_chars: int = 4000) -> dict[str, Any]:
        target = self._resolve_path(path)
        if not target.exists():
            return {"ok": False, "error": f"File not found: {target}"}
        if not target.is_file():
            return {"ok": False, "error": f"Not a file: {target}"}
        try:
            data = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return {"ok": False, "error": f"Read failed: {exc}"}
        clipped = data[:max(1, int(max_chars))]
        return {
            "ok": True,
            "path": str(target),
            "content": clipped,
            "truncated": len(data) > len(clipped),
        }

    def _fn_write_file(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
        make_dirs: bool = True,
    ) -> dict[str, Any]:
        target = self._resolve_path(path)
        existed = target.exists()
        if existed and target.is_dir():
            return {"ok": False, "error": f"Cannot write to directory: {target}"}
        if existed and not overwrite:
            return {"ok": False, "error": f"File already exists: {target}"}
        try:
            if make_dirs:
                target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "error": f"Write failed: {exc}"}
        return {
            "ok": True,
            "path": str(target),
            "created": not existed,
            "overwritten": existed,
            "chars_written": len(str(content)),
        }

    def _fn_edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        target = self._resolve_path(path)
        if not target.exists():
            return {"ok": False, "error": f"File not found: {target}"}
        if not target.is_file():
            return {"ok": False, "error": f"Not a file: {target}"}
        if not old_text:
            return {"ok": False, "error": "old_text is required"}
        try:
            data = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return {"ok": False, "error": f"Read failed: {exc}"}

        matches = data.count(old_text)
        if matches == 0:
            return {"ok": False, "error": f"old_text not found in file: {target}"}

        replacements = matches if replace_all else 1
        updated = data.replace(old_text, new_text, replacements)
        try:
            target.write_text(updated, encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "error": f"Write failed: {exc}"}

        return {
            "ok": True,
            "path": str(target),
            "replacements": replacements,
        }

    def _fn_run_shell(self, command: str, timeout_seconds: int = 20) -> dict[str, Any]:
        timeout = max(1, int(timeout_seconds))
        try:
            completed = subprocess.run(
                command,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except Exception as exc:
            return {"ok": False, "error": f"Command failed: {exc}"}
        return {
            "ok": True,
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout[:4000],
            "stderr": completed.stderr[:4000],
        }

    def _fn_get_env(self, name: str) -> dict[str, Any]:
        return {"ok": True, "name": name, "value": os.getenv(name)}

    def _fn_end_run(self, summary: str = "") -> dict[str, Any]:
        return {
            "ok": True,
            "end_run": True,
            "summary": str(summary or "").strip(),
        }

    def _fn_web_search(self, query: str, max_results: int = 5, timeout_seconds: int = 20) -> dict[str, Any]:
        q = str(query or "").strip()
        if not q:
            return {"ok": False, "error": "query is required"}

        try:
            limit = max(1, min(int(max_results), 10))
        except (TypeError, ValueError):
            limit = 5

        try:
            timeout = max(5, min(int(timeout_seconds), 120))
        except (TypeError, ValueError):
            timeout = 20

        try:
            results, search_url = self._bing_search_with_playwright(
                query=q,
                max_results=limit,
                timeout_seconds=timeout,
            )
        except ImportError:
            return {
                "ok": False,
                "error": (
                    "Playwright is not installed. Install with 'pip install playwright' "
                    "and run 'playwright install chromium'."
                ),
            }
        except Exception as exc:
            return {"ok": False, "error": f"Bing search failed: {exc}"}

        return {
            "ok": True,
            "engine": "bing",
            "query": q,
            "search_url": search_url,
            "count": len(results),
            "results": results,
        }

    def _bing_search_with_playwright(
        self,
        *,
        query: str,
        max_results: int,
        timeout_seconds: int,
    ) -> tuple[list[dict[str, str]], str]:
        from playwright.sync_api import sync_playwright

        search_url = f"https://www.bing.com/search?q={quote_plus(query)}&count={max_results}"
        timeout_ms = max(1, timeout_seconds) * 1000

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(search_url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_timeout(250)
                items = page.locator("li.b_algo")
                total = items.count()

                parsed: list[dict[str, str]] = []
                for idx in range(min(total, max_results)):
                    item = items.nth(idx)
                    link = item.locator("h2 a").first
                    href = (link.get_attribute("href") or "").strip()
                    if not href:
                        continue
                    title = (link.inner_text() or "").strip()
                    snippet = ""
                    snippet_nodes = item.locator("p")
                    if snippet_nodes.count() > 0:
                        snippet = (snippet_nodes.first.inner_text() or "").strip()
                    parsed.append({"title": title, "url": href, "snippet": snippet})

                return parsed, search_url
            finally:
                browser.close()

    def list_builtin_functions(self) -> list[str]:
        return sorted(self._functions.keys())

    def get_function_specs(self) -> list[AgentFunctionSpec]:
        return list(self._functions.values())

    def get_agent_functions(self, format_mode: str = "tools") -> list[dict[str, Any]]:
        mode = str(format_mode).strip().lower()
        if mode == "functions":
            return self.get_agent_functions_legacy()
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._functions.values()
        ]

    def get_agent_functions_legacy(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in self._functions.values()
        ]

    def execute_agent_function(self, name: str, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        target = self._functions.get(str(name))
        if target is None:
            return {"ok": False, "error": f"Unknown function: {name}"}

        if arguments is None:
            kwargs: dict[str, Any] = {}
        elif isinstance(arguments, str):
            raw = arguments.strip()
            if not raw:
                kwargs = {}
            else:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    return {"ok": False, "error": f"Invalid JSON arguments: {exc}"}
                if not isinstance(parsed, dict):
                    return {"ok": False, "error": "Function arguments must be a JSON object"}
                kwargs = parsed
        elif isinstance(arguments, dict):
            kwargs = arguments
        else:
            return {"ok": False, "error": "Unsupported arguments type"}

        try:
            result = target.handler(**kwargs)
        except TypeError as exc:
            return {"ok": False, "error": f"Bad arguments for {name}: {exc}"}
        except Exception as exc:  # pragma: no cover - defensive runtime path
            return {"ok": False, "error": f"{name} failed: {exc}"}

        if isinstance(result, dict):
            return result
        return {"ok": True, "result": result}
