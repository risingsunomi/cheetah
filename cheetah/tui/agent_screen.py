from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.markup import escape
from transformers import AutoTokenizer

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.events import Mount
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Checkbox, Footer, Header, Input, Label, RichLog, Select, Static, TextArea

from cheetah.logging_utils import get_logger
from cheetah.agent.functions import AgentFunctions
from cheetah.agent.prompt_loader import (
    PROMPT_TEMPLATE_NAME,
    list_agent_prompt_names,
    load_agent_prompt_template,
    normalize_prompt_name,
    render_agent_system_prompt,
    save_agent_prompt_template,
)
from cheetah.models.llm.backend import (
    detect_quantization_mode,
    get_backend_device,
    get_llm_backend,
    load_model_for_backend,
    resolve_model_assets_for_backend,
)
from cheetah.orchestration.peer_client import PeerClient
from cheetah.tui.help_screen import HelpScreen
from cheetah.tui.helpers import (
    apply_chat_template_with_thinking,
    default_enable_thinking,
    memory_abort_reason,
    relieve_memory_pressure,
)
from cheetah.orchestration.distributed_inference import (
    MemoryPressureError,
    build_peer_load_plan,
    clear_model_shards_on_peers,
    distributed_shard_log_messages,
    format_shard_span,
    load_model_shards_on_peers,
    shard_plan_budget_errors,
    streaming_generate_with_peers,
    total_layers_from_model_config,
    validate_peer_runtime_fingerprints,
)
from cheetah.tui.widget.model_picker_screen import ModelPickerScreen

logger = get_logger(__name__)


class AgentScreen(Screen[None]):
    """Config-driven CoT agent control screen."""

    CSS_PATH = Path(__file__).with_name("agent_screen.tcss")
    BINDINGS = [
        ("escape", "pop_screen", "Back"),
        ("b", "pop_screen", "Back"),
        ("h", "open_help", "Help"),
    ]

    def __init__(
        self,
        peer_client: PeerClient,
        default_model: str | None = None,
        offline: bool = False,
    ) -> None:
        super().__init__()
        self._peer_client = peer_client
        self._offline = offline
        self._llm_backend: str = get_llm_backend()
        self._model_id: str = default_model or ""

        self._model: Optional[Any] = None
        self._model_config: Optional[dict[str, Any]] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_cache_path: Optional[Path] = None
        self._model_loaded: bool = False
        self._model_is_quantized: bool = False
        self._agent_running: bool = False
        self._function_format: str = "tools"
        self._agent_functions: list[dict[str, Any]] = []
        self._agent_messages: list[dict[str, str]] = []
        self._agent_task: Optional[asyncio.Task[str]] = None
        self._agent_runtime = AgentFunctions()
        self._agent_functions = self._agent_runtime.get_agent_functions(format_mode=self._function_format)
        self._agent_name: str = (os.getenv("TC_AGENT_NAME") or "cot-agent").strip() or "cot-agent"
        self._agent_instructions: str = (os.getenv("TC_AGENT_INSTRUCTIONS") or "").strip()
        self._endless_mode: bool = self._env_flag("TC_AGENT_ENDLESS_MODE", False)
        self._agent_max_steps: int = self._env_int("TC_AGENT_MAX_STEPS", 10, minimum=1)
        prompt_name = os.getenv("TC_AGENT_SYSTEM_PROMPT") or PROMPT_TEMPLATE_NAME
        try:
            self._agent_prompt_name: str = normalize_prompt_name(prompt_name)
        except ValueError:
            self._agent_prompt_name = PROMPT_TEMPLATE_NAME
        self._last_agent_response: str = ""

        self._model_label: Optional[Label] = None
        self._backend_label: Optional[Label] = None
        self._state_label: Optional[Label] = None
        self._config_summary: Optional[Static] = None
        self._functions_summary: Optional[Static] = None
        self._load_button: Optional[Button] = None
        self._start_button: Optional[Button] = None
        self._stop_button: Optional[Button] = None
        self._config_button: Optional[Button] = None
        self._cli_button: Optional[Button] = None
        self._agent_log: Optional[RichLog] = None

        self._gen_overrides: Dict[str, float | int | bool] = {}
        self._gen_inputs: dict[str, Input] = {}
        self._thinking_checkbox: Optional[Checkbox] = None
        self._endless_checkbox: Optional[Checkbox] = None
        self._instructions_area: Optional[TextArea] = None

    def compose(self) -> ComposeResult:
        effective = self._resolved_gen_config()
        yield Header(show_clock=True)
        with Container(id="agent-root"):
            with Container(id="agent-body"):
                with Container(id="agent-main"):
                    agent_log = RichLog(id="agent-log", markup=True, auto_scroll=True, wrap=True, highlight=True)
                    self._agent_log = agent_log
                    yield agent_log

                with VerticalScroll(id="agent-side"):
                    with Container(id="agent-run-actions"):
                        start_button = Button("Start", id="agent-start", variant="primary")
                        self._start_button = start_button
                        yield start_button
                        stop_button = Button("Stop", id="agent-stop", variant="error")
                        stop_button.disabled = True
                        self._stop_button = stop_button
                        yield stop_button

                    with Static(id="agent-model-panel"):
                        yield Label("Model", classes="panel-title")
                        model_value = Label(self._model_id or "None selected", id="agent-model-value")
                        self._model_label = model_value
                        yield model_value
                        backend_value = Label(self._llm_backend, id="agent-backend-value")
                        self._backend_label = backend_value
                        yield backend_value

                    with Container(id="agent-model-actions"):
                        yield Button("Select Model", id="agent-open-model-picker")

                    with Container(id="agent-config-actions"):
                        config_button = Button("Agent Config", id="agent-open-config", variant="primary")
                        self._config_button = config_button
                        yield config_button

                    with Static(id="agent-instructions-panel"):
                        yield Label("Instructions", classes="panel-title")
                        instructions_area = TextArea(
                            self._agent_instructions,
                            id="agent-instructions",
                            soft_wrap=True,
                            show_line_numbers=False,
                            placeholder="Describe goals, constraints, and desired output for the agent.",
                        )
                        self._instructions_area = instructions_area
                        yield instructions_area

                    with Static(id="agent-gen-panel"):
                        yield Label("Generation", classes="panel-title")
                        with Container(classes="agent-gen-row"):
                            yield Label("Temp", classes="agent-gen-label")
                            yield self._make_gen_input("temperature", effective["temperature"])
                        with Container(classes="agent-gen-row"):
                            yield Label("Top K", classes="agent-gen-label")
                            yield self._make_gen_input("top_k", effective["top_k"])
                        with Container(classes="agent-gen-row"):
                            yield Label("Top P", classes="agent-gen-label")
                            yield self._make_gen_input("top_p", effective["top_p"])
                        with Container(classes="agent-gen-row"):
                            yield Label("Repeat", classes="agent-gen-label")
                            yield self._make_gen_input("repetition_penalty", effective["repetition_penalty"])
                        with Container(classes="agent-gen-row"):
                            yield Label("Alpha F", classes="agent-gen-label")
                            yield self._make_gen_input("alpha_f", effective["alpha_f"])
                        with Container(classes="agent-gen-row"):
                            yield Label("Alpha P", classes="agent-gen-label")
                            yield self._make_gen_input("alpha_p", effective["alpha_p"])
                        with Container(classes="agent-gen-row"):
                            yield Label("Max Steps", classes="agent-gen-label")
                            yield self._make_gen_input("max_steps", effective["max_steps"])
                        thinking_checkbox = Checkbox("Enable Thinking", id="agent-gen-enable-thinking")
                        thinking_checkbox.value = bool(effective["enable_thinking"])
                        self._thinking_checkbox = thinking_checkbox
                        yield thinking_checkbox
                        endless_checkbox = Checkbox("Endless Mode", id="agent-gen-endless-mode")
                        endless_checkbox.value = bool(self._endless_mode)
                        self._endless_checkbox = endless_checkbox
                        yield endless_checkbox

                    
                    
                    with Static(id="agent-status-panel"):
                        yield Label("Status", classes="panel-title")
                        state_value = Label("Idle", id="agent-state-value")
                        self._state_label = state_value
                        yield state_value
                        yield Static("Available Functions", classes="panel-title")
                        functions_summary = Label("0", id="agent-functions-value")
                        self._functions_summary = functions_summary
                        yield functions_summary
                    
        yield Footer()

    async def on_mount(self, _: Mount) -> None:
        self._sync_functions_with_runtime()
        self._refresh_agent_config_summary()
        self._refresh_backend_label()
        self._refresh_state_label()
        self._refresh_functions_summary()
        self._log("Agent screen ready.")

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_open_help(self) -> None:
        self.app.push_screen(HelpScreen("Agent Help", self._help_text()))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "agent-open-model-picker":
            self._open_model_picker()
        elif button_id == "agent-open-config":
            self._open_agent_config()
        elif button_id == "agent-load-model":
            await self._start_model_load()
        elif button_id == "agent-open-cli":
            self._open_cli_menu()
        elif button_id == "agent-start":
            await self._start_agent()
        elif button_id == "agent-stop":
            self._stop_agent()

    def _open_model_picker(self) -> None:
        self.app.push_screen(ModelPickerScreen(self._model_id or ""), self._handle_model_selected)

    @staticmethod
    def _help_text() -> str:
        return "\n".join(
            [
                "Agent Screen",
                "- Start/Stop controls run looping agent execution.",
                "- Agent Config opens the name and system prompt editor.",
                "- Max Steps controls non-endless loop length.",
                "- Builtin tools are loaded from agent/functions.json and code handlers.",
                "- CLI Access runs one-off shell commands.",
                "- Endless Mode ignores end_run and loops until manual Stop.",
                "- h opens this help screen.",
                "- b / Esc returns to previous menu.",
            ]
        )

    def _handle_model_selected(self, result: Optional[str]) -> None:
        if not result:
            return
        selected = result.strip()
        if not selected:
            return
        if selected != self._model_id:
            self._clear_model(update_log=False)
        self._model_id = selected
        if self._model_label is not None:
            self._model_label.update(selected)
        self._log(f"Model set to '{selected}'.")

    def _open_agent_config(self) -> None:
        self.app.push_screen(
            AgentConfigModal(
                name=self._agent_name,
                prompt_name=self._agent_prompt_name,
                prompt_names=self._available_prompt_names(),
            ),
            self._apply_agent_config,
        )

    def _apply_agent_config(self, result: Optional[dict[str, Any]]) -> None:
        if not result:
            return
        self._agent_name = str(result.get("name", self._agent_name)).strip() or "cot-agent"
        prompt_name = str(result.get("prompt_name", self._agent_prompt_name)).strip() or self._agent_prompt_name
        try:
            self._agent_prompt_name = normalize_prompt_name(prompt_name)
        except ValueError:
            self._log(f"Invalid prompt selection '{prompt_name}', keeping '{self._agent_prompt_name}'.")
        self._refresh_agent_config_summary()
        self._refresh_state_label()
        self._log("Agent config updated.")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        checkbox_id = getattr(event.checkbox, "id", None)
        if checkbox_id == "agent-gen-enable-thinking":
            self._gen_overrides["enable_thinking"] = bool(event.value)
            return
        if checkbox_id == "agent-gen-endless-mode":
            self._endless_mode = bool(event.value)
            self._refresh_agent_config_summary()
            return

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "agent-instructions":
            return
        self._agent_instructions = event.text_area.text.strip()
        self._refresh_agent_config_summary()

    def _open_cli_menu(self) -> None:
        self.app.push_screen(AgentCLIModal(), self._handle_cli_command)

    def _handle_cli_command(self, command: Optional[str]) -> None:
        if command is None:
            return
        cmd = command.strip()
        if not cmd:
            self._log("CLI command was empty.")
            return
        asyncio.create_task(self._run_cli_command(cmd))

    async def _run_cli_command(self, command: str) -> None:
        self._log(f"[CLI] $ {command}")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            rc = int(proc.returncode or 0)
        except Exception as exc:
            self._log(f"[CLI] Failed: {exc}")
            return

        out_text = stdout.decode("utf-8", errors="replace").strip()
        err_text = stderr.decode("utf-8", errors="replace").strip()
        if out_text:
            self._log(f"[CLI][stdout] {out_text[:3000]}")
        if err_text:
            self._log(f"[CLI][stderr] {err_text[:3000]}")
        self._log(f"[CLI] Exit code: {rc}")

    def _resolved_gen_config(self) -> dict[str, float | int | bool]:
        config = self._model_config if isinstance(self._model_config, dict) else {}

        def _as_float(value: Any, default: float) -> float:
            try:
                return default if value is None else float(value)
            except (TypeError, ValueError):
                return default

        def _as_int(value: Any, default: int) -> int:
            try:
                return default if value is None else int(value)
            except (TypeError, ValueError):
                return default

        return {
            "temperature": _as_float(self._gen_overrides.get("temperature", config.get("temperature")), 1.0),
            "top_k": _as_int(self._gen_overrides.get("top_k", config.get("top_k")), 0),
            "top_p": _as_float(self._gen_overrides.get("top_p", config.get("top_p")), 0.8),
            "repetition_penalty": _as_float(
                self._gen_overrides.get("repetition_penalty", config.get("repetition_penalty")),
                1.0,
            ),
            "alpha_f": _as_float(self._gen_overrides.get("alpha_f", 0.0), 0.0),
            "alpha_p": _as_float(self._gen_overrides.get("alpha_p", 0.0), 0.0),
            "max_steps": _as_int(self._gen_overrides.get("max_steps", self._agent_max_steps), self._agent_max_steps),
            "enable_thinking": bool(
                self._gen_overrides.get(
                    "enable_thinking",
                    default_enable_thinking(
                        model_path=self._model_cache_path,
                        tokenizer=self._tokenizer,
                    ),
                )
            ),
        }

    def _effective_gen_config(self) -> dict[str, float | int | bool]:
        effective = self._resolved_gen_config()
        parsers = {
            "temperature": float,
            "top_k": int,
            "top_p": float,
            "repetition_penalty": float,
            "alpha_f": float,
            "alpha_p": float,
            "max_steps": int,
        }
        for key, parser in parsers.items():
            widget = self._gen_inputs.get(key)
            if widget is None:
                continue
            raw = widget.value.strip()
            if not raw:
                continue
            try:
                effective[key] = parser(raw)
            except ValueError:
                continue
        try:
            max_steps = int(effective.get("max_steps", self._agent_max_steps))
        except (TypeError, ValueError):
            max_steps = self._agent_max_steps
        self._agent_max_steps = max(1, max_steps)
        effective["max_steps"] = self._agent_max_steps
        if self._thinking_checkbox is not None:
            effective["enable_thinking"] = bool(self._thinking_checkbox.value)
        return effective

    async def _start_model_load(self) -> None:
        if not self._model_id:
            self._log("Select a model first.")
            return
        if self._model is not None and self._tokenizer is not None:
            self._log("Model already loaded.")
            return
        self._set_load_button_enabled(False)
        self._llm_backend = get_llm_backend()
        self._refresh_backend_label()
        self._log(f"Loading model '{self._model_id}' with backend '{self._llm_backend}'...")
        peer_plan: dict[str, Any] | None = None
        try:
            preview_config, preview_model_path = await resolve_model_assets_for_backend(
                model_id=self._model_id,
                offline_mode=self._offline,
                backend=self._llm_backend,
            )
            total_layers = total_layers_from_model_config(preview_config)
            peer_plan = build_peer_load_plan(
                self._peer_client,
                model_name=self._model_id or "model",
                total_layers=total_layers,
                model_path=preview_model_path,
                model_config=preview_config,
                backend=self._llm_backend,
            )
            for line in distributed_shard_log_messages(
                self._peer_client,
                model_name=self._model_id or "model",
                total_layers=total_layers,
                model_path=preview_model_path,
                model_config=preview_config,
                backend=self._llm_backend,
            ):
                self._log(line)
            budget_errors = peer_plan.get("budget_errors") or shard_plan_budget_errors(peer_plan.get("peers", []))
            if budget_errors:
                raise RuntimeError("Shard plan exceeds reported memory: " + "; ".join(budget_errors))
            local_shard = peer_plan.get("local_shard") if peer_plan else None
            self._model, self._model_config, self._tokenizer, self._model_cache_path, elapsed = await self._load_model(
                shard=local_shard,
            )
        except Exception as exc:
            self._clear_model(update_log=False, clear_remote=False)
            if peer_plan is not None and peer_plan.get("distributed"):
                await asyncio.to_thread(
                    clear_model_shards_on_peers,
                    self._peer_client,
                    peers=peer_plan.get("remote_peers"),
                    model_id=self._model_id or "",
                )
            self._log(f"Model load failed: {exc}")
            logger.exception("Agent model load failed")
        else:
            try:
                self._model_is_quantized, quant_mode = detect_quantization_mode(
                    self._model_config,
                    backend=self._llm_backend,
                )
                mode_label = f"quantized ({quant_mode})" if self._model_is_quantized else "standard"
                local_shard = getattr(self._model, "shard", None)
                register_runtime = getattr(self._peer_client, "register_generation_runtime", None)
                if callable(register_runtime):
                    register_runtime(
                        model=self._model,
                        tokenizer=self._tokenizer,
                        backend=self._llm_backend,
                        model_id=self._model_id or "",
                        model_config=self._model_config,
                        model_path=str(self._model_cache_path or ""),
                        shard=getattr(self._model, "shard", None),
                    )
                if peer_plan is not None and peer_plan.get("distributed"):
                    remote_peer_count = len(peer_plan.get("remote_peers", []))
                    if remote_peer_count > 0:
                        self._log(f"Waiting for {remote_peer_count} peer shard(s) to finish loading...")
                    peer_load_plan = await asyncio.to_thread(
                        load_model_shards_on_peers,
                        self._peer_client,
                        model_id=self._model_id or "",
                        backend=self._llm_backend,
                        offline_mode=self._offline,
                        total_layers=total_layers_from_model_config(self._model_config),
                        peers=peer_plan.get("peers"),
                        model_path=self._model_cache_path,
                        model_config=self._model_config,
                    )
                    mismatches = validate_peer_runtime_fingerprints(
                        peer_load_plan.get("remote_results", []),
                        local_model_config=self._model_config,
                        local_model_path=self._model_cache_path,
                    )
                    if mismatches:
                        raise RuntimeError("; ".join(mismatches))
                    for entry in peer_load_plan.get("remote_results", []):
                        peer = entry.get("peer")
                        response = entry.get("response", {}) if isinstance(entry.get("response"), dict) else {}
                        label = self._peer_label_for_log(peer)
                        status = "already ready" if response.get("already_loaded") else "ready"
                        try:
                            peer_elapsed = float(response.get("elapsed", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            peer_elapsed = 0.0
                        shard = getattr(peer, "shard", None)
                        self._log(f"{label} {status} in {peer_elapsed:.1f}s: {format_shard_span(shard)}.")
                self._model_loaded = True
                if self._thinking_checkbox is not None and "enable_thinking" not in self._gen_overrides:
                    self._thinking_checkbox.value = default_enable_thinking(
                        model_path=self._model_cache_path,
                        tokenizer=self._tokenizer,
                    )
                if peer_plan is not None and peer_plan.get("distributed") and local_shard is not None:
                    self._log(
                        f"Local shard ready in {elapsed:.1f}s. Backend: {self._llm_backend}. "
                        f"Mode: {mode_label}. {format_shard_span(local_shard)}."
                    )
                    self._log(f"Shard-aware model ready on {len(peer_plan.get('peers', []))} nodes.")
                else:
                    self._log(f"Model ready in {elapsed:.1f}s. Backend: {self._llm_backend}. Mode: {mode_label}.")
            except Exception as exc:
                self._clear_model(update_log=False, clear_remote=False)
                if peer_plan is not None and peer_plan.get("distributed"):
                    await asyncio.to_thread(
                        clear_model_shards_on_peers,
                        self._peer_client,
                        peers=peer_plan.get("remote_peers"),
                        model_id=self._model_id or "",
                    )
                self._log(f"Model load failed: {exc}")
                logger.exception("Agent model load failed")
        finally:
            self._set_load_button_enabled(True)
            self._refresh_state_label()

    async def _load_model(self, *, shard: Any = None) -> tuple[Any, dict[str, Any], AutoTokenizer, Path, float]:
        self._llm_backend = get_llm_backend()
        start = time.time()
        model, model_config, tokenizer, model_path = await load_model_for_backend(
            model_id=self._model_id,
            shard=shard,
            weight_device=None,
            offline_mode=self._offline,
            backend=self._llm_backend,
        )
        elapsed = time.time() - start
        return model, model_config, tokenizer, model_path, elapsed

    async def _start_agent(self) -> None:
        if self._agent_running:
            return
        if not self._model_loaded:
            await self._start_model_load()
            if not self._model_loaded:
                return

        name = self._agent_name or "cot-agent"
        instructions = self._current_agent_instructions()
        if not instructions:
            self._log("Add agent instructions before starting.")
            return
        self._agent_instructions = instructions

        self._agent_running = True
        self._set_agent_controls_running(True)
        self._refresh_state_label()

        gen_cfg = self._effective_gen_config()
        max_steps = int(gen_cfg["max_steps"])
        self._log(f"Agent '{name}' started.")
        self._log(f"Instructions: {instructions}")
        self._log(
            "Generation settings: "
            f"temp={gen_cfg['temperature']}, "
            f"top_k={gen_cfg['top_k']}, "
            f"top_p={gen_cfg['top_p']}, "
            f"thinking={gen_cfg['enable_thinking']}, "
            f"alpha_f={gen_cfg['alpha_f']}, "
            f"alpha_p={gen_cfg['alpha_p']}, "
            f"max_steps={max_steps}"
        )
        self._log(
            f"Available functions: {len(self._agent_functions)}"
        )
        self._log(
            "Endless mode: enabled (ignores end_run)."
            if self._endless_mode
            else "Endless mode: disabled (end_run stops the loop)."
        )
        self._agent_messages = self._build_initial_messages(name=name, instructions=instructions)
        self._last_agent_response = ""
        self._agent_task = asyncio.create_task(self._agent_loop())

    def _stop_agent(self) -> None:
        if not self._agent_running:
            return
        self._agent_running = False
        if self._agent_task is not None and not self._agent_task.done():
            self._agent_task.cancel()
        self._agent_task = None
        self._set_agent_controls_running(False)
        self._refresh_state_label()
        self._log("Agent stopped.")

    def _make_gen_input(self, key: str, value: float | int) -> Input:
        input_widget = Input(id=f"agent-gen-{key}", placeholder=str(value))
        input_widget.value = str(value)
        self._gen_inputs[key] = input_widget
        return input_widget

    def _build_initial_messages(self, *, name: str, instructions: str) -> list[dict[str, str]]:
        enabled_functions = sorted(self._enabled_function_names())
        try:
            system_prompt = render_agent_system_prompt(
                prompt_name=self._agent_prompt_name,
                name=name,
                agent_prompt=instructions,
                endless_mode=self._endless_mode,
                enabled_functions=enabled_functions,
                function_format=self._function_format,
                tool_summary=self._tool_prompt_summary(),
            )
        except Exception as exc:
            logger.exception("Failed to render agent prompt '%s'", self._agent_prompt_name)
            self._log(
                f"Prompt '{self._agent_prompt_name}' failed to render ({exc}); "
                f"falling back to '{PROMPT_TEMPLATE_NAME}'."
            )
            system_prompt = render_agent_system_prompt(
                prompt_name=PROMPT_TEMPLATE_NAME,
                name=name,
                agent_prompt=instructions,
                endless_mode=self._endless_mode,
                enabled_functions=enabled_functions,
                function_format=self._function_format,
                tool_summary=self._tool_prompt_summary(),
            )

        self._log(f"Initial system prompt:\n{system_prompt}")
        return [
            {"role": "system", "content": system_prompt},
        ]

    async def _agent_loop(self) -> str:
        max_steps = max(1, int(self._agent_max_steps))
        max_memory_recoveries = self._env_int("TC_AGENT_MAX_MEMORY_RECOVERIES", 2, minimum=0)

        final_reply = ""
        recovery_attempts = 0
        try:
            step = 0
            while self._agent_running:
                reason = self._memory_abort_reason()
                if reason:
                    if recovery_attempts < max_memory_recoveries and self._recover_from_memory_pressure(reason):
                        recovery_attempts += 1
                        await asyncio.sleep(0)
                        continue
                    self._log(reason)
                    break

                step += 1
                if not self._endless_mode and step > max_steps:
                    self._log(f"Reached max steps ({max_steps}); stopping loop.")
                    break

                try:
                    reply = await asyncio.to_thread(self._generate_agent_reply, self._agent_messages)
                except MemoryPressureError as exc:
                    if recovery_attempts < max_memory_recoveries and self._recover_from_memory_pressure(
                        str(exc),
                        step=step,
                    ):
                        recovery_attempts += 1
                        await asyncio.sleep(0)
                        continue
                    self._log(str(exc))
                    break
                if not reply:
                    self._log(f"[agent][step {step}] Empty response; stopping loop.")
                    break

                recovery_attempts = 0
                final_reply = reply

                payload = self._extract_agent_payload(reply)
                if payload is None:
                    self._log(f"[agent][step {step}] raw response\n{reply}")
                else:
                    self._log_agent_response_json(step, payload)

                function_call = self._extract_function_call_from_payload(payload)
                if function_call is None:
                    self._log(f"[agent][step {step}] No function call found; continuing loop.")
                    compact_reply = self._compact_agent_reply_for_memory(payload)
                    if compact_reply is not None:
                        self._agent_messages.append(compact_reply)
                    nudge = (
                        "Return one JSON object with thoughts and ability. "
                        "Use ability.name and ability.args. "
                        "If the task is complete, use end_run."
                    )
                    self._agent_messages.append({"role": "user", "content": nudge})
                    await asyncio.sleep(0)
                    continue

                function_name, arguments = function_call
                compact_reply = self._compact_agent_reply_for_memory(payload, function_name=function_name, arguments=arguments)
                if compact_reply is not None:
                    self._agent_messages.append(compact_reply)
                call_json = json.dumps(
                    {"name": function_name, "arguments": arguments},
                    ensure_ascii=True,
                )
                self._log(f"[function.call]\n{call_json}")
                if function_name not in self._enabled_function_names():
                    result = {
                        "ok": False,
                        "error": f"Function '{function_name}' is not enabled.",
                        "enabled_functions": sorted(self._enabled_function_names()),
                    }
                else:
                    result = await asyncio.to_thread(
                        self._agent_runtime.execute_agent_function,
                        function_name,
                        arguments,
                    )

                result_summary = self._summarize_function_result(function_name, arguments, result)
                self._log(f"[function.result] {result_summary}")

                is_end_run = function_name == "end_run" and bool(result.get("ok")) and bool(result.get("end_run"))
                if is_end_run and not self._endless_mode:
                    self._log("end_run received; stopping loop.")
                    break
                if is_end_run and self._endless_mode:
                    self._log("end_run received but ignored because Endless Mode is enabled.")

                self._agent_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Function result: {result_summary}\n"
                            "Choose the next ability, or use end_run if the task is complete."
                        ),
                    }
                )
                await asyncio.sleep(0)

            if self._agent_running:
                self._log("Agent loop completed.")
            self._last_agent_response = final_reply
        except asyncio.CancelledError:
            self._log("Agent loop canceled.")
            raise
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self._log(f"Agent loop failed: {exc}")
            logger.exception("Agent loop failed")
            self._last_agent_response = final_reply
        finally:
            self._agent_running = False
            self._agent_task = None
            self._set_agent_controls_running(False)
            self._refresh_state_label()
        return final_reply

    def _enabled_function_names(self) -> set[str]:
        names: set[str] = set()
        for entry in self._agent_functions:
            if not isinstance(entry, dict):
                continue
            if "function" in entry and isinstance(entry.get("function"), dict):
                name = entry["function"].get("name")
            else:
                name = entry.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
        if not names:
            names = set(self._agent_runtime.list_builtin_functions())
        return names

    def _tool_prompt_summary(self) -> str:
        lines: list[str] = []
        for spec in self._agent_runtime.get_function_specs():
            properties = spec.parameters.get("properties", {})
            required = set(spec.parameters.get("required", []))
            arg_parts: list[str] = []
            for arg_name in properties.keys():
                suffix = "*" if arg_name in required else "?"
                arg_parts.append(f"{arg_name}{suffix}")
            signature = f"{spec.name}({', '.join(arg_parts)})" if arg_parts else f"{spec.name}()"
            lines.append(f"- {signature}: {spec.description}")
        return "\n".join(lines) if lines else "- none"

    def _context_window_tokens(self) -> int:
        config = self._model_config if isinstance(self._model_config, dict) else {}
        configured = config.get("max_seq_len", 2048)
        try:
            context_window = int(configured)
        except (TypeError, ValueError):
            context_window = 2048
        return max(256, context_window)

    def _response_reserve_tokens(self, context_window: int) -> int:
        reserve = os.getenv("TC_AGENT_MAX_RESP_LEN", "256")
        try:
            reserve_int = int(reserve)
        except (TypeError, ValueError):
            reserve_int = 256
        return max(32, min(reserve_int, context_window - 1))

    def _token_count_for_messages(self, messages: list[dict[str, str]]) -> int:
        if self._tokenizer is None:
            return 0
        template = apply_chat_template_with_thinking(
            self._tokenizer,
            messages,
            add_generation_prompt=True,
            tokenize=False,
            model_path=self._model_cache_path,
            enable_thinking=bool(self._effective_gen_config()["enable_thinking"]),
        )
        enc = self._tokenizer(template, return_tensors="np")
        return int(enc["input_ids"].shape[1])

    def _prepare_agent_prompt(self, messages: list[dict[str, str]]) -> tuple[dict[str, Any], int, list[dict[str, str]]]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        context_window = self._context_window_tokens()
        reserve = self._response_reserve_tokens(context_window)
        input_budget = max(1, context_window - reserve)

        selected: list[dict[str, str]] = []
        for message in reversed(messages):
            candidate = [message, *selected]
            token_count = self._token_count_for_messages(candidate)
            if token_count <= input_budget or not selected:
                selected = candidate
            else:
                break

        template = apply_chat_template_with_thinking(
            self._tokenizer,
            selected,
            add_generation_prompt=True,
            tokenize=False,
            model_path=self._model_cache_path,
            enable_thinking=bool(self._effective_gen_config()["enable_thinking"]),
        )
        enc = self._tokenizer(template, return_tensors="np")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens > input_budget:
            input_ids = input_ids[:, -input_budget:]
            attention_mask = attention_mask[:, -input_budget:]
            prompt_tokens = input_budget

        max_new_tokens = max(1, min(reserve, context_window - prompt_tokens))
        return {"input_ids": input_ids, "attention_mask": attention_mask}, max_new_tokens, selected

    def _generate_agent_reply(self, messages: list[dict[str, str]]) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model/tokenizer not loaded")

        enc, max_new_tokens, selected = self._prepare_agent_prompt(messages)
        if selected and selected != messages:
            messages[:] = selected

        gen_cfg = self._effective_gen_config()
        temp = float(gen_cfg["temperature"])
        top_k = int(gen_cfg["top_k"])
        top_p = float(gen_cfg["top_p"])
        repetition_penalty = float(gen_cfg["repetition_penalty"])
        alpha_f = float(gen_cfg["alpha_f"])
        alpha_p = float(gen_cfg["alpha_p"])

        if hasattr(self._model, "reset_kv_cache"):
            self._model.reset_kv_cache()

        if self._llm_backend == "torch":
            import torch

            device = self._torch_runtime_device()
            input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long, device=device)
        else:
            import tinygrad as tg

            input_ids = tg.Tensor(enc["input_ids"])
            attention_mask = tg.Tensor(enc["attention_mask"])

        out_tokens, _ = streaming_generate_with_peers(
            self._peer_client,
            self._model,
            input_ids,
            attention_mask,
            self._tokenizer,
            max_new_tokens=max_new_tokens,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            alpha_f=alpha_f,
            alpha_p=alpha_p,
            verbose=False,
            on_token=None,
            abort_check=self._memory_abort_reason,
        )
        if not out_tokens:
            return ""
        return self._tokenizer.decode(out_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _torch_runtime_device() -> str:
        device = str(get_backend_device("torch", default="cpu") or "cpu").strip().lower()
        if device in {"metal", "mps"}:
            return "mps"
        if device.startswith("cuda"):
            return device
        return "cpu"

    def _memory_abort_reason(self) -> str | None:
        return memory_abort_reason("agent loop")

    def _recover_from_memory_pressure(self, reason: str, *, step: int | None = None) -> bool:
        prefix = f"[agent][step {step}] " if step is not None else "[agent] "
        dropped = self._compact_agent_messages_for_memory_pressure()
        relieve_memory_pressure(self._model)
        remaining = self._memory_abort_reason()

        actions: list[str] = []
        if dropped:
            actions.append(f"compacted {dropped} older messages")
        actions.append("cleared runtime caches")
        self._log(f"{prefix}Memory pressure detected. {', '.join(actions)}.")

        if remaining:
            self._log(f"{prefix}Memory pressure persists after recovery: {remaining}")
            return False

        self._log(f"{prefix}Recovered from memory pressure; retrying with reduced context.")
        return True

    def _compact_agent_messages_for_memory_pressure(self) -> int:
        keep_tail = self._env_int("TC_AGENT_MEMORY_TRIM_KEEP_MESSAGES", 4, minimum=2)
        head_count = min(2, len(self._agent_messages))
        if len(self._agent_messages) <= head_count + keep_tail:
            return 0

        tail_start = max(head_count, len(self._agent_messages) - keep_tail)
        trimmed = self._agent_messages[head_count:tail_start]
        if not trimmed:
            return 0

        summary_lines: list[str] = []
        for message in trimmed[-6:]:
            role = str(message.get("role", "unknown")).strip() or "unknown"
            content = " ".join(str(message.get("content", "")).split())
            if len(content) > 160:
                content = content[:157] + "..."
            summary_lines.append(f"- {role}: {content or '<empty>'}")

        summary = (
            "Earlier turns were compacted due to memory pressure. "
            "Continue from the recent context and preserve the active task."
        )
        if summary_lines:
            summary += "\nCompacted context summary:\n" + "\n".join(summary_lines)

        self._agent_messages = [
            *self._agent_messages[:head_count],
            {"role": "user", "content": summary},
            *self._agent_messages[tail_start:],
        ]
        return len(trimmed)

    def _extract_agent_payload(self, text: str) -> dict[str, Any] | None:
        for candidate in self._json_candidates(text):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if self._extract_function_call_from_payload(payload) is not None or "thoughts" in payload:
                return payload
        compact_payload = self._payload_from_compact_agent_text(text)
        if compact_payload is not None:
            return compact_payload
        return None

    def _extract_function_call(self, text: str) -> tuple[str, dict[str, Any] | str] | None:
        return self._extract_function_call_from_payload(self._extract_agent_payload(text))

    def _extract_function_call_from_payload(
        self,
        payload: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any] | str] | None:
        if not isinstance(payload, dict):
            return None

        name = None
        arguments: dict[str, Any] | str = {}

        ability = payload.get("ability")
        if isinstance(ability, dict):
            name = ability.get("name")
            arguments = ability.get("args", {})

        function_call = payload.get("function_call")
        if name is None and isinstance(function_call, dict):
            name = function_call.get("name")
            arguments = function_call.get("arguments", {})

        if name is None:
            tool_calls = payload.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                first = tool_calls[0]
                if isinstance(first, dict):
                    fn = first.get("function", {})
                    if isinstance(fn, dict):
                        name = fn.get("name")
                        arguments = fn.get("arguments", {})

        if name is None and isinstance(payload.get("name"), str):
            name = payload.get("name")
            arguments = payload.get("arguments", {})

        if isinstance(name, str) and name.strip():
            return name.strip(), arguments
        return None

    def _compact_agent_reply_for_memory(
        self,
        payload: dict[str, Any] | None,
        *,
        function_name: str | None = None,
        arguments: dict[str, Any] | str | None = None,
    ) -> dict[str, str] | None:
        if not isinstance(payload, dict):
            return None

        thoughts = payload.get("thoughts")
        compact_thoughts: dict[str, str] = {}
        if isinstance(thoughts, dict):
            step_completed = self._truncate_text(str(thoughts.get("step completed", "")).strip(), 160)
            speak = self._truncate_text(str(thoughts.get("speak", "")).strip(), 160)
            if step_completed:
                compact_thoughts["step completed"] = step_completed
            if speak:
                compact_thoughts["speak"] = speak

        ability = payload.get("ability")
        ability_name = function_name
        ability_args = arguments
        if ability_name is None and isinstance(ability, dict):
            raw_name = ability.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                ability_name = raw_name.strip()
            if ability_args is None:
                ability_args = ability.get("args", {})

        compact_payload: dict[str, Any] = {}
        if compact_thoughts:
            compact_payload["thoughts"] = compact_thoughts
        if ability_name:
            compact_payload["ability"] = {
                "name": ability_name,
                "args": {} if ability_args is None else ability_args,
            }

        if not compact_payload:
            return None
        return {
            "role": "assistant",
            "content": json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":")),
        }

    def _log_agent_response_json(self, step: int, payload: dict[str, Any]) -> None:
        try:
            rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            rendered = str(payload)
        self._log(f"[agent][step {step}] thought/action JSON\n{rendered}")

    def _summarize_function_result(
        self,
        function_name: str,
        arguments: dict[str, Any] | str,
        result: dict[str, Any],
    ) -> str:
        if not bool(result.get("ok")):
            return self._truncate_text(f"{function_name} failed: {result.get('error', 'unknown error')}", 220)

        if function_name == "write_file":
            return self._truncate_text(
                f"write_file ok path={result.get('path')} chars={result.get('chars_written')} "
                f"created={result.get('created')} overwritten={result.get('overwritten')}",
                220,
            )
        if function_name == "edit_file":
            return self._truncate_text(
                f"edit_file ok path={result.get('path')} replacements={result.get('replacements')}",
                220,
            )
        if function_name == "read_file":
            content = self._truncate_text(str(result.get("content", "")).replace("\n", "\\n"), 120)
            return self._truncate_text(
                f"read_file ok path={result.get('path')} truncated={result.get('truncated')} content={content}",
                220,
            )
        if function_name == "list_dir":
            entries = result.get("entries")
            count = len(entries) if isinstance(entries, list) else 0
            return self._truncate_text(f"list_dir ok path={result.get('path')} entries={count}", 220)
        if function_name == "run_shell":
            stdout = self._truncate_text(str(result.get("stdout", "")).replace("\n", "\\n"), 80)
            stderr = self._truncate_text(str(result.get("stderr", "")).replace("\n", "\\n"), 80)
            return self._truncate_text(
                f"run_shell ok returncode={result.get('returncode')} stdout={stdout} stderr={stderr}",
                220,
            )
        if function_name == "get_env":
            return self._truncate_text(f"get_env ok name={result.get('name')} value={result.get('value')}", 220)
        if function_name == "web_search":
            return self._truncate_text(
                f"web_search ok query={result.get('query')} count={result.get('count')}",
                220,
            )
        if function_name == "end_run":
            return self._truncate_text(f"end_run ok summary={result.get('summary', '')}", 220)
        return self._truncate_text(
            f"{function_name} ok result={self._compact_json_text(result, limit=180)}",
            220,
        )

    @staticmethod
    def _compact_json_text(value: Any, *, limit: int = 180) -> str:
        try:
            text = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        except (TypeError, ValueError):
            text = str(value)
        return AgentScreen._truncate_text(text, limit)

    @staticmethod
    def _truncate_text(value: str, limit: int) -> str:
        text = " ".join(str(value).split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    @staticmethod
    def _json_candidates(text: str) -> list[str]:
        candidates: list[str] = []
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
            if block.strip():
                candidates.append(block.strip())

        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append(stripped)

        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            candidates.append(brace_match.group(0).strip())

        # Preserve order while removing duplicates.
        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    @staticmethod
    def _payload_from_compact_agent_text(text: str) -> dict[str, Any] | None:
        stripped = " ".join(str(text).strip().splitlines())
        if "ability=" not in stripped:
            return None

        ability_name = ""
        args: dict[str, Any] | str = {}
        speak = ""
        step_completed = ""
        for part in re.split(r"\s+\|\s+", stripped):
            if part.startswith("ability="):
                ability_name = part[len("ability=") :].strip()
            elif part.startswith("args="):
                raw_args = part[len("args=") :].strip()
                if raw_args:
                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        query_match = re.search(r'"query"\s*:\s*"([^"]+)"', raw_args)
                        args = {"query": query_match.group(1)} if query_match else raw_args
                    else:
                        args = parsed_args if isinstance(parsed_args, dict) else raw_args
            elif part.startswith("speak="):
                speak = part[len("speak=") :].strip()
            elif part.startswith("step="):
                step_completed = part[len("step=") :].strip()

        if not ability_name:
            return None

        thoughts: dict[str, str] = {}
        if step_completed:
            thoughts["step completed"] = AgentScreen._truncate_text(step_completed, 160)
        if speak:
            thoughts["speak"] = AgentScreen._truncate_text(speak, 160)
        return {
            "thoughts": thoughts,
            "ability": {
                "name": ability_name,
                "args": args,
            },
        }

    def _set_agent_controls_running(self, running: bool) -> None:
        if self._start_button is not None:
            self._start_button.disabled = running
        if self._stop_button is not None:
            self._stop_button.disabled = not running
        if self._config_button is not None:
            self._config_button.disabled = running
        if self._cli_button is not None:
            self._cli_button.disabled = running

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
        raw = os.getenv(name)
        if raw is None:
            value = default
        else:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                value = default
        if minimum is not None:
            value = max(minimum, value)
        return value

    def _sync_functions_with_runtime(self) -> None:
        self._agent_functions = self._agent_runtime.get_agent_functions(format_mode=self._function_format)

    def _available_prompt_names(self) -> list[str]:
        try:
            prompts = list_agent_prompt_names()
        except Exception:
            logger.exception("Failed to list agent prompts")
            prompts = []
        if self._agent_prompt_name not in prompts:
            prompts.append(self._agent_prompt_name)
        if PROMPT_TEMPLATE_NAME not in prompts:
            prompts.insert(0, PROMPT_TEMPLATE_NAME)
        return sorted(dict.fromkeys(prompts), key=lambda name: (name != PROMPT_TEMPLATE_NAME, name))

    def _current_agent_instructions(self) -> str:
        if self._instructions_area is not None:
            return self._instructions_area.text.strip()
        return self._agent_instructions.strip()

    def _refresh_agent_config_summary(self) -> None:
        if self._config_summary is None:
            return
        preview = " ".join(self._current_agent_instructions().split())
        if len(preview) > 180:
            preview = preview[:177] + "..."
        if not preview:
            preview = "<not set>"
        self._config_summary.update(
            "\n".join(
                [
                    f"Name: {self._agent_name}",
                    f"Prompt: {self._agent_prompt_name}",
                    f"Endless Mode: {'On' if self._endless_mode else 'Off'}",
                    "Instructions:",
                    preview,
                ]
            )
        )

    def _refresh_functions_summary(self) -> None:
        if self._functions_summary is None:
            return
        specs = self._agent_runtime.get_function_specs()
        self._functions_summary.update(f"{len(specs)}")

    def _set_load_button_enabled(self, enabled: bool) -> None:
        if self._load_button is not None:
            self._load_button.disabled = not enabled

    def _peer_label_for_log(self, peer: Any) -> str:
        if peer is None:
            return "peer"
        peer_id = str(getattr(peer, "peer_client_id", "") or "peer").strip()
        host = str(getattr(peer, "ip_address", "") or getattr(peer, "address", "")).strip()
        if not host or host == peer_id:
            return peer_id
        return f"{peer_id} ({host})"

    def _clear_model(self, *, update_log: bool = True, clear_remote: bool = True) -> None:
        existing_model = self._model
        if self._agent_running:
            self._stop_agent()
        clear_runtime = getattr(self._peer_client, "clear_generation_runtime", None)
        if callable(clear_runtime):
            clear_runtime(model=existing_model)
        if clear_remote:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                clear_model_shards_on_peers(
                    self._peer_client,
                    model_id=self._model_id or "",
                )
            else:
                loop.create_task(
                    asyncio.to_thread(
                        clear_model_shards_on_peers,
                        self._peer_client,
                        model_id=self._model_id or "",
                    )
                )
        self._model = None
        self._model_config = None
        self._tokenizer = None
        self._model_cache_path = None
        self._model_loaded = False
        self._model_is_quantized = False
        if update_log:
            self._log("Model cleared.")
        self._refresh_state_label()

    def _refresh_backend_label(self) -> None:
        if self._backend_label is not None:
            self._backend_label.update(f"Backend: {self._llm_backend}")

    def _refresh_state_label(self) -> None:
        if self._state_label is None:
            return
        if self._agent_running:
            state = "Running"
        elif self._model_loaded:
            state = "Ready"
        else:
            state = "Idle"
        self._state_label.update(state)

    def _log(self, message: str) -> None:
        if self._agent_log is None:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._agent_log.write(f"({stamp}) {escape(message)}")
        self._agent_log.scroll_end(animate=False, force=True)


class AgentConfigModal(ModalScreen[Optional[dict[str, Any]]]):
    BINDINGS = [
        Binding("escape", "dismiss_modal", "Close", show=False, priority=True),
        Binding("b", "dismiss_modal", "Close", show=False, priority=True),
    ]

    def __init__(
        self,
        *,
        name: str,
        prompt_name: str,
        prompt_names: list[str],
    ) -> None:
        super().__init__(id="agent-config-modal")
        self._initial_name = name
        self._initial_prompt_name = prompt_name
        self._prompt_names = prompt_names
        self._name_input: Optional[Input] = None
        self._prompt_select: Optional[Select[str]] = None
        self._prompt_editor: Optional[TextArea] = None
        self._status: Optional[Label] = None
        try:
            self._current_prompt_name = normalize_prompt_name(prompt_name)
        except ValueError:
            self._current_prompt_name = PROMPT_TEMPLATE_NAME

    def compose(self) -> ComposeResult:
        with Container(id="agent-config-modal-container"):
            with Container(id="agent-config-header"):
                yield Button("X", id="agent-config-close", classes="modal-close")
            yield Label("Name", classes="agent-field-label")
            name_input = Input(id="agent-config-name", placeholder="e.g. research-assistant")
            self._name_input = name_input
            yield name_input

            yield Label("System Prompt", classes="agent-field-label")
            prompt_select = Select(
                [(prompt_name, prompt_name) for prompt_name in self._prompt_names],
                value=self._current_prompt_name,
                allow_blank=False,
                id="agent-config-prompt-select",
            )
            self._prompt_select = prompt_select
            yield prompt_select

            yield Label("Prompt Template", classes="agent-field-label")
            prompt_editor = TextArea(
                "",
                id="agent-config-prompt-editor",
                soft_wrap=True,
                show_line_numbers=False,
                placeholder="Edit the Jinja system prompt template here.",
            )
            self._prompt_editor = prompt_editor
            yield prompt_editor

            with Container(id="agent-config-prompt-actions"):
                yield Button("Save Prompt", id="agent-config-save-prompt")
                yield Button("Save As", id="agent-config-save-prompt-as", variant="primary")

            status = Label("", id="agent-config-status")
            self._status = status
            yield status

    def on_mount(self, _: Mount) -> None:
        if self._name_input is not None:
            self._name_input.value = self._initial_name
            self.call_after_refresh(self._name_input.focus)
        self._load_prompt_into_editor(self._current_prompt_name)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "agent-config-prompt-select":
            return
        value = event.value
        if not isinstance(value, str) or not value.strip():
            return
        self._load_prompt_into_editor(value)

    def action_dismiss_modal(self) -> None:
        self._dismiss_with_current_config()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agent-config-close":
            self._dismiss_with_current_config()
            return
        if event.button.id == "agent-config-save-prompt":
            self._save_prompt()
            return
        if event.button.id == "agent-config-save-prompt-as":
            self.app.push_screen(
                PromptNameModal(
                    title="Save Prompt As",
                    initial=self._current_prompt_name,
                ),
                self._handle_save_prompt_as,
            )
            return

    def _dismiss_with_current_config(self) -> None:
        name = self._name_input.value.strip() if self._name_input is not None else ""
        self.dismiss(
            {
                "name": name or "cot-agent",
                "prompt_name": self._current_prompt_name,
            }
        )

    def _load_prompt_into_editor(self, prompt_name: str) -> None:
        try:
            resolved_name = normalize_prompt_name(prompt_name)
            content = load_agent_prompt_template(resolved_name)
        except Exception as exc:
            self._set_status(f"Failed to load prompt '{prompt_name}': {exc}")
            return

        self._current_prompt_name = resolved_name
        if self._prompt_select is not None and self._prompt_select.value != resolved_name:
            self._prompt_select.value = resolved_name
        if self._prompt_editor is not None:
            self._prompt_editor.load_text(content)
        self._set_status("")

    def _save_prompt(self) -> str | None:
        content = self._prompt_editor.text if self._prompt_editor is not None else ""
        try:
            saved_name = save_agent_prompt_template(self._current_prompt_name, content)
        except Exception as exc:
            self._set_status(f"Failed to save prompt '{self._current_prompt_name}': {exc}")
            return None
        self._refresh_prompt_options(saved_name)
        self._set_status("")
        return saved_name

    def _handle_save_prompt_as(self, result: Optional[str]) -> None:
        if result is None:
            return
        self._save_prompt_as(result)

    def _save_prompt_as(self, target_name: str) -> str | None:
        target_name = target_name.strip()
        if not target_name:
            self._set_status("Enter a prompt name for Save As.")
            return None
        content = self._prompt_editor.text if self._prompt_editor is not None else ""
        try:
            saved_name = save_agent_prompt_template(target_name, content)
        except Exception as exc:
            self._set_status(f"Failed to save prompt as '{target_name}': {exc}")
            return None
        self._current_prompt_name = saved_name
        self._refresh_prompt_options(saved_name)
        self._set_status("")
        return saved_name

    def _refresh_prompt_options(self, selected_name: str) -> None:
        if self._prompt_select is None:
            return
        prompt_names = list_agent_prompt_names()
        if selected_name not in prompt_names:
            prompt_names.append(selected_name)
        self._prompt_names = sorted(dict.fromkeys(prompt_names), key=lambda name: (name != PROMPT_TEMPLATE_NAME, name))
        self._prompt_select.set_options([(prompt_name, prompt_name) for prompt_name in self._prompt_names])
        self._prompt_select.value = selected_name

    def _set_status(self, message: str) -> None:
        if self._status is not None:
            self._status.update(message)


class PromptNameModal(ModalScreen[Optional[str]]):
    BINDINGS = [
        Binding("escape", "dismiss_modal", "Close", show=False, priority=True),
        Binding("b", "dismiss_modal", "Close", show=False, priority=True),
    ]

    def __init__(self, *, title: str, initial: str = "") -> None:
        super().__init__(id="agent-prompt-name-modal")
        self._title = title
        self._initial = initial
        self._name_input: Optional[Input] = None
        self._status: Optional[Label] = None

    def compose(self) -> ComposeResult:
        with Container(id="agent-prompt-name-modal-container"):
            with Container(id="agent-prompt-name-header"):
                yield Label(self._title, id="agent-prompt-name-title")
                yield Button("X", id="agent-prompt-name-close", classes="modal-close")
            name_input = Input(id="agent-prompt-name-field", placeholder="prompt_name.j2")
            self._name_input = name_input
            yield name_input
            status = Label("", id="agent-prompt-name-status")
            self._status = status
            yield status
            with Container(id="agent-prompt-name-buttons"):
                yield Button("Cancel", id="agent-prompt-name-cancel")
                yield Button("Save", id="agent-prompt-name-save", variant="primary")

    def on_mount(self, _: Mount) -> None:
        if self._name_input is not None:
            self._name_input.value = self._initial
            self.call_after_refresh(self._name_input.focus)

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id in {"agent-prompt-name-cancel", "agent-prompt-name-close"}:
            self.dismiss(None)
            return
        if event.button.id != "agent-prompt-name-save":
            return
        value = self._name_input.value.strip() if self._name_input is not None else ""
        if not value:
            self._set_status("Enter a prompt name.")
            return
        try:
            value = normalize_prompt_name(value)
        except ValueError as exc:
            self._set_status(str(exc))
            return
        self.dismiss(value)

    def _set_status(self, message: str) -> None:
        if self._status is not None:
            self._status.update(message)
