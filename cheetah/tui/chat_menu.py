# Chat menu UI
# written with openai codex assistance
from __future__ import annotations

import asyncio
from datetime import datetime
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import traceback

from transformers import AutoTokenizer
from rich.markup import escape

from cheetah.models.llm.backend import (
    detect_quantization_mode,
    get_backend_device,
    get_llm_backend,
    load_model_for_backend,
    resolve_model_assets_for_backend,
)
from cheetah.tui.widget.model_picker_screen import ModelPickerScreen
from cheetah.tui.chat_log_storage import ChatLogStorage, ChatLogSummary, ChatMessage
from cheetah.orchestration.peer_client import PeerClient
from cheetah.tui.orchestration_screen import OrchestrationScreen
from cheetah.tui.help_screen import HelpScreen
from cheetah.tui.helpers import (
    apply_chat_template_with_thinking,
    default_enable_thinking,
    memory_abort_reason,
    split_thinking_response,
)
from cheetah.orchestration.distributed_inference import (
    MemoryPressureError,
    build_peer_load_plan,
    clear_model_shards_on_peers,
    distributed_shard_log_messages,
    format_shard_span,
    load_model_shards_on_peers,
    streaming_generate_with_peers,
    total_layers_from_model_config,
    validate_peer_runtime_fingerprints,
)

from textual.app import ComposeResult
from textual.containers import Container
from textual.events import Mount
from textual.geometry import Size
from textual.message import Message
from textual.message_pump import MessagePump
from textual.screen import Screen, ModalScreen
from textual.widgets import ( 
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
    LoadingIndicator
)

from cheetah.logging_utils import get_logger

logger = get_logger(__name__)
MAX_RESTORED_MESSAGES = 20
MAX_SEQ_LEN = 2048
MAX_RESP_LEN = 400

class ChatModelSelected(Message):
    def __init__(self, sender: MessagePump, model_id: str) -> None:
        super().__init__(sender)
        self.model_id = model_id

class ChatScreen(Screen[None]):
    """Primary chat interface."""

    CSS_PATH = Path(__file__).with_name("chat_menu.tcss")

    BINDINGS = [("escape", "pop_screen", "Back"),
        ("b", "pop_screen", "Back"),
        ("ctrl+s", "open_model_picker", "Select Model"),
        ("ctrl+n", "open_orchestration", "Network Nodes"),
        ("h", "open_help", "Help"),
    ]

    def __init__(self, peer_client: PeerClient, default_model: str | None = None, offline: bool = False) -> None:
        super().__init__()
        self._peer_client: PeerClient = peer_client
        self._offline: bool = offline
        self._llm_backend: str = get_llm_backend()
        self._model_id: str = default_model or ""
        self._tok_stats: float = 0.0
        self._model: Optional[Any] = None
        self._model_config: Optional[object] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_cache_path: Optional[Path] = None
        self._history: List[dict[str, str]] = []
        self._generating_resp: bool = False
        self.out_tokens: List[int] = []
        self._model_loaded: bool = False
        self._model_is_quantized: bool = False
        self._log_storage = ChatLogStorage()
        self._current_log_id: Optional[int] = None
        
        self._chat_log: Optional[RichLog] = None
        self._chat_input: Optional[Input] = None
        self._stats_label: Optional[Label] = None
        self._model_label: Optional[Label] = None
        self._chat_log_list: Optional[ListView] = None
        self._load_button: Optional[Button] = None
        self._peer_label: Optional[Label] = None
        self._torch_peer_notice_shown: bool = False
        self._gen_overrides: Dict[str, float | int | bool] = {}
        self._streaming_reply_open: bool = False
        self._streaming_reply_timestamp: Optional[str] = None
        self._streamed_chars: int = 0
        self._streaming_reply_text: str = ""
        self._streaming_reply_prefix: str = ""
        self._streaming_rendered_lines: int = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield LoadingIndicator(id="chat-loading-indicator")
        with Container(id="chat-root"):
            with Container(id="chat-body"):
                with Container(id="chat-history"):
                    yield Label("Chat Logs", classes="panel-title")
                    history_list = ListView(id="chat-log-list")
                    self._chat_log_list = history_list
                    yield history_list
                    with Container(id="chat-log-actions"):
                        yield Button("New Log", id="new-chat-log", variant="primary")
                        yield Button("Load Log", id="load-chat-log")
                        yield Button("Rename Log", id="rename-chat-log")
                        yield Button("Delete Log", id="delete-chat-log", variant="error")
                yield RichLog(id="chat-log", markup=True, auto_scroll=True, wrap=True, highlight=True)
                with Container(id="chat-side"):
                    with Static(id="model-panel"):
                        yield Label("Model", classes="panel-title")
                        model_value = Label(self._model_id or "<select>", id="model-value")
                        self._model_label = model_value
                        yield model_value
                        yield Button("Select Model", id="open-model-picker")
                        yield Button("Gen Config", id="open-gen-config")
                        load_button = Button("Load Model", id="load-model")
                        self._load_button = load_button
                        yield load_button
                        yield Button("Clear Model", id="clear-model", variant="error")
                    with Static(id="nodes-panel"):
                        yield Label("Nodes", classes="panel-title")
                        yield Label("Self", id="nodes-value")
                    with Static(id="stats-panel"):
                        yield Label("Processing", classes="panel-title")
                        stats_value = Label("Tokens/sec: --", id="stats-value")
                        self._stats_label = stats_value
                        yield stats_value
            yield Input(placeholder="", id="chat-input")
        yield Footer()

    async def on_mount(self, _: Mount) -> None:
        self._chat_log = self.query_one("#chat-log", RichLog)
        self._chat_input = self.query_one("#chat-input", Input)
        # if self._chat_input is not None:
        #     self.call_after_refresh(self._chat_input.focus)
        if self._model_label is not None:
            self._model_label.update(self._model_id or "<select>")
        await self._initialize_chat_logs()
        # Slow down peer discovery to avoid UI pauses while typing.
        await asyncio.to_thread(self._get_peer_count)
        self.set_interval(5.0, self._get_peer_count)

    def action_open_model_picker(self) -> None:
        self._open_model_picker()

    def action_open_orchestration(self) -> None:
        self.app.push_screen(OrchestrationScreen(self._peer_client))

    def action_open_help(self) -> None:
        self.app.push_screen(HelpScreen("Chat Help", self._help_text()))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open-model-picker":
            self._open_model_picker()
        elif event.button.id == "open-gen-config":
            self._open_gen_config()
        elif event.button.id == "load-model":
            self._chat_input.placeholder = "Loading model..."
            if not self._model_id:
                self._log_sys_msg("Select a model first.")
                return
            if self._model is not None and self._tokenizer is not None:
                self._log_sys_msg("Model already loaded.")
                return
            
            if not self._model_id:
                self._log_sys_msg("No model selected to load.")
                return

            self._set_load_button_enabled(False)
            
            self._llm_backend = get_llm_backend()
            self._log_sys_msg(
                f"Loading model '{self._model_id}' with backend '{self._llm_backend}'..."
            )
            await self._start_model_load()
            self._chat_input.placeholder = ""
        elif event.button.id == "clear-model":
            self._clear_model(persist=True)
        elif event.button.id == "new-chat-log":
            await self._create_new_log()
        elif event.button.id == "load-chat-log":
            await self._load_selected_chat_log()
        elif event.button.id == "open-orchestration":
            self.app.push_screen(OrchestrationScreen(self._peer_client))
        elif event.button.id == "rename-chat-log":
            self._prompt_rename_selected_log()
        elif event.button.id == "delete-chat-log":
            self._confirm_delete_selected_log()

    def _open_model_picker(self) -> None:
        self.app.push_screen(ModelPickerScreen(self._model_id or ""), self._handle_model_selected)

    def _open_gen_config(self) -> None:
        effective = self._effective_gen_config()
        modal = GenerationConfigModal(
            max_new_tokens=int(effective["max_new_tokens"]),
            temperature=effective["temperature"],
            top_k=int(effective["top_k"]),
            top_p=effective["top_p"],
            repetition_penalty=effective["repetition_penalty"],
            alpha_f=effective["alpha_f"],
            alpha_p=effective["alpha_p"],
            enable_thinking=bool(effective["enable_thinking"]),
        )
        self.app.push_screen(modal, self._apply_gen_config)

    def _apply_gen_config(self, result: Optional[dict[str, float | int | bool]]) -> None:
        if result is None:
            return
        self._gen_overrides.update(result)
        self._log_effective_gen_config()

    def _log_effective_gen_config(self) -> None:
        effective = self._effective_gen_config()
        context_window = self._context_window_tokens()
        self._log_sys_msg(
            "Gen config updated: "
            f"context_window={context_window}, "
            f"max_new_tokens={int(effective['max_new_tokens'])}, "
            f"temp={float(effective['temperature'])}, "
            f"top_k={int(effective['top_k'])}, "
            f"top_p={float(effective['top_p'])}, "
            f"repetition_penalty={float(effective['repetition_penalty'])}, "
            f"alpha_f={float(effective['alpha_f'])}, "
            f"alpha_p={float(effective['alpha_p'])}, "
            f"thinking={bool(effective['enable_thinking'])}",
            persist=False,
        )

    def _effective_gen_config(self) -> dict[str, float | int]:
        config = self._model_config if isinstance(self._model_config, dict) else {}

        def _as_float(value: Any, default: float) -> float:
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _as_int(value: Any, default: int) -> int:
            if value is None:
                return default
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        thinking_override = self._gen_overrides.get("enable_thinking")
        if thinking_override is None:
            thinking_enabled = default_enable_thinking(
                model_path=self._model_cache_path,
                tokenizer=self._tokenizer,
            )
        else:
            thinking_enabled = bool(thinking_override)

        return {
            "max_new_tokens": _as_int(self._gen_overrides.get("max_new_tokens"), self._default_max_new_tokens()),
            "temperature": _as_float(self._gen_overrides.get("temperature", config.get("temperature")), 1.0),
            "top_k": _as_int(self._gen_overrides.get("top_k", config.get("top_k")), 0),
            "top_p": _as_float(self._gen_overrides.get("top_p", config.get("top_p")), 0.8),
            "repetition_penalty": _as_float(
                self._gen_overrides.get("repetition_penalty", config.get("repetition_penalty")),
                1.0,
            ),
            "alpha_f": _as_float(self._gen_overrides.get("alpha_f", 0.0), 0.0),
            "alpha_p": _as_float(self._gen_overrides.get("alpha_p", 0.0), 0.0),
            "enable_thinking": thinking_enabled,
        }

    def _context_window_tokens(self) -> int:
        config = self._model_config if isinstance(self._model_config, dict) else {}
        configured = config.get("max_seq_len", MAX_SEQ_LEN)
        
        try:
            context_window = int(configured)
        except (TypeError, ValueError):
            context_window = MAX_SEQ_LEN
        
        if context_window <= 0:
            context_window = MAX_SEQ_LEN

        return context_window

    def _default_max_new_tokens(self) -> int:
        config = self._model_config if isinstance(self._model_config, dict) else {}
        try:
            reserve_int = int(config.get("max_new_tokens", MAX_RESP_LEN))
        except (TypeError, ValueError):
            reserve_int = MAX_RESP_LEN
        return max(1, reserve_int)

    def _response_reserve_tokens(self, context_window: int) -> int:
        reserve_int = int(self._effective_gen_config()["max_new_tokens"])
        return max(1, min(reserve_int, context_window - 1))

    def _token_count_for_messages(self, messages: List[dict[str, str]]) -> int:
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

    def _prepare_generation_prompt(
        self,
    ) -> tuple[Dict[str, Any], int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded")

        context_window = self._context_window_tokens()
        reserve = self._response_reserve_tokens(context_window)
        input_budget = max(1, context_window - reserve)

        # Keep most recent turns that fit budget.
        selected: List[dict[str, str]] = []
        for message in reversed(self._history):
            candidate = [message, *selected]
            candidate_tokens = self._token_count_for_messages(candidate)
            if candidate_tokens <= input_budget or not selected:
                selected = candidate
                continue
            break
        logger.debug("Selected messages for prompt: %s", selected)
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

        # Last-resort clamp if a single recent message is still too long.
        if prompt_tokens > input_budget:
            input_ids = input_ids[:, -input_budget:]
            attention_mask = attention_mask[:, -input_budget:]
            prompt_tokens = input_budget

        if selected and selected != self._history:
            self._history = selected

        max_new_tokens = max(1, min(reserve, context_window - prompt_tokens))
        logger.debug(
            "Prepared prompt tokens=%d context=%d reserve=%d max_new=%d history=%d",
            prompt_tokens,
            context_window,
            reserve,
            max_new_tokens,
            len(self._history),
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}, max_new_tokens

    def _trim_history_for_context(self) -> None:
        # Use the same token-budget logic as generation to keep only model-relevant history.
        if self._tokenizer is None or not self._history:
            return
        try:
            self._prepare_generation_prompt()
        except Exception:
            # If tokenizer/template fails, keep existing history unchanged.
            return

    def _handle_model_selected(self, result: Optional[str]) -> None:
        if not result:
            return
        result = result.strip()
        if not result:
            return
        if result != self._model_id:
            self._gen_overrides.pop("max_new_tokens", None)
            self._model = None
            self._model_config = None
            self._tokenizer = None
            self._model_cache_path = None
            self._history.clear()
        self._model_id = result
        if self._model_label is not None:
            self._model_label.update(result)
        asyncio.create_task(self._update_current_log_model_async())
        self._log_sys_msg(f"Model set to '{result}'.", persist=False)
        self._log_sys_msg("Click 'Load Model' to download weights into memory.", persist=False)

    def action_pop_screen(self) -> None:
        """Allow Esc/b bindings to exit the chat screen cleanly."""
        self.app.pop_screen()
        # Defer model cleanup so screen pop isn't blocked by tensor teardown.
        asyncio.create_task(self._clear_model_after_pop())

    async def _clear_model_after_pop(self) -> None:
        await asyncio.sleep(0)
        self._clear_model(persist=False, reset_kv_cache=False)

    def _get_peer_count(self) -> None:
        if self.app is None:
            return
        count = self._peer_client.peer_count()
        self.app.title = f"[Nodes: {count}]"
    
    def _instruct_begin_chat(self) -> None:
        for history in self._history:
            if history["role"] in {"system", "user"}:
                return

        context_window = self._context_window_tokens()
        self._history.append({
            "role": "system",
            "content": (
                "You are a helpful assistant. Keep responses concise and stay within "
                f"the active context window ({context_window} tokens)."
            ),
        })

    @staticmethod
    def _help_text() -> str:
        return "\n".join(
            [
                "Chat Screen",
                "- Enter: Send message",
                "- Ctrl+S: Select model",
                "- Ctrl+N: Open network nodes",
                "- h: Open help",
                "- b / Esc: Back to main menu",
                "",
                "Tips",
                "- Use 'Load Model' before chatting.",
                "- 'Gen Config' changes temperature/top-k/top-p and penalties.",
            ]
        )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            self._instruct_begin_chat()
            
            content = event.value.strip()
            if not content:
                return
            event.input.value = ""

            if self._generating_resp:
                self._log_sys_msg("Model is generating a response; please wait...")
                return

            if self._current_log_id is None:
                self._log_sys_msg("Create or load a chat log before chatting.")
                return
            if not self._model_id:
                self._log_sys_msg("Select a model before chatting (Ctrl+S).")
                return
            if not self._model_loaded:
                await self._start_model_load()
                if not self._model_loaded:
                    return

            self._append_user(content)
            self._history.append({"role": "user", "content": content})

            self._start_model_stream()
            event.input.placeholder = "Generating response..."
            self._set_chat_input_enabled(False)
            self._generating_resp = True
            
            asyncio.create_task(self._run_generation())

    async def _run_generation(self) -> None:
        try:
            try:
                out_tokens, elapsed = await asyncio.to_thread(
                    self._generate_response,
                    self._stream_model_token_from_thread,
                )
            except MemoryPressureError as exc:
                self._finish_model_stream_line()
                self._streaming_reply_timestamp = None
                self._streamed_chars = 0
                self._log_sys_msg(str(exc))
                return
            except Exception as exc:
                self._finish_model_stream_line()
                self._streaming_reply_timestamp = None
                self._streamed_chars = 0
                self._log_sys_msg(f"Error during generation: {exc}")
                self._log_sys_msg(f"Traceback: {traceback.format_exc()}")
                return

            token_count = len(out_tokens)
            tok_rate = (token_count / elapsed) if elapsed > 0 else float("inf")
            self._tok_stats = tok_rate if token_count else 0.0
            if self._stats_label is not None:
                display_rate = self._tok_stats if token_count else 0.0
                self._stats_label.update(f"Tokens/sec: {display_rate:.1f}")

            reply = self._decode_tokens(out_tokens, skip_special_tokens=False)
            self._finalize_model_stream(reply)
        finally:
            self._generating_resp = False
            if self._chat_log is not None:
                # Nudge focus/refresh so final streamed text is painted reliably.
                self._chat_log.focus()
                self._chat_log.refresh()
            if self._chat_input is not None:
                self._chat_input.placeholder = ""
                self._set_chat_input_enabled(True)
                self.call_after_refresh(self._chat_input.focus)

    def _append_user(
        self,
        content: str,
        persist: bool = True,
        timestamp: Optional[str] = None
    ) -> None:
        if self._chat_log is None:
            return
        entry_time = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = escape(content)
        self._write_to_chat_log(
            f"({entry_time}) [bold][#3bd6ee]User[/][/bold]: {text}\n"
        )
        if persist:
            self._record_message("user", content, entry_time)
        self._chat_log.scroll_end(animate=False, force=True)

    def _append_system(
        self,
        content: str,
        persist: bool = True,
        timestamp: Optional[str] = None
    ) -> None:
        if self._chat_log is None:
            return
        entry_time = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = escape(content)
        self._write_to_chat_log(
            f"({entry_time}) [bold][#00FF00]System[/][/bold]: {text}\n"
        )
        if persist:
            self._record_message("system", content, entry_time)
        self._chat_log.scroll_end(animate=False, force=True)

    def _append_model(
        self,
        content: str,
        persist: bool = True,
        timestamp: Optional[str] = None
    ) -> None:
        if self._chat_log is None:
            return
        entry_time = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        markup, _, final_text = self._build_model_log_markup(
            content,
            timestamp=entry_time,
            show_empty_placeholder=True,
        )
        self._write_to_chat_log(markup + "\n")
        if persist:
            self._record_message("assistant", final_text or "(empty response)", entry_time)
        self._chat_log.scroll_end(animate=False, force=True)

    def _append_model_thinking(
        self,
        content: str,
        persist: bool = True,
        timestamp: Optional[str] = None,
    ) -> None:
        if self._chat_log is None:
            return
        thinking_text = content.strip()
        if not thinking_text:
            return
        entry_time = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_to_chat_log(
            f"{self._model_log_prefix(entry_time)} [italic](thinking)[/italic]: "
            f"[italic]{escape(thinking_text)}[/italic]\n"
        )
        if persist:
            self._record_message("assistant_thinking", thinking_text, entry_time)
        self._chat_log.scroll_end(animate=False, force=True)

    def _write_to_chat_log(self, message: str) -> None:
        if self._chat_log is None:
            return
        self._chat_log.write(message)
        scroll_end = getattr(self._chat_log, "scroll_end", None)
        if callable(scroll_end):
            scroll_end(animate=False)
            return
        action_scroll_end = getattr(self._chat_log, "action_scroll_end", None)
        if callable(action_scroll_end):
            action_scroll_end()

    def _set_chat_input_enabled(self, enabled: bool) -> None:
        if self._chat_input is not None:
            self._chat_input.disabled = not enabled

    def _start_model_stream(self) -> None:
        if self._chat_log is None:
            return
        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._streaming_reply_timestamp = entry_time
        self._streaming_reply_open = True
        self._streamed_chars = 0
        self._streaming_reply_text = ""
        self._streaming_reply_prefix = self._model_log_prefix(entry_time)
        self._streaming_rendered_lines = 0
        self._render_stream_line()

    def _append_model_stream_chunk(self, chunk: str) -> None:
        if not self._streaming_reply_open:
            return
        if not chunk:
            return
        self._streamed_chars += len(chunk)
        self._streaming_reply_text += chunk
        self._render_stream_line()

    def _render_stream_line(self) -> None:
        log = self._chat_log
        if log is None or not self._streaming_reply_open:
            return
        if self._streaming_rendered_lines > 0:
            remove_count = min(self._streaming_rendered_lines, len(log.lines))
            if remove_count > 0:
                del log.lines[-remove_count:]
                line_cache = getattr(log, "_line_cache", None)
                if line_cache is not None:
                    line_cache.clear()
                # Keep RichLog internals consistent after manual line removal.
                widest = int(getattr(log, "_widest_line_width", 0))
                log.virtual_size = Size(widest, len(log.lines))
        before = len(log.lines)
        entry_time = self._streaming_reply_timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        markup, _, _ = self._build_model_log_markup(
            self._streaming_reply_text,
            timestamp=entry_time,
            show_empty_placeholder=False,
        )
        log.write(markup)
        self._streaming_rendered_lines = max(0, len(log.lines) - before)
        log.refresh()
        log.scroll_end(animate=False, force=True)

    def _finish_model_stream_line(self) -> None:
        if not self._streaming_reply_open:
            return
        self._streaming_reply_open = False
        self._streaming_reply_text = ""
        self._streaming_reply_prefix = ""
        self._streaming_rendered_lines = 0

    def _finalize_model_stream(self, reply: str) -> None:
        # Replace the live streamed text with the final full decode so the
        # visible chat line always matches persisted chat history.
        if self._streaming_reply_open:
            self._streaming_reply_text = reply
            self._render_stream_line()

        self._finish_model_stream_line()
        # Visual break between assistant messages.
        self._write_to_chat_log("\n")
        entry_time = self._streaming_reply_timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        thinking_text, final_reply = split_thinking_response(reply, tokenizer=self._tokenizer)
        stored_reply = final_reply or "(empty response)"
        if thinking_text:
            self._record_message("assistant_thinking", thinking_text, entry_time)
        self._record_message("assistant", stored_reply, entry_time)
        self._history.append({"role": "assistant", "content": stored_reply})
        self._trim_history_for_context()
        self._streaming_reply_timestamp = None
        self._streamed_chars = 0

    def _stream_model_token_from_thread(self, token: int) -> None:
        piece = self._decode_token_piece(token)
        if not piece:
            return
        self.app.call_from_thread(self._append_model_stream_chunk, piece)

    def _decode_token_piece(self, token: int) -> str:
        tokenizer = self._tokenizer
        if tokenizer is None:
            return ""
        if token == tokenizer.eos_token_id:
            self.refresh()
            return ""
        return self._decode_tokens([token], skip_special_tokens=False)

    def _decode_tokens(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        tokenizer = self._tokenizer
        if tokenizer is None or not tokens:
            return ""
        try:
            return tokenizer.decode(
                tokens,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_log_prefix(self, timestamp: str) -> str:
        label = escape(self._model_id or "Model")
        return f"({timestamp}) [bold][#ff0000]{label}[/][/bold]"

    def _build_model_log_markup(
        self,
        raw_content: str,
        *,
        timestamp: str,
        show_empty_placeholder: bool,
    ) -> tuple[str, str, str]:
        thinking_text, final_text = split_thinking_response(raw_content, tokenizer=self._tokenizer)
        prefix = self._model_log_prefix(timestamp)
        lines: list[str] = []
        if thinking_text:
            lines.append(
                f"{prefix} [italic](thinking)[/italic]: [italic]{escape(thinking_text)}[/italic]"
            )
        if final_text:
            lines.append(f"{prefix}: {escape(final_text)}")
        elif show_empty_placeholder and not thinking_text:
            lines.append(f"{prefix}: {escape('(empty response)')}")
        elif not lines:
            lines.append(f"{prefix}: ")
        return "\n".join(lines), thinking_text, final_text

    def _handle_peer_generate_token_request(self, message: dict) -> dict:
        payload = message.get("payload", {})
        if not isinstance(payload, dict):
            return {"error": "invalid payload"}
        if self._model is None or self._tokenizer is None:
            return {"error": "model not loaded"}

        from cheetah.models.shard import Shard
        from cheetah.orchestration.model_engine import ModelEngine

        backend = "torch" if self._llm_backend == "torch" else "tinygrad"
        sender_peer_id = str(payload.get("sender_peer_id", "") or "").strip()
        request_tokens = self._flow_token_count(payload)
        if sender_peer_id:
            self._record_peer_flow(sender_peer_id, self._peer_client.peer_client_id, request_tokens, phase="request")
            self._mark_peer_seen(sender_peer_id)

        try:
            input_ids_data = self._payload_to_2d_list(payload.get("input_ids"))
            attention_mask_data = self._payload_to_2d_list(payload.get("attention_mask"))
            position_ids_data = self._payload_to_2d_list(payload.get("position_ids"))
            hidden_state_data = self._payload_to_2d_list(payload.get("hidden_state"))
            hidden_state = None if hidden_state_data in ([], [[]]) else self._payload_to_tensor(
                hidden_state_data,
                backend=backend,
                integer=False,
            )
            position_ids = None if position_ids_data in ([], [[]]) else self._payload_to_tensor(
                position_ids_data,
                backend=backend,
                integer=True,
            )

            input_ids = self._payload_to_tensor(input_ids_data, backend=backend, integer=True)
            attention_mask = self._payload_to_tensor(attention_mask_data, backend=backend, integer=True)

            shard = getattr(self._model, "shard", None)
            shard_payload = payload.get("shard")
            if isinstance(shard_payload, dict):
                model_name = str(
                    shard_payload.get("model_name")
                    or getattr(shard, "model_name", "")
                    or self._model_id
                    or "model"
                )
                start_layer = int(shard_payload.get("start_layer", getattr(shard, "start_layer", 0)) or 0)
                end_layer = int(shard_payload.get("end_layer", getattr(shard, "end_layer", 0)) or 0)
                total_layers = int(
                    shard_payload.get("total_layers", getattr(shard, "total_layers", end_layer)) or end_layer
                )
                shard = Shard(model_name, start_layer, end_layer, total_layers)

            engine = ModelEngine(shard=shard) if shard is not None else ModelEngine()
            response = engine.get_tokens(
                self._model,
                input_ids,
                attention_mask,
                self._tokenizer,
                hidden_state=hidden_state,
                position_ids=position_ids,
                prefill=bool(payload.get("prefill", False)),
                temp=float(payload.get("temp", 1.0) or 1.0),
                top_k=int(payload.get("top_k", 0) or 0),
                top_p=float(payload.get("top_p", 0.8) or 0.8),
                alpha_f=float(payload.get("alpha_f", 0.0) or 0.0),
                alpha_p=float(payload.get("alpha_p", 0.0) or 0.0),
                repetition_penalty=float(payload.get("repetition_penalty", 1.0) or 1.0),
                seen_tokens=[int(tok) for tok in payload.get("seen_tokens", []) or []],
                trace=bool(payload.get("trace", False)),
            )
            if sender_peer_id:
                response_tokens = self._flow_token_count(response)
                self._record_peer_flow(self._peer_client.peer_client_id, sender_peer_id, response_tokens, phase="response")
            return response
        except Exception as exc:
            logger.exception("Peer token generation failed: %s", exc)
            return {"error": str(exc)}

    @staticmethod
    def _payload_to_2d_list(value: Any) -> list[list[Any]]:
        if value is None:
            return [[]]
        if isinstance(value, list):
            if not value:
                return [[]]
            if isinstance(value[0], list):
                return value
            return [value]
        return [[]]

    @staticmethod
    def _flow_token_count(message: Any) -> int:
        payload = message.get("payload", message) if isinstance(message, dict) else message
        if not isinstance(payload, dict):
            return 0
        for key in ("attention_mask", "position_ids", "input_ids", "hidden_state"):
            count = ChatScreen._token_count_from_payload_value(payload.get(key))
            if count > 0:
                return count
        if payload.get("token") is not None or payload.get("tensor") is not None:
            return 1
        return 0

    @staticmethod
    def _token_count_from_payload_value(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, list):
            if not value:
                return 0
            if isinstance(value[0], list):
                return max((len(row) for row in value if isinstance(row, list)), default=0)
            return len(value)
        if isinstance(value, dict):
            shape = value.get("shape")
            if isinstance(shape, list) and shape:
                index = 1 if len(shape) >= 2 else 0
                try:
                    return max(int(shape[index]), 0)
                except (TypeError, ValueError):
                    return 0
        try:
            shape = tuple(getattr(value, "shape", ()) or ())
        except Exception:
            shape = ()
        if len(shape) >= 2:
            try:
                return max(int(shape[1]), 0)
            except (TypeError, ValueError):
                return 0
        if len(shape) == 1:
            try:
                return max(int(shape[0]), 0)
            except (TypeError, ValueError):
                return 0
        return 0

    def _record_peer_flow(self, source: str, target: str, tokens: int, *, phase: str) -> None:
        recorder = getattr(self._peer_client, "record_flow", None)
        if callable(recorder):
            recorder(source, target, tokens, phase=phase)

    def _mark_peer_seen(self, peer_client_id: str) -> None:
        marker = getattr(self._peer_client, "mark_peer_seen", None)
        if callable(marker):
            marker(peer_client_id)

    def _payload_to_tensor(self, data: list[list[Any]], *, backend: str, integer: bool) -> Any:
        if backend == "torch":
            import torch

            device = self._torch_runtime_device()
            dtype = torch.long if integer else torch.float32
            return torch.tensor(data, dtype=dtype, device=device)

        import tinygrad as tg

        dtype = tg.dtypes.int32 if integer else None
        device = get_backend_device("tinygrad", default="CPU")
        assert device is not None
        tensor = tg.Tensor(data, device=device)
        if dtype is not None:
            tensor = tensor.cast(dtype)
        return tensor

    @staticmethod
    def _torch_runtime_device() -> str:
        device = str(get_backend_device("torch", default="cpu") or "cpu").strip().lower()
        if device in {"metal", "mps"}:
            return "mps"
        if device.startswith("cuda"):
            return device
        return "cpu"

    def _record_message(self, role: str, content: str, timestamp: str) -> None:
        if self._current_log_id is None:
            return
        try:
            self._log_storage.append_message(self._current_log_id, role, content, timestamp)
        except Exception as exc:  # pragma: no cover - logging defensive path
            logger.exception("Failed to record chat message: %s", exc)

    def _set_load_button_enabled(self, enabled: bool) -> None:
        if self._load_button is not None:
            self._load_button.disabled = not enabled

    async def _initialize_chat_logs(self) -> None:
        summaries = self._log_storage.list_logs()
        if not summaries:
            default_name = datetime.now().strftime("Session %Y-%m-%d %H:%M:%S")
            new_id = self._log_storage.create_log(default_name, self._model_id or "")
            summaries = self._log_storage.list_logs()
            self._current_log_id = new_id
        else:
            self._current_log_id = summaries[0]["id"]
        summaries = await self._refresh_chat_log_list(select_id=self._current_log_id, summaries=summaries)
        if self._current_log_id is None:
            return
        current_log = self._log_storage.get_log(self._current_log_id)
        if current_log is not None:
            stored_model = current_log["model_id"] or ""
            if stored_model:
                self._model_id = stored_model
                if self._model_label is not None:
                    self._model_label.update(self._model_id or "<select>")
        messages = self._log_storage.get_messages(self._current_log_id, limit=MAX_RESTORED_MESSAGES)
        if self._chat_log is not None:
            self._chat_log.clear()
        self._history.clear()
        self._restore_messages_to_view(messages, persist=False)

    async def _refresh_chat_log_list(
        self,
        select_id: Optional[int] = None,
        summaries: Optional[List[ChatLogSummary]] = None
    ) -> List[ChatLogSummary]:
        if summaries is None:
            summaries = self._log_storage.list_logs()
        list_view = self._chat_log_list
        if list_view is None:
            return summaries
        await list_view.clear()
        for summary in summaries:
            item = self._build_log_item(summary)
            await list_view.mount(item)
        target_id = select_id
        if target_id is None and summaries:
            target_id = summaries[0]["id"]
        if target_id is not None:
            self._highlight_log(target_id)
        return summaries

    def _build_log_item(self, summary: ChatLogSummary) -> ListItem:
        label_text = self._format_log_item_label(summary)
        label = Label(label_text)
        return ListItem(label, id=f"log-{summary['id']}")

    @staticmethod
    def _format_log_item_label(summary: ChatLogSummary) -> str:
        name = summary["name"]
        model = summary["model_id"] or "no model"
        stamp = summary["updated_at"]
        stamp = stamp.split(".")[0]
        return f"{name} ({model}) [{stamp}]"

    def _highlight_log(self, log_id: int) -> None:
        if self._chat_log_list is None:
            return
        for index, child in enumerate(self._chat_log_list.children):
            if getattr(child, "id", "") == f"log-{log_id}":
                try:
                    self._chat_log_list.index = index  # type: ignore[attr-defined]
                except AttributeError:
                    pass
                break

    def _get_selected_log_id(self) -> Optional[int]:
        if self._chat_log_list is None:
            return None
        index = getattr(self._chat_log_list, "index", None)
        if index is None:
            return None
        try:
            child = self._chat_log_list.children[index]
        except IndexError:
            return None
        return self._log_id_from_widget(child)

    async def _create_new_log(self) -> None:
        name = datetime.now().strftime("Session %Y-%m-%d %H:%M:%S")
        try:
            new_id = self._log_storage.create_log(name, self._model_id or "")
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log_sys_msg(f"Failed to create new log: {exc}")
            return
        self._current_log_id = new_id
        await self._refresh_chat_log_list(select_id=new_id)
        if self._chat_log is not None:
            self._chat_log.clear()
        self._history.clear()
        self._log_sys_msg(f"Created log '{name}'.")
        self.call_after_refresh(lambda: self._open_log_name_modal(new_id, "Name Chat Log", name))

    def _prompt_rename_selected_log(self) -> None:
        log_id = self._get_selected_log_id()
        if log_id is None:
            log_id = self._current_log_id
        if log_id is None:
            self._log_sys_msg("Select a chat log to rename.")
            return
        summary = self._log_storage.get_log(log_id)
        if summary is None:
            self._log_sys_msg("Chat log not found.")
            return
        self._open_log_name_modal(log_id, "Rename Chat Log", summary["name"])

    def _open_log_name_modal(self, log_id: int, title: str, initial: str) -> None:
        modal = ChatLogNameModal(title, initial)
        self.app.push_screen(modal, lambda value: self._apply_log_name(log_id, value))

    def _apply_log_name(self, log_id: int, result: Optional[str]) -> None:
        if result is None:
            return
        name = result.strip()
        if not name:
            self._log_sys_msg("Log name cannot be blank.")
            return
        try:
            self._log_storage.rename_log(log_id, name)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log_sys_msg(f"Failed to rename chat log: {exc}")
            return
        self._log_sys_msg(f"Log renamed to '{name}'.")
        self._current_log_id = log_id
        asyncio.create_task(self._refresh_chat_log_list(select_id=log_id))

    def _confirm_delete_selected_log(self) -> None:
        log_id = self._get_selected_log_id()
        if log_id is None:
            log_id = self._current_log_id
        if log_id is None:
            self._log_sys_msg("Select a chat log to delete.")
            return
        summary = self._log_storage.get_log(log_id)
        if summary is None:
            self._log_sys_msg("Chat log not found.")
            return
        modal = ConfirmModal(
            "Delete Chat Log",
            f"Delete '{summary['name']}'? This cannot be undone.",
            confirm_label="Delete",
            confirm_variant="error",
        )
        self.app.push_screen(modal, lambda confirmed: self._finalize_delete_log(log_id, confirmed))

    def _finalize_delete_log(self, log_id: int, confirmed: Optional[bool]) -> None:
        if not confirmed:
            return
        try:
            self._log_storage.delete_log(log_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log_sys_msg(f"Failed to delete chat log: {exc}")
            return
        removed_current = self._current_log_id == log_id
        if removed_current:
            self._current_log_id = None
            self._history.clear()
            if self._chat_log is not None:
                self._chat_log.clear()
            self._clear_model(persist=False)
        self._log_sys_msg("Chat log deleted.")
        asyncio.create_task(self._post_delete_refresh(removed_current))

    async def _post_delete_refresh(self, reload_current: bool) -> None:
        summaries = self._log_storage.list_logs()
        if not summaries:
            default_name = datetime.now().strftime("Session %Y-%m-%d %H:%M:%S")
            new_id = self._log_storage.create_log(default_name, self._model_id or "")
            summaries = self._log_storage.list_logs()
            self._current_log_id = new_id
            reload_current = True
        valid_ids = {summary["id"] for summary in summaries}
        if self._current_log_id not in valid_ids:
            self._current_log_id = summaries[0]["id"] if summaries else None
            reload_current = True
        await self._refresh_chat_log_list(select_id=self._current_log_id, summaries=summaries)
        if reload_current and self._current_log_id is not None:
            summary = self._log_storage.get_log(self._current_log_id)
            if summary is not None:
                self._model_id = summary["model_id"] or ""
                if self._model_label is not None:
                    self._model_label.update(self._model_id or "<select>")
            messages = self._log_storage.get_messages(self._current_log_id, limit=MAX_RESTORED_MESSAGES)
            if self._chat_log is not None:
                self._chat_log.clear()
            self._history.clear()
            self._restore_messages_to_view(messages, persist=False)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        log_id = self._log_id_from_widget(event.item)
        if log_id is not None:
            self._current_log_id = log_id

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        log_id = self._log_id_from_widget(event.item)
        if log_id is None:
            return
        event.stop()
        self._handle_chat_log_load(log_id)

    async def _load_selected_chat_log(self) -> None:
        log_id = self._get_selected_log_id()
        if log_id is None:
            self._log_sys_msg("Select a chat log to load.")
            return
        self._handle_chat_log_load(log_id)

    def _handle_chat_log_load(self, log_id: int) -> None:
        summary = self._log_storage.get_log(log_id)
        if summary is None:
            self._log_sys_msg("Chat log not found.")
            return
        messages = self._log_storage.get_messages(log_id, limit=MAX_RESTORED_MESSAGES)
        self._current_log_id = log_id
        self._highlight_log(log_id)
        self._clear_model(persist=False)
        if self._chat_log is not None:
            self._chat_log.clear()
        self._history.clear()
        self._model_id = summary["model_id"] or ""
        if self._model_label is not None:
            self._model_label.update(self._model_id or "<select>")
        self._restore_messages_to_view(messages, persist=False)
        if self._model_id:
            self._start_model_load()
        else:
            self._log_sys_msg("No model associated with this log. Select one to continue.")

    def _restore_messages_to_view(self, messages: List[ChatMessage], persist: bool) -> None:
        for message in messages:
            role = message["role"]
            content = message["content"]
            timestamp = message["timestamp"]
            if role == "user":
                self._append_user(content, persist=persist, timestamp=timestamp)
                self._history.append({"role": "user", "content": content})
            elif role == "assistant_thinking":
                self._append_model_thinking(content, persist=persist, timestamp=timestamp)
            elif role == "assistant":
                self._append_model(content, persist=persist, timestamp=timestamp)
                self._history.append({"role": "assistant", "content": content})
            else:
                self._append_system(content=content, persist=persist, timestamp=timestamp)
        self._trim_history_for_context()

    async def _update_current_log_model_async(self) -> None:
        if self._current_log_id is None:
            return
        try:
            self._log_storage.set_log_model(self._current_log_id, self._model_id or "")
            await self._refresh_chat_log_list(select_id=self._current_log_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to update log model mapping: %s", exc)

    @staticmethod
    def _parse_log_item_id(item_id: Optional[str]) -> Optional[int]:
        if not item_id:
            return None
        if not item_id.startswith("log-"):
            return None
        try:
            return int(item_id.split("-", 1)[1])
        except (IndexError, ValueError):
            return None

    def _log_id_from_widget(self, widget: Optional[ListItem]) -> Optional[int]:
        if widget is None:
            return None
        return self._parse_log_item_id(getattr(widget, "id", None))

    async def _start_model_load(self) -> None:
        peer_plan: dict[str, Any] | None = None
        try:
            preview_config, _ = await resolve_model_assets_for_backend(
                model_id=self._model_id,
                offline_mode=self._offline,
                backend=self._llm_backend,
                progress_callback=self._log_model_download_progress,
            )
            total_layers = total_layers_from_model_config(preview_config)
            peer_plan = build_peer_load_plan(
                self._peer_client,
                model_name=self._model_id or "model",
                total_layers=total_layers,
            )
            for line in distributed_shard_log_messages(
                self._peer_client,
                model_name=self._model_id or "model",
                total_layers=total_layers,
            ):
                self._log_sys_msg(line)

            local_shard = peer_plan.get("local_shard") if peer_plan else None
            self._model, self._model_config, self._tokenizer, self._model_cache_path, elapsed = await self._load_model(
                shard=local_shard,
            )
        except Exception as exc:
            self._clear_model(persist=False, clear_remote=False)
            if peer_plan is not None and peer_plan.get("distributed"):
                await asyncio.to_thread(
                    clear_model_shards_on_peers,
                    self._peer_client,
                    peers=peer_plan.get("remote_peers"),
                    model_id=self._model_id or "",
                )
            self._log_sys_msg(f"Model load failed: {exc}\n traceback: {traceback.format_exc()}")
            self._set_load_button_enabled(True)
            if self._chat_input is not None:
                self._chat_input.focus()
        else:
            try:
                self._gen_overrides.pop("max_new_tokens", None)
                self._model_is_quantized, quant_mode = detect_quantization_mode(
                    self._model_config,
                    backend=self._llm_backend,
                )
                mode_label = f"quantized ({quant_mode})" if self._model_is_quantized else "standard"
                local_shard = getattr(self._model, "shard", None)
                if peer_plan is not None and peer_plan.get("distributed") and local_shard is not None:
                    ready_msg = (
                        f"Local shard ready in {elapsed:.1f}s. Backend: {self._llm_backend}. "
                        f"Mode: {mode_label}. {format_shard_span(local_shard)}."
                    )
                else:
                    ready_msg = (
                        f"Model ready in {elapsed:.1f}s. Backend: {self._llm_backend}. Mode: {mode_label}."
                    )
                self._log_sys_msg(ready_msg)
                self._log_effective_gen_config()
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
                        self._log_sys_msg(f"Waiting for {remote_peer_count} peer shard(s) to finish loading...")
                    peer_load_plan = await asyncio.to_thread(
                        load_model_shards_on_peers,
                        self._peer_client,
                        model_id=self._model_id or "",
                        backend=self._llm_backend,
                        offline_mode=self._offline,
                        total_layers=total_layers_from_model_config(self._model_config),
                        peers=peer_plan.get("peers"),
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
                        self._log_sys_msg(f"{label} {status} in {peer_elapsed:.1f}s: {format_shard_span(shard)}.")
                    self._log_sys_msg(
                        f"Shard-aware model ready on {len(peer_plan.get('peers', []))} nodes."
                    )
                self._model_loaded = True
                self._set_load_button_enabled(True)
            except Exception as exc:
                self._clear_model(persist=False, clear_remote=False)
                if peer_plan is not None and peer_plan.get("distributed"):
                    await asyncio.to_thread(
                        clear_model_shards_on_peers,
                        self._peer_client,
                        peers=peer_plan.get("remote_peers"),
                        model_id=self._model_id or "",
                    )
                self._log_sys_msg(f"Model load failed: {exc}\n traceback: {traceback.format_exc()}")
                self._set_load_button_enabled(True)
                if self._chat_input is not None:
                    self._chat_input.focus()

    def _log_sys_msg(self, message: str, *, persist: bool = False) -> None:
        self._append_system(message, persist=persist)

    async def _log_model_download_progress(self, message: str) -> None:
        self._log_sys_msg(f"[download] {message}", persist=False)

    async def _load_model(self, *, shard: Any = None) -> tuple[Any, object, AutoTokenizer, Path, float]:
        try:
            self._llm_backend = get_llm_backend()
            start = time.time()
            model, model_config, tokenizer, model_path = await load_model_for_backend(
                model_id=self._model_id,
                shard=shard,
                weight_device=None,
                offline_mode=self._offline,
                backend=self._llm_backend,
                progress_callback=self._log_model_download_progress,
            )
            elapsed = time.time() - start
            return model, model_config, tokenizer, model_path, elapsed
        except Exception as exc:
            logger.exception("Error loading model '%s': %s", self._model_id, exc)
            raise

    def _generate_response(
        self,
        on_token: Optional[Callable[[int], None]] = None,
    ) -> tuple[list[int], float]:
        enc, max_new_tokens = self._prepare_generation_prompt()
        if self._llm_backend == "torch":
            return self._generate_response_torch(
                enc=enc,
                max_new_tokens=max_new_tokens,
                on_token=on_token,
            )
        return self._generate_response_tinygrad(
            enc=enc,
            max_new_tokens=max_new_tokens,
            on_token=on_token,
        )

    def _generate_response_tinygrad(
        self,
        enc: Dict[str, Any],
        max_new_tokens: int = 4096,
        on_token: Optional[Callable[[int], None]] = None,
    ) -> tuple[list[int], float]:
        import tinygrad as tg

        input_ids = tg.Tensor(enc["input_ids"])
        attention_mask = tg.Tensor(enc["attention_mask"])

        if hasattr(self._model, "reset_kv_cache"):
            self._model.reset_kv_cache()

        max_new = max_new_tokens
        gen_cfg = self._effective_gen_config()
        temp = float(gen_cfg["temperature"])
        top_k = int(gen_cfg["top_k"])
        top_p = float(gen_cfg["top_p"])
        repetition_penalty = float(gen_cfg["repetition_penalty"])
        alpha_f = float(gen_cfg["alpha_f"])
        alpha_p = float(gen_cfg["alpha_p"])

        self._peer_client.in_use = True
        try:
            result = streaming_generate_with_peers(
                self._peer_client,
                self._model,
                input_ids,
                attention_mask,
                self._tokenizer,
                max_new_tokens=max_new,
                temp=temp,
                top_k=top_k,
                top_p=top_p,
                alpha_f=alpha_f,
                alpha_p=alpha_p,
                repetition_penalty=repetition_penalty,
                on_token=on_token,
                abort_check=self._memory_abort_reason,
            )
        finally:
            self._peer_client.in_use = False
        if result is None:
            raise RuntimeError("streaming_generate_with_peers returned no output")
        return result

    def _generate_response_torch(
        self,
        enc: Dict[str, Any],
        max_new_tokens: int = 4096,
        on_token: Optional[Callable[[int], None]] = None,
    ) -> tuple[list[int], float]:
        import torch

        device = self._torch_runtime_device()
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long, device=device)

        if hasattr(self._model, "reset_kv_cache"):
            self._model.reset_kv_cache()

        max_new = max_new_tokens
        gen_cfg = self._effective_gen_config()
        temp = float(gen_cfg["temperature"])
        top_k = int(gen_cfg["top_k"])
        top_p = float(gen_cfg["top_p"])
        repetition_penalty = float(gen_cfg["repetition_penalty"])
        alpha_f = float(gen_cfg["alpha_f"])
        alpha_p = float(gen_cfg["alpha_p"])

        self._peer_client.in_use = True
        try:
            result = streaming_generate_with_peers(
                self._peer_client,
                self._model,
                input_ids,
                attention_mask,
                self._tokenizer,
                max_new_tokens=max_new,
                temp=temp,
                top_k=top_k,
                top_p=top_p,
                alpha_f=alpha_f,
                alpha_p=alpha_p,
                repetition_penalty=repetition_penalty,
                on_token=on_token,
                abort_check=self._memory_abort_reason,
            )
        finally:
            self._peer_client.in_use = False
        if result is None:
            raise RuntimeError("streaming_generate_with_peers returned no output")
        return result

    def _memory_abort_reason(self) -> str | None:
        return memory_abort_reason("chat generation")

    def _peer_label_for_log(self, peer: Any) -> str:
        if peer is None:
            return "peer"
        peer_id = str(getattr(peer, "peer_client_id", "") or "peer").strip()
        host = str(getattr(peer, "ip_address", "") or getattr(peer, "address", "")).strip()
        if not host or host == peer_id:
            return peer_id
        return f"{peer_id} ({host})"

    def _clear_model(
        self,
        *,
        persist: bool = False,
        reset_kv_cache: bool = False,
        clear_remote: bool = True,
    ) -> None:
        existing_model = getattr(self, "_model", None)
        if persist:
            self._log_sys_msg("Clearing loaded model.", persist=True)
        if (
            reset_kv_cache
            and hasattr(self, "_model")
            and self._model is not None
            and hasattr(self._model, "reset_kv_cache")
        ):
            self._model.reset_kv_cache()
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
        self._torch_peer_notice_shown = False
        self._gen_overrides.pop("max_new_tokens", None)
        self._history.clear()
        self._generating_resp = False
        self._streaming_reply_open = False
        self._streaming_reply_timestamp = None
        self._streamed_chars = 0
        self._streaming_reply_text = ""
        self._streaming_reply_prefix = ""
        self._streaming_rendered_lines = 0
        self._set_chat_input_enabled(True)
        self._set_load_button_enabled(True)
        if persist:
            self._log_sys_msg("Model cleared. Select and load a model to continue.", persist=persist)

    def action_toggle_loading(self) -> None:
        # This calls for loading indicator by chaning css visibility
        loading_indicator = self.query_one("#chat-loading-indicator", LoadingIndicator)
        if loading_indicator is not None:
            current_visibility = loading_indicator.styles.visibility
            loading_indicator.styles.visibility = "visible" if current_visibility == "hidden" else "hidden"

class ChatLogNameModal(ModalScreen[Optional[str]]):
    """Modal dialog to capture a chat log name."""

    def __init__(self, title: str, initial: str = "") -> None:
        super().__init__(id="chat-log-name-modal")
        self._title = title
        self._initial = initial
        self._chat_input: Optional[Input] = None

    def compose(self) -> ComposeResult:
        with Container(id="chat-log-name-modal-container"):
            yield Label(self._title, id="chat-log-name-title")
            self._chat_input = Input(id="chat-log-name-field", placeholder="Enter log name…")
            yield self._chat_input
            with Container(id="chat-log-name-buttons"):
                yield Button("Cancel", id="chat-log-name-cancel")
                yield Button("Save", id="chat-log-name-save", variant="primary")

    def on_mount(self) -> None:
        if self._chat_input is not None:
            self._chat_input.value = self._initial
            # self.call_after_refresh(self._chat_input.focus)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat-log-name-cancel":
            self.dismiss(None)
        elif event.button.id == "chat-log-name-save":
            value = self._chat_input.value.strip() if self._chat_input else ""
            self.dismiss(value or None)


class GenerationConfigModal(ModalScreen[Optional[dict[str, float | int | bool]]]):
    """Modal dialog to edit generation hyperparameters."""

    def __init__(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        alpha_f: float,
        alpha_p: float,
        enable_thinking: bool,
    ) -> None:
        super().__init__(id="chat-gen-config-modal")
        self._defaults = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "alpha_f": alpha_f,
            "alpha_p": alpha_p,
        }
        self._inputs: dict[str, Input] = {}
        self._thinking_checkbox: Optional[Checkbox] = None
        self._enable_thinking = enable_thinking
        self._status: Optional[Label] = None

    def compose(self) -> ComposeResult:
        with Container(id="chat-gen-config-modal-container"):
            yield Label("Generation Config", id="chat-gen-config-title")
            yield Label("max_new_tokens", classes="chat-gen-config-label")
            max_new_tokens_input = Input(id="chat-gen-config-max-new-tokens", placeholder="e.g. 512")
            self._inputs["max_new_tokens"] = max_new_tokens_input
            yield max_new_tokens_input

            yield Label("temperature", classes="chat-gen-config-label")
            temp_input = Input(id="chat-gen-config-temperature", placeholder="e.g. 0.7")
            self._inputs["temperature"] = temp_input
            yield temp_input

            yield Label("top_k", classes="chat-gen-config-label")
            top_k_input = Input(id="chat-gen-config-top-k", placeholder="e.g. 40")
            self._inputs["top_k"] = top_k_input
            yield top_k_input

            yield Label("top_p", classes="chat-gen-config-label")
            top_p_input = Input(id="chat-gen-config-top-p", placeholder="e.g. 0.9")
            self._inputs["top_p"] = top_p_input
            yield top_p_input

            yield Label("repetition_penalty", classes="chat-gen-config-label")
            repetition_penalty_input = Input(
                id="chat-gen-config-repetition-penalty",
                placeholder="e.g. 1.05",
            )
            self._inputs["repetition_penalty"] = repetition_penalty_input
            yield repetition_penalty_input

            yield Label("alpha_f", classes="chat-gen-config-label")
            alpha_f_input = Input(id="chat-gen-config-alpha-f", placeholder="e.g. 0.0")
            self._inputs["alpha_f"] = alpha_f_input
            yield alpha_f_input

            yield Label("alpha_p", classes="chat-gen-config-label")
            alpha_p_input = Input(id="chat-gen-config-alpha-p", placeholder="e.g. 0.0")
            self._inputs["alpha_p"] = alpha_p_input
            yield alpha_p_input

            thinking_checkbox = Checkbox("Enable Thinking", id="chat-gen-config-enable-thinking")
            self._thinking_checkbox = thinking_checkbox
            yield thinking_checkbox

            status = Label("", id="chat-gen-config-status")
            self._status = status
            yield status

            with Container(id="chat-gen-config-buttons"):
                yield Button("Cancel", id="chat-gen-config-cancel")
                yield Button("Save", id="chat-gen-config-save", variant="primary")

    def on_mount(self) -> None:
        for key, widget in self._inputs.items():
            widget.value = str(self._defaults[key])
        if self._thinking_checkbox is not None:
            self._thinking_checkbox.value = self._enable_thinking

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat-gen-config-cancel":
            self.dismiss(None)
            return
        if event.button.id != "chat-gen-config-save":
            return

        try:
            max_new_tokens = int(self._inputs["max_new_tokens"].value.strip())
            temperature = float(self._inputs["temperature"].value.strip())
            top_k = int(self._inputs["top_k"].value.strip())
            top_p = float(self._inputs["top_p"].value.strip())
            repetition_penalty = float(self._inputs["repetition_penalty"].value.strip())
            alpha_f = float(self._inputs["alpha_f"].value.strip())
            alpha_p = float(self._inputs["alpha_p"].value.strip())
        except ValueError:
            self._set_status("Invalid number format.")
            return

        if max_new_tokens < 1:
            self._set_status("max_new_tokens must be >= 1.")
            return
        if temperature < 0:
            self._set_status("temperature must be >= 0.")
            return
        if top_k < 0:
            self._set_status("top_k must be >= 0.")
            return
        if not (0.0 <= top_p <= 1.0):
            self._set_status("top_p must be between 0 and 1.")
            return
        if repetition_penalty <= 0.0:
            self._set_status("repetition_penalty must be > 0.")
            return
        if alpha_f < 0 or alpha_p < 0:
            self._set_status("alpha_f and alpha_p must be >= 0.")
            return

        self.dismiss(
            {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "alpha_f": alpha_f,
                "alpha_p": alpha_p,
                "enable_thinking": bool(self._thinking_checkbox.value) if self._thinking_checkbox is not None else False,
            }
        )

    def _set_status(self, message: str) -> None:
        if self._status is not None:
            self._status.update(message)


class ConfirmModal(ModalScreen[Optional[bool]]):
    """Generic confirmation modal."""

    def __init__(
        self,
        title: str,
        message: str,
        *,
        confirm_label: str = "OK",
        confirm_variant: str = "primary",
    ) -> None:
        super().__init__(id="chat-log-confirm-modal")
        self._title = title
        self._message = message
        self._confirm_label = confirm_label
        self._confirm_variant = confirm_variant

    def compose(self) -> ComposeResult:
        with Container(id="chat-log-confirm-modal-container"):
            yield Label(self._title, id="chat-log-confirm-title")
            yield Static(self._message, id="chat-log-confirm-message")
            with Container(id="chat-log-confirm-buttons"):
                yield Button("Cancel", id="chat-log-confirm-cancel")
                yield Button(self._confirm_label, id="chat-log-confirm-accept", variant=self._confirm_variant)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat-log-confirm-cancel":
            self.dismiss(False)
        elif event.button.id == "chat-log-confirm-accept":
            self.dismiss(True)
