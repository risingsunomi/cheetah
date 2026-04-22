import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from textual.app import App
from textual.widgets import Button, Checkbox, Input, Select, Static, TextArea

from cheetah.tui.agent_screen import AgentScreen


class _AgentScreenHost(App[None]):
    def compose(self):
        yield Static("host")


class TestAgentScreen(unittest.IsolatedAsyncioTestCase):
    async def test_agent_screen_mounts_without_query_error(self) -> None:
        app = _AgentScreenHost()

        async with app.run_test() as pilot:
            app.push_screen(AgentScreen(peer_client=object()))
            await pilot.pause()

            screen = app.screen
            self.assertIsInstance(screen, AgentScreen)
            self.assertIsNotNone(screen._agent_log)
            self.assertIsNotNone(screen._config_button)
            self.assertIsNotNone(screen._functions_summary)
            self.assertEqual(
                set(screen._gen_inputs),
                {"temperature", "top_k", "top_p", "repetition_penalty", "alpha_f", "alpha_p"},
            )
            self.assertIsNotNone(screen.query_one("#agent-gen-enable-thinking", Checkbox))
            self.assertIsNotNone(screen.query_one("#agent-gen-endless-mode", Checkbox))
            self.assertIsNotNone(screen.query_one("#agent-instructions", TextArea))
            self.assertIsNotNone(screen.query_one("#agent-open-model-picker", Button))
            self.assertEqual(len(list(screen.query("#agent-clear-model"))), 0)
            self.assertEqual(len(list(screen.query("#agent-open-gen-config"))), 0)

    async def test_agent_config_modal_exposes_prompt_selection_and_editor(self) -> None:
        app = _AgentScreenHost()

        async with app.run_test() as pilot:
            screen = AgentScreen(peer_client=object())
            app.push_screen(screen)
            await pilot.pause()
            screen._open_agent_config()
            await pilot.pause()

            modal = app.screen
            self.assertIsNotNone(modal.query_one("#agent-config-prompt-select", Select))
            self.assertIsNotNone(modal.query_one("#agent-config-prompt-editor", TextArea))
            self.assertIsNotNone(modal.query_one("#agent-config-save-prompt", Button))
            self.assertIsNotNone(modal.query_one("#agent-config-save-prompt-as", Button))
            self.assertEqual(len(list(modal.query("#agent-config-cancel"))), 0)
            self.assertEqual(len(list(modal.query("#agent-config-save"))), 0)
            self.assertEqual(len(list(modal.query("#agent-config-endless-mode"))), 0)
            self.assertEqual(len(list(modal.query("#agent-config-instructions"))), 0)

    async def test_agent_config_modal_closes_on_escape(self) -> None:
        app = _AgentScreenHost()

        async with app.run_test() as pilot:
            screen = AgentScreen(peer_client=object())
            app.push_screen(screen)
            await pilot.pause()
            screen._open_agent_config()
            await pilot.pause()

            modal = app.screen
            name_input = modal.query_one("#agent-config-name", Input)
            name_input.value = "renamed-agent"
            await pilot.press("escape")
            await pilot.pause()

            self.assertIs(app.screen, screen)
            self.assertEqual(screen._agent_name, "renamed-agent")

    async def test_main_screen_instructions_editor_updates_agent_instructions(self) -> None:
        app = _AgentScreenHost()

        async with app.run_test() as pilot:
            screen = AgentScreen(peer_client=object())
            screen._agent_instructions = "original instructions"
            app.push_screen(screen)
            await pilot.pause()

            instructions = screen.query_one("#agent-instructions", TextArea)
            instructions.load_text("changed instructions")
            await pilot.pause()

            self.assertEqual(screen._agent_instructions, "changed instructions")

    async def test_agent_config_save_as_opens_prompt_name_modal(self) -> None:
        app = _AgentScreenHost()

        async with app.run_test() as pilot:
            screen = AgentScreen(peer_client=object())
            app.push_screen(screen)
            await pilot.pause()
            screen._open_agent_config()
            await pilot.pause()

            modal = app.screen
            save_as = modal.query_one("#agent-config-save-prompt-as", Button)
            save_as.press()
            await pilot.pause()

            save_as_modal = app.screen
            self.assertIsNotNone(save_as_modal.query_one("#agent-prompt-name-field", Input))


class TestAgentScreenPrompt(unittest.TestCase):
    def test_handle_model_selected_clears_loaded_model_on_switch(self) -> None:
        screen = AgentScreen(peer_client=object())
        screen._model_id = "model-a"
        screen._model = object()
        screen._tokenizer = object()
        calls: list[bool] = []

        def _track_clear(*, update_log: bool = True) -> None:
            calls.append(update_log)

        screen._clear_model = _track_clear  # type: ignore[method-assign]

        screen._handle_model_selected("model-b")

        self.assertEqual(screen._model_id, "model-b")
        self.assertEqual(calls, [False])

    def test_build_initial_messages_uses_structured_agent_loop_prompt(self) -> None:
        screen = AgentScreen(peer_client=object())

        messages = screen._build_initial_messages(
            name="cot-agent",
            instructions="Create a file and stop when done.",
        )

        self.assertEqual(messages[0]["role"], "system")
        system_prompt = messages[0]["content"]
        self.assertIn("Respond in this JSON format", system_prompt)
        self.assertIn('"thoughts"', system_prompt)
        self.assertIn('"ability"', system_prompt)
        self.assertIn('"step completed"', system_prompt)
        self.assertIn("set `ability.name` to `end_run`", system_prompt)
        self.assertIn("Ability signatures:", system_prompt)
        self.assertNotIn("Function schema payload", system_prompt)

    def test_build_initial_messages_uses_selected_prompt_name(self) -> None:
        screen = AgentScreen(peer_client=object())
        screen._agent_prompt_name = "custom_prompt.j2"

        with patch(
            "cheetah.tui.agent_screen.render_agent_system_prompt",
            return_value="custom system prompt",
        ) as render_prompt:
            messages = screen._build_initial_messages(
                name="cot-agent",
                instructions="Create a file and stop when done.",
            )

        self.assertEqual(messages, [{"role": "system", "content": "custom system prompt"}])
        self.assertEqual(render_prompt.call_args.kwargs["prompt_name"], "custom_prompt.j2")

    def test_extract_function_call_reads_ability_format(self) -> None:
        screen = AgentScreen(peer_client=object())

        result = screen._extract_function_call(
            """
            {
              "thoughts": {
                "text": "Need to create the file",
                "reasoning": "The task asks for a file write",
                "criticism": "None",
                "step completed": "Selected the correct action",
                "plan": "- write file\\n- end run",
                "speak": "Writing the file now"
              },
              "ability": {
                "name": "write_file",
                "args": {
                  "path": "notes.txt",
                  "content": "hello"
                }
              }
            }
            """
        )

        self.assertEqual(
            result,
            ("write_file", {"path": "notes.txt", "content": "hello"}),
        )

    def test_compact_agent_reply_for_memory_drops_verbose_thoughts(self) -> None:
        screen = AgentScreen(peer_client=object())

        compact = screen._compact_agent_reply_for_memory(
            {
                "thoughts": {
                    "text": "Long hidden thought",
                    "reasoning": "Long reasoning",
                    "criticism": "Long criticism",
                    "step completed": "Prepared the write action",
                    "plan": "- one\n- two",
                    "speak": "Writing the file now",
                },
                "ability": {
                    "name": "write_file",
                    "args": {"path": "notes.txt", "content": "hello"},
                },
            },
            function_name="write_file",
            arguments={"path": "notes.txt", "content": "hello"},
        )

        self.assertIsNotNone(compact)
        assert compact is not None
        self.assertEqual(compact["role"], "assistant")
        self.assertIn("ability=write_file", compact["content"])
        self.assertIn("speak=Writing the file now", compact["content"])
        self.assertNotIn("reasoning", compact["content"])

    def test_apply_agent_config_updates_prompt_name(self) -> None:
        screen = AgentScreen(peer_client=object())

        screen._apply_agent_config(
            {
                "name": "cot-agent",
                "prompt_name": "custom_prompt.j2",
            }
        )

        self.assertEqual(screen._agent_prompt_name, "custom_prompt.j2")


class TestAgentScreenModelLoad(unittest.IsolatedAsyncioTestCase):
    async def test_start_model_load_uses_local_shard_and_waits_for_peers(self) -> None:
        self_peer = SimpleNamespace(
            peer_client_id="self",
            ip_address="192.168.0.10",
            port=8765,
            gpu_vram="8",
            cpu_ram="8",
            gpu_flops=0.0,
        )
        remote_peer = SimpleNamespace(
            peer_client_id="peer-1",
            ip_address="192.168.0.20",
            port=8765,
            gpu_vram="8",
            cpu_ram="8",
            gpu_flops=0.0,
        )
        runtime_calls: list[dict[str, object]] = []
        peer_client = SimpleNamespace(
            peer_client_id="self",
            get_peers=lambda include_self=True: [self_peer, remote_peer] if include_self else [remote_peer],
            register_generation_runtime=lambda **kwargs: runtime_calls.append(kwargs),
            clear_generation_runtime=lambda **kwargs: None,
        )
        screen = AgentScreen(peer_client=peer_client)
        screen._model_id = "demo/model"
        screen._set_load_button_enabled = lambda enabled: None
        screen._refresh_backend_label = lambda: None
        screen._refresh_state_label = lambda: None
        log_messages: list[str] = []
        screen._log = lambda message: log_messages.append(message)

        captured: dict[str, object] = {}

        async def fake_load_model(*, shard=None):
            captured["shard"] = shard
            return (
                SimpleNamespace(shard=shard),
                {"num_layers": 8, "max_seq_len": 640},
                object(),
                Path("/tmp/model"),
                0.5,
            )

        screen._load_model = fake_load_model

        with (
            patch(
                "cheetah.tui.agent_screen.resolve_model_assets_for_backend",
                new=AsyncMock(return_value=({"num_layers": 8, "max_seq_len": 640}, Path("/tmp/model"))),
            ),
            patch("cheetah.tui.agent_screen.detect_quantization_mode", return_value=(False, "standard")),
            patch(
                "cheetah.tui.agent_screen.load_model_shards_on_peers",
                return_value={
                    "remote_results": [
                        {
                            "peer": remote_peer,
                            "response": {"elapsed": 1.2, "already_loaded": False},
                        }
                    ]
                },
            ) as load_peers,
        ):
            await screen._start_model_load()

        shard = captured.get("shard")
        self.assertIsNotNone(shard)
        assert shard is not None
        self.assertEqual(shard.start_layer, 0)
        self.assertEqual(shard.end_layer, 4)
        self.assertTrue(screen._model_loaded)
        load_peers.assert_called_once()
        self.assertEqual(runtime_calls[0]["model_id"], "demo/model")
        self.assertEqual(runtime_calls[0]["shard"].start_layer, 0)
        self.assertIn("Waiting for 1 peer shard(s) to finish loading...", log_messages)
        self.assertIn("Shard-aware model ready on 2 nodes.", log_messages)


if __name__ == "__main__":
    unittest.main()
