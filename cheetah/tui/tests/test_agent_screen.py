import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
