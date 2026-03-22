from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cheetah.agent.prompt_loader import (
    list_agent_prompt_names,
    load_agent_prompt_template,
    normalize_prompt_name,
    render_agent_system_prompt,
    save_agent_prompt_template,
)


class TestPromptLoader(unittest.TestCase):
    def test_render_agent_system_prompt_uses_template(self) -> None:
        rendered = render_agent_system_prompt(
            name="cot-agent",
            endless_mode=False,
            enabled_functions=["write_file", "end_run"],
            function_format="tools",
            tool_summary="- write_file(path*, content*): Write text\n- end_run(summary?): Stop the run",
            agent_prompt="Create a file and stop.",
        )

        self.assertIn("You are 'cot-agent'", rendered)
        self.assertIn("Respond in this JSON format", rendered)
        self.assertIn('"thoughts"', rendered)
        self.assertIn('"ability"', rendered)
        self.assertIn("write_file, end_run", rendered)
        self.assertIn("Ability signatures:", rendered)
        self.assertIn("prefer `web_search` instead of creating a script", rendered)
        self.assertNotIn("Function schema payload", rendered)

    def test_save_and_load_agent_prompt_template_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_name = save_agent_prompt_template(
                "custom_prompt",
                "You are '{{ name }}'.",
                prompt_dir=tmpdir,
            )

            self.assertEqual(saved_name, "custom_prompt.j2")
            self.assertEqual(
                load_agent_prompt_template(saved_name, prompt_dir=tmpdir),
                "You are '{{ name }}'.",
            )
            self.assertEqual(list_agent_prompt_names(prompt_dir=tmpdir), ["custom_prompt.j2"])

    def test_render_agent_system_prompt_supports_custom_prompt_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "custom_prompt.j2").write_text(
                "Agent '{{ name }}' says {{ agent_prompt }}",
                encoding="utf-8",
            )

            rendered = render_agent_system_prompt(
                prompt_name="custom_prompt.j2",
                prompt_dir=tmpdir,
                name="cot-agent",
                endless_mode=False,
                enabled_functions=[],
                function_format="tools",
                tool_summary="- none",
                agent_prompt="hello",
            )

            self.assertEqual(rendered, "Agent 'cot-agent' says hello")

    def test_normalize_prompt_name_adds_extension(self) -> None:
        self.assertEqual(normalize_prompt_name("custom_prompt"), "custom_prompt.j2")


if __name__ == "__main__":
    unittest.main()
