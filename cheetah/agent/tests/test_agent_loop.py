from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cheetah.tui.agent_screen import AgentScreen
from cheetah.orchestration.distributed_inference import MemoryPressureError


class TestAgentLoop(unittest.IsolatedAsyncioTestCase):
    async def test_agent_loop_executes_functions_until_end_run(self) -> None:
        screen = AgentScreen(peer_client=object())

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "notes" / "summary.txt"
            replies = iter(
                [
                    json.dumps(
                        {
                            "thoughts": {
                                "text": "Need to write the requested file",
                                "reasoning": "The task is a file creation task",
                                "criticism": "Need to stop after writing",
                                "step completed": "Chose the next action",
                                "plan": "- write file\n- end run",
                                "speak": "Writing the file now",
                            },
                            "ability": {
                                "name": "write_file",
                                "args": {
                                    "path": str(target_path),
                                    "content": "Washington, D.C. is the capital of the USA.\n",
                                    "make_dirs": True,
                                },
                            }
                        }
                    ),
                    json.dumps(
                        {
                            "thoughts": {
                                "text": "The file has been written",
                                "reasoning": "The task is complete",
                                "criticism": "None",
                                "step completed": "Finished the requested file write",
                                "plan": "- end run",
                                "speak": "Done",
                            },
                            "ability": {
                                "name": "end_run",
                                "args": {"summary": "File written."},
                            }
                        }
                    ),
                ]
            )

            def _fake_generate_agent_reply(messages: list[dict[str, str]]) -> str:
                self.assertTrue(messages)
                return next(replies)

            screen._agent_running = True
            screen._agent_messages = screen._build_initial_messages(
                name="cot-agent",
                instructions="Write the capital of the USA to a file and stop.",
            )

            with patch.object(screen, "_generate_agent_reply", side_effect=_fake_generate_agent_reply):
                with patch.object(screen, "_memory_abort_reason", return_value=None):
                    final_reply = await screen._agent_loop()

            self.assertEqual(final_reply, screen._last_agent_response)
            self.assertFalse(screen._agent_running)
            self.assertTrue(target_path.exists())
            self.assertEqual(target_path.read_text(encoding="utf-8"), "Washington, D.C. is the capital of the USA.\n")
            self.assertIn('"name": "end_run"', final_reply)
            self.assertTrue(
                any(
                    message["role"] == "user" and message["content"].startswith("Function result: write_file ok")
                    for message in screen._agent_messages
                )
            )
            self.assertTrue(
                any(
                    message["role"] == "assistant"
                    and json.loads(message["content"])["ability"]["name"] == "write_file"
                    for message in screen._agent_messages
                )
            )

    async def test_agent_loop_recovers_from_memory_pressure_by_compacting_messages(self) -> None:
        screen = AgentScreen(peer_client=object())
        screen._agent_running = True
        screen._agent_messages = screen._build_initial_messages(
            name="cot-agent",
            instructions="Keep working even if memory gets tight.",
        )
        for idx in range(8):
            role = "assistant" if idx % 2 else "user"
            screen._agent_messages.append(
                {
                    "role": role,
                    "content": f"long message {idx} " * 20,
                }
            )

        original_len = len(screen._agent_messages)
        call_count = 0

        def _fake_generate_agent_reply(messages: list[dict[str, str]]) -> str:
            nonlocal call_count
            self.assertTrue(messages)
            call_count += 1
            if call_count == 1:
                raise MemoryPressureError("Memory guard triggered (agent loop)")
            return json.dumps(
                {
                    "thoughts": {
                        "text": "Recovered and ready to finish",
                        "reasoning": "No more actions are needed",
                        "criticism": "None",
                        "step completed": "Recovered from memory pressure",
                        "plan": "- end run",
                        "speak": "Done",
                    },
                    "ability": {
                        "name": "end_run",
                        "args": {"summary": "Recovered and finished."},
                    }
                }
            )

        with patch.object(screen, "_generate_agent_reply", side_effect=_fake_generate_agent_reply):
            with patch.object(screen, "_memory_abort_reason", return_value=None):
                with patch("cheetah.tui.agent_screen.relieve_memory_pressure") as relieve_mock:
                    final_reply = await screen._agent_loop()

        self.assertEqual(call_count, 2)
        self.assertEqual(relieve_mock.call_count, 1)
        self.assertFalse(screen._agent_running)
        self.assertIn('"name": "end_run"', final_reply)
        self.assertLess(len(screen._agent_messages), original_len + 1)
        self.assertTrue(
            any(
                "Earlier turns were compacted due to memory pressure" in message["content"]
                for message in screen._agent_messages
                if message["role"] == "user"
            )
        )


if __name__ == "__main__":
    unittest.main()
