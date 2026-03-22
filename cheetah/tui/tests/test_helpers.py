from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cheetah.tui import helpers


def _gib(value: float) -> int:
    return int(value * (1024 ** 3))


class _FakePsutil:
    def __init__(self, *, ram_percent: float, available_gb: float, swap_percent: float) -> None:
        self._virtual_memory = SimpleNamespace(percent=ram_percent, available=_gib(available_gb))
        self._swap_memory = SimpleNamespace(percent=swap_percent)

    def virtual_memory(self):
        return self._virtual_memory

    def swap_memory(self):
        return self._swap_memory


class _DummyTokenizer:
    def __init__(self, *, chat_template: str = "") -> None:
        self.chat_template = chat_template
        self.calls: list[dict[str, object]] = []
        self.all_special_tokens = ["<s>", "</s>", "<think>", "</think>"]

    def apply_chat_template(self, messages, *, add_generation_prompt: bool, tokenize: bool, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "kwargs": kwargs,
            }
        )
        return "template"


class TestMemoryAbortReason(unittest.TestCase):
    def test_swap_alone_does_not_abort_when_ram_and_available_are_healthy(self) -> None:
        fake_psutil = _FakePsutil(ram_percent=79.0, available_gb=5.03, swap_percent=93.3)
        with patch.dict(
            os.environ,
            {
                "TC_MEM_MAX_PERCENT": "92",
                "TC_MEM_MIN_AVAILABLE_GB": "0.75",
            },
            clear=False,
        ):
            with patch.object(helpers, "psutil", fake_psutil):
                self.assertIsNone(helpers.memory_abort_reason("agent loop"))

    def test_low_available_ram_aborts(self) -> None:
        fake_psutil = _FakePsutil(ram_percent=79.0, available_gb=0.5, swap_percent=93.3)
        with patch.dict(
            os.environ,
            {
                "TC_MEM_MAX_PERCENT": "92",
                "TC_MEM_MIN_AVAILABLE_GB": "0.75",
            },
            clear=False,
        ):
            with patch.object(helpers, "psutil", fake_psutil):
                reason = helpers.memory_abort_reason("agent loop")
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("Available RAM 0.50 GiB <= 0.75 GiB", reason)
        self.assertNotIn("Swap usage", reason)

    def test_high_ram_percent_aborts(self) -> None:
        fake_psutil = _FakePsutil(ram_percent=94.0, available_gb=5.03, swap_percent=99.2)
        with patch.dict(
            os.environ,
            {
                "TC_MEM_MAX_PERCENT": "92",
                "TC_MEM_MIN_AVAILABLE_GB": "0.75",
            },
            clear=False,
        ):
            with patch.object(helpers, "psutil", fake_psutil):
                reason = helpers.memory_abort_reason("agent loop")
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("RAM usage 94.0% >= 92.0%", reason)
        self.assertNotIn("Swap usage", reason)


class TestThinkingHelpers(unittest.TestCase):
    def test_default_enable_thinking_detects_think_tokens_in_tokenizer_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "tokenizer_config.json"
            config_path.write_text(
                """
                {
                  "added_tokens_decoder": {
                    "1": {"content": "<think>"},
                    "2": {"content": "</think>"}
                  }
                }
                """,
                encoding="utf-8",
            )
            self.assertTrue(helpers.default_enable_thinking(model_path=tmpdir))

    def test_apply_chat_template_with_thinking_passes_enable_thinking_for_supported_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "tokenizer_config.json"
            config_path.write_text(
                """
                {
                  "added_tokens_decoder": {
                    "1": {"content": "<think>"},
                    "2": {"content": "</think>"}
                  }
                }
                """,
                encoding="utf-8",
            )
            tokenizer = _DummyTokenizer(chat_template="")

            rendered = helpers.apply_chat_template_with_thinking(
                tokenizer,
                [{"role": "user", "content": "hi"}],
                add_generation_prompt=True,
                tokenize=False,
                model_path=tmpdir,
                enable_thinking=True,
            )

            self.assertEqual(rendered, "template")
            self.assertEqual(tokenizer.calls[0]["kwargs"], {"enable_thinking": True})

    def test_apply_chat_template_with_thinking_skips_flag_for_unsupported_models(self) -> None:
        tokenizer = _DummyTokenizer(chat_template="")

        rendered = helpers.apply_chat_template_with_thinking(
            tokenizer,
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
            tokenize=False,
            model_path=None,
            enable_thinking=True,
        )

        self.assertEqual(rendered, "template")
        self.assertEqual(tokenizer.calls[0]["kwargs"], {})

    def test_split_thinking_response_separates_thinking_from_final_reply(self) -> None:
        tokenizer = _DummyTokenizer(chat_template="")

        thinking, final = helpers.split_thinking_response(
            "<s><think>reasoning step</think>The final answer</s>",
            tokenizer=tokenizer,
        )

        self.assertEqual(thinking, "reasoning step")
        self.assertEqual(final, "The final answer")

    def test_split_thinking_response_keeps_plain_reply_when_no_thinking_tags_exist(self) -> None:
        tokenizer = _DummyTokenizer(chat_template="")

        thinking, final = helpers.split_thinking_response(
            "<s>Just the answer</s>",
            tokenizer=tokenizer,
        )

        self.assertEqual(thinking, "")
        self.assertEqual(final, "Just the answer")


if __name__ == "__main__":
    unittest.main()
