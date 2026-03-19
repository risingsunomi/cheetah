import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tiny_cheetah.tui.chat_menu import ChatScreen


class _ThinkingTokenizer:
    all_special_tokens = ["<s>", "</s>", "<think>", "</think>"]


class TestChatScreenThinkingRender(unittest.TestCase):
    def test_build_model_log_markup_shows_thinking_before_reply(self) -> None:
        screen = ChatScreen(peer_client=object())
        screen._model_id = "Qwen/Test"
        screen._tokenizer = _ThinkingTokenizer()

        markup, thinking, final = screen._build_model_log_markup(
            "<s><think>brief reasoning</think>The final answer</s>",
            timestamp="2026-03-16 12:00:00",
            show_empty_placeholder=True,
        )

        self.assertEqual(thinking, "brief reasoning")
        self.assertEqual(final, "The final answer")
        self.assertIn("[italic](thinking)[/italic]", markup)
        self.assertIn("[italic]brief reasoning[/italic]", markup)
        self.assertIn(": The final answer", markup)
        self.assertLess(markup.index("[italic](thinking)[/italic]"), markup.index(": The final answer"))


class TestChatScreenGenerationLimits(unittest.TestCase):
    def test_response_reserve_tokens_accepts_string_env(self) -> None:
        screen = ChatScreen(peer_client=object())

        with patch.dict("os.environ", {"TC_MAX_RESP_LEN": "192"}, clear=False):
            self.assertEqual(screen._response_reserve_tokens(512), 192)

    def test_response_reserve_tokens_falls_back_on_invalid_env(self) -> None:
        screen = ChatScreen(peer_client=object())

        with patch.dict("os.environ", {"TC_MAX_RESP_LEN": "not-an-int"}, clear=False):
            self.assertEqual(screen._response_reserve_tokens(512), 192)


class TestChatScreenLogSelection(unittest.IsolatedAsyncioTestCase):
    async def test_load_selected_chat_log_uses_sync_handler(self) -> None:
        screen = ChatScreen(peer_client=object())
        calls: list[int] = []
        screen._get_selected_log_id = lambda: 18
        screen._handle_chat_log_load = lambda log_id: calls.append(log_id)

        await screen._load_selected_chat_log()

        self.assertEqual(calls, [18])

    async def test_list_view_selected_uses_sync_handler(self) -> None:
        screen = ChatScreen(peer_client=object())
        calls: list[int] = []
        stopped: list[bool] = []
        screen._handle_chat_log_load = lambda log_id: calls.append(log_id)

        event = SimpleNamespace(
            item=SimpleNamespace(id="log-18"),
            stop=lambda: stopped.append(True),
        )

        await screen.on_list_view_selected(event)

        self.assertEqual(calls, [18])
        self.assertEqual(stopped, [True])


if __name__ == "__main__":
    unittest.main()
