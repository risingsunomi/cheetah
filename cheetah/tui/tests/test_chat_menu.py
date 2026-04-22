from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from cheetah.tui.chat_menu import ChatScreen


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
    def test_response_reserve_tokens_defaults_to_400(self) -> None:
        screen = ChatScreen(peer_client=object())

        self.assertEqual(screen._response_reserve_tokens(512), 400)

    def test_response_reserve_tokens_uses_model_generation_limit(self) -> None:
        screen = ChatScreen(peer_client=object())
        screen._model_config = {"max_new_tokens": 640, "max_seq_len": 2048}

        self.assertEqual(screen._response_reserve_tokens(1024), 640)

    def test_response_reserve_tokens_falls_back_on_invalid_model_config(self) -> None:
        screen = ChatScreen(peer_client=object())
        screen._model_config = {"max_new_tokens": "not-an-int", "max_seq_len": 2048}

        self.assertEqual(screen._response_reserve_tokens(512), 400)

    def test_effective_gen_config_exposes_max_new_tokens(self) -> None:
        screen = ChatScreen(peer_client=object())
        screen._model_config = {"max_new_tokens": 320, "max_seq_len": 2048}

        self.assertEqual(screen._effective_gen_config()["max_new_tokens"], 320)

    def test_response_reserve_tokens_prefers_chat_gen_override(self) -> None:
        screen = ChatScreen(peer_client=object())
        screen._model_config = {"max_new_tokens": 192, "max_seq_len": 2048}
        screen._gen_overrides["max_new_tokens"] = 384

        self.assertEqual(screen._response_reserve_tokens(512), 384)

    def test_response_reserve_tokens_clamps_to_context_window(self) -> None:
        screen = ChatScreen(peer_client=object())
        screen._gen_overrides["max_new_tokens"] = 2048

        self.assertEqual(screen._response_reserve_tokens(128), 127)

    def test_model_change_resets_max_new_tokens_override(self) -> None:
        screen = ChatScreen(peer_client=object())
        screen._model_id = "old-model"
        screen._gen_overrides["max_new_tokens"] = 768
        screen._gen_overrides["temperature"] = 0.2

        with patch("asyncio.create_task", side_effect=lambda coro: coro.close()):
            screen._handle_model_selected("new-model")

        self.assertNotIn("max_new_tokens", screen._gen_overrides)
        self.assertEqual(screen._gen_overrides["temperature"], 0.2)


class TestChatScreenModelLoad(unittest.IsolatedAsyncioTestCase):
    async def test_start_model_load_resets_max_new_tokens_override_and_logs_effective_config(self) -> None:
        screen = ChatScreen(peer_client=object())
        screen._gen_overrides["max_new_tokens"] = 128
        screen._gen_overrides["temperature"] = 0.3
        screen._gen_overrides["enable_thinking"] = False
        screen._set_load_button_enabled = lambda enabled: None
        log_messages: list[str] = []
        screen._log_sys_msg = lambda message, **kwargs: log_messages.append(message)

        async def fake_load_model(*, shard=None):
            self.assertIsNone(shard)
            return object(), {"max_seq_len": 640}, object(), "/tmp/model", 0.5

        screen._load_model = fake_load_model

        with (
            patch(
                "cheetah.tui.chat_menu.resolve_model_assets_for_backend",
                new=AsyncMock(return_value=({"max_seq_len": 640}, Path("/tmp/model"))),
            ),
            patch("cheetah.tui.chat_menu.detect_quantization_mode", return_value=(False, "standard")),
        ):
            await screen._start_model_load()

        self.assertNotIn("max_new_tokens", screen._gen_overrides)
        self.assertEqual(screen._gen_overrides["temperature"], 0.3)
        self.assertEqual(screen._effective_gen_config()["max_new_tokens"], 400)
        self.assertIn("Model ready in 0.5s. Backend: tinygrad. Mode: standard.", log_messages)
        self.assertIn("Gen config updated: context_window=640, max_new_tokens=400, temp=0.3", log_messages[1])

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
        screen = ChatScreen(peer_client=peer_client)
        screen._model_id = "demo/model"
        screen._set_load_button_enabled = lambda enabled: None
        log_messages: list[str] = []
        screen._log_sys_msg = lambda message, **kwargs: log_messages.append(message)

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
                "cheetah.tui.chat_menu.resolve_model_assets_for_backend",
                new=AsyncMock(return_value=({"num_layers": 8, "max_seq_len": 640}, Path("/tmp/model"))),
            ),
            patch("cheetah.tui.chat_menu.detect_quantization_mode", return_value=(False, "standard")),
            patch(
                "cheetah.tui.chat_menu.load_model_shards_on_peers",
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
