from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tinygrad as tg
import numpy as np

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
        self.eos_token_id = 1

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


class _StreamingDummyModel:
    def __init__(self) -> None:
        self.prefill_calls = 0
        self.decode_calls = 0
        self.reset_calls = 0
        self.start_positions: list[int] = []

    def reset_kv_cache(self) -> None:
        self.reset_calls += 1

    def __call__(
        self,
        x: tg.Tensor,
        attention_mask: tg.Tensor | None = None,
        position_ids: tg.Tensor | None = None,
        hidden_state: tg.Tensor | None = None,
    ) -> tg.Tensor:
        del hidden_state
        if attention_mask is None or position_ids is None:
            raise AssertionError("prefill should include attention_mask and position_ids")
        self.prefill_calls += 1
        batch_size, seq_len = x.shape
        logits = tg.Tensor.zeros((batch_size, seq_len, 4), device=x.device)
        logits = logits + tg.Tensor([[[0.0, 0.0, 1.0, 0.0]]], device=x.device)
        return logits

    def decode_token(
        self,
        x: tg.Tensor,
        position_ids: tg.Tensor | None = None,
        *,
        start_pos: int | tg.UOp | None = None,
    ) -> tg.Tensor:
        del position_ids
        if isinstance(start_pos, tg.UOp):
            raise AssertionError("streaming_generate should pass a concrete start_pos to the model wrapper")
        self.decode_calls += 1
        self.start_positions.append(int(start_pos))
        batch_size, seq_len = x.shape
        logits = tg.Tensor.zeros((batch_size, seq_len, 4), device=x.device)
        logits = logits + tg.Tensor([[[0.0, 1.0, 0.0, 0.0]]], device=x.device)
        return logits


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


class TestStreamingGenerateTinygrad(unittest.TestCase):
    def test_streaming_generate_uses_decode_token_fast_path(self) -> None:
        model = _StreamingDummyModel()
        tokenizer = _DummyTokenizer(chat_template="")

        out, elapsed = helpers.streaming_generate(
            model,
            input_ids=tg.Tensor([[10, 11]]),
            attention_mask=tg.Tensor([[1, 1]]),
            tokenizer=tokenizer,
            max_new_tokens=4,
            temp=0.0,
            top_k=0,
            top_p=1.0,
        )

        self.assertEqual(out, [2, 1])
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(model.reset_calls, 1)
        self.assertEqual(model.prefill_calls, 1)
        self.assertEqual(model.decode_calls, 1)
        self.assertEqual(model.start_positions, [2])

    def test_streaming_generate_with_peers_uses_prefill_timeout_for_peer_rpc(self) -> None:
        class _FakeEngine:
            def __init__(self) -> None:
                self.shard = None

            def get_tokens(self, *args, **kwargs):
                del args, kwargs
                return {
                    "hidden_state": {
                        "buffer": "zczMPQ==",
                        "shape": [1, 1, 1],
                        "dtype": "float32",
                    },
                    "attention_mask": {
                        "buffer": "AQAAAAEAAAA=",
                        "shape": [1, 2],
                        "dtype": "int32",
                    },
                    "position_ids": {
                        "buffer": "AAAAAAAAAAABAAAA",
                        "shape": [1, 2],
                        "dtype": "int32",
                    },
                    "end_token": False,
                }

            def recv_tokens(self, payload, tokenizer=None, backend=None):
                del tokenizer, backend
                normalized = dict(payload)
                if isinstance(payload.get("hidden_state"), dict):
                    normalized["hidden_state"] = np.array([[[0.1]]], dtype=np.float32).tolist()
                if isinstance(payload.get("attention_mask"), dict):
                    normalized["attention_mask"] = [[1, 1]]
                if isinstance(payload.get("position_ids"), dict):
                    normalized["position_ids"] = [[0, 1]]
                return normalized

        class _PeerClientStub:
            def __init__(self) -> None:
                self.peer_client_id = "self"
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
                self.peer_device = self_peer
                self._peers = [self_peer, remote_peer]
                self.calls: list[dict[str, object]] = []

            def get_peers(self, include_self: bool = False):
                return list(self._peers if include_self else self._peers[1:])

            def send_payload(self, message, *, expect_reply=True, address=None, timeout=None):
                self.calls.append(
                    {
                        "message": message,
                        "expect_reply": expect_reply,
                        "address": address,
                        "timeout": timeout,
                    }
                )
                return {
                    "token": 1,
                    "attention_mask": [[1, 1]],
                    "position_ids": [[0, 1]],
                    "end_token": True,
                }

        peer_client = _PeerClientStub()
        model = SimpleNamespace(config={"num_layers": 8}, shard=None)
        tokenizer = _DummyTokenizer(chat_template="")

        with patch.object(helpers, "_local_engine", return_value=_FakeEngine()):
            with patch.dict(os.environ, {"TC_PEER_PREFILL_TIMEOUT_SECONDS": "45"}, clear=False):
                out, _ = helpers.streaming_generate_with_peers(
                    peer_client,
                    model,
                    input_ids=tg.Tensor([[10, 11]]),
                    attention_mask=tg.Tensor([[1, 1]]),
                    tokenizer=tokenizer,
                    max_new_tokens=1,
                    temp=0.0,
                    top_k=0,
                    top_p=1.0,
                )

        self.assertEqual(out, [1])
        self.assertEqual(len(peer_client.calls), 1)
        self.assertEqual(peer_client.calls[0]["timeout"], 45.0)
        sent_payload = peer_client.calls[0]["message"]["payload"]
        self.assertIsInstance(sent_payload["hidden_state"], dict)
        self.assertEqual(sent_payload["hidden_state"]["dtype"], "float32")
        self.assertIsInstance(sent_payload["attention_mask"], dict)


class TestDistributedShardLogging(unittest.TestCase):
    def test_distributed_shard_plan_messages_list_local_then_remote_peers(self) -> None:
        peers = [
            SimpleNamespace(peer_client_id="self", ip_address="192.168.0.10", gpu_vram="8", cpu_ram="16", gpu_flops=0.0),
            SimpleNamespace(peer_client_id="peer-1", ip_address="192.168.0.20", gpu_vram="4", cpu_ram="8", gpu_flops=0.0),
            SimpleNamespace(peer_client_id="peer-2", ip_address="192.168.0.30", gpu_vram="2", cpu_ram="4", gpu_flops=0.0),
        ]

        lines = helpers.distributed_shard_plan_messages(
            peers,
            local_peer_id="self",
            model_name="demo",
            total_layers=12,
        )

        self.assertEqual(lines[0], "Using 3 nodes for shard-aware execution.")
        self.assertIn("Loading local shard self (192.168.0.10):", lines[1])
        self.assertIn("Loading shard on peer peer-1 (192.168.0.20):", lines[2])
        self.assertIn("Loading shard on peer peer-2 (192.168.0.30):", lines[3])

    def test_build_peer_load_plan_returns_local_and_remote_shards(self) -> None:
        self_peer = SimpleNamespace(
            peer_client_id="self",
            ip_address="192.168.0.10",
            gpu_vram="8",
            cpu_ram="8",
            gpu_flops=0.0,
        )
        remote_peer = SimpleNamespace(
            peer_client_id="peer-1",
            ip_address="192.168.0.20",
            gpu_vram="8",
            cpu_ram="8",
            gpu_flops=0.0,
        )
        peer_client = SimpleNamespace(
            peer_client_id="self",
            get_peers=lambda include_self=True: [self_peer, remote_peer] if include_self else [remote_peer],
        )

        plan = helpers.build_peer_load_plan(
            peer_client,
            model_name="demo",
            total_layers=9,
        )

        self.assertTrue(plan["distributed"])
        self.assertEqual(len(plan["peers"]), 2)
        self.assertEqual(len(plan["remote_peers"]), 1)
        self.assertEqual(plan["local_shard"].start_layer, 0)
        self.assertEqual(plan["local_shard"].end_layer, 4)
        self.assertEqual(plan["remote_peers"][0].shard.start_layer, 4)
        self.assertEqual(plan["remote_peers"][0].shard.end_layer, 8)

    def test_validate_peer_runtime_fingerprints_detects_mismatch(self) -> None:
        remote_peer = SimpleNamespace(peer_client_id="peer-1", ip_address="192.168.0.20")

        with patch.object(
            helpers,
            "local_runtime_fingerprints",
            return_value={
                "config_fingerprint": "local-config",
                "tokenizer_fingerprint": "local-tokenizer",
            },
        ):
            mismatches = helpers.validate_peer_runtime_fingerprints(
                [
                    {
                        "peer": remote_peer,
                        "response": {
                            "fingerprint_protocol": 2,
                            "config_fingerprint": "remote-config",
                            "tokenizer_fingerprint": "remote-tokenizer",
                        },
                    }
                ],
                local_model_config={"num_layers": 8},
                local_model_path="/tmp/model",
            )

        self.assertIn("peer-1 (192.168.0.20) model config fingerprint mismatch", mismatches)
        self.assertIn("peer-1 (192.168.0.20) tokenizer fingerprint mismatch", mismatches)

    def test_validate_peer_runtime_fingerprints_flags_older_peer_protocol(self) -> None:
        remote_peer = SimpleNamespace(peer_client_id="peer-1", ip_address="192.168.0.20")

        with patch.object(
            helpers,
            "local_runtime_fingerprints",
            return_value={
                "config_fingerprint": "local-config",
                "tokenizer_fingerprint": "local-tokenizer",
            },
        ):
            mismatches = helpers.validate_peer_runtime_fingerprints(
                [
                    {
                        "peer": remote_peer,
                        "response": {
                            "config_fingerprint": "legacy-config",
                            "tokenizer_fingerprint": "legacy-tokenizer",
                        },
                    }
                ],
                local_model_config={"num_layers": 8},
                local_model_path="/tmp/model",
            )

        self.assertEqual(
            mismatches,
            ["peer-1 (192.168.0.20) is running an older peer fingerprint protocol; update and restart that node"],
        )


if __name__ == "__main__":
    unittest.main()
