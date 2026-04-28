import unittest
from argparse import Namespace
from contextlib import redirect_stdout
import io
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from textual.widgets import Label

try:
    import tinygrad as tg
except Exception:
    tg = None

try:
    import torch
except Exception:
    torch = None

from cheetah.tui.train_menu import (
    Batch,
    TrainingCancelled,
    TrainingProcess,
    TrainScreen,
    _train_epoch,
    _build_training_namespace,
    _ensure_required_keys,
    _prepare_dataset_corpus,
    _stream_corpus_batches,
    default_training_settings,
)
from cheetah.orchestration.distributed_inference import distributed_shard_plan_messages
from cheetah.tui.training_path_screen import TrainingPathScreen
from cheetah.tui.training_path_types import TrainingNode


class _PeerClientStub:
    def __init__(self, peers):
        self._peers = list(peers)
        self.peer_client_id = "self"

    def get_peers(self, include_self: bool = False):
        return list(self._peers)

    def peer_count(self) -> int:
        return len(self._peers)


class _TokenizerStub:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [1, 2, 3, 4, 5] if text.strip() else []


if torch is not None:
    class _TorchTrainModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4))
            self.decode_calls = 0

        def forward(self, x, attention_mask=None, position_ids=None):
            del attention_mask, position_ids
            return self.weight.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1)

        def decode_token(self, *args, **kwargs):
            del args, kwargs
            self.decode_calls += 1
            raise AssertionError("torch training should not use decode_token")


if tg is not None:
    class _TinygradTrainModel:
        def __init__(self) -> None:
            self.weight = tg.Tensor.zeros(4, requires_grad=True)
            self.decode_calls = 0

        def __call__(self, x, attention_mask=None, position_ids=None):
            del attention_mask, position_ids
            return self.weight.reshape(1, 1, -1).expand((x.shape[0], x.shape[1], self.weight.shape[0]))

        def decode_token(self, *args, **kwargs):
            del args, kwargs
            self.decode_calls += 1
            raise AssertionError("tinygrad training should not use decode_token")


class TestTrainScreen(unittest.TestCase):
    def test_default_training_settings_reads_env(self) -> None:
        env = {
            "TC_TRAINING_MODEL_ID": "unsloth/Llama-3.2-1B-Instruct",
            "TC_TRAINING_DATASET_ID": "NousResearch/Hermes-3-Dataset",
            "TC_TRAINING_MAX_DATASET_ENTRIES": "100",
            "TC_TRAINING_SEQ_LENGTH": "1024",
            "TC_TRAINING_BATCH_SIZE": "4",
            "TC_TRAINING_EPOCHS": "3",
            "TC_TRAINING_LR": "5e-5",
            "TC_TRAINING_DEVICE": "METAL",
            "TC_TRAINING_GRADIENT_ACCUMULATION": "2",
            "TC_TRAINING_SAVE_DIR": "checkpoints/demo",
            "TC_OFFLINE_MODE": "true",
            "TC_TRAINING_FINETUNE": "false",
        }

        with patch.dict("os.environ", env, clear=False):
            settings = default_training_settings()

        self.assertEqual(settings["model-id"], "unsloth/Llama-3.2-1B-Instruct")
        self.assertEqual(settings["dataset-id"], "NousResearch/Hermes-3-Dataset")
        self.assertEqual(settings["max-dataset-entries"], "100")
        self.assertEqual(settings["seq-length"], "1024")
        self.assertEqual(settings["batch-size"], "4")
        self.assertEqual(settings["epochs"], "3")
        self.assertEqual(settings["lr"], "5e-5")
        self.assertEqual(settings["device"], "METAL")
        self.assertEqual(settings["gradient-accumulation"], "2")
        self.assertEqual(settings["save-dir"], "checkpoints/demo")
        self.assertTrue(settings["offline"])
        self.assertTrue(settings["from-scratch"])

    def test_update_resource_usage_uses_current_peer_client_api(self) -> None:
        peers = [
            SimpleNamespace(gpu_model="Apple M4 Pro GPU", tg_device="MPS", cpu_model="Apple M4 Pro"),
            SimpleNamespace(gpu_model="", tg_device="CUDA", cpu_model=""),
            SimpleNamespace(gpu_model="", tg_device="CUDA", cpu_model=""),
            SimpleNamespace(gpu_model="", tg_device="", cpu_model="EPYC"),
        ]
        screen = TrainScreen(peer_client=_PeerClientStub(peers))
        screen._resource_labels = {
            "cpu": Label(),
            "ram": Label(),
            "gpu": Label(),
            "peers": Label(),
        }

        self.assertEqual(screen._aggregate_devices(), ["Apple M4 Pro GPU", "CUDA", "EPYC"])
        screen._update_resource_usage()

    def test_build_training_settings_uses_selected_backend(self) -> None:
        screen = TrainScreen(peer_client=_PeerClientStub([]))
        screen._settings["model-id"] = "Qwen/Qwen2.5-0.5B-Instruct"
        screen._settings["config-path"] = "stale/config.json"
        screen._settings["device"] = "mps"

        with patch("cheetah.tui.train_menu.get_llm_backend", return_value="torch"):
            settings = screen._build_training_settings()

        self.assertIsNotNone(settings)
        assert settings is not None
        self.assertEqual(settings["backend"], "torch")
        self.assertEqual(settings["model-id"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertNotIn("config-path", settings)
        self.assertEqual(settings["weights-dir"], "")

    def test_build_training_settings_includes_peer_snapshot(self) -> None:
        peers = [
            SimpleNamespace(peer_client_id="self", ip_address="192.168.0.10", gpu_vram="8", cpu_ram="16", gpu_flops=0.0),
            SimpleNamespace(peer_client_id="peer-1", ip_address="192.168.0.20", gpu_vram="4", cpu_ram="8", gpu_flops=0.0),
        ]
        screen = TrainScreen(peer_client=_PeerClientStub(peers))
        screen._settings["model-id"] = "Qwen/Qwen2.5-0.5B-Instruct"

        with patch("cheetah.tui.train_menu.get_llm_backend", return_value="torch"):
            settings = screen._build_training_settings()

        self.assertIsNotNone(settings)
        assert settings is not None
        self.assertEqual(settings["local-peer-id"], "self")
        snapshot = settings["peer-snapshot"]
        self.assertIsInstance(snapshot, list)
        self.assertEqual(len(snapshot), 2)
        self.assertEqual(snapshot[0]["peer_client_id"], "self")
        self.assertEqual(snapshot[1]["peer_client_id"], "peer-1")

    def test_build_training_settings_snapshot_still_supports_shard_plan_logging(self) -> None:
        peers = [
            SimpleNamespace(peer_client_id="self", ip_address="192.168.0.10", gpu_vram="8", cpu_ram="8", gpu_flops=0.0),
            SimpleNamespace(peer_client_id="peer-1", ip_address="192.168.0.20", gpu_vram="8", cpu_ram="8", gpu_flops=0.0),
        ]
        screen = TrainScreen(peer_client=_PeerClientStub(peers))
        screen._settings["model-id"] = "Qwen/Qwen2.5-0.5B-Instruct"

        with patch("cheetah.tui.train_menu.get_llm_backend", return_value="torch"):
            settings = screen._build_training_settings()

        self.assertIsNotNone(settings)
        assert settings is not None
        lines = distributed_shard_plan_messages(
            settings["peer-snapshot"],
            local_peer_id=settings["local-peer-id"],
            model_name="demo",
            total_layers=9,
        )

        self.assertEqual(lines[0], "Using 2 nodes for shard-aware execution.")
        self.assertIn("Loading local shard self (192.168.0.10):", lines[1])
        self.assertIn("Loading shard on peer peer-1 (192.168.0.20):", lines[2])

    def test_build_training_settings_chains_previous_step_for_finetune(self) -> None:
        with TemporaryDirectory() as tmp:
            screen = TrainScreen(peer_client=_PeerClientStub([]))
            screen._settings["model-id"] = "Qwen/Qwen2.5-0.5B-Instruct"
            screen._settings["dataset-id"] = "NousResearch/Hermes-3-Dataset"
            screen._settings["save-dir"] = str(Path(tmp) / "runs")
            screen._node_steps = [
                TrainingNode("Base Training", settings={"from-scratch": True}),
                TrainingNode("Fine Tune", settings={"from-scratch": False}),
            ]
            previous_dir = Path(tmp) / "runs" / "step_01_base-training"
            previous_dir.mkdir(parents=True, exist_ok=True)
            screen._node_output_dirs[0] = previous_dir

            with patch("cheetah.tui.train_menu.get_llm_backend", return_value="torch"):
                settings = screen._build_training_settings(node_index=1)

        self.assertIsNotNone(settings)
        assert settings is not None
        self.assertEqual(settings["weights-dir"], str(previous_dir))
        self.assertEqual(settings["from-scratch"], False)
        self.assertIn("step_02_fine-tune", str(settings["save-dir"]))

    def test_build_training_settings_applies_step_specific_data_and_tuning(self) -> None:
        with TemporaryDirectory() as tmp:
            screen = TrainScreen(peer_client=_PeerClientStub([]))
            screen._settings["model-id"] = "Qwen/Qwen2.5-0.5B-Instruct"
            screen._settings["dataset-id"] = "NousResearch/Hermes-3-Dataset"
            screen._settings["epochs"] = "1"
            screen._settings["lr"] = "1e-4"
            screen._settings["seq-length"] = "256"
            screen._settings["batch-size"] = "2"
            screen._settings["gradient-accumulation"] = "1"
            screen._settings["save-dir"] = str(Path(tmp) / "runs")
            screen._node_steps = [
                TrainingNode("Base Training", settings={"from-scratch": True}),
                TrainingNode(
                    "Fine Tune",
                    settings={
                        "from-scratch": False,
                        "dataset-id": "custom/fine-tune-dataset",
                        "data-path": "",
                        "max-dataset-entries": "50",
                        "epochs": "3",
                        "lr": "5e-5",
                        "seq-length": "512",
                        "batch-size": "4",
                        "gradient-accumulation": "2",
                    },
                ),
            ]
            previous_dir = Path(tmp) / "runs" / "step_01_base-training"
            previous_dir.mkdir(parents=True, exist_ok=True)
            screen._node_output_dirs[0] = previous_dir

            with patch("cheetah.tui.train_menu.get_llm_backend", return_value="torch"):
                settings = screen._build_training_settings(node_index=1)

        self.assertIsNotNone(settings)
        assert settings is not None
        self.assertEqual(settings["weights-dir"], str(previous_dir))
        self.assertEqual(settings["dataset-id"], "custom/fine-tune-dataset")
        self.assertEqual(settings["data-path"], "")
        self.assertEqual(settings["max-dataset-entries"], "50")
        self.assertEqual(settings["epochs"], "3")
        self.assertEqual(settings["lr"], "5e-5")
        self.assertEqual(settings["seq-length"], "512")
        self.assertEqual(settings["batch-size"], "4")
        self.assertEqual(settings["gradient-accumulation"], "2")

    def test_build_training_env_sets_backend_specific_device(self) -> None:
        settings = {
            "backend": "torch",
            "model-id": "Qwen/Qwen2.5-0.5B-Instruct",
            "data-path": "",
            "dataset-id": "",
            "max-dataset-entries": "25",
            "seq-length": "256",
            "batch-size": "2",
            "epochs": "1",
            "lr": "1e-4",
            "device": "metal",
            "gradient-accumulation": "1",
            "save-dir": "",
            "offline": False,
            "from-scratch": False,
        }

        args = _build_training_namespace(settings)

        self.assertEqual(args.backend, "torch")
        self.assertEqual(args.device, "mps")
        self.assertEqual(args.max_dataset_entries, 25)

    def test_ensure_required_keys_supports_config_wrappers(self) -> None:
        wrapper = Namespace(config={})

        _ensure_required_keys(wrapper)

        self.assertIn("attn_scale", wrapper.config)
        self.assertIn("temperature", wrapper.config)

    def test_stream_corpus_batches_reports_tokenization_progress(self) -> None:
        with TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "corpus.txt"
            data_path.write_text(("hello world\n" * 40), encoding="utf-8")

            stream = io.StringIO()
            with redirect_stdout(stream):
                batches = list(
                    _stream_corpus_batches(
                        _TokenizerStub(),
                        data_path,
                        seq_length=4096,
                        batch_size=2,
                        backend="torch",
                        device="cpu",
                    )
                )

        self.assertEqual(batches, [])
        output = stream.getvalue()
        self.assertIn("[data] tokenizing progress: 0%", output)
        self.assertIn("[data] tokenizing progress: 100%", output)
        self.assertIn("lines read", output)

    def test_prepare_dataset_corpus_uses_limit_specific_cache_file(self) -> None:
        with TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "dataset"
            output_dir = Path(tmp) / "processed"
            dataset_root.mkdir(parents=True, exist_ok=True)
            jsonl_path = dataset_root / "data.jsonl"
            jsonl_path.write_text(
                "\n".join(
                    [
                        '{"messages":[{"role":"user","content":"one"},{"role":"assistant","content":"a"}]}',
                        '{"messages":[{"role":"user","content":"two"},{"role":"assistant","content":"b"}]}',
                        '{"messages":[{"role":"user","content":"three"},{"role":"assistant","content":"c"}]}',
                    ]
                ),
                encoding="utf-8",
            )

            corpus_limit = _prepare_dataset_corpus(dataset_root, output_dir, limit=2)
            corpus_all = _prepare_dataset_corpus(dataset_root, output_dir, limit=None)
            limited_text = corpus_limit.read_text(encoding="utf-8")

            self.assertNotEqual(corpus_limit.name, corpus_all.name)
            self.assertEqual(corpus_limit.name, "dataset_corpus_limit_2.txt")
            self.assertEqual(corpus_all.name, "dataset_corpus_all.txt")
            self.assertIn("one", limited_text)
            self.assertIn("two", limited_text)
            self.assertNotIn("three", limited_text)

    def test_training_process_terminate_logs_runtime_cleanup(self) -> None:
        runtime_model = object()

        def fake_run_training_job(settings, stop_event, runtime_state) -> None:
            del settings
            runtime_state["model"] = runtime_model
            runtime_state["optimizer"] = object()
            runtime_state["tokenizer"] = object()
            while not stop_event.is_set():
                time.sleep(0.01)
            raise TrainingCancelled()

        with (
            patch("cheetah.tui.train_menu._run_training_job", side_effect=fake_run_training_job),
            patch("cheetah.tui.train_menu.relieve_memory_pressure") as relieve_memory,
        ):
            training = TrainingProcess({"backend": "torch"})
            training.start()
            time.sleep(0.05)
            training.terminate()
            assert training._thread is not None
            training._thread.join(timeout=1.0)

            drained = [line for line in training.drain() if line is not None]

        self.assertIn(
            "[info] Stop requested. Waiting for training loop to reach a safe cleanup point...",
            drained,
        )
        self.assertIn("[info] Releasing training runtime...", drained)
        self.assertIn("[info] Training runtime cleared.", drained)
        relieve_memory.assert_called_once_with(runtime_model)

    def test_training_path_add_step_defaults_to_finetune_previous(self) -> None:
        screen = TrainingPathScreen([TrainingNode("Base Training")])
        with patch.object(screen, "_refresh_views"):
            screen._on_add_result("Fine Tune")

        self.assertEqual(len(screen._path_nodes), 2)
        self.assertFalse(screen._path_nodes[1].settings.get("from-scratch", True))

    def test_training_path_step_settings_switch_data_source_and_preserve_mode(self) -> None:
        screen = TrainingPathScreen(
            [
                TrainingNode("Base Training"),
                TrainingNode(
                    "Fine Tune",
                    settings={"from-scratch": False, "data-path": "/tmp/base.txt", "lr": "1e-4"},
                ),
            ]
        )
        with patch.object(screen, "_refresh_views"):
            screen._on_step_settings_result(
                1,
                {
                    "dataset-id": "custom/fine-tune-dataset",
                    "data-path": "",
                    "max-dataset-entries": "100",
                    "epochs": "2",
                    "lr": "",
                    "seq-length": "",
                    "batch-size": "",
                    "gradient-accumulation": "",
                },
            )

        node = screen._path_nodes[1]
        self.assertFalse(node.settings["from-scratch"])
        self.assertEqual(node.settings["dataset-id"], "custom/fine-tune-dataset")
        self.assertEqual(node.settings["data-path"], "")
        self.assertEqual(node.settings["max-dataset-entries"], "100")
        self.assertEqual(node.settings["epochs"], "2")
        self.assertNotIn("lr", node.settings)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_train_epoch_torch_uses_forward_path_and_restores_mode(self) -> None:
        model = _TorchTrainModel()
        model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        batch = Batch(
            input_ids=torch.tensor([[0, 1]], dtype=torch.long),
            labels=torch.tensor([[1, 2]], dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            position_ids=torch.tensor([[0, 1]], dtype=torch.long),
        )

        loss = _train_epoch(model, optimizer, [batch], grad_accum=1, backend="torch")

        self.assertTrue(loss >= 0.0)
        self.assertFalse(model.training)
        self.assertEqual(model.decode_calls, 0)

    @unittest.skipIf(tg is None, "tinygrad is not installed")
    def test_train_epoch_tinygrad_uses_forward_path_and_restores_training_flag(self) -> None:
        model = _TinygradTrainModel()
        optimizer = tg.nn.optim.Adam([model.weight], lr=0.1)
        batch = Batch(
            input_ids=tg.Tensor([[0, 1]], dtype=tg.dtypes.int32),
            labels=tg.Tensor([[1, 2]], dtype=tg.dtypes.int32),
            attention_mask=tg.Tensor([[1, 1]], dtype=tg.dtypes.int32),
            position_ids=tg.Tensor([[0, 1]], dtype=tg.dtypes.int32),
        )
        original_training = tg.Tensor.training

        loss = _train_epoch(model, optimizer, [batch], grad_accum=1, backend="tinygrad")

        self.assertTrue(loss >= 0.0)
        self.assertEqual(model.decode_calls, 0)
        self.assertEqual(tg.Tensor.training, original_training)


if __name__ == "__main__":
    unittest.main()
