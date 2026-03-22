import os
import unittest
from unittest import mock

from cheetah.models.llm.backend import (
    LLM_BACKEND_ENV,
    TINYGRAD_DEVICE_ENV,
    TORCH_DEVICE_ENV,
    get_backend_device,
    set_backend_device,
    set_llm_backend,
)


class TestBackendDeviceEnv(unittest.TestCase):
    def test_get_backend_device_prefers_backend_specific_env(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                TORCH_DEVICE_ENV: "mps",
                TINYGRAD_DEVICE_ENV: "METAL",
            },
            clear=True,
        ):
            self.assertEqual(get_backend_device("torch"), "mps")
            self.assertEqual(get_backend_device("tinygrad"), "METAL")

    def test_set_llm_backend_syncs_legacy_device_to_selected_backend(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            set_backend_device("METAL", backend="tinygrad")
            set_backend_device("mps", backend="torch")

            self.assertEqual(set_llm_backend("torch"), "torch")
            self.assertEqual(set_llm_backend("tinygrad"), "tinygrad")

    def test_set_backend_device_only_syncs_legacy_for_active_backend(self) -> None:
        with mock.patch.dict(os.environ, {LLM_BACKEND_ENV: "torch"}, clear=True):
            set_backend_device("METAL", backend="tinygrad")
            set_backend_device("mps", backend="torch")