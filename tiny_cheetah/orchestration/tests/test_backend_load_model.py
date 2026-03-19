import unittest
from unittest import mock

from tiny_cheetah.models.llm import backend as llm_backend


class TestBackendLoadModelForBackend(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_progress_callback(self) -> None:
        helper_module = mock.Mock()
        helper_module.load_model = mock.AsyncMock(return_value=("model", {}, "tokenizer", "/tmp/model"))
        progress_callback = mock.AsyncMock()

        with mock.patch.object(llm_backend, "backend_helpers_module", return_value=helper_module):
            result = await llm_backend.load_model_for_backend(
                model_id="demo/model",
                backend="torch",
                progress_callback=progress_callback,
            )

        helper_module.load_model.assert_awaited_once_with(
            model_id="demo/model",
            shard=None,
            weight_device=None,
            offline_mode=False,
            progress_callback=progress_callback,
        )
        self.assertEqual(result, ("model", {}, "tokenizer", "/tmp/model"))
