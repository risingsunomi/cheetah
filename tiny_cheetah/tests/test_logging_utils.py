from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiny_cheetah import logging_utils


class _FakeDatetime:
    @classmethod
    def now(cls):
        class _FakeNow:
            def strftime(self, fmt: str) -> str:
                if fmt == "%Y%m%d":
                    return "20260315"
                raise AssertionError(f"Unexpected format: {fmt}")

        return _FakeNow()


class TestLoggingUtils(unittest.TestCase):
    def test_daily_log_path_and_latest_indicator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"TC_LOG_DIR": tmpdir}, clear=False):
                with patch.object(logging_utils, "datetime", _FakeDatetime):
                    log_path = logging_utils._log_path()

                self.assertEqual(log_path, Path(tmpdir) / "tiny_cheetah_20260315.log")
                log_path.write_text("log\n", encoding="utf-8")
                logging_utils._update_latest_log_indicator(log_path)

            latest_link = Path(tmpdir) / "tiny_cheetah_latest.log"
            marker_file = Path(tmpdir) / "LATEST_LOG"

            self.assertTrue(marker_file.exists())
            marker_text = marker_file.read_text(encoding="utf-8")
            self.assertIn("tiny_cheetah_20260315.log", marker_text)

            if latest_link.exists() or latest_link.is_symlink():
                self.assertTrue(latest_link.is_symlink())
                self.assertEqual(latest_link.resolve(), log_path.resolve())


if __name__ == "__main__":
    unittest.main()
