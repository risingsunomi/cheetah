# Testing device info
import re
import subprocess
import unittest
from cheetah.orchestration.device_info import collect_host_info
from cheetah.logging_utils import get_logger

logger = get_logger(__name__)

class TestDeviceInfo(unittest.TestCase):
    def test_collect_host_info(self):
        info = collect_host_info()
        logger.info("Collected host info: %s", info)
        self.assertIsNotNone(info)

if __name__ == "__main__":
    unittest.main()