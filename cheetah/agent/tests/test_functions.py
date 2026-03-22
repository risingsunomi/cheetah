from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from cheetah.agent.functions import AgentFunctions


class TestAgentFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = AgentFunctions()

    def test_list_builtin_functions_contains_expected(self) -> None:
        names = set(self.runtime.list_builtin_functions())
        self.assertIn("list_dir", names)
        self.assertIn("read_file", names)
        self.assertIn("write_file", names)
        self.assertIn("edit_file", names)
        self.assertIn("run_shell", names)
        self.assertIn("get_env", names)
        self.assertIn("end_run", names)
        self.assertIn("web_search", names)

    def test_get_agent_functions_tools_format(self) -> None:
        tools = self.runtime.get_agent_functions(format_mode="tools")
        self.assertTrue(tools)
        first = tools[0]
        self.assertEqual(first.get("type"), "function")
        self.assertIn("function", first)

    def test_get_agent_functions_legacy_format(self) -> None:
        funcs = self.runtime.get_agent_functions(format_mode="functions")
        self.assertTrue(funcs)
        first = funcs[0]
        self.assertIn("name", first)
        self.assertIn("description", first)
        self.assertIn("parameters", first)
        self.assertNotIn("function", first)

    def test_execute_unknown_function(self) -> None:
        result = self.runtime.execute_agent_function("does_not_exist", {})
        self.assertFalse(result.get("ok", True))
        self.assertIn("Unknown function", str(result.get("error", "")))

    def test_execute_function_invalid_json_arguments(self) -> None:
        result = self.runtime.execute_agent_function("list_dir", "{")
        self.assertFalse(result.get("ok", True))
        self.assertIn("Invalid JSON arguments", str(result.get("error", "")))

    def test_execute_list_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "a.txt"), "w", encoding="utf-8").close()
            open(os.path.join(tmpdir, "b.txt"), "w", encoding="utf-8").close()
            result = self.runtime.execute_agent_function("list_dir", {"path": tmpdir})
            self.assertTrue(result.get("ok"))
            entries = result.get("entries", [])
            self.assertIn("a.txt", entries)
            self.assertIn("b.txt", entries)

    def test_execute_read_file_with_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "sample.txt")
            with open(file_path, "w", encoding="utf-8") as handle:
                handle.write("abcdef")

            result = self.runtime.execute_agent_function(
                "read_file",
                {"path": file_path, "max_chars": 3},
            )
            self.assertTrue(result.get("ok"))
            self.assertEqual(result.get("content"), "abc")
            self.assertTrue(result.get("truncated"))

    def test_execute_get_env(self) -> None:
        with patch.dict(os.environ, {"TC_AGENT_TEST_ENV": "hello"}):
            result = self.runtime.execute_agent_function("get_env", {"name": "TC_AGENT_TEST_ENV"})
            self.assertTrue(result.get("ok"))
            self.assertEqual(result.get("value"), "hello")

    def test_execute_write_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "nested", "sample.txt")
            result = self.runtime.execute_agent_function(
                "write_file",
                {
                    "path": file_path,
                    "content": "hello world",
                    "make_dirs": True,
                },
            )
            self.assertTrue(result.get("ok"))
            with open(file_path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "hello world")

    def test_execute_edit_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "sample.txt")
            with open(file_path, "w", encoding="utf-8") as handle:
                handle.write("alpha beta alpha")

            result = self.runtime.execute_agent_function(
                "edit_file",
                {
                    "path": file_path,
                    "old_text": "alpha",
                    "new_text": "omega",
                    "replace_all": True,
                },
            )
            self.assertTrue(result.get("ok"))
            self.assertEqual(result.get("replacements"), 2)
            with open(file_path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "omega beta omega")

    def test_execute_run_shell(self) -> None:
        result = self.runtime.execute_agent_function("run_shell", {"command": "echo hello"})
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("returncode"), 0)
        self.assertIn("hello", str(result.get("stdout", "")))

    def test_execute_end_run(self) -> None:
        result = self.runtime.execute_agent_function("end_run", {"summary": "task complete"})
        self.assertTrue(result.get("ok"))
        self.assertTrue(result.get("end_run"))
        self.assertEqual(result.get("summary"), "task complete")

    def test_execute_web_search(self) -> None:
        mock_results = [
            {
                "title": "Example",
                "url": "https://example.com",
                "snippet": "Example snippet.",
            }
        ]
        mock_url = "https://www.bing.com/search?q=example&count=1"
        with patch.object(
            self.runtime,
            "_bing_search_with_playwright",
            return_value=(mock_results, mock_url),
        ):
            result = self.runtime.execute_agent_function(
                "web_search",
                {"query": "example", "max_results": 1, "timeout_seconds": 10},
            )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("engine"), "bing")
        self.assertEqual(result.get("count"), 1)
        self.assertEqual(result.get("search_url"), mock_url)
        self.assertEqual(result.get("results"), mock_results)

    def test_execute_web_search_requires_query(self) -> None:
        result = self.runtime.execute_agent_function("web_search", {"query": "   "})
        self.assertFalse(result.get("ok", True))
        self.assertIn("query is required", str(result.get("error", "")))

    def test_execute_web_search_missing_playwright(self) -> None:
        with patch.object(self.runtime, "_bing_search_with_playwright", side_effect=ImportError):
            result = self.runtime.execute_agent_function("web_search", {"query": "example"})
        self.assertFalse(result.get("ok", True))
        self.assertIn("Playwright is not installed", str(result.get("error", "")))


if __name__ == "__main__":
    unittest.main()
