from __future__ import annotations

import unittest

from tiny_cheetah.agent.model import AgentFunctionSpec


def _noop() -> None:
    return None


class TestAgentModel(unittest.TestCase):
    def test_agent_function_spec_dataclass(self) -> None:
        spec = AgentFunctionSpec(
            name="noop",
            description="No-op",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_noop,
        )
        self.assertEqual(spec.name, "noop")
        self.assertEqual(spec.description, "No-op")
        self.assertTrue(callable(spec.handler))


if __name__ == "__main__":
    unittest.main()
