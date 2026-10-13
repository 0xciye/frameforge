import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frameforge_engine  # noqa: E402


class EngineProtocolTests(unittest.TestCase):
    def test_health_response_contains_engine_capabilities(self):
        response = frameforge_engine.handle_request({"id": "test-1", "command": "health"})

        self.assertEqual(response["id"], "test-1")
        self.assertTrue(response["ok"])
        self.assertEqual(response["data"]["status"], "ok")
        self.assertIn("health", response["data"]["capabilities"])

    def test_unknown_command_returns_protocol_error(self):
        response = frameforge_engine.handle_request({"id": 2, "command": "detect"})

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "unknown_command")

    def test_non_object_request_returns_protocol_error(self):
        response = frameforge_engine.handle_request(["health"])

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
