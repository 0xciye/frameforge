#!/usr/bin/env python3
"""FrameForge's local JSON-lines engine entry point."""

from __future__ import annotations

import json
import platform
import sys
from typing import Any, Dict


def handle_request(request: Any) -> Dict[str, Any]:
    """Return one protocol response for a decoded JSON request."""
    if not isinstance(request, dict):
        return {
            "id": None,
            "ok": False,
            "error": {"code": "invalid_request", "message": "Request must be a JSON object."},
        }

    request_id = request.get("id")
    command = request.get("command")
    if command == "health":
        return {
            "id": request_id,
            "ok": True,
            "data": {
                "status": "ok",
                "engine": "FrameForge Engine",
                "python_version": platform.python_version(),
                "capabilities": ["health"],
            },
        }

    return {
        "id": request_id,
        "ok": False,
        "error": {
            "code": "unknown_command",
            "message": "Supported command: health.",
        },
    }


def serve() -> int:
    """Read requests from stdin and write one JSON response per line."""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
        except json.JSONDecodeError as error:
            response = {
                "id": None,
                "ok": False,
                "error": {"code": "invalid_json", "message": "Request is not valid JSON."},
            }
            print("Invalid JSON request: {0}".format(error), file=sys.stderr)
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
