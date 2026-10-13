#!/usr/bin/env python3
"""FrameForge's local JSON-lines engine entry point."""

from __future__ import annotations

import json
import platform
import sys
from typing import Any, Callable, Dict, Optional

from frameforge.media import extract_frames, inspect_model, inspect_sources, preview_center_crop, preview_frame
from frameforge.project import create_project, load_project, save_annotations, save_project


def handle_request(request: Any, progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
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
                "capabilities": [
                    "health",
                    "inspect_sources",
                    "preview_frame",
                    "preview_center_crop",
                    "extract_frames",
                    "save_annotations",
                    "project_persistence",
                    "model_inspection",
                ],
            },
        }

    arguments = request.get("arguments", {})
    if not isinstance(arguments, dict):
        return {
            "id": request_id,
            "ok": False,
            "error": {"code": "invalid_arguments", "message": "Arguments must be a JSON object."},
        }

    try:
        if command == "inspect_sources":
            result = inspect_sources(arguments.get("paths", []))
        elif command == "inspect_model":
            result = inspect_model(arguments["path"])
        elif command == "preview_frame":
            result = preview_frame(arguments["path"], arguments.get("frame_index", 0))
        elif command == "preview_center_crop":
            result = preview_center_crop(
                arguments["path"],
                arguments.get("frame_index", 0),
                arguments["width"],
                arguments["height"],
            )
        elif command == "extract_frames":
            result = extract_frames(
                arguments.get("paths", []),
                arguments["output_dir"],
                arguments.get("options", {}),
                progress,
            )
        elif command == "create_project":
            result = create_project(arguments["path"], arguments.get("name", "Dự án mới"))
        elif command == "save_project":
            result = save_project(arguments["path"], arguments["project"])
        elif command == "save_annotations":
            result = save_annotations(
                arguments["path"],
                arguments["image_path"],
                arguments["annotations"],
                arguments["status"],
                arguments["image_fingerprint"],
            )
        elif command == "load_project":
            result = load_project(arguments["path"])
        else:
            return {
                "id": request_id,
                "ok": False,
                "error": {
                    "code": "unknown_command",
                    "message": "Supported commands: health, inspect_sources, inspect_model, preview_frame, preview_center_crop, extract_frames, create_project, save_project, save_annotations, load_project.",
                },
            }
    except (KeyError, OSError, ValueError, TypeError) as error:
        return {
            "id": request_id,
            "ok": False,
            "error": {"code": "processing_error", "message": str(error)},
        }
    return {"id": request_id, "ok": True, "data": result}


def serve() -> int:
    """Read requests from stdin and write one JSON response per line."""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            request_id = request.get("id") if isinstance(request, dict) else None

            def emit_progress(event: Dict[str, Any]) -> None:
                sys.stdout.write(json.dumps({"id": request_id, "ok": True, "event": event}, separators=(",", ":")) + "\n")
                sys.stdout.flush()

            response = handle_request(request, emit_progress)
        except json.JSONDecodeError as error:
            response = {
                "id": None,
                "ok": False,
                "error": {"code": "invalid_json", "message": "Request is not valid JSON."},
            }
            print("Invalid JSON request: {0}".format(error), file=sys.stderr)
        except Exception as error:
            request_id = request.get("id") if isinstance(request, dict) else None
            response = {
                "id": request_id,
                "ok": False,
                "error": {"code": "internal_error", "message": str(error)},
            }
            print("Engine request failed: {0}".format(error), file=sys.stderr)
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
