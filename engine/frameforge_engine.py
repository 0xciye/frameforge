#!/usr/bin/env python3
"""FrameForge's local JSON-lines engine entry point."""

from __future__ import annotations

import json
import platform
import sys
from typing import Any, Callable, Dict, Optional

from frameforge.dataset import export_yolo_dataset, validate_dataset
from frameforge.discovery import scan_project_folder
from frameforge.media import extract_frames, inspect_model, inspect_sources, preview_center_crop, preview_frame
from frameforge.project import create_project, load_project, save_annotations, save_project
from frameforge.session import load_session, save_session
from frameforge.training import inspect_training_dataset, list_training_runs, register_discovered_run, train_yolo_queue, training_capabilities


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
                    "dataset_validation",
                    "yolo_export",
                    "project_persistence",
                    "model_inspection",
                    "training_capabilities",
                    "training_dataset_inspection",
                    "yolo11n_training",
                    "training_registry",
                    "external_training_registry",
                    "smart_folder_discovery",
                    "session_recovery",
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
        elif command == "validate_dataset":
            result = validate_dataset(
                arguments["path"],
                arguments.get("split_mode", "source"),
                arguments.get("train_ratio", 0.8),
                arguments.get("val_ratio", 0.1),
                arguments.get("test_ratio", 0.1),
                arguments.get("seed", 42),
                arguments.get("segment_seconds", 30),
            )
        elif command == "export_yolo_dataset":
            result = export_yolo_dataset(
                arguments["path"],
                arguments["output_dir"],
                arguments.get("split_mode", "source"),
                arguments.get("train_ratio", 0.8),
                arguments.get("val_ratio", 0.1),
                arguments.get("test_ratio", 0.1),
                arguments.get("seed", 42),
                arguments.get("segment_seconds", 30),
            )
        elif command == "load_project":
            result = load_project(arguments["path"])
        elif command == "save_session":
            result = save_session(arguments["project"], arguments["session"])
        elif command == "load_session":
            result = load_session()
        elif command == "training_capabilities":
            result = training_capabilities(arguments.get("output_dir"))
        elif command == "inspect_training_dataset":
            result = inspect_training_dataset(arguments["data_yaml"], progress)
        elif command == "list_training_runs":
            result = list_training_runs()
        elif command == "register_discovered_run":
            result = register_discovered_run(arguments["discovery_root"], arguments["run"])
        elif command == "scan_project_folder":
            result = scan_project_folder(
                arguments["path"],
                arguments.get("max_depth", 8),
                arguments.get("max_files", 50_000),
                arguments.get("ignored_dirs"),
                arguments.get("cancel_file"),
                progress,
            )
        elif command == "train_yolo_queue":
            result = train_yolo_queue(arguments, progress)
        else:
            return {
                "id": request_id,
                "ok": False,
                "error": {
                    "code": "unknown_command",
                    "message": "Supported commands: health, inspect_sources, inspect_model, preview_frame, preview_center_crop, extract_frames, create_project, save_project, save_annotations, validate_dataset, export_yolo_dataset, load_project, save_session, load_session, training_capabilities, inspect_training_dataset, list_training_runs, register_discovered_run, scan_project_folder, train_yolo_queue.",
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
    protocol_stdout = sys.stdout
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            request_id = request.get("id") if isinstance(request, dict) else None

            def emit_progress(event: Dict[str, Any]) -> None:
                protocol_stdout.write(json.dumps({"id": request_id, "ok": True, "event": event}, separators=(",", ":")) + "\n")
                protocol_stdout.flush()

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
