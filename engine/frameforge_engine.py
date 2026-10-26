#!/usr/bin/env python3
"""FrameForge's local JSON-lines engine entry point."""

from __future__ import annotations

import json
import platform
import sys
from typing import Any, Callable, Dict, Optional

from frameforge.dataset import export_yolo_dataset, validate_dataset
from frameforge.deletion import delete_images
from frameforge.inference import auto_label_project, predict_crops, save_extraction_suggestions
from frameforge.labelme import import_images, import_labelme_folder
from frameforge.media import extract_frames, image_thumbnails, inspect_sources, preview_center_crop, preview_frame
from frameforge.relocate import relocate_images
from frameforge.review import review_images
from frameforge.models import delete_model, import_yolo_checkpoint, inference_capabilities, list_models
from frameforge.project import (
    create_project,
    load_project,
    review_ai_negative_candidate,
    save_ai_suggestions,
    save_annotations,
    save_project,
    set_project_classes,
)
from frameforge.session import load_session, save_session
from frameforge.sidecar import sync_project as sync_sidecars, write_project as write_sidecars


READS_SIDECARS = {"load_project", "validate_dataset", "export_yolo_dataset", "auto_label_project", "review_images"}
WRITES_ONE_SIDECAR = {"save_annotations", "save_ai_suggestions", "review_ai_negative_candidate"}
WRITES_ALL_SIDECARS = {"save_project", "set_project_classes", "review_images"}


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
                    "image_thumbnails",
                    "extract_frames",
                    "save_annotations",
                    "ai_annotation_review",
                    "dataset_validation",
                    "yolo_export",
                    "project_persistence",
                    "project_class_schema",
                    "inference_capabilities",
                    "yolo11n_inference",
                    "project_batch_inference",
                    "extraction_detection",
                    "candidate_negative_review",
                    "model_registry",
                    "trusted_model_import",
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
        # Labels live in the JSON beside each image: read files changed outside FrameForge first.
        if command in READS_SIDECARS:
            sync_sidecars(arguments["path"])
        if command == "inspect_sources":
            result = inspect_sources(arguments.get("paths", []))
        elif command == "preview_frame":
            result = preview_frame(arguments["path"], arguments.get("frame_index", 0), lossless=arguments.get("lossless") is True, options=arguments.get("options"))
        elif command == "preview_center_crop":
            result = preview_center_crop(
                arguments["path"],
                arguments.get("frame_index", 0),
                arguments["width"],
                arguments["height"],
            )
        elif command == "image_thumbnails":
            result = image_thumbnails(arguments.get("paths"))
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
        elif command == "save_ai_suggestions":
            result = save_ai_suggestions(
                arguments["path"],
                arguments["image_path"],
                arguments["predictions"],
                arguments["image_fingerprint"],
                arguments["checkpoint"],
                arguments.get("skipped_large_objects", 0),
                arguments.get("replace") is True,
            )
        elif command == "set_project_classes":
            result = set_project_classes(arguments["path"], arguments["class_names"])
        elif command == "delete_images":
            result = delete_images(arguments["path"], arguments["image_paths"])
        elif command == "relocate_images":
            result = relocate_images(arguments["path"], arguments.get("root"))
        elif command == "review_images":
            result = review_images(arguments["path"], arguments["image_paths"])
        elif command == "import_images":
            result = import_images(
                arguments["path"],
                arguments["paths"],
                arguments.get("mark_reviewed", False),
                arguments.get("dry_run", False),
            )
        elif command == "import_labelme_folder":
            result = import_labelme_folder(
                arguments["path"],
                arguments["folder"],
                arguments.get("mark_reviewed", False),
            )
        elif command == "review_ai_negative_candidate":
            result = review_ai_negative_candidate(
                arguments["path"],
                arguments["candidate_id"],
                arguments["action"],
                arguments.get("image_fingerprint"),
            )
        elif command == "validate_dataset":
            result = validate_dataset(
                arguments["path"],
                arguments.get("split_mode", "image"),
                arguments.get("train_ratio", 0.8),
                arguments.get("val_ratio", 0.1),
                arguments.get("test_ratio", 0.1),
                arguments.get("seed", 42),
                arguments.get("segment_seconds", 30),
                arguments.get("negative_ratio", 0.1),
                arguments.get("required_class_ids"),
            )
        elif command == "export_yolo_dataset":
            result = export_yolo_dataset(
                arguments["path"],
                arguments["output_dir"],
                arguments.get("split_mode", "image"),
                arguments.get("train_ratio", 0.8),
                arguments.get("val_ratio", 0.1),
                arguments.get("test_ratio", 0.1),
                arguments.get("seed", 42),
                arguments.get("segment_seconds", 30),
                arguments.get("negative_ratio", 0.1),
                arguments.get("required_class_ids"),
            )
        elif command == "load_project":
            result = load_project(arguments["path"])
        elif command == "save_session":
            result = save_session(arguments["project"], arguments["session"])
        elif command == "load_session":
            result = load_session()
        elif command == "inference_capabilities":
            result = inference_capabilities()
        elif command == "list_models":
            result = list_models()
        elif command == "delete_model":
            result = delete_model(arguments.get("run_id"))
        elif command == "import_yolo_checkpoint":
            result = import_yolo_checkpoint(arguments.get("path"), arguments.get("trust_source"))
        elif command == "predict_crops":
            result = predict_crops(
                arguments["checkpoint"],
                arguments["images"],
                arguments.get("confidence_threshold", 0.25),
                arguments.get("iou_threshold", 0.7),
                arguments.get("class_ids"),
                arguments.get("cancel_file"),
                progress,
            )
        elif command == "save_extraction_suggestions":
            try:
                result = save_extraction_suggestions(
                    arguments["path"],
                    arguments["checkpoint"],
                    arguments.get("checkpoint_sha256"),
                    arguments["items"],
                )
            finally:
                # Chunks committed before a failure still belong in the JSON files.
                write_sidecars(arguments["path"], [
                    item["image_path"] for item in arguments["items"]
                    if isinstance(item, dict) and isinstance(item.get("image_path"), str)
                ])
        elif command == "auto_label_project":
            try:
                result = auto_label_project(
                    arguments["path"],
                    arguments["checkpoint"],
                    arguments.get("confidence_threshold", 0.25),
                    arguments.get("iou_threshold", 0.7),
                    arguments.get("cancel_file"),
                    progress,
                    arguments.get("replace") is True,
                )
            finally:
                # Chunks saved before a cancel or failure still belong in the JSON files.
                write_sidecars(arguments["path"])
        else:
            return {
                "id": request_id,
                "ok": False,
                "error": {
                    "code": "unknown_command",
                    "message": "Supported commands: health, inspect_sources, preview_frame, preview_center_crop, image_thumbnails, extract_frames, create_project, save_project, save_annotations, save_ai_suggestions, save_extraction_suggestions, set_project_classes, review_images, relocate_images, import_images, import_labelme_folder, review_ai_negative_candidate, load_project, save_session, load_session, inference_capabilities, list_models, delete_model, import_yolo_checkpoint, predict_crops, auto_label_project, validate_dataset, export_yolo_dataset.",
                },
            }
        if command in WRITES_ONE_SIDECAR:
            written = write_sidecars(arguments["path"], [result.get("image_path") or arguments["image_path"]])
        elif command in WRITES_ALL_SIDECARS:
            written = write_sidecars(arguments["path"])
        else:
            written = None
        if written and written["errors"] and isinstance(result, dict):
            result["json_errors"] = written["errors"]
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
