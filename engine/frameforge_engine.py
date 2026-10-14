#!/usr/bin/env python3
"""FrameForge's local JSON-lines engine entry point."""

from __future__ import annotations

import json
import platform
import sys
from typing import Any, Callable, Dict, Optional

from frameforge.dataset import export_yolo_dataset, validate_dataset
from frameforge.discovery import prepare_discovered_dataset, scan_project_folder
from frameforge.inference import auto_label_project, predict_crops
from frameforge.media import extract_frames, inspect_model, inspect_sources, preview_center_crop, preview_frame
from frameforge.project import create_project, load_project, review_ai_negative_candidate, save_ai_suggestions, save_annotations, save_project
from frameforge.session import load_session, save_session
from frameforge.training import (
    compare_model_challenger,
    evaluate_yolo_checkpoint,
    get_model_promotion_state,
    get_evaluation_targets,
    inspect_training_dataset,
    import_yolo_checkpoint,
    list_model_evaluations,
    list_training_runs,
    promote_model_challenger,
    register_discovered_run,
    rollback_model_promotion,
    save_evaluation_targets,
    train_yolo_queue,
    training_capabilities,
)


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
                    "ai_annotation_review",
                    "dataset_validation",
                    "yolo_export",
                    "project_persistence",
                    "model_inspection",
                    "training_capabilities",
                    "training_dataset_inspection",
                    "yolo11n_training",
                    "yolo11n_inference",
                    "project_batch_inference",
                    "candidate_negative_review",
                    "training_registry",
                    "trusted_model_import",
                    "fixed_set_evaluation",
                    "evaluation_targets",
                    "champion_challenger_management",
                    "external_training_registry",
                    "smart_folder_discovery",
                    "workspace_folder_import",
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
        elif command == "save_ai_suggestions":
            result = save_ai_suggestions(
                arguments["path"],
                arguments["image_path"],
                arguments["predictions"],
                arguments["image_fingerprint"],
                arguments["checkpoint"],
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
                arguments.get("split_mode", "source"),
                arguments.get("train_ratio", 0.8),
                arguments.get("val_ratio", 0.1),
                arguments.get("test_ratio", 0.1),
                arguments.get("seed", 42),
                arguments.get("segment_seconds", 30),
                arguments.get("parent_manifest_path"),
                arguments.get("replay_strategy", "all"),
                arguments.get("freeze_evaluation_set", False),
                arguments.get("negative_ratio", 0.1),
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
                arguments.get("parent_manifest_path"),
                arguments.get("replay_strategy", "all"),
                arguments.get("freeze_evaluation_set", False),
                arguments.get("negative_ratio", 0.1),
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
        elif command == "import_yolo_checkpoint":
            result = import_yolo_checkpoint(arguments.get("path"), arguments.get("trust_source"))
        elif command == "evaluate_yolo_checkpoint":
            result = evaluate_yolo_checkpoint(arguments, progress)
        elif command == "list_model_evaluations":
            result = list_model_evaluations(arguments.get("run_id"))
        elif command == "get_evaluation_targets":
            result = get_evaluation_targets()
        elif command == "save_evaluation_targets":
            result = save_evaluation_targets(arguments)
        elif command == "get_model_promotion_state":
            result = get_model_promotion_state()
        elif command == "compare_model_challenger":
            result = compare_model_challenger(arguments)
        elif command == "promote_model_challenger":
            result = promote_model_challenger(arguments)
        elif command == "rollback_model_promotion":
            result = rollback_model_promotion()
        elif command == "register_discovered_run":
            result = register_discovered_run(arguments["discovery_root"], arguments["run"])
        elif command == "prepare_discovered_dataset":
            result = prepare_discovered_dataset(arguments["discovery_root"], arguments["data_yaml"])
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
        elif command == "auto_label_project":
            result = auto_label_project(
                arguments["path"],
                arguments["checkpoint"],
                arguments.get("confidence_threshold", 0.25),
                arguments.get("iou_threshold", 0.7),
                arguments.get("cancel_file"),
                progress,
            )
        else:
            return {
                "id": request_id,
                "ok": False,
                "error": {
                    "code": "unknown_command",
                    "message": "Supported commands: health, inspect_sources, inspect_model, preview_frame, preview_center_crop, extract_frames, create_project, save_project, save_annotations, save_ai_suggestions, review_ai_negative_candidate, auto_label_project, validate_dataset, export_yolo_dataset, load_project, save_session, load_session, training_capabilities, inspect_training_dataset, list_training_runs, evaluate_yolo_checkpoint, list_model_evaluations, get_evaluation_targets, save_evaluation_targets, get_model_promotion_state, compare_model_challenger, promote_model_challenger, rollback_model_promotion, register_discovered_run, prepare_discovered_dataset, scan_project_folder, train_yolo_queue, predict_crops.",
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
