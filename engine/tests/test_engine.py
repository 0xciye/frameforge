import base64
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

import frameforge_engine  # noqa: E402
from frameforge.media import _RecentHashes, _center_crop, _letterbox, extract_frames, inspect_sources, preview_center_crop, preview_frame, probe_media
from frameforge.dataset import _balanced_replay_groups, export_yolo_dataset, validate_dataset
from frameforge.project import create_project, load_project, review_ai_negative_candidate, save_ai_suggestions, save_annotations, save_project
from frameforge import training


def create_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened(), "MJPG test fixture codec is unavailable"
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def distinct_frame(index: int) -> np.ndarray:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    cv2.rectangle(frame, (index * 5, 8), (min(index * 5 + 18, 63), 38), (80 + index * 12, 190, 240), -1)
    cv2.putText(frame, str(index), (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return frame


def create_dataset_project(tmp_path: Path, invalid_class: bool = False) -> tuple[Path, dict[str, Path]]:
    image_root = tmp_path / "images"
    source_a = image_root / "source-a"
    source_b = image_root / "source-b"
    source_a.mkdir(parents=True)
    source_b.mkdir(parents=True)
    image_paths = {
        "a_positive": source_a / "same.png",
        "a_negative": source_a / "negative.png",
        "b_positive": source_b / "same.png",
        "b_candidate": source_b / "candidate.png",
    }
    for index, path in enumerate(image_paths.values()):
        frame = np.full((48, 64, 3), 40 + index * 35, dtype=np.uint8)
        cv2.imwrite(str(path), frame)

    project_path = tmp_path / "dataset.ffproj"
    project = {
        "id": "dataset-project-id",
        "name": "Fixture dataset",
        "sources": [
            {"path": str(tmp_path / "source-a.mp4"), "kind": "video", "fps": 10.0},
            {"path": str(tmp_path / "source-b.mp4"), "kind": "video", "fps": 10.0},
        ],
        "settings": {"crop_mode": "center", "output_width": 320, "output_height": 320},
        "class_names": ["head", "body"],
        "images": [
            {
                "path": str(image_paths["a_positive"]), "source_path": str(tmp_path / "source-a.mp4"),
                "source_id": "source-a", "frame_index": 10, "timestamp_seconds": 1.0,
                "width": 64, "height": 48, "status": "reviewed",
                "annotations": [{
                    "id": "a-box", "class_id": 99 if invalid_class else 0,
                    "x_min": 8, "y_min": 4, "x_max": 24, "y_max": 20,
                    "review_status": "reviewed",
                }],
            },
            {
                "path": str(image_paths["a_negative"]), "source_path": str(tmp_path / "source-a.mp4"),
                "source_id": "source-a", "frame_index": 20, "timestamp_seconds": 2.0,
                "width": 64, "height": 48, "status": "reviewed", "annotations": [],
            },
            {
                "path": str(image_paths["b_positive"]), "source_path": str(tmp_path / "source-b.mp4"),
                "source_id": "source-b", "frame_index": 10, "timestamp_seconds": 1.0,
                "width": 64, "height": 48, "status": "reviewed",
                "annotations": [{
                    "id": "b-box", "class_id": 1,
                    "x_min": 16, "y_min": 12, "x_max": 48, "y_max": 36,
                    "review_status": "reviewed",
                }],
            },
            {
                "path": str(image_paths["b_candidate"]), "source_path": str(tmp_path / "source-b.mp4"),
                "source_id": "source-b", "frame_index": 20, "timestamp_seconds": 2.0,
                "width": 64, "height": 48, "status": "ready", "annotations": [],
            },
        ],
    }
    save_project(str(project_path), project)
    return project_path, image_paths


def create_reviewed_lineage_project(tmp_path: Path, reviewed: bool = True) -> Path:
    project_path, _ = create_dataset_project(tmp_path)
    project = load_project(str(project_path))
    for image in project["images"]:
        image["status"] = "reviewed" if reviewed else "ready"
    for index, class_id in enumerate((0, 1)):
        source_id = "source-{0}".format(chr(ord("c") + index))
        source_path = tmp_path / (source_id + ".mp4")
        image_path = tmp_path / "images" / source_id / "positive.png"
        image_path.parent.mkdir(parents=True)
        cv2.imwrite(str(image_path), np.full((48, 64, 3), 150 + index * 40, dtype=np.uint8))
        project["sources"].append({"path": str(source_path), "kind": "video", "fps": 10.0})
        project["images"].append({
            "path": str(image_path), "source_path": str(source_path), "source_id": source_id,
            "frame_index": 5, "timestamp_seconds": 0.5, "width": 64, "height": 48,
            "status": "reviewed" if reviewed else "ready",
            "annotations": [{
                "id": source_id + "-box", "class_id": class_id,
                "x_min": 8, "y_min": 5, "x_max": 32, "y_max": 36,
                "review_status": "reviewed",
            }],
        })
    save_project(str(project_path), project)
    return project_path


def test_recent_hash_index_matches_threshold_and_evicts_old_values():
    index = _RecentHashes(threshold=2, capacity=2)
    first = 0
    second = 0b11
    third = 1 << 20

    assert index.contains_near(first) is False
    index.add(first)
    assert index.contains_near(second) is True
    index.add(third)
    index.add(1 << 40)

    assert index.contains_near(second) is False
    assert index.contains_near(third) is True
    assert len(index.values) == 2


def test_letterbox_applies_user_selected_rgb_padding_as_bgr():
    source = np.full((48, 64, 3), 80, dtype=np.uint8)

    output = _letterbox(source, 32, 32, (0, 0, 255))

    assert output.shape == (32, 32, 3)
    assert output[0, 0].tolist() == [0, 0, 255]


def test_exact_center_crop_uses_floor_geometry_without_resizing():
    frame = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)

    crop, rect = _center_crop(frame, 5, 3)

    assert rect == {"x": 3, "y": 2, "width": 5, "height": 3}
    assert crop.shape == (3, 5, 3)
    assert np.array_equal(crop, frame[2:5, 3:8])


def test_exact_center_crop_rejects_crop_larger_than_source():
    with pytest.raises(ValueError, match="vượt quá khung nguồn"):
        _center_crop(np.zeros((24, 32, 3), dtype=np.uint8), 33, 24)


def test_exact_center_crop_accepts_one_pixel_custom_dimensions():
    frame = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)

    crop, rect = _center_crop(frame, 1, 1)

    assert rect == {"x": 3, "y": 2, "width": 1, "height": 1}
    assert np.array_equal(crop, frame[2:3, 3:4])


def test_health_response_contains_engine_capabilities():
    response = frameforge_engine.handle_request({"id": "test-1", "command": "health"})

    assert response["id"] == "test-1"
    assert response["ok"] is True
    assert response["data"]["engine"] == "FrameForge Engine"
    assert "health" in response["data"]["capabilities"]


def test_fhd_video_is_accepted_and_higher_resolution_video_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeCapture:
        def __init__(self, frame: np.ndarray):
            self.frame = frame

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FPS:
                return 30.0
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 1.0
            return 0.0

        def set(self, _property_id: int, _value: float) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            return True, self.frame

        def release(self) -> None:
            pass

    fhd_path = tmp_path / "fhd.mp4"
    oversized_path = tmp_path / "oversized.mp4"
    fhd_path.touch()
    oversized_path.touch()
    current_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: FakeCapture(current_frame))

    assert (probe_media(str(fhd_path))["width"], probe_media(str(fhd_path))["height"]) == (1920, 1080)

    for height, width in [(1081, 1920), (1080, 1921), (2160, 3840)]:
        current_frame = np.zeros((height, width, 3), dtype=np.uint8)
        inspection = inspect_sources([str(oversized_path)])
        assert inspection["sources"] == []
        assert "tối đa FHD" in inspection["errors"][0]["message"]

    with pytest.raises(ValueError, match="tối đa FHD"):
        preview_frame(str(oversized_path))
    extraction = extract_frames(
        [str(oversized_path)],
        str(tmp_path / "oversized-output"),
        {"crop_mode": "center", "output_width": 320, "output_height": 320},
    )
    assert extraction["failed_sources"] == 1
    assert "tối đa FHD" in extraction["errors"][0]["message"]


def test_unknown_command_returns_structured_error():
    response = frameforge_engine.handle_request({"id": 2, "command": "train_model"})

    assert response["ok"] is False
    assert response["error"]["code"] == "unknown_command"


def test_model_inspection_command_routes_to_onnx_inspector(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(frameforge_engine, "inspect_model", lambda path: {"path": path, "format": "ONNX"})

    response = frameforge_engine.handle_request({
        "id": "model-1", "command": "inspect_model", "arguments": {"path": "/models/detector.onnx"}
    })

    assert response["ok"] is True
    assert response["data"] == {"path": "/models/detector.onnx", "format": "ONNX"}


def test_center_crop_preview_command_routes_to_crop_preview(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        frameforge_engine,
        "preview_center_crop",
        lambda path, frame_index, width, height: {"path": path, "frame_index": frame_index, "width": width, "height": height},
    )

    response = frameforge_engine.handle_request({
        "id": "crop-preview-1",
        "command": "preview_center_crop",
        "arguments": {"path": "/media/clip.mp4", "frame_index": 6, "width": 320, "height": 256},
    })

    assert response["ok"] is True
    assert response["data"] == {"path": "/media/clip.mp4", "frame_index": 6, "width": 320, "height": 256}


def test_smart_folder_discovery_command_routes_scan_arguments(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        frameforge_engine,
        "scan_project_folder",
        lambda path, max_depth, max_files, ignored_dirs, cancel_file, progress: {
            "path": path,
            "max_depth": max_depth,
            "max_files": max_files,
            "ignored_dirs": ignored_dirs,
            "cancel_file": cancel_file,
            "progress_enabled": progress is not None,
        },
    )

    response = frameforge_engine.handle_request({
        "id": "discovery-1",
        "command": "scan_project_folder",
        "arguments": {"path": "/project", "max_depth": 4, "max_files": 1000, "ignored_dirs": ["cache"]},
    }, progress=lambda _event: None)

    assert response["ok"] is True
    assert response["data"] == {
        "path": "/project", "max_depth": 4, "max_files": 1000,
        "ignored_dirs": ["cache"], "cancel_file": None, "progress_enabled": True,
    }


def test_discovered_training_run_command_persists_scan_record(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        frameforge_engine,
        "register_discovered_run",
        lambda root, run: {"root": root, "run_id": run["experiment_id"], "checkpoint_trusted": False},
    )

    response = frameforge_engine.handle_request({
        "id": "discovered-run-1",
        "command": "register_discovered_run",
        "arguments": {"discovery_root": "/project", "run": {"experiment_id": "run-123"}},
    })

    assert response["ok"] is True
    assert response["data"] == {"root": "/project", "run_id": "run-123", "checkpoint_trusted": False}


def test_yolo_inference_command_routes_threshold_and_cancellation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        frameforge_engine,
        "predict_crops",
        lambda checkpoint, images, confidence, iou, classes, cancel_file, progress: {
            "checkpoint": checkpoint,
            "images": images,
            "confidence": confidence,
            "iou": iou,
            "classes": classes,
            "cancel_file": cancel_file,
            "progress_enabled": progress is not None,
        },
    )

    response = frameforge_engine.handle_request({
        "id": "infer-1",
        "command": "predict_crops",
        "arguments": {
            "checkpoint": "/models/best.pt",
            "images": ["/images/crop.jpg"],
            "confidence_threshold": 0.5,
            "iou_threshold": 0.6,
            "class_ids": [1],
            "cancel_file": "/app/inference/cancel/request.cancel",
        },
    }, progress=lambda _event: None)

    assert response["ok"] is True
    assert response["data"] == {
        "checkpoint": "/models/best.pt",
        "images": ["/images/crop.jpg"],
        "confidence": 0.5,
        "iou": 0.6,
        "classes": [1],
        "cancel_file": "/app/inference/cancel/request.cancel",
        "progress_enabled": True,
    }


def test_media_inspection_and_preview_read_real_image(tmp_path: Path):
    image_path = tmp_path / "ảnh thử.png"
    image = distinct_frame(2)
    assert cv2.imwrite(str(image_path), image)

    result = inspect_sources([str(image_path)])
    preview = preview_frame(str(image_path))

    assert result["sources"][0]["name"] == image_path.name
    assert result["sources"][0]["kind"] == "image"
    assert (result["sources"][0]["width"], result["sources"][0]["height"]) == (64, 48)
    assert preview["data_url"].startswith("data:image/jpeg;base64,")
    assert preview["image_fingerprint"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert (preview["width"], preview["height"]) == (64, 48)

    crop_preview = preview_center_crop(str(image_path), 0, 32, 24)
    encoded = crop_preview["data_url"].split(",", 1)[1]
    cropped = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert crop_preview["crop_rect"] == {"x": 16, "y": 12, "width": 32, "height": 24}
    assert (crop_preview["width"], crop_preview["height"]) == (32, 24)
    assert cropped.shape[:2] == (24, 32)


def test_video_sampling_filters_and_output_dimensions(tmp_path: Path):
    video_path = tmp_path / "sample.avi"
    output_path = tmp_path / "output"
    create_video(video_path, [distinct_frame(index) for index in range(8)])

    result = extract_frames(
        [str(video_path)],
        str(output_path),
        {
            "interval": 2,
            "interval_unit": "frames",
            "filter_duplicates": False,
            "filter_blurry": False,
            "crop_mode": "letterbox",
            "output_width": 32,
            "output_height": 32,
            "image_format": "jpg",
            "jpeg_quality": 90,
        },
    )

    assert result["candidate_frames"] == 4
    assert result["exported_images"] == 4
    assert result["files"][0]["thumbnail_data_url"].startswith("data:image/jpeg;base64,")
    assert len(list(output_path.glob("*.jpg"))) == 4
    for file in output_path.glob("*.jpg"):
        image = cv2.imread(str(file))
        assert image.shape[:2] == (32, 32)


def test_exact_center_crop_extraction_records_geometry_and_source_identity(tmp_path: Path):
    video_path = tmp_path / "center.avi"
    output_path = tmp_path / "center-output"
    create_video(video_path, [distinct_frame(2)])

    result = extract_frames(
        [str(video_path)],
        str(output_path),
        {
            "interval": 1,
            "interval_unit": "frames",
            "filter_duplicates": False,
            "filter_blurry": False,
            "crop_mode": "center",
            "output_width": 32,
            "output_height": 24,
            "image_format": "png",
        },
    )

    image = result["files"][0]
    cropped = cv2.imread(image["path"])
    assert cropped.shape[:2] == (24, 32)
    assert image["crop_rect"] == {"x": 16, "y": 12, "width": 32, "height": 24}
    assert len(image["source_id"]) == 64
    assert image["extraction_settings"]["crop_mode"] == "center"
    assert result["candidate_crops"] == result["accepted_crops"] == 1


def test_blur_and_duplicate_filters_measure_extracted_crops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    video_path = tmp_path / "filter-crops.avi"
    output_path = tmp_path / "filter-crops-output"
    create_video(video_path, [distinct_frame(index) for index in range(3)])
    measured_shapes = []

    def sharpness(crop: np.ndarray) -> float:
        measured_shapes.append(crop.shape[:2])
        return 1.0

    def perceptual_hash(crop: np.ndarray) -> int:
        measured_shapes.append(crop.shape[:2])
        return 0

    monkeypatch.setattr("frameforge.media._blur_score", sharpness)
    monkeypatch.setattr("frameforge.media._dhash", perceptual_hash)

    result = extract_frames(
        [str(video_path)],
        str(output_path),
        {
            "interval": 1,
            "interval_unit": "frames",
            "filter_duplicates": True,
            "duplicate_threshold": 0,
            "filter_blurry": True,
            "blur_threshold": 0,
            "crop_mode": "center",
            "output_width": 32,
            "output_height": 24,
            "image_format": "jpg",
        },
    )

    assert measured_shapes and set(measured_shapes) == {(24, 32)}
    assert result["candidate_frames"] == 3
    assert result["candidate_crops"] == 3
    assert result["duplicate_crops"] == 2
    assert result["exported_images"] == 1


def test_duplicate_and_blur_filters_reject_unusable_frames(tmp_path: Path):
    video_path = tmp_path / "still.avi"
    output_path = tmp_path / "filtered"
    still = np.full((48, 64, 3), 120, dtype=np.uint8)
    create_video(video_path, [still] * 5)

    result = extract_frames(
        [str(video_path)],
        str(output_path),
        {
            "interval": 1,
            "interval_unit": "frames",
            "filter_duplicates": True,
            "duplicate_threshold": 0,
            "filter_blurry": True,
            "blur_threshold": 10,
            "crop_mode": "letterbox",
            "output_width": 32,
            "output_height": 32,
            "image_format": "png",
            "jpeg_quality": 90,
        },
    )

    assert result["candidate_frames"] == 5
    assert result["blurry_crops"] == 5
    assert result["exported_images"] == 0


def test_max_frame_count_applies_across_multiple_sources(tmp_path: Path):
    first = tmp_path / "first.avi"
    second = tmp_path / "second.avi"
    output = tmp_path / "limited"
    create_video(first, [distinct_frame(index) for index in range(4)])
    create_video(second, [distinct_frame(index + 4) for index in range(4)])

    result = extract_frames(
        [str(first), str(second)],
        str(output),
        {
            "interval": 1,
            "interval_unit": "frames",
            "max_frames": 3,
            "filter_duplicates": False,
            "filter_blurry": False,
            "crop_mode": "letterbox",
            "output_width": 32,
            "output_height": 32,
            "image_format": "jpg",
        },
    )

    assert result["candidate_frames"] == 3
    assert result["exported_images"] == 3


def test_grid_crop_creates_overlapping_tiles(tmp_path: Path):
    video_path = tmp_path / "grid.avi"
    output_path = tmp_path / "tiles"
    create_video(video_path, [distinct_frame(3)])

    result = extract_frames(
        [str(video_path)],
        str(output_path),
        {
            "interval": 1,
            "interval_unit": "frames",
            "crop_mode": "grid",
            "overlap_x": 0.5,
            "overlap_y": 0.5,
            "output_width": 32,
            "output_height": 32,
            "image_format": "jpg",
            "jpeg_quality": 90,
        },
    )

    assert result["exported_images"] == 6
    assert len(list(output_path.glob("*.jpg"))) == 6


def test_sqlite_project_round_trips_settings_images_and_annotations(tmp_path: Path):
    project_path = tmp_path / "demo.ffproj"
    created = create_project(str(project_path), "Dự án thử")
    project = {
        "id": created["id"],
        "name": "Dự án thử",
        "sources": [{"path": "/tmp/source.mp4", "name": "source.mp4", "kind": "video", "width": 640, "height": 360}],
        "settings": {"interval": 5, "crop_mode": "letterbox"},
        "images": [{
            "path": "/tmp/frame.jpg", "source_path": "/tmp/source.mp4", "frame_index": 10,
            "source_id": "source-identity", "timestamp_seconds": 1.25,
            "crop_rect": {"x": 1120, "y": 560, "width": 320, "height": 320},
            "sharpness_score": 82.5, "extraction_settings": {"crop_mode": "center"},
            "width": 320, "height": 320, "status": "ready", "annotations": [{
                "id": "annotation-1", "class_id": 0, "x_min": 10, "y_min": 20,
                "x_max": 100, "y_max": 120, "confidence": 0.9, "source": "ai",
                "review_status": "unreviewed",
            }],
        }],
        "class_names": ["person"],
    }
    save_project(str(project_path), project)

    reopened = load_project(str(project_path))

    assert reopened["id"] == created["id"]
    assert reopened["settings"] == project["settings"]
    assert reopened["sources"][0]["path"] == "/tmp/source.mp4"
    assert reopened["images"][0]["annotations"][0]["class_id"] == 0
    assert reopened["images"][0]["source_id"] == "source-identity"
    assert reopened["images"][0]["timestamp_seconds"] == 1.25
    assert reopened["images"][0]["crop_rect"] == project["images"][0]["crop_rect"]
    assert reopened["images"][0]["sharpness_score"] == 82.5
    assert reopened["images"][0]["extraction_settings"] == {"crop_mode": "center"}
    assert reopened["class_names"] == ["person"]


def _create_annotation_project(tmp_path: Path) -> tuple[Path, str]:
    project_path = tmp_path / "labels.ffproj"
    created = create_project(str(project_path), "Labels")
    image_file = tmp_path / "frame.jpg"
    image_file.write_bytes(b"fixture image bytes")
    image_path = str(image_file)
    save_project(str(project_path), {
        "id": created["id"],
        "name": "Labels",
        "sources": [],
        "settings": {},
        "images": [{
            "path": image_path,
            "width": 1920,
            "height": 1080,
            "status": "ready",
            "annotations": [],
        }],
        "class_names": ["head", "body"],
    })
    return project_path, image_path


def _register_annotation_checkpoint(tmp_path: Path) -> Path:
    output_dir = tmp_path / "output" / "frameforge-training" / "queue" / "run"
    weights = output_dir / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "best.pt"
    checkpoint.write_bytes(b"verified checkpoint fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training._persist_run({
        "run_id": "run-annotation-fixture",
        "source": "frameforge_training",
        "checkpoint_trusted": True,
        "queue_id": "queue-annotation-fixture",
        "status": "completed",
        "model": "YOLO11n",
        "architecture": "yolo11n",
        "task": "detect",
        "class_names": ["head", "body"],
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest,
        "requested_imgsz": 64,
        "created_at": training._now(),
    })
    return checkpoint


def test_new_project_starts_with_stable_head_and_body_classes(tmp_path: Path):
    project_path = tmp_path / "defaults.ffproj"

    create_project(str(project_path), "Defaults")

    assert load_project(str(project_path))["class_names"] == ["head", "body"]


def test_save_annotations_command_persists_pixel_boxes_and_review_state(tmp_path: Path):
    project_path, image_path = _create_annotation_project(tmp_path)
    request = {
        "id": "save-label-1",
        "command": "save_annotations",
        "arguments": {
            "path": str(project_path),
            "image_path": image_path,
            "annotations": [{
                "id": "head-1",
                "class_id": 0,
                "x_min": 120.5,
                "y_min": 84,
                "x_max": 640,
                "y_max": 420.25,
                "confidence": None,
                "source": "manual",
                "review_status": "reviewed",
            }],
            "status": "reviewed",
            "image_fingerprint": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        },
    }

    response = frameforge_engine.handle_request(request)
    reopened = load_project(str(project_path))
    image = reopened["images"][0]

    assert response["ok"] is True
    assert response["data"]["annotation_count"] == 1
    assert image["status"] == "reviewed"
    assert image["annotations"][0]["class_id"] == 0
    assert image["annotations"][0]["x_min"] == 120.5
    assert image["annotations"][0]["y_max"] == 420.25
    assert image["annotations"][0]["review_status"] == "reviewed"


def test_save_annotations_rejects_invalid_boxes_without_replacing_saved_labels(tmp_path: Path):
    project_path, image_path = _create_annotation_project(tmp_path)
    saved = [{
        "id": "saved-box",
        "class_id": 0,
        "x_min": 10,
        "y_min": 20,
        "x_max": 100,
        "y_max": 120,
        "source": "manual",
        "review_status": "unreviewed",
    }]
    fingerprint = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    save_annotations(str(project_path), image_path, saved, "ready", fingerprint)

    for invalid in (
        {**saved[0], "x_max": 10},
        {**saved[0], "x_max": 1921},
        {**saved[0], "class_id": 2},
        {**saved[0], "x_min": float("nan")},
    ):
        with pytest.raises(ValueError):
            save_annotations(str(project_path), image_path, [invalid], "ready", fingerprint)
        reopened = load_project(str(project_path))
        assert reopened["images"][0]["annotations"][0]["id"] == "saved-box"


def test_save_annotations_detects_image_changes_after_preview(tmp_path: Path):
    project_path, image_path = _create_annotation_project(tmp_path)
    old_fingerprint = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    Path(image_path).write_bytes(b"replacement image bytes")

    with pytest.raises(ValueError, match="đã thay đổi"):
        save_annotations(str(project_path), image_path, [], "ready", old_fingerprint)


def test_ai_suggestions_keep_provenance_and_can_be_retried_after_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    project_path, image_path = _create_annotation_project(tmp_path)
    checkpoint = _register_annotation_checkpoint(tmp_path)
    fingerprint = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    prediction = {"class_id": 0, "bbox_xyxy": [10, 20, 120, 140], "confidence": 0.8}

    first = save_ai_suggestions(str(project_path), image_path, [prediction], fingerprint, str(checkpoint))
    ai_box = first["annotations"][0]
    assert ai_box["review_status"] == "needs_review"
    assert ai_box["source_model_id"] == "run-annotation-fixture"
    assert ai_box["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    manual_box = {
        "id": "manual-box", "class_id": 1, "x_min": 200, "y_min": 200,
        "x_max": 300, "y_max": 300, "source": "manual", "review_status": "unreviewed",
    }
    saved = save_annotations(str(project_path), image_path, [manual_box], "ready", fingerprint)
    assert {item["source"] for item in saved["annotations"]} == {"manual", "ai"}
    reopened = load_project(str(project_path))["images"][0]["annotations"]
    reopened_ai = next(item for item in reopened if item["source"] == "ai")
    assert reopened_ai["source_model_id"] == "run-annotation-fixture"

    with pytest.raises(ValueError, match="provenance"):
        save_annotations(
            str(project_path), image_path,
            [manual_box, {**reopened_ai, "checkpoint_sha256": "b" * 64}],
            "ready", fingerprint,
        )

    rejected = {**reopened_ai, "review_status": "rejected"}
    save_annotations(str(project_path), image_path, [manual_box, rejected], "ready", fingerprint)
    second = save_ai_suggestions(str(project_path), image_path, [prediction], fingerprint, str(checkpoint))
    assert second["annotation_count"] == 1
    reopened = load_project(str(project_path))["images"][0]["annotations"]
    assert {item["review_status"] for item in reopened if item["source"] == "ai"} == {"rejected", "needs_review"}


def test_empty_inference_candidate_requires_review_before_becoming_negative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    project_path, image_path = _create_annotation_project(tmp_path)
    checkpoint = _register_annotation_checkpoint(tmp_path)
    fingerprint = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()

    saved = save_ai_suggestions(str(project_path), image_path, [], fingerprint, str(checkpoint))
    candidate = saved["negative_candidate"]
    assert candidate["review_status"] == "needs_review"
    assert load_project(str(project_path))["images"][0]["status"] == "ready"
    with pytest.raises(ValueError, match="preview"):
        review_ai_negative_candidate(str(project_path), candidate["id"], "confirmed_negative", "0" * 64)

    rejected = review_ai_negative_candidate(str(project_path), candidate["id"], "rejected")
    assert rejected["negative_candidate"]["review_status"] == "rejected"
    retry = save_ai_suggestions(str(project_path), image_path, [], fingerprint, str(checkpoint))
    accepted = review_ai_negative_candidate(
        str(project_path), retry["negative_candidate"]["id"], "confirmed_negative", fingerprint,
    )
    reopened = load_project(str(project_path))["images"][0]
    assert accepted["status"] == "reviewed"
    assert reopened["status"] == "reviewed"
    assert reopened["negative_candidates"][-1]["review_status"] == "confirmed_negative"


def test_pending_negative_candidate_cannot_be_marked_reviewed_through_annotation_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    project_path, image_path = _create_annotation_project(tmp_path)
    checkpoint = _register_annotation_checkpoint(tmp_path)
    fingerprint = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    saved = save_ai_suggestions(str(project_path), image_path, [], fingerprint, str(checkpoint))

    with pytest.raises(ValueError, match="candidate AI"):
        save_annotations(str(project_path), image_path, [], "reviewed", fingerprint)

    assert load_project(str(project_path))["images"][0]["negative_candidates"][0]["id"] == saved["negative_candidate"]["id"]


@pytest.mark.parametrize("prediction", [
    {"class_id": 0, "bbox_xyxy": [True, 20, 120, 140], "confidence": 0.8},
    {"class_id": 0, "bbox_xyxy": [10, 20, 120, 140], "confidence": True},
    {"class_id": 0, "bbox_xyxy": ["10", 20, 120, 140], "confidence": 0.8},
    {"class_id": 0, "bbox_xyxy": [10, 20, 120, 140], "confidence": "0.8"},
])
def test_ai_suggestions_reject_non_numeric_box_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prediction: dict,
):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    project_path, image_path = _create_annotation_project(tmp_path)
    checkpoint = _register_annotation_checkpoint(tmp_path)
    fingerprint = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="phải là số"):
        save_ai_suggestions(str(project_path), image_path, [prediction], fingerprint, str(checkpoint))

    assert load_project(str(project_path))["images"][0]["annotations"] == []


def test_schema_v1_project_loads_and_migrates_on_save(tmp_path: Path):
    project_path = tmp_path / "legacy.ffproj"
    connection = sqlite3.connect(project_path)
    connection.executescript(
        """
        CREATE TABLE project_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE sources (path TEXT PRIMARY KEY, position INTEGER NOT NULL, kind TEXT NOT NULL, metadata_json TEXT NOT NULL);
        CREATE TABLE project_settings (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
        CREATE TABLE images (path TEXT PRIMARY KEY, source_path TEXT, frame_index INTEGER, width INTEGER NOT NULL, height INTEGER NOT NULL, status TEXT NOT NULL);
        CREATE TABLE classes (class_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE annotations (
            annotation_id TEXT PRIMARY KEY, image_path TEXT NOT NULL REFERENCES images(path) ON DELETE CASCADE,
            class_id INTEGER NOT NULL, x_min REAL NOT NULL, y_min REAL NOT NULL, x_max REAL NOT NULL, y_max REAL NOT NULL,
            confidence REAL, source TEXT NOT NULL, review_status TEXT NOT NULL
        );
        INSERT INTO project_meta VALUES('schema_version', '1');
        INSERT INTO project_meta VALUES('project_id', 'legacy-project');
        INSERT INTO project_meta VALUES('application', 'FrameForge');
        INSERT INTO project_meta VALUES('name', 'Legacy');
        INSERT INTO sources VALUES('/tmp/source.mp4', 0, 'video', '{"path":"/tmp/source.mp4","name":"source.mp4","kind":"video"}');
        INSERT INTO images VALUES('/tmp/old-frame.jpg', '/tmp/source.mp4', 2, 640, 360, 'ready');
        """
    )
    connection.commit()
    connection.close()

    loaded = load_project(str(project_path))
    loaded["images"][0].update({
        "source_id": "source-identity",
        "timestamp_seconds": 0.2,
        "crop_rect": {"x": 16, "y": 12, "width": 32, "height": 24},
        "extraction_settings": {"crop_mode": "center"},
    })
    save_project(str(project_path), loaded)

    reopened = load_project(str(project_path))
    assert reopened["images"][0]["path"] == "/tmp/old-frame.jpg"
    assert reopened["images"][0]["crop_rect"] == {"x": 16, "y": 12, "width": 32, "height": 24}
    connection = sqlite3.connect(project_path)
    version = connection.execute("SELECT value FROM project_meta WHERE key = 'schema_version'").fetchone()[0]
    connection.close()
    assert version == "4"


def test_save_project_preserves_an_unrelated_existing_file(tmp_path: Path):
    path = tmp_path / "notes.ffproj"
    original = "not a FrameForge project"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="không phải là dự án FrameForge"):
        save_project(str(path), {"name": "Không ghi đè"})

    assert path.read_text(encoding="utf-8") == original


def test_jsonl_process_returns_one_matching_reply(tmp_path: Path):
    request = {"id": "integration-1", "command": "health"}
    result = subprocess.run(
        [sys.executable, str(ENGINE_ROOT / "frameforge_engine.py")],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )

    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert replies == [frameforge_engine.handle_request(request)]


def test_jsonl_extraction_streams_progress_before_final_reply(tmp_path: Path):
    video_path = tmp_path / "protocol.avi"
    output_path = tmp_path / "protocol-output"
    create_video(video_path, [distinct_frame(index) for index in range(3)])
    request = {
        "id": "extract-1",
        "command": "extract_frames",
        "arguments": {
            "paths": [str(video_path)],
            "output_dir": str(output_path),
            "options": {
                "interval": 1,
                "interval_unit": "frames",
                "filter_duplicates": False,
                "filter_blurry": False,
                "crop_mode": "letterbox",
                "output_width": 32,
                "output_height": 32,
                "image_format": "jpg",
            },
        },
    }

    result = subprocess.run(
        [sys.executable, str(ENGINE_ROOT / "frameforge_engine.py")],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    replies = [json.loads(line) for line in result.stdout.splitlines()]

    assert replies[0]["id"] == "extract-1"
    assert "event" in replies[0]
    assert replies[-1]["ok"] is True
    assert replies[-1]["data"]["exported_images"] == 3
    assert len(list(output_path.glob("*.jpg"))) == 3


def test_dataset_validation_groups_by_source_and_excludes_unreviewed_negatives(tmp_path: Path):
    project_path, _ = create_dataset_project(tmp_path)

    report = validate_dataset(
        str(project_path), "source", train_ratio=0.5, val_ratio=0.5, test_ratio=0, seed=23,
    )

    assert report["valid"] is True
    assert report["image_count"] == 4
    assert report["included_image_count"] == 3
    assert report["positive_image_count"] == 2
    assert report["reviewed_negative_count"] == 1
    assert report["candidate_negative_excluded_count"] == 1
    assert report["annotation_count"] == 2
    assert report["split_counts"]["train"] > 0
    assert report["split_counts"]["val"] > 0
    assert report["split_counts"]["test"] == 0
    assert any("chưa review" in warning for warning in report["warnings"])
    assert report["duplicate_basename_groups"] == 1


def test_dataset_export_excludes_unreviewed_manual_annotations(tmp_path: Path):
    project_path, image_paths = create_dataset_project(tmp_path)
    project = load_project(str(project_path))
    unreviewed = next(item for item in project["images"] if item["path"] == str(image_paths["a_positive"]))
    unreviewed["status"] = "ready"
    unreviewed["annotations"][0]["review_status"] = "unreviewed"
    save_project(str(project_path), project)

    report = validate_dataset(str(project_path), "source", 0.5, 0.5, 0, 23)

    assert report["valid"] is True
    assert report["included_image_count"] == 2
    assert report["positive_image_count"] == 1
    assert report["unreviewed_annotation_excluded_count"] == 1

    output_root = tmp_path / "review-gated-exports"
    output_root.mkdir()
    result = export_yolo_dataset(str(project_path), str(output_root), "source", 0.5, 0.5, 0, 23)
    manifest = json.loads((Path(result["path"]) / "manifest.json").read_text(encoding="utf-8"))
    assert all(item["source_image"] != str(image_paths["a_positive"]) for item in manifest["images"])


def test_dataset_lineage_excludes_unreviewed_training_annotations_from_parent(tmp_path: Path):
    project_path = create_reviewed_lineage_project(tmp_path)
    output_root = tmp_path / "parent-exports"
    output_root.mkdir()
    base = export_yolo_dataset(
        str(project_path), str(output_root), "source", 0.5, 0.25, 0.25, 23,
        freeze_evaluation_set=True,
    )
    manifest_path = Path(base["path"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_positive = next(
        item for item in manifest["images"] if item["split"] == "train" and item["annotation_count"] > 0
    )
    train_positive["annotation_provenance"][0]["review_status"] = "unreviewed"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    report = validate_dataset(
        str(project_path), "source", 0.8, 0.2, 0, 23,
        parent_manifest_path=str(manifest_path), replay_strategy="all",
    )

    assert report["valid"] is True
    assert report["unreviewed_parent_train_excluded_count"] == 1
    assert any("ảnh train trong parent" in warning for warning in report["warnings"])


def test_dataset_lineage_rejects_unreviewed_annotations_in_frozen_parent(tmp_path: Path):
    project_path = create_reviewed_lineage_project(tmp_path)
    output_root = tmp_path / "parent-exports"
    output_root.mkdir()
    base = export_yolo_dataset(
        str(project_path), str(output_root), "source", 0.5, 0.25, 0.25, 23,
        freeze_evaluation_set=True,
    )
    manifest_path = Path(base["path"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation_positive = next(
        item for item in manifest["images"]
        if item["split"] in {"val", "test"} and item["annotation_count"] > 0
    )
    evaluation_positive["annotation_provenance"][0]["review_status"] = "unreviewed"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    report = validate_dataset(
        str(project_path), "source", 0.8, 0.2, 0, 23,
        parent_manifest_path=str(manifest_path), replay_strategy="all",
    )

    assert report["valid"] is False
    assert any("Fixed validation/test" in error for error in report["errors"])


def test_dataset_export_excludes_pending_ai_as_box_and_reviewed_negative(tmp_path: Path):
    project_path, image_paths = create_dataset_project(tmp_path)
    project = load_project(str(project_path))
    positive = next(item for item in project["images"] if item["path"] == str(image_paths["a_positive"]))
    positive["annotations"].append({
        "id": "reviewed-ai", "class_id": 1, "x_min": 32, "y_min": 8,
        "x_max": 55, "y_max": 40, "confidence": 0.95, "source": "ai",
        "review_status": "reviewed", "source_model_id": "run-fixture",
        "checkpoint_sha256": "b" * 64,
    })
    image = next(item for item in project["images"] if item["path"] == str(image_paths["a_negative"]))
    image["annotations"] = [{
        "id": "pending-ai", "class_id": 0, "x_min": 4, "y_min": 4,
        "x_max": 30, "y_max": 30, "confidence": 0.9, "source": "ai",
        "review_status": "needs_review", "source_model_id": "run-fixture",
        "checkpoint_sha256": "a" * 64,
    }]
    save_project(str(project_path), project)

    report = validate_dataset(str(project_path), "source", 0.5, 0.5, 0, 23)
    assert report["valid"] is True
    assert report["pending_suggestion_count"] == 1
    assert report["reviewed_negative_count"] == 0
    assert report["candidate_negative_excluded_count"] == 2
    assert report["included_image_count"] == 2
    assert report["annotation_count"] == 3

    output_root = tmp_path / "exports"
    output_root.mkdir()
    result = export_yolo_dataset(str(project_path), str(output_root), "source", 0.5, 0.5, 0, 23)
    manifest = json.loads((Path(result["path"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["pending_suggestion_count"] == 1
    assert all(item["source_image"] != str(image_paths["a_negative"]) for item in manifest["images"])
    exported_positive = next(item for item in manifest["images"] if item["source_image"] == str(image_paths["a_positive"]))
    ai_provenance = next(item for item in exported_positive["annotation_provenance"] if item["source"] == "ai")
    assert ai_provenance["review_status"] == "reviewed"
    assert ai_provenance["source_model_id"] == "run-fixture"
    assert ai_provenance["checkpoint_sha256"] == "b" * 64


def test_dataset_requires_review_for_ai_candidate_negative_and_records_provenance(tmp_path: Path):
    project_path, image_paths = create_dataset_project(tmp_path)
    project = load_project(str(project_path))
    candidate_image = next(item for item in project["images"] if item["path"] == str(image_paths["a_negative"]))
    fingerprint = hashlib.sha256(image_paths["a_negative"].read_bytes()).hexdigest()
    candidate_image["negative_candidates"] = [{
        "id": "candidate-negative-1",
        "source_model_id": "run-negative-fixture",
        "checkpoint_sha256": "c" * 64,
        "image_fingerprint": fingerprint,
        "review_status": "needs_review",
        "created_at": "2026-10-14T00:00:00Z",
        "reviewed_at": None,
    }]
    candidate_image["status"] = "reviewed"
    save_project(str(project_path), project)

    pending_report = validate_dataset(str(project_path), "source", 0.5, 0.5, 0, 23)
    assert pending_report["pending_negative_candidate_count"] == 1
    assert pending_report["included_image_count"] == 2

    review_ai_negative_candidate(
        str(project_path), "candidate-negative-1", "confirmed_negative", fingerprint,
    )
    confirmed_report = validate_dataset(str(project_path), "source", 0.5, 0.5, 0, 23)
    assert confirmed_report["reviewed_negative_count"] == 1
    output_root = tmp_path / "negative-exports"
    output_root.mkdir()
    result = export_yolo_dataset(str(project_path), str(output_root), "source", 0.5, 0.5, 0, 23)
    manifest = json.loads((Path(result["path"]) / "manifest.json").read_text(encoding="utf-8"))
    exported = next(item for item in manifest["images"] if item["source_image"] == str(image_paths["a_negative"]))
    assert exported["negative_candidate_provenance"][0]["review_status"] == "confirmed_negative"
    assert exported["negative_candidate_provenance"][0]["checkpoint_sha256"] == "c" * 64


def test_yolo_export_writes_normalized_labels_manifest_and_unique_version(tmp_path: Path):
    project_path, image_paths = create_dataset_project(tmp_path)
    output_root = tmp_path / "exports"
    output_root.mkdir()

    result = export_yolo_dataset(
        str(project_path), str(output_root), "source", 0.5, 0.5, 0, 23,
    )
    root = Path(result["path"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    data_config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))

    assert result["image_count"] == 3
    assert data_config["train"] == "images/train"
    assert data_config["val"] == "images/val"
    assert data_config["names"] == {0: "head", 1: "body"}
    assert len(manifest["images"]) == 3
    assert all(
        item["split"] == "train" for item in manifest["images"]
        if item["source_id"] == "source-a"
    ) or all(
        item["split"] == "val" for item in manifest["images"]
        if item["source_id"] == "source-a"
    )
    assert not any(item["source_image"] == str(image_paths["b_candidate"]) for item in manifest["images"])
    positive = next(item for item in manifest["images"] if item["source_image"] == str(image_paths["a_positive"]))
    label_text = (root / positive["exported_label"]).read_text(encoding="utf-8")
    assert label_text == "0 0.25000000 0.25000000 0.25000000 0.33333333\n"
    negative = next(item for item in manifest["images"] if item["source_image"] == str(image_paths["a_negative"]))
    assert (root / negative["exported_label"]).read_text(encoding="utf-8") == ""
    assert (root / positive["exported_image"]).read_bytes() == image_paths["a_positive"].read_bytes()

    next_version = export_yolo_dataset(
        str(project_path), str(output_root), "source", 0.5, 0.5, 0, 23,
    )
    assert next_version["path"] != result["path"]
    assert len(list(output_root.glob("frameforge_dataset_*"))) == 2


def test_frozen_validation_and_test_require_reviewed_images(tmp_path: Path):
    project_path = create_reviewed_lineage_project(tmp_path, reviewed=False)
    report = validate_dataset(
        str(project_path), "source", 0.5, 0.25, 0.25, 23, freeze_evaluation_set=True,
    )

    assert report["valid"] is False
    assert any("mọi ảnh đã được người dùng review" in error for error in report["errors"])


def test_frozen_validation_and_test_require_nonempty_group_splits(tmp_path: Path):
    project_path, _ = create_dataset_project(tmp_path)
    project = load_project(str(project_path))
    for image in project["images"]:
        image["status"] = "reviewed"
    save_project(str(project_path), project)

    report = validate_dataset(
        str(project_path), "source", 0.5, 0.25, 0.25, 23, freeze_evaluation_set=True,
    )

    assert report["valid"] is False
    assert any("validation và test sau khi chia nhóm" in error for error in report["errors"])


def test_dataset_lineage_reuses_frozen_splits_and_supports_new_only_replay(tmp_path: Path):
    project_path = create_reviewed_lineage_project(tmp_path)
    output_root = tmp_path / "lineage-exports"
    output_root.mkdir()
    base = export_yolo_dataset(
        str(project_path), str(output_root), "source", 0.5, 0.25, 0.25, 23,
        freeze_evaluation_set=True,
    )
    base_root = Path(base["path"])
    base_manifest = json.loads((base_root / "manifest.json").read_text(encoding="utf-8"))
    assert base["version"] == 1
    assert base_manifest["fixed_validation_set"]["enabled"] is True
    assert base_manifest["frozen_test_set"]["enabled"] is True
    assert len(base_manifest["annotation_revision"]) == 64
    assert all(len(item["label_sha256"]) == 64 and item["group_key"] for item in base_manifest["images"])

    project = load_project(str(project_path))
    frozen_test_source = next(item["source_image"] for item in base_manifest["images"] if item["split"] == "test")
    changed_test_image = next(item for item in project["images"] if item["path"] == frozen_test_source)
    changed_test_image["status"] = "ready"
    changed_test_image["annotations"] = []
    new_source = tmp_path / "source-e.mp4"
    new_image = tmp_path / "images" / "source-e" / "new.png"
    new_image.parent.mkdir(parents=True)
    cv2.imwrite(str(new_image), np.full((48, 64, 3), 235, dtype=np.uint8))
    project["sources"].append({"path": str(new_source), "kind": "video", "fps": 10.0})
    project["images"].append({
        "path": str(new_image), "source_path": str(new_source), "source_id": "source-e",
        "frame_index": 1, "timestamp_seconds": 0.1, "width": 64, "height": 48,
        "status": "reviewed", "annotations": [{
            "id": "source-e-box", "class_id": 0,
            "x_min": 10, "y_min": 4, "x_max": 34, "y_max": 35,
            "review_status": "reviewed",
        }],
    })
    save_project(str(project_path), project)

    report = validate_dataset(
        str(project_path), "source", 0.8, 0.2, 0.0, 23,
        parent_manifest_path=str(base_root / "manifest.json"), replay_strategy="new",
    )
    assert report["valid"] is True
    assert report["new_image_count"] == 1
    assert report["replayed_image_count"] == 0
    assert report["fixed_validation_count"] == base["split_counts"]["val"]
    assert report["frozen_test_count"] == base["split_counts"]["test"]

    child = export_yolo_dataset(
        str(project_path), str(output_root), "source", 0.8, 0.2, 0.0, 23,
        parent_manifest_path=str(base_root / "manifest.json"), replay_strategy="new",
    )
    child_manifest = json.loads((Path(child["path"]) / "manifest.json").read_text(encoding="utf-8"))
    assert child["version"] == 2
    assert child["parent_dataset_id"] == base["dataset_id"]
    assert child_manifest["parent_manifest_sha256"] == hashlib.sha256((base_root / "manifest.json").read_bytes()).hexdigest()
    assert child_manifest["replay_strategy"] == "new"
    for split in ("val", "test"):
        before = sorted((item["sha256"], item["label_sha256"]) for item in base_manifest["images"] if item["split"] == split)
        after = sorted((item["sha256"], item["label_sha256"]) for item in child_manifest["images"] if item["split"] == split)
        assert after == before
    parent_train_digests = {item["sha256"] for item in base_manifest["images"] if item["split"] == "train"}
    child_train_digests = {item["sha256"] for item in child_manifest["images"] if item["split"] == "train"}
    assert parent_train_digests.isdisjoint(child_train_digests)
    assert child_train_digests == {hashlib.sha256(new_image.read_bytes()).hexdigest()}

    all_replay = validate_dataset(
        str(project_path), "source", 0.8, 0.2, 0.0, 23,
        parent_manifest_path=str(base_root / "manifest.json"), replay_strategy="all",
    )
    assert all_replay["valid"] is True
    assert all_replay["replayed_image_count"] == base["split_counts"]["train"]


def test_balanced_replay_selects_deterministic_class_groups():
    samples = [
        {"group_key": "a", "annotations": [{"class_id": 0}]},
        {"group_key": "b", "annotations": [{"class_id": 0}]},
        {"group_key": "c", "annotations": [{"class_id": 0}]},
        {"group_key": "d", "annotations": [{"class_id": 1}]},
        {"group_key": "negative", "annotations": []},
    ]

    first = _balanced_replay_groups(samples, 23)
    second = _balanced_replay_groups(samples, 23)

    assert first == second
    assert "d" in first
    assert len(first & {"a", "b", "c"}) == 1
    assert len(first & {"negative"}) == 1


def test_dataset_validation_blocks_invalid_class_ids_and_corrupt_images(tmp_path: Path):
    invalid_class_project, _ = create_dataset_project(tmp_path / "invalid-class", invalid_class=True)
    invalid_class_report = validate_dataset(
        str(invalid_class_project), "source", train_ratio=0.5, val_ratio=0.5, test_ratio=0,
    )
    assert invalid_class_report["valid"] is False
    assert any("Class ID" in error for error in invalid_class_report["errors"])

    corrupt_project, corrupt_paths = create_dataset_project(tmp_path / "corrupt")
    corrupt_paths["a_positive"].write_bytes(b"not an image")
    corrupt_report = validate_dataset(
        str(corrupt_project), "source", train_ratio=0.5, val_ratio=0.5, test_ratio=0,
    )
    assert corrupt_report["valid"] is False
    assert any("hỏng hoặc không đọc được" in error for error in corrupt_report["errors"])
