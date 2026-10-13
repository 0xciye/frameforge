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

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

import frameforge_engine  # noqa: E402
from frameforge.media import _RecentHashes, _center_crop, _letterbox, extract_frames, inspect_sources, preview_center_crop, preview_frame
from frameforge.project import create_project, load_project, save_annotations, save_project


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
    assert version == "2"


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
