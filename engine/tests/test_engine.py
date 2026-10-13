import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

import frameforge_engine  # noqa: E402
from frameforge.media import _RecentHashes, _letterbox, extract_frames, inspect_sources, preview_frame
from frameforge.project import create_project, load_project, save_project


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
    assert (preview["width"], preview["height"]) == (64, 48)


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
    assert result["blurry_frames"] == 5
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
            "width": 416, "height": 416, "status": "ready", "annotations": [{
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
    assert reopened["class_names"] == ["person"]


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
