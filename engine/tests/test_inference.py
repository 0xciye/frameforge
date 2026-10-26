import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

from frameforge import inference, models
from frameforge.project import create_project, load_project, save_project


def register_checkpoint(tmp_path: Path) -> Path:
    output_dir = tmp_path / "output" / "imported" / "run"
    weights = output_dir / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "best.pt"
    checkpoint.write_bytes(b"verified checkpoint fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    models._persist_run({
        "run_id": "run-inference-fixture",
        "source": "frameforge_import",
        "checkpoint_trusted": True,
        "queue_id": "queue-inference-fixture",
        "status": "completed",
        "model": "YOLO11n",
        "architecture": "yolo11n",
        "task": "detect",
        "class_names": ["head", "body"],
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest,
        "requested_imgsz": 64,
        "created_at": models._now(),
    })
    return checkpoint


def create_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def install_fake_yolo(monkeypatch: pytest.MonkeyPatch, names=None, prediction_boxes=None):
    xyxy, confidence, classes = prediction_boxes or (
        [[2, 2, 10, 12], [3, 4, 12, 14], [5, 6, 20, 22]],
        [0.5, 0.499, 0.8],
        [0, 0, 1],
    )

    class FakeBoxes:
        def __init__(self):
            self.xyxy = np.array(xyxy, dtype=np.float32)
            self.conf = np.array(confidence, dtype=np.float32)
            self.cls = np.array(classes, dtype=np.float32)

    class FakeYOLO:
        instances = []

        def __init__(self, source, task):
            self.source = source
            self.task = task
            self.names = names or {0: "head", 1: "body"}
            self.calls = []
            self.__class__.instances.append(self)

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            return [SimpleNamespace(orig_shape=(24, 32), boxes=FakeBoxes()) for _ in kwargs["source"]]

    module = ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setattr(inference, "inference_capabilities", lambda: {
        "available": True, "preferred_device": "cpu",
    })
    return FakeYOLO


def test_predict_crops_applies_inclusive_threshold_class_filter_and_pixel_boxes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = register_checkpoint(tmp_path)
    image = create_image(tmp_path / "crop.jpg")
    FakeYOLO = install_fake_yolo(monkeypatch)
    progress = []

    result = inference.predict_crops(
        str(checkpoint), [str(image)], confidence_threshold=0.5, iou_threshold=0.7,
        class_ids=[0], progress=progress.append,
    )

    assert result["status"] == "completed"
    assert result["device_actual"] == "cpu"
    assert result["results"][0]["width"] == 32
    assert result["results"][0]["height"] == 24
    assert result["results"][0]["predictions"] == [{
        "class_id": 0,
        "class_name": "head",
        "confidence": pytest.approx(0.5),
        "bbox_xyxy": pytest.approx([2, 2, 10, 12]),
    }]
    call = FakeYOLO.instances[0].calls[0]
    assert call["conf"] < 0.5
    assert call["iou"] == 0.7
    assert call["classes"] == [0]
    assert progress[-1]["stage"] == "inference_completed"


def test_predict_crops_skips_boxes_clipped_by_crop_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = register_checkpoint(tmp_path)
    image = create_image(tmp_path / "crop.jpg")
    install_fake_yolo(monkeypatch, prediction_boxes=(
        [[0, 2, 10, 12], [3, 4, 12, 14]],
        [0.9, 0.8],
        [0, 1],
    ))

    result = inference.predict_crops(str(checkpoint), [str(image)])

    assert result["skipped_large_objects"] == 1
    assert len(result["results"][0]["predictions"]) == 1
    assert result["results"][0]["predictions"][0]["class_id"] == 1
    assert result["results"][0]["skipped_large_objects"] == 1


def test_predict_crops_rejects_unregistered_and_changed_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    image = create_image(tmp_path / "crop.jpg")
    unregistered = tmp_path / "unregistered.pt"
    unregistered.write_bytes(b"arbitrary model")
    FakeYOLO = install_fake_yolo(monkeypatch)

    with pytest.raises(ValueError, match="registry FrameForge"):
        inference.predict_crops(str(unregistered), [str(image)])

    checkpoint = register_checkpoint(tmp_path)
    checkpoint.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="checksum đã thay đổi"):
        inference.predict_crops(str(checkpoint), [str(image)])

    assert FakeYOLO.instances == []


def test_predict_crops_rejects_checkpoint_class_schema_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = register_checkpoint(tmp_path)
    image = create_image(tmp_path / "crop.jpg")
    FakeYOLO = install_fake_yolo(monkeypatch, {0: "person", 1: "car"})

    with pytest.raises(ValueError, match="Schema class"):
        inference.predict_crops(str(checkpoint), [str(image)])

    assert FakeYOLO.instances[0].calls == []


def make_project(tmp_path: Path, count: int) -> tuple[Path, list[Path]]:
    project_path = tmp_path / "batch.ffproj"
    created = create_project(str(project_path), "Batch")
    images = [create_image(tmp_path / "images" / f"crop-{index:04}.jpg") for index in range(count)]
    save_project(str(project_path), {
        "id": created["id"], "name": "Batch", "sources": [], "settings": {},
        "class_names": ["head", "body"],
        "images": [{
            "path": str(path), "width": 32, "height": 24, "status": "ready", "annotations": [],
        } for path in images],
    })
    return project_path, images


def test_auto_label_project_loads_model_once_and_saves_large_projects_in_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = register_checkpoint(tmp_path)
    project_path, image_paths = make_project(tmp_path, 130)
    FakeYOLO = install_fake_yolo(monkeypatch)
    progress = []

    result = inference.auto_label_project(str(project_path), str(checkpoint), progress=progress.append)

    assert result["status"] == "completed"
    assert result["processed_images"] == len(image_paths)
    assert result["images_with_detections"] == len(image_paths)
    assert result["annotation_count"] == 3 * len(image_paths)
    assert len(FakeYOLO.instances) == 1
    assert len(FakeYOLO.instances[0].calls) == 9
    assert sum(event["stage"] == "project_inference_saved" for event in progress) == 3
    loaded = load_project(str(project_path))
    assert all(len(image["annotations"]) == 3 for image in loaded["images"])
    assert all(annotation["review_status"] == "needs_review" for image in loaded["images"] for annotation in image["annotations"])


def test_auto_label_project_rechecks_fingerprints_before_saving(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = register_checkpoint(tmp_path)
    project_path, image_paths = make_project(tmp_path, 3)
    FakeYOLO = install_fake_yolo(monkeypatch)
    predict = FakeYOLO.instances
    original_predict = FakeYOLO.predict
    changed = {"done": False}

    def mutate_during_inference(self, **kwargs):
        if not changed["done"]:
            Path(kwargs["source"][0]).write_bytes(b"changed during inference")
            changed["done"] = True
        return original_predict(self, **kwargs)

    monkeypatch.setattr(FakeYOLO, "predict", mutate_during_inference)

    result = inference.auto_label_project(str(project_path), str(checkpoint))

    assert result["processed_images"] == 2
    assert result["skipped_changed"] == 1
    assert len(predict) == 1
    loaded = load_project(str(project_path))
    by_path = {image["path"]: image for image in loaded["images"]}
    assert by_path[str(image_paths[0])]["annotations"] == []
    assert all(by_path[str(path)]["annotations"] for path in image_paths[1:])


def test_auto_label_project_does_not_turn_skipped_large_objects_into_negatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = register_checkpoint(tmp_path)
    project_path, image_paths = make_project(tmp_path, 1)
    install_fake_yolo(monkeypatch, prediction_boxes=(
        [[0, 2, 32, 20]],
        [0.9],
        [0],
    ))

    result = inference.auto_label_project(str(project_path), str(checkpoint))

    assert result["skipped_large_objects"] == 1
    assert result["candidate_negative_count"] == 0
    loaded = load_project(str(project_path))
    assert loaded["images"][0]["path"] == str(image_paths[0])
    assert loaded["images"][0]["annotations"] == []
    assert loaded["images"][0]["negative_candidates"] == []


def test_auto_label_project_does_not_rerun_edge_only_images_with_same_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = register_checkpoint(tmp_path)
    project_path, image_paths = make_project(tmp_path, 1)
    FakeYOLO = install_fake_yolo(monkeypatch, prediction_boxes=([[0, 2, 32, 20]], [0.9], [0]))

    first = inference.auto_label_project(str(project_path), str(checkpoint))
    second = inference.auto_label_project(str(project_path), str(checkpoint))

    assert first["skipped_large_objects"] == 1
    assert second["processed_images"] == 0
    assert second["skipped_attempted"] == 1
    assert len(FakeYOLO.instances) == 1
    attempts = load_project(str(project_path))["images"][0]["inference_attempts"]
    assert [attempt["skipped_large_objects"] for attempt in attempts] == [1]

    image_paths[0].write_bytes(cv2.imencode(".jpg", np.full((24, 32, 3), 90, dtype=np.uint8))[1].tobytes())
    third = inference.auto_label_project(str(project_path), str(checkpoint))
    assert third["skipped_attempted"] == 0
    assert third["processed_images"] == 1


def test_inference_uses_registered_checkpoint_imgsz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = register_checkpoint(tmp_path)
    image = create_image(tmp_path / "crop.jpg")
    project_path, _ = make_project(tmp_path, 1)
    FakeYOLO = install_fake_yolo(monkeypatch)

    inference.predict_crops(str(checkpoint), [str(image)])
    inference.auto_label_project(str(project_path), str(checkpoint))

    calls = [call for instance in FakeYOLO.instances for call in instance.calls]
    assert calls and all(call["imgsz"] == 64 for call in calls)


def test_duplicate_boxes_of_one_object_keep_the_most_confident():
    from frameforge.inference import _drop_duplicate_boxes

    def box(class_id, confidence, xyxy):
        return {"class_id": class_id, "class_name": str(class_id), "confidence": confidence, "bbox_xyxy": xyxy}

    tall = box(1, 0.44, [10.0, 10.0, 30.0, 73.0])     # 20 x 63 body
    short = box(1, 0.61, [10.0, 10.0, 29.0, 46.0])    # 19 x 36 body on the same person (IoU ~0.54, inside ~1.0)
    head = box(0, 0.55, [14.0, 4.0, 20.0, 11.0])      # other class: never merged with a body
    other_body = box(1, 0.50, [60.0, 10.0, 80.0, 73.0])  # a second person
    kept = _drop_duplicate_boxes([tall, short, head, other_body])

    assert [item["confidence"] for item in kept] == [0.61, 0.55, 0.50]
    assert tall not in kept


def test_touching_boxes_of_two_objects_both_stay():
    from frameforge.inference import _drop_duplicate_boxes

    left = {"class_id": 0, "class_name": "head", "confidence": 0.9, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}
    right = {"class_id": 0, "class_name": "head", "confidence": 0.8, "bbox_xyxy": [7.0, 0.0, 17.0, 10.0]}  # IoU 0.18
    assert len(_drop_duplicate_boxes([left, right])) == 2


def test_suggest_again_replaces_every_existing_box(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from frameforge.project import save_ai_suggestions

    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = register_checkpoint(tmp_path)
    project_path, image_paths = make_project(tmp_path, 2)
    project = load_project(str(project_path))
    project["images"][0]["status"] = "reviewed"
    project["images"][0]["annotations"] = [{
        "id": "manual-1", "class_id": 1, "x_min": 1, "y_min": 1, "x_max": 9, "y_max": 9,
        "source": "manual", "review_status": "reviewed",
    }]
    save_project(str(project_path), project)
    install_fake_yolo(monkeypatch)

    # Without replace the labeled image is left alone.
    first = inference.auto_label_project(str(project_path), str(checkpoint))
    assert first["processed_images"] == 1
    labeled = load_project(str(project_path))["images"][0]
    assert [item["id"] for item in labeled["annotations"]] == ["manual-1"]

    again = inference.auto_label_project(str(project_path), str(checkpoint), replace=True)
    assert again["processed_images"] == 2
    images = load_project(str(project_path))["images"]
    for image in images:
        assert len(image["annotations"]) == 3
        assert all(item["source"] == "ai" and item["review_status"] == "needs_review" for item in image["annotations"])
        assert image["status"] == "ready"

    # One image: the same checkpoint again is refused, unless it replaces the boxes.
    fingerprint = hashlib.sha256(image_paths[1].read_bytes()).hexdigest()
    prediction = [{"class_id": 0, "confidence": 0.7, "bbox_xyxy": [4, 4, 14, 14]}]
    with pytest.raises(ValueError):
        save_ai_suggestions(str(project_path), str(image_paths[1]), prediction, fingerprint, str(checkpoint))
    saved = save_ai_suggestions(str(project_path), str(image_paths[1]), prediction, fingerprint, str(checkpoint), replace=True)
    assert saved["replaced"] is True
    image = next(item for item in load_project(str(project_path))["images"] if item["path"] == str(image_paths[1]))
    assert [item["x_min"] for item in image["annotations"]] == [4.0]


def test_delete_model_removes_registry_entry_and_only_its_own_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    outside = register_checkpoint(tmp_path)  # output_dir outside FrameForge's model folder
    inside_dir = models._model_root() / "imported" / "run-inside"
    (inside_dir / "weights").mkdir(parents=True)
    inside = inside_dir / "weights" / "best.pt"
    inside.write_bytes(b"inside checkpoint")
    models._persist_run({
        "run_id": "run-inside", "source": "frameforge_import", "checkpoint_trusted": True, "status": "completed",
        "class_names": ["head", "body"], "output_dir": str(inside_dir), "checkpoint_path": str(inside),
        "checkpoint_sha256": hashlib.sha256(inside.read_bytes()).hexdigest(), "requested_imgsz": 64,
        "created_at": models._now(),
    })

    models.delete_model("run-inside")
    models.delete_model("run-inference-fixture")

    assert not inside_dir.exists()
    assert outside.exists()
    assert models.list_models()["runs"] == []
    with pytest.raises(ValueError):
        models.delete_model("run-inside")


def install_brightness_yolo(monkeypatch: pytest.MonkeyPatch):
    """Fake model that finds one head in bright crops and nothing in dark ones."""

    class FakeBoxes:
        def __init__(self, found: bool):
            self.xyxy = np.array([[4, 4, 20, 18]] if found else [], dtype=np.float32).reshape(-1, 4)
            self.conf = np.array([0.9] if found else [], dtype=np.float32)
            self.cls = np.array([0] if found else [], dtype=np.float32)

    class FakeYOLO:
        instances = []

        def __init__(self, source, task):
            self.names = {0: "head", 1: "body"}
            self.calls = []
            self.__class__.instances.append(self)

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            return [
                SimpleNamespace(orig_shape=frame.shape[:2], boxes=FakeBoxes(float(frame.mean()) > 100))
                for frame in kwargs["source"]
            ]

    module = ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setattr(inference, "inference_capabilities", lambda: {"available": True, "preferred_device": "cpu"})
    return FakeYOLO


def extraction_sources(tmp_path: Path) -> list[str]:
    sources = []
    for name, value in (("bright", 200), ("dark", 10)):
        path = tmp_path / "sources" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), np.full((48, 64, 3), value, dtype=np.uint8))
        sources.append(str(path))
    return sources


def extraction_options(checkpoint: Path, keep_empty: bool = False, class_names=("head", "body")) -> dict:
    return {
        "crop_mode": "center", "output_width": 32, "output_height": 24, "image_format": "png",
        "detection": {
            "checkpoint": str(checkpoint), "confidence_threshold": 0.35,
            "class_names": list(class_names), "keep_empty": keep_empty,
        },
    }


def test_extraction_keeps_only_crops_with_objects_and_returns_their_boxes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from frameforge.media import extract_frames

    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = register_checkpoint(tmp_path)
    FakeYOLO = install_brightness_yolo(monkeypatch)

    result = extract_frames(extraction_sources(tmp_path), str(tmp_path / "out"), extraction_options(checkpoint))

    assert len(FakeYOLO.instances) == 1
    assert result["no_object_crops"] == 1
    assert result["detected_images"] == 1
    assert [Path(image["path"]).name for image in result["images"]] == ["bright_frame_000000_center.png"]
    assert not (tmp_path / "out" / "dark_frame_000000_center.png").exists()
    detected = result["images"][0]["ai_detection"]
    assert detected["image_fingerprint"] == hashlib.sha256(Path(result["images"][0]["path"]).read_bytes()).hexdigest()
    assert detected["predictions"] == [{"class_id": 0, "confidence": pytest.approx(0.9), "bbox_xyxy": pytest.approx([4, 4, 20, 18])}]
    assert result["detection"]["source_model_id"] == "run-inference-fixture"
    assert "checkpoint" not in result["extraction_settings"]["detection"]
    assert "ai_detection" not in result["files"][0]


def test_extraction_can_keep_empty_crops_for_negative_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from frameforge.media import extract_frames

    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = register_checkpoint(tmp_path)
    install_brightness_yolo(monkeypatch)

    result = extract_frames(extraction_sources(tmp_path), str(tmp_path / "out"), extraction_options(checkpoint, keep_empty=True))

    assert result["no_object_crops"] == 0
    assert result["exported_images"] == 2
    assert [len(image["ai_detection"]["predictions"]) for image in result["images"]] == [1, 0]


def test_extraction_detection_rejects_class_mismatch_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from frameforge.media import extract_frames

    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = register_checkpoint(tmp_path)
    install_brightness_yolo(monkeypatch)

    with pytest.raises(ValueError, match="class dự án"):
        extract_frames(extraction_sources(tmp_path), str(tmp_path / "out"), extraction_options(checkpoint, class_names=("person",)))
    assert not any((tmp_path / "out").glob("*.png"))


def test_extraction_suggestions_are_saved_for_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from frameforge.media import extract_frames

    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = register_checkpoint(tmp_path)
    install_brightness_yolo(monkeypatch)
    result = extract_frames(extraction_sources(tmp_path), str(tmp_path / "out"), extraction_options(checkpoint, keep_empty=True))
    project_path = tmp_path / "extract.ffproj"
    created = create_project(str(project_path), "Extract")
    save_project(str(project_path), {
        "id": created["id"], "name": "Extract", "sources": [], "settings": {}, "class_names": ["head", "body"],
        "images": [{"path": image["path"], "width": image["width"], "height": image["height"], "status": "ready"} for image in result["images"]],
    })
    items = [{"image_path": image["path"], **image["ai_detection"]} for image in result["images"]]

    with pytest.raises(ValueError, match="đã thay đổi"):
        inference.save_extraction_suggestions(str(project_path), str(checkpoint), "0" * 64, items)
    saved = inference.save_extraction_suggestions(
        str(project_path), str(checkpoint), result["detection"]["checkpoint_sha256"], items,
    )

    assert saved["annotation_count"] == 1
    assert saved["candidate_negative_count"] == 1
    loaded = {Path(image["path"]).name: image for image in load_project(str(project_path))["images"]}
    box = loaded["bright_frame_000000_center.png"]["annotations"][0]
    assert (box["source"], box["review_status"], box["source_model_id"]) == ("ai", "needs_review", "run-inference-fixture")
    assert loaded["dark_frame_000000_center.png"]["negative_candidates"][0]["review_status"] == "needs_review"
