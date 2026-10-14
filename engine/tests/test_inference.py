import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

from frameforge import inference, training


def register_checkpoint(tmp_path: Path) -> Path:
    output_dir = tmp_path / "output" / "frameforge-training" / "queue" / "run"
    weights = output_dir / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "best.pt"
    checkpoint.write_bytes(b"verified checkpoint fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training._persist_run({
        "run_id": "run-inference-fixture",
        "source": "frameforge_training",
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
        "created_at": training._now(),
    })
    return checkpoint


def create_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def install_fake_yolo(monkeypatch: pytest.MonkeyPatch, names=None):
    class FakeBoxes:
        xyxy = np.array([[1, 2, 10, 12], [3, 4, 12, 14], [5, 6, 20, 22]], dtype=np.float32)
        conf = np.array([0.5, 0.499, 0.8], dtype=np.float32)
        cls = np.array([0, 0, 1], dtype=np.float32)

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
    monkeypatch.setattr(inference, "training_capabilities", lambda: {
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
        "bbox_xyxy": pytest.approx([1, 2, 10, 12]),
    }]
    call = FakeYOLO.instances[0].calls[0]
    assert call["conf"] < 0.5
    assert call["iou"] == 0.7
    assert call["classes"] == [0]
    assert progress[-1]["stage"] == "inference_completed"


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
