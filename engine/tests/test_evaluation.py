import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

from frameforge import training
import frameforge_engine


def _make_fixed_dataset(root: Path, *, leakage: bool = False) -> Path:
    rows = []
    image_hashes = {"val": [], "test": []}
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        value = 100 if split == "val" else 120 if split == "test" else 100 if leakage else 80
        image_path = root / "images" / split / "sample.jpg"
        assert cv2.imwrite(str(image_path), np.full((32, 32, 3), value, dtype=np.uint8))
        label_path = root / "labels" / split / "sample.txt"
        label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        label_digest = hashlib.sha256(label_path.read_bytes()).hexdigest()
        if split in image_hashes:
            image_hashes[split].append(image_digest)
        rows.append({
            "exported_image": f"images/{split}/sample.jpg",
            "exported_label": f"labels/{split}/sample.txt",
            "sha256": image_digest,
            "label_sha256": label_digest,
            "split": split,
            "group_key": f"source-{split}",
            "status": "reviewed",
            "width": 32,
            "height": 32,
            "annotation_count": 1,
            "annotation_provenance": [{"class_id": 0, "source": "manual", "review_status": "reviewed"}],
        })
    manifest = {
        "format": "FrameForge YOLO dataset",
        "dataset_id": "fixed-dataset",
        "version": 1,
        "class_mapping": {"0": "head"},
        "fixed_validation_set": {"enabled": True, "set_id": "fixed-dataset:validation", "image_sha256s": image_hashes["val"]},
        "frozen_test_set": {"enabled": True, "set_id": "fixed-dataset:test", "image_sha256s": image_hashes["test"]},
        "images": rows,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": ".", "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "head"},
    }), encoding="utf-8")
    return data_yaml


def _register_checkpoint(root: Path, class_names: list[str] | None = None, run_id: str = "run-1") -> Path:
    run_dir = root / run_id
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "best.pt"
    checkpoint.write_bytes(b"internal checkpoint")
    record = {
        "run_id": run_id, "queue_id": "queue-" + run_id, "created_at": "2026-10-14T00:00:00+00:00",
        "status": "completed", "source": "frameforge_training", "checkpoint_trusted": True,
        "model": "YOLO11n", "architecture": "yolo11n", "task": "detect",
        "requested_imgsz": 64, "effective_imgsz": 64, "output_dir": str(run_dir),
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": training._sha256_file(checkpoint),
        "class_names": class_names or ["head"],
    }
    training._persist_run(record)
    return checkpoint


def _fake_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    class_names: list[str] | None = None,
    metric_overrides: dict[str, float] | None = None,
    class_metric_overrides: dict[str, float] | None = None,
):
    class FakeModel:
        task = "detect"
        names = {index: name for index, name in enumerate(class_names or ["head"])}
        calls = []

        def __init__(self):
            self.callbacks = {}

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

        def val(self, **kwargs):
            self.calls.append(kwargs)
            for callback in self.callbacks.values():
                callback(SimpleNamespace(dataloader=None, batch_i=0))
            values = {
                "metrics/precision(B)": 0.8, "metrics/recall(B)": 0.6,
                "metrics/mAP50(B)": 0.7, "metrics/mAP50-95(B)": 0.45,
            }
            values.update(metric_overrides or {})
            summary = {
                "Class": "head", "Box-P": 0.8, "Box-R": 0.6, "Box-F1": 0.68571,
                "mAP50": 0.7, "mAP50-95": 0.45,
            }
            summary.update(class_metric_overrides or {})
            return SimpleNamespace(
                results_dict=values,
                summary=lambda: [summary],
            )

    monkeypatch.setattr(training, "_load_yolo_model", lambda _path: FakeModel())
    monkeypatch.setattr(training, "training_capabilities", lambda: {
        "devices": {"cpu": True, "mps": False, "cuda": False}, "preferred_device": "cpu",
        "hardware": {"processor": "test"}, "ultralytics_version": "test-version",
    })
    return FakeModel


def test_fixed_set_evaluation_persists_metrics_conditions_targets_and_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_yaml = _make_fixed_dataset(tmp_path / "dataset")
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = _register_checkpoint(tmp_path / "run-root")
    FakeModel = _fake_evaluator(monkeypatch)
    saved_targets = training.save_evaluation_targets({
        "data_yaml": str(data_yaml), "targets": {"precision": 0.75, "p95_latency_ms": 20, "max_model_size_mb": 1},
        "per_class": {"head": {"recall": 0.7}},
    })

    result = training.evaluate_yolo_checkpoint({
        "checkpoint": str(checkpoint), "data_yaml": str(data_yaml), "split": "val",
        "imgsz": 64, "confidence_threshold": 0.01, "iou_threshold": 0.65, "batch": 2, "device": "cpu",
    })

    assert result["fixed_set_id"] == "fixed-dataset:validation"
    assert result["metrics"]["f1"] == pytest.approx(0.6857142857)
    assert result["metrics"]["model_size_mb"] > 0
    assert result["metrics"]["p95_latency_ms"] is None
    assert result["target_status"]["overall"]["precision"] == "met"
    assert result["target_status"]["overall"]["p95_latency_ms"] == "not_measured"
    assert result["target_status"]["overall"]["max_model_size_mb"] == "met"
    assert result["target_status"]["per_class"]["head"]["recall"] == "not_met"
    assert result["conditions"]["nms_iou_threshold"] == 0.65
    assert FakeModel.calls[0]["split"] == "val"
    assert training.get_evaluation_targets()["targets"] == saved_targets["targets"]
    assert training.list_model_evaluations("run-1")["evaluations"][0]["evaluation_id"] == result["evaluation_id"]


@pytest.mark.parametrize("asset", ["image", "label"])
def test_fixed_set_evaluation_rejects_changed_image_or_label_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asset: str):
    dataset = tmp_path / "dataset"
    data_yaml = _make_fixed_dataset(dataset)
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = _register_checkpoint(tmp_path / "run-root")
    _fake_evaluator(monkeypatch)
    target = dataset / ("images/val/sample.jpg" if asset == "image" else "labels/val/sample.txt")
    target.write_bytes(target.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="SHA-256"):
        training.evaluate_yolo_checkpoint({"checkpoint": str(checkpoint), "data_yaml": str(data_yaml), "imgsz": 64})


def test_fixed_set_evaluation_rejects_train_evaluation_fingerprint_leakage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_yaml = _make_fixed_dataset(tmp_path / "dataset", leakage=True)
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = _register_checkpoint(tmp_path / "run-root")
    _fake_evaluator(monkeypatch)

    with pytest.raises(ValueError, match="rò rỉ"):
        training.evaluate_yolo_checkpoint({"checkpoint": str(checkpoint), "data_yaml": str(data_yaml), "imgsz": 64})


def test_fixed_set_evaluation_refuses_unfrozen_sets_and_untrusted_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_yaml = _make_fixed_dataset(tmp_path / "dataset")
    manifest_path = data_yaml.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fixed_validation_set"]["enabled"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = _register_checkpoint(tmp_path / "run-root")
    _fake_evaluator(monkeypatch)

    with pytest.raises(ValueError, match="chưa có validation cố định"):
        training.evaluate_yolo_checkpoint({"checkpoint": str(checkpoint), "data_yaml": str(data_yaml), "imgsz": 64})
    with pytest.raises(ValueError, match="nội bộ"):
        training.evaluate_yolo_checkpoint({"checkpoint": str(data_yaml), "data_yaml": str(data_yaml), "imgsz": 64})


def test_fixed_set_evaluation_marks_nonfinite_metric_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_yaml = _make_fixed_dataset(tmp_path / "dataset")
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = _register_checkpoint(tmp_path / "run-root")
    _fake_evaluator(monkeypatch, metric_overrides={"metrics/precision(B)": float("nan")})
    training.save_evaluation_targets({"data_yaml": str(data_yaml), "targets": {"precision": 0.7}, "per_class": {}})

    result = training.evaluate_yolo_checkpoint({"checkpoint": str(checkpoint), "data_yaml": str(data_yaml), "imgsz": 64})

    assert result["metrics"]["precision"] is None
    assert result["target_status"]["overall"]["precision"] == "invalid_measurement"


def test_fixed_set_evaluation_cancellation_does_not_persist_partial_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_yaml = _make_fixed_dataset(tmp_path / "dataset")
    app_data = tmp_path / "app-data"
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(app_data))
    checkpoint = _register_checkpoint(tmp_path / "run-root")
    _fake_evaluator(monkeypatch)
    cancel_file = app_data / "training" / "cancel" / "request.cancel"
    cancel_file.parent.mkdir(parents=True)
    cancel_file.write_text("cancel", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Đã hủy đánh giá"):
        training.evaluate_yolo_checkpoint({
            "checkpoint": str(checkpoint), "data_yaml": str(data_yaml), "imgsz": 64, "cancel_file": str(cancel_file),
        })
    assert training.list_model_evaluations("run-1")["evaluations"] == []


def _run_fake_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_yaml: Path,
    run_id: str,
    *,
    metric_overrides: dict[str, float] | None = None,
    class_metric_overrides: dict[str, float] | None = None,
    split: str = "val",
    imgsz: int = 64,
):
    checkpoint = _register_checkpoint(tmp_path / "runs", run_id=run_id)
    _fake_evaluator(
        monkeypatch,
        metric_overrides=metric_overrides,
        class_metric_overrides=class_metric_overrides,
    )
    return training.evaluate_yolo_checkpoint({
        "checkpoint": str(checkpoint), "data_yaml": str(data_yaml), "split": split,
        "imgsz": imgsz, "confidence_threshold": 0.01, "iou_threshold": 0.65, "batch": 2, "device": "cpu",
    })


def test_champion_challenger_promote_and_rollback_are_explicit_and_persistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    data_yaml = _make_fixed_dataset(tmp_path / "dataset")
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    champion_eval = _run_fake_evaluation(tmp_path, monkeypatch, data_yaml, "run-1")

    state = training.get_model_promotion_state()
    assert state["champion"] is None
    assert state["challenger"]["evaluation_id"] == champion_eval["evaluation_id"]
    assert state["can_rollback"] is False

    comparison = training.compare_model_challenger({"max_regression": 0})
    assert comparison["can_promote"] is True
    assert comparison["recommendation"] == "initial_champion"
    state = training.promote_model_challenger({"max_regression": 0})
    assert state["champion"]["evaluation_id"] == champion_eval["evaluation_id"]
    assert state["challenger"] is None
    assert state["history"][0]["action"] == "promoted"

    next_eval = _run_fake_evaluation(
        tmp_path, monkeypatch, data_yaml, "run-2",
        metric_overrides={
            "metrics/precision(B)": 0.82, "metrics/recall(B)": 0.65,
            "metrics/mAP50(B)": 0.73, "metrics/mAP50-95(B)": 0.48,
        },
        class_metric_overrides={
            "Box-P": 0.82, "Box-R": 0.65, "Box-F1": 0.725,
            "mAP50": 0.73, "mAP50-95": 0.48,
        },
    )
    comparison = training.compare_model_challenger({"max_regression": 0})
    assert comparison["comparable"] is True
    assert comparison["can_promote"] is True
    assert comparison["overall"]["map50"]["delta"] == pytest.approx(0.03)
    state = training.promote_model_challenger({"max_regression": 0})
    assert state["champion"]["evaluation_id"] == next_eval["evaluation_id"]
    assert state["can_rollback"] is True

    restored = training.rollback_model_promotion()
    assert restored["champion"]["evaluation_id"] == champion_eval["evaluation_id"]
    assert restored["challenger"]["evaluation_id"] == next_eval["evaluation_id"]
    assert restored["can_rollback"] is False
    assert restored["history"][0]["action"] == "rolled_back"


def test_challenger_cannot_promote_when_quality_regresses_or_targets_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    data_yaml = _make_fixed_dataset(tmp_path / "dataset")
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    champion_eval = _run_fake_evaluation(tmp_path, monkeypatch, data_yaml, "run-1")
    training.promote_model_challenger({})
    training.save_evaluation_targets({
        "data_yaml": str(data_yaml), "targets": {"precision": 0.9}, "per_class": {},
    })
    challenger_eval = _run_fake_evaluation(
        tmp_path, monkeypatch, data_yaml, "run-2",
        metric_overrides={
            "metrics/precision(B)": 0.8, "metrics/recall(B)": 0.3,
            "metrics/mAP50(B)": 0.65, "metrics/mAP50-95(B)": 0.4,
        },
        class_metric_overrides={
            "Box-P": 0.8, "Box-R": 0.3, "Box-F1": 0.436,
            "mAP50": 0.65, "mAP50-95": 0.4,
        },
    )
    comparison = training.compare_model_challenger({"max_regression": 0})
    assert comparison["can_promote"] is False
    assert comparison["recommendation"] == "retain_champion"
    assert any("recall" in reason for reason in comparison["regressions"])
    assert any("precision" in reason for reason in comparison["target_failures"])
    with pytest.raises(ValueError, match="Chưa thể promote"):
        training.promote_model_challenger({"max_regression": 0})
    state = training.get_model_promotion_state()
    assert state["champion"]["evaluation_id"] == champion_eval["evaluation_id"]
    assert state["challenger"]["evaluation_id"] == challenger_eval["evaluation_id"]


def test_challenger_comparison_rejects_different_fixed_conditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    data_yaml = _make_fixed_dataset(tmp_path / "dataset")
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    champion_eval = _run_fake_evaluation(tmp_path, monkeypatch, data_yaml, "run-1")
    training.promote_model_challenger({})
    _run_fake_evaluation(tmp_path, monkeypatch, data_yaml, "run-2", imgsz=128)

    comparison = training.compare_model_challenger({})
    assert comparison["comparable"] is False
    assert comparison["can_promote"] is False
    assert any("imgsz" in reason for reason in comparison["blockers"])
    assert training.get_model_promotion_state()["champion"]["evaluation_id"] == champion_eval["evaluation_id"]


def test_changed_checkpoint_blocks_promotion_and_rollback_preserves_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    data_yaml = _make_fixed_dataset(tmp_path / "dataset")
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    champion_eval = _run_fake_evaluation(tmp_path, monkeypatch, data_yaml, "run-1")
    checkpoint = Path(champion_eval["checkpoint_path"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")

    comparison = training.compare_model_challenger({})
    assert comparison["can_promote"] is False
    assert any("checksum" in reason for reason in comparison["blockers"])
    with pytest.raises(ValueError, match="Chưa thể promote"):
        training.promote_model_challenger({})
    assert training.get_model_promotion_state()["champion"] is None


def test_existing_validation_history_is_migrated_to_pending_challenger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    data_yaml = _make_fixed_dataset(tmp_path / "dataset")
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    evaluation = _run_fake_evaluation(tmp_path, monkeypatch, data_yaml, "run-1")
    connection = training._registry_connection()
    try:
        connection.execute("DELETE FROM model_roles WHERE role_id='default'")
        connection.commit()
    finally:
        connection.close()

    state = training.get_model_promotion_state()

    assert state["challenger"]["evaluation_id"] == evaluation["evaluation_id"]


def test_engine_routes_model_promotion_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))

    response = frameforge_engine.handle_request({
        "id": "promotion-state", "command": "get_model_promotion_state", "arguments": {},
    })

    assert response["id"] == "promotion-state"
    assert response["ok"] is True
    assert response["data"]["champion"] is None
    assert response["data"]["challenger"] is None
