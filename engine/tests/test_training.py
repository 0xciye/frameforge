import csv
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

from frameforge import training


def create_yolo_dataset(root: Path, with_validation: bool = True) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    splits = ["train", "val"] if with_validation else ["train"]
    for split in splits:
        for index in range(2):
            image_path = root / "images" / split / f"{index}.jpg"
            image = np.full((64, 64, 3), 50 + index * 30, dtype=np.uint8)
            assert cv2.imwrite(str(image_path), image)
            (root / "labels" / split / f"{index}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": ".",
        "train": "images/train",
        "val": "images/val" if with_validation else None,
        "names": {0: "head"},
    }), encoding="utf-8")
    return data_yaml


def test_training_dataset_preflight_checks_images_labels_and_fingerprint(tmp_path: Path):
    data_yaml = create_yolo_dataset(tmp_path / "dataset")

    summary = training.inspect_training_dataset(str(data_yaml))

    assert summary["valid"] is True
    assert summary["class_names"] == ["head"]
    assert summary["image_counts"] == {"train": 2, "val": 2}
    assert summary["instance_counts"] == {"train": 2, "val": 2}
    assert len(summary["dataset_fingerprint"]) == 64

    (data_yaml.parent / "labels" / "train" / "0.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    changed = training.inspect_training_dataset(str(data_yaml))
    assert changed["dataset_fingerprint"] != summary["dataset_fingerprint"]


def test_training_dataset_preflight_rejects_invalid_and_corrupt_labels(tmp_path: Path):
    data_yaml = create_yolo_dataset(tmp_path / "dataset")
    (data_yaml.parent / "labels" / "train" / "0.txt").write_text("3 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (data_yaml.parent / "images" / "val" / "1.jpg").write_bytes(b"not an image")

    summary = training.inspect_training_dataset(str(data_yaml))

    assert summary["valid"] is False
    assert any("Class ID ngoài" in error for error in summary["errors"])
    assert any("không giải mã được" in error for error in summary["errors"])


def test_training_dataset_preflight_requires_train_and_validation_images(tmp_path: Path):
    data_yaml = create_yolo_dataset(tmp_path / "dataset", with_validation=False)

    summary = training.inspect_training_dataset(str(data_yaml))

    assert summary["valid"] is False
    assert any("Split val không có ảnh" in error for error in summary["errors"])


def test_training_capabilities_reports_missing_optional_stack(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(training.importlib.util, "find_spec", lambda _name: None)

    result = training.training_capabilities()

    assert result["available"] is False
    assert result["missing_packages"] == ["torch", "ultralytics"]
    assert result["devices"]["cpu"] is True
    assert result["preferred_device"] == "cpu"


def install_fake_training_runtime(monkeypatch: pytest.MonkeyPatch, cancel_marker: Path | None = None):
    class FakeStride:
        def max(self):
            return self

        def item(self):
            return 32

    class FakeProcess:
        def cpu_percent(self, _interval):
            return 12.5

        def memory_info(self):
            return SimpleNamespace(rss=123456)

    fake_psutil = ModuleType("psutil")
    fake_psutil.virtual_memory = lambda: SimpleNamespace(total=16 * 1024**3, available=8 * 1024**3)
    fake_psutil.cpu_count = lambda logical=False: 8
    fake_psutil.Process = FakeProcess
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    class FakeMPS:
        def is_built(self):
            return True

        def is_available(self):
            return True

        def get_name(self):
            return "Test MPS"

    fake_torch = ModuleType("torch")
    fake_torch.backends = SimpleNamespace(mps=FakeMPS())
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    fake_torch.mps = SimpleNamespace(current_allocated_memory=lambda: 4096)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        training.importlib.util,
        "find_spec",
        lambda name: object() if name in {"torch", "ultralytics"} else None,
    )

    class FakeYOLO:
        sources: list[str] = []
        image_sizes: list[int] = []
        train_parameters: list[dict] = []
        marker_to_create: Path | None = cancel_marker

        def __init__(self, source):
            self.__class__.sources.append(source)
            self.task = "detect"
            self.model = SimpleNamespace(stride=FakeStride())
            self.callbacks = {}

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

        def train(self, **kwargs):
            self.__class__.image_sizes.append(kwargs["imgsz"])
            self.__class__.train_parameters.append(kwargs.copy())
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            (save_dir / "weights").mkdir(parents=True)
            best = save_dir / "weights" / "best.pt"
            last = save_dir / "weights" / "last.pt"
            best.write_bytes(b"best checkpoint")
            last.write_bytes(b"last checkpoint")
            with (save_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["metrics/mAP50(B)", "train/box_loss"])
                writer.writeheader()
                writer.writerow({"metrics/mAP50(B)": "0.75", "train/box_loss": "0.2"})
            trainer = SimpleNamespace(
                save_dir=save_dir,
                best=best,
                last=last,
                device="mps:0",
                epoch=0,
                epochs=kwargs["epochs"],
                batch_i=0,
                train_loader=[0],
                batch_size=kwargs["batch"],
                metrics={"metrics/mAP50(B)": 0.75},
                stopper=SimpleNamespace(possible_stop=False),
                stop=False,
            )
            self.trainer = trainer
            for name in ("on_train_start", "on_train_epoch_start", "on_train_batch_end", "on_model_save"):
                self.callbacks[name](trainer)
            if self.__class__.marker_to_create:
                self.__class__.marker_to_create.write_text("cancel", encoding="utf-8")
                self.__class__.marker_to_create = None
            self.callbacks["on_fit_epoch_end"](trainer)
            return SimpleNamespace()

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    return FakeYOLO


def test_training_preflight_and_queue_support_over_20k_images_and_300_epochs_per_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    dataset = tmp_path / "large-dataset"
    train_images = dataset / "images" / "train"
    val_images = dataset / "images" / "val"
    train_labels = dataset / "labels" / "train"
    val_labels = dataset / "labels" / "val"
    for folder in (train_images, val_images, train_labels, val_labels):
        folder.mkdir(parents=True)
    ok, encoded = cv2.imencode(".jpg", np.full((320, 320, 3), 80, dtype=np.uint8))
    assert ok
    image_bytes = encoded.tobytes()
    for index in range(20_001):
        (train_images / f"image-{index:05}.jpg").write_bytes(image_bytes)
    for index in range(2):
        (val_images / f"image-{index}.jpg").write_bytes(image_bytes)
    (train_labels / "image-00000.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (val_labels / "image-0.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": ".", "train": "images/train", "val": "images/val", "names": {0: "head"},
    }), encoding="utf-8")

    inspected = training.inspect_training_dataset(str(data_yaml))
    assert inspected["valid"] is True
    assert inspected["total_images"] == 20_003
    assert inspected["image_counts"] == {"train": 20_001, "val": 2}

    output_dir = tmp_path / "training-output"
    output_dir.mkdir()
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    FakeYOLO = install_fake_training_runtime(monkeypatch)
    result = training.train_yolo_queue({
        "data_yaml": str(data_yaml),
        "output_dir": str(output_dir),
        "sizes": [64, 96],
        "device": "auto",
        "settings": {"epochs": 301, "batch": 2, "workers": 0, "save_period": 50},
    })

    assert result["status"] == "completed"
    assert len(FakeYOLO.train_parameters) == 2
    assert [call["epochs"] for call in FakeYOLO.train_parameters] == [301, 301]
    assert [call["imgsz"] for call in FakeYOLO.train_parameters] == [64, 96]
    assert [call["save_period"] for call in FakeYOLO.train_parameters] == [50, 50]


def test_multi_resolution_queue_reloads_same_base_and_registers_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_yaml = create_yolo_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    FakeYOLO = install_fake_training_runtime(monkeypatch)
    progress_events = []

    result = training.train_yolo_queue({
        "data_yaml": str(data_yaml),
        "output_dir": str(output_dir),
        "sizes": [128, 160],
        "device": "auto",
        "settings": {"epochs": 1, "batch": 2, "workers": 0},
    }, progress_events.append)

    assert result["status"] == "completed"
    assert FakeYOLO.sources == ["yolo11n.pt", "yolo11n.pt"]
    assert FakeYOLO.image_sizes == [128, 160]
    assert [job["status"] for job in result["jobs"]] == ["completed", "completed"]
    assert all(job["device_actual"] == "mps:0" for job in result["jobs"])
    assert all(job["checkpoint_path"].endswith("best.pt") for job in result["jobs"])
    assert all(job["metrics"]["metrics/mAP50(B)"] == 0.75 for job in result["jobs"])
    assert any(event["stage"] == "epoch_completed" for event in progress_events)
    registry = training.list_training_runs()
    assert registry["available_count"] == 2
    assert all(run["checkpoint_checksum_matches"] is True for run in registry["runs"])
    assert all(run["checkpoint_trusted"] is True for run in registry["runs"])

    parent = result["jobs"][0]
    fine_tune_output = tmp_path / "fine-tune-output"
    fine_tune_output.mkdir()
    fine_tune = training.train_yolo_queue({
        "data_yaml": str(data_yaml),
        "output_dir": str(fine_tune_output),
        "sizes": [128],
        "device": "auto",
        "base_model": parent["checkpoint_path"],
        "settings": {"epochs": 1, "batch": 2, "workers": 0},
    })
    assert fine_tune["status"] == "completed"
    assert fine_tune["jobs"][0]["parent_model_id"] == parent["run_id"]
    assert FakeYOLO.sources[-1] == parent["checkpoint_path"]


def test_training_cancel_waits_for_epoch_checkpoint_and_keeps_partial_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_yaml = create_yolo_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    app_data = tmp_path / "app-data"
    marker = app_data / "training" / "cancel" / "request.cancel"
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(app_data))
    install_fake_training_runtime(monkeypatch, cancel_marker=marker)

    result = training.train_yolo_queue({
        "data_yaml": str(data_yaml),
        "output_dir": str(output_dir),
        "sizes": [128, 160],
        "device": "mps",
        "settings": {"epochs": 3, "batch": 2, "workers": 0},
        "cancel_file": str(marker),
    })

    assert result["status"] == "cancelled"
    assert result["jobs"][0]["status"] == "cancelled"
    assert Path(result["jobs"][0]["checkpoint_path"]).is_file()
    assert result["jobs"][1]["status"] == "not_started"


def test_imported_discovery_run_is_persisted_but_checkpoint_stays_untrusted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "external"
    run_dir = root / "runs" / "custom-name"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    best = weights / "best.pt"
    best.write_bytes(b"untrusted serialized model")
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))

    imported = training.register_discovered_run(str(root), {
        "experiment_id": "run-0123456789abcdef0123",
        "source_dir": str(run_dir),
        "name": "custom-name",
        "training_imgsz": 192,
        "task_claimed": "detect",
        "metrics_history": [{"epoch": 0, "map50": 0.5}],
        "checkpoints": [{"path": str(best), "kind": "best"}],
    })

    assert imported["source"] == "external_discovery"
    assert imported["checkpoint_trusted"] is False
    registry = training.list_training_runs()
    saved = next(run for run in registry["runs"] if run["run_id"] == imported["run_id"])
    assert saved["checkpoint_exists"] is True
    assert saved["checkpoint_checksum_matches"] is None
    assert saved["checkpoint_usable"] is False
    assert registry["available_count"] == 0
    assert training._registered_checkpoint_run_id(best.resolve()) is None


def test_training_rejects_unregistered_local_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_yaml = create_yolo_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    checkpoint = tmp_path / "unregistered.pt"
    checkpoint.write_bytes(b"arbitrary serialized model")
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    FakeYOLO = install_fake_training_runtime(monkeypatch)

    with pytest.raises(ValueError, match="nội bộ đã đăng ký"):
        training.train_yolo_queue({
            "data_yaml": str(data_yaml),
            "output_dir": str(output_dir),
            "sizes": [128],
            "device": "auto",
            "base_model": str(checkpoint),
            "settings": {"epochs": 1, "batch": 2, "workers": 0},
        })

    assert FakeYOLO.sources == []


def test_training_rejects_registered_external_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_yaml = create_yolo_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    root = tmp_path / "external"
    run_dir = root / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "best.pt"
    checkpoint.write_bytes(b"external model")
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    FakeYOLO = install_fake_training_runtime(monkeypatch)
    training.register_discovered_run(str(root), {
        "experiment_id": "run-0123456789abcdef0123",
        "source_dir": str(run_dir),
        "name": "external-run",
        "checkpoints": [{"path": str(checkpoint), "kind": "best"}],
    })

    with pytest.raises(ValueError, match="nội bộ đã đăng ký"):
        training.train_yolo_queue({
            "data_yaml": str(data_yaml),
            "output_dir": str(output_dir),
            "sizes": [128],
            "device": "auto",
            "base_model": str(checkpoint),
            "settings": {"epochs": 1, "batch": 2, "workers": 0},
        })

    assert FakeYOLO.sources == []


def test_imported_discovery_run_rejects_paths_outside_scan_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "external"
    root.mkdir(parents=True)
    run_dir = tmp_path / "outside" / "run"
    run_dir.mkdir(parents=True)
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))

    with pytest.raises(ValueError, match="ngoài thư mục"):
        training.register_discovered_run(str(root), {
            "experiment_id": "run-0123456789abcdef0123",
            "source_dir": str(run_dir),
            "checkpoints": [],
        })
