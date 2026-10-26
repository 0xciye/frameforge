import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

from frameforge import models

def install_fake_import_model(
    monkeypatch: pytest.MonkeyPatch, *, task="detect", yaml_file="yolo11.yaml", scale="n", imgsz=320,
):
    class FakeYOLO:
        calls = []
        loaded_bytes = []

        def __init__(self, source):
            self.calls.append(source)
            self.loaded_bytes.append(Path(source).read_bytes())
            self.task = task
            self.names = {0: "head", 1: "body"}
            self.model = SimpleNamespace(yaml={"yaml_file": yaml_file, "scale": scale})
            self.overrides = {} if imgsz is None else {"imgsz": imgsz}

    module = ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return FakeYOLO


def test_import_yolo_checkpoint_requires_explicit_trust_and_registers_verified_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"trusted YOLO11n checkpoint fixture")
    FakeYOLO = install_fake_import_model(monkeypatch)

    with pytest.raises(ValueError, match="xác nhận bạn tin cậy"):
        models.import_yolo_checkpoint(str(checkpoint), False)
    assert FakeYOLO.calls == []

    imported = models.import_yolo_checkpoint(str(checkpoint), True)

    managed = Path(imported["checkpoint_path"])
    loaded_copy = Path(FakeYOLO.calls[0])
    assert imported["display_name"] == checkpoint.name
    assert loaded_copy != checkpoint
    assert loaded_copy.suffix == ".pt"
    assert loaded_copy.parent.parent.name == imported["run_id"]
    assert FakeYOLO.loaded_bytes == [checkpoint.read_bytes()]
    assert managed.is_file()
    assert managed.read_bytes() == checkpoint.read_bytes()
    record = models.verified_checkpoint_record(managed)
    assert record is not None
    assert record["source"] == "frameforge_import"
    assert record["display_name"] == checkpoint.name
    assert record["class_names"] == ["head", "body"]
    assert record["requested_imgsz"] == record["effective_imgsz"] == imported["requested_imgsz"] == 320
    assert record["imgsz_source"] == imported["imgsz_source"] == "checkpoint_metadata"
    assert record["origin_sha256"] == imported["checkpoint_sha256"]
    listed = models.list_models()["runs"]
    imported_run = next(item for item in listed if item["run_id"] == imported["run_id"])
    assert imported_run["checkpoint_usable"] is True
    assert imported_run["checkpoint_checksum_matches"] is True


def test_import_checkpoint_without_imgsz_uses_and_labels_ultralytics_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = tmp_path / "no-imgsz.pt"
    checkpoint.write_bytes(b"trusted YOLO11n checkpoint fixture")
    install_fake_import_model(monkeypatch, imgsz=None)

    imported = models.import_yolo_checkpoint(str(checkpoint), True)
    record = models.verified_checkpoint_record(Path(imported["checkpoint_path"]))

    assert record is not None
    assert imported["requested_imgsz"] == record["requested_imgsz"] == record["effective_imgsz"] == 640
    assert imported["imgsz_source"] == record["imgsz_source"] == "ultralytics_default"


def test_import_checkpoint_uses_adjacent_run_args_when_checkpoint_metadata_omits_imgsz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    run_dir = tmp_path / "workspace" / "runs" / "detect" / "train9"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    checkpoint = weights_dir / "best.pt"
    checkpoint.write_bytes(b"trusted YOLO11n checkpoint fixture")
    (run_dir / "args.yaml").write_text("imgsz: 384\n", encoding="utf-8")
    install_fake_import_model(monkeypatch, imgsz=None)

    imported = models.import_yolo_checkpoint(str(checkpoint), True)
    record = models.verified_checkpoint_record(Path(imported["checkpoint_path"]))

    assert record is not None
    assert imported["requested_imgsz"] == record["requested_imgsz"] == 384
    assert imported["imgsz_source"] == record["imgsz_source"] == "run_args_yaml"
    assert record["imgsz_source_path"] == str(run_dir / "args.yaml")


@pytest.mark.parametrize(
    ("task", "yaml_file", "scale", "message"),
    [
        ("segment", "yolo11.yaml", "n", "task object detection"),
        ("detect", "yolov8n.yaml", "n", "kiến trúc YOLO11n"),
        ("detect", "yolo11.yaml", "s", "kiến trúc YOLO11n"),
        ("detect", "yolo11x.yaml", "n", "kiến trúc YOLO11n"),
    ],
)
def test_import_yolo_checkpoint_rejects_wrong_task_or_architecture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task, yaml_file, scale, message,
):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = tmp_path / "unsupported.pt"
    checkpoint.write_bytes(b"unsupported checkpoint fixture")
    install_fake_import_model(monkeypatch, task=task, yaml_file=yaml_file, scale=scale)

    with pytest.raises(ValueError, match=message):
        models.import_yolo_checkpoint(str(checkpoint), True)

    assert models.list_models()["runs"] == []


def test_import_yolo_checkpoint_rejects_oversized_file_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "oversized.pt"
    with checkpoint.open("wb") as stream:
        stream.truncate(models.MAX_IMPORTED_CHECKPOINT_BYTES + 1)
    FakeYOLO = install_fake_import_model(monkeypatch)

    with pytest.raises(ValueError, match="không vượt quá 2 GiB"):
        models.import_yolo_checkpoint(str(checkpoint), True)

    assert FakeYOLO.calls == []


def test_import_yolo_checkpoint_rejects_managed_copy_changed_during_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    checkpoint = tmp_path / "changing.pt"
    checkpoint.write_bytes(b"original checkpoint")

    class MutatingYOLO:
        def __init__(self, source):
            Path(source).write_bytes(b"changed during load")
            self.task = "detect"
            self.names = {0: "head"}
            self.model = SimpleNamespace(yaml={"yaml_file": "yolo11.yaml", "scale": "n"})

    module = ModuleType("ultralytics")
    module.YOLO = MutatingYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)

    with pytest.raises(ValueError, match="Bản sao checkpoint thay đổi"):
        models.import_yolo_checkpoint(str(checkpoint), True)

    assert checkpoint.read_bytes() == b"original checkpoint"
    assert models.list_models()["runs"] == []


def test_inference_capabilities_reports_missing_optional_stack(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(models.importlib.util, "find_spec", lambda _name: None)

    result = models.inference_capabilities()

    assert result["available"] is False
    assert result["missing_packages"] == ["torch", "ultralytics"]
    assert result["devices"]["cpu"] is True
    assert result["preferred_device"] == "cpu"



def test_untrusted_registry_record_is_not_usable_for_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    run_dir = tmp_path / "external-run"
    checkpoint = run_dir / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"external checkpoint")
    models._persist_run({
        "run_id": "external",
        "source": "external_discovery",
        "checkpoint_trusted": False,
        "queue_id": "external",
        "status": "completed",
        "model": "YOLO11n",
        "architecture": "yolo11n",
        "task": "detect",
        "requested_imgsz": 640,
        "output_dir": str(run_dir),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": models._sha256_file(checkpoint),
        "class_names": ["head"],
        "created_at": models._now(),
    })

    assert models.verified_checkpoint_record(checkpoint.resolve()) is None
    listed = models.list_models()["runs"]
    assert listed[0]["checkpoint_usable"] is False
