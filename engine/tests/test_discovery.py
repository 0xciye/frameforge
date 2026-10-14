import csv
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

from frameforge.discovery import prepare_discovered_dataset, scan_project_folder
from frameforge.training import inspect_training_dataset


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_scan_reconstructs_nested_dataset_and_run_without_trusting_checkpoints(tmp_path: Path):
    root = tmp_path / "workspace"
    dataset = root / "data" / "detection_dataset"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "images" / "train" / "frame.jpg").write_bytes(b"image index only")
    (dataset / "labels" / "train" / "frame.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    write_yaml(dataset / "data.yaml", {
        "path": ".",
        "train": "images/train",
        "names": {0: "head", 1: "body"},
    })

    run = root / "runs" / "detect" / "train7"
    write_yaml(run / "args.yaml", {
        "task": "detect",
        "model": "yolo11n.pt",
        "data": "../../../data/detection_dataset/data.yaml",
        "imgsz": 192,
        "epochs": 2,
        "created": date(2026, 10, 14),
    })
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"untrusted checkpoint bytes")
    (run / "weights" / "last.pt").write_bytes(b"untrusted checkpoint bytes")
    with (run / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "epoch", "train/box_loss", "metrics/precision(B)", "metrics/recall(B)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        ])
        writer.writeheader()
        writer.writerow({
            "epoch": 0, "train/box_loss": 1.2, "metrics/precision(B)": 0.5,
            "metrics/recall(B)": 0.6, "metrics/mAP50(B)": 0.55, "metrics/mAP50-95(B)": 0.3,
        })
        writer.writerow({
            "epoch": 1, "train/box_loss": 0.7, "metrics/precision(B)": 0.7,
            "metrics/recall(B)": 0.8, "metrics/mAP50(B)": 0.75, "metrics/mAP50-95(B)": 0.4,
        })

    progress = []
    result = scan_project_folder(str(root), progress=progress.append)

    assert result["status"] == "complete"
    assert len(result["datasets"]) == 1
    candidate = result["datasets"][0]
    assert candidate["status"] == "candidate"
    assert candidate["class_names"] == ["head", "body"]
    assert candidate["splits"][0]["image_count"] == 1
    assert candidate["splits"][0]["labeled_count"] == 1
    assert not candidate["warnings"]

    assert len(result["runs"]) == 1
    run_info = result["runs"][0]
    assert run_info["name"] == "train7"
    assert run_info["training_imgsz"] == 192
    assert run_info["dataset_reference"] == str(dataset / "data.yaml")
    assert run_info["metrics_history"][-1]["map50"] == 0.75
    assert run_info["best_metrics"]["map50_95"] == 0.4
    assert run_info["run_config"]["created"] == "2026-10-14"
    assert run_info["best_checkpoint"]["trust_required"] is True
    assert run_info["best_checkpoint"]["compatibility"] == "unknown"
    assert result["exported_models"] == []
    assert progress[-1]["datasets_found"] == 1
    assert progress[-1]["runs_found"] == 1
    json.dumps(result)


def test_scan_blocks_dataset_paths_outside_selected_root(tmp_path: Path):
    root = tmp_path / "scan-root"
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    (outside / "image.jpg").write_bytes(b"must not inspect")
    write_yaml(root / "dataset.yaml", {
        "path": ".",
        "train": str(outside),
        "names": ["object"],
    })

    result = scan_project_folder(str(root))

    assert result["datasets"][0]["status"] == "blocked_external"
    assert result["datasets"][0]["splits"][0]["image_count"] is None
    assert any("ngoài" in warning for warning in result["datasets"][0]["splits"][0]["warnings"])


def test_scan_uses_configurable_ignored_directory_names(tmp_path: Path):
    root = tmp_path / "scan-root"
    write_yaml(root / "node_modules" / "example" / "data.yaml", {"path": ".", "train": "images"})

    default_scan = scan_project_folder(str(root))
    configured_scan = scan_project_folder(str(root), ignored_dirs=[])

    assert default_scan["datasets"] == []
    assert len(configured_scan["datasets"]) == 1


def test_scan_skips_symlink_directories_and_reports_file_limit(tmp_path: Path):
    root = tmp_path / "scan-root"
    child = root / "child"
    child.mkdir(parents=True)
    for index in range(4):
        (child / f"file-{index}.bin").write_bytes(b"x")
    try:
        (child / "loop").symlink_to(root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    complete = scan_project_folder(str(root))
    limited = scan_project_folder(str(root), max_files=2)

    assert complete["symlinks_skipped"] == 1
    assert limited["status"] == "partial"
    assert limited["files_seen"] == 2
    assert any("giới hạn 2 file" in warning for warning in limited["warnings"])


def test_scan_reports_depth_limit_and_cancellation(tmp_path: Path):
    root = tmp_path / "scan-root"
    deep = root / "one" / "two"
    deep.mkdir(parents=True)
    (deep / "hidden.bin").write_bytes(b"x")

    limited = scan_project_folder(str(root), max_depth=1)
    assert limited["status"] == "partial"
    assert limited["files_seen"] == 0
    assert any("sâu hơn giới hạn" in warning for warning in limited["warnings"])

    cancel_file = tmp_path / "cancel" / "request.cancel"
    progress = []

    def cancel_on_start(event):
        progress.append(event)
        if event["stage"] == "discovery_started":
            cancel_file.parent.mkdir(parents=True)
            cancel_file.write_text("cancel", encoding="utf-8")

    cancelled = scan_project_folder(str(root), cancel_file=str(cancel_file), progress=cancel_on_start)
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancelled"] is True
    assert progress[-1]["stage"] == "discovery_cancelled"


def test_scan_uses_safe_yaml_loader_for_untrusted_tags(tmp_path: Path):
    root = tmp_path / "scan-root"
    marker = tmp_path / "executed.txt"
    dataset_yaml = root / "data.yaml"
    dataset_yaml.parent.mkdir(parents=True)
    dataset_yaml.write_text(
        "!!python/object/apply:os.system ['touch {0}']".format(marker),
        encoding="utf-8",
    )

    result = scan_project_folder(str(root))

    assert marker.exists() is False
    assert result["datasets"][0]["status"] == "invalid_metadata"
    assert "YAML" in result["datasets"][0]["warnings"][0]


def test_imported_workspace_rebases_old_machine_paths_and_keeps_source_yaml_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "transferred-workspace"
    dataset = root / "dataset"
    for split in ("train", "val"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        image_path = image_dir / "frame.jpg"
        assert cv2.imwrite(str(image_path), np.full((64, 64, 3), 127, dtype=np.uint8))
        (label_dir / "frame.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml = dataset / "data.yaml"
    original = yaml.safe_dump({
        "path": r"C:\Users\source\workspace\dataset",
        "train": r"C:\Users\source\workspace\dataset\images\train",
        "val": r"C:\Users\source\workspace\dataset\images\val",
        "names": {0: "object"},
        "download": "must not be copied into the imported config",
    }, sort_keys=False)
    data_yaml.write_text(original, encoding="utf-8")
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))

    scanned = scan_project_folder(str(root))
    assert scanned["datasets"][0]["status"] == "candidate"
    assert any("đã được đối chiếu" in item for item in scanned["datasets"][0]["warnings"])

    prepared = prepare_discovered_dataset(str(root), str(data_yaml))
    normalized_path = Path(prepared["data_yaml"])
    normalized = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))
    assert prepared["paths_rebased"] is True
    assert normalized["path"] == str(dataset)
    assert normalized["train"] == str(dataset / "images" / "train")
    assert normalized["val"] == str(dataset / "images" / "val")
    assert "download" not in normalized
    assert data_yaml.read_text(encoding="utf-8") == original
    report = inspect_training_dataset(str(normalized_path))
    assert report["valid"] is True
    assert report["imported_workspace"] is True
    assert report["image_counts"] == {"train": 1, "val": 1}


def test_transferred_workspace_finds_dataset_when_yaml_is_outside_dataset_and_keeps_checkpoint_untrusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "transferred-workspace"
    dataset = root / "assets" / "dataset"
    for split in ("train", "val"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        assert cv2.imwrite(str(image_dir / "frame.jpg"), np.full((64, 64, 3), 127, dtype=np.uint8))
        (label_dir / "frame.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    data_yaml = root / "config" / "data.yaml"
    source_dataset = r"C:\Users\source\workspace\dataset"
    write_yaml(data_yaml, {
        "path": source_dataset,
        "train": source_dataset + r"\images\train",
        "val": source_dataset + r"\images\val",
        "names": ["object"],
    })
    checkpoint = root / "runs" / "detect" / "train1" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint_bytes = b"untrusted checkpoint fixture"
    checkpoint.write_bytes(checkpoint_bytes)
    write_yaml(checkpoint.parents[1] / "args.yaml", {
        "task": "detect",
        "model": "yolo11n.pt",
        "data": str(data_yaml),
        "imgsz": 640,
    })
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))

    scanned = scan_project_folder(str(root))
    assert scanned["datasets"][0]["status"] == "candidate"
    assert any("đã được đối chiếu" in item for item in scanned["datasets"][0]["warnings"])
    run_checkpoint = scanned["runs"][0]["best_checkpoint"]
    assert run_checkpoint["path"] == str(checkpoint)
    assert run_checkpoint["trust_required"] is True
    assert run_checkpoint["sha256"] is None

    prepared = prepare_discovered_dataset(str(root), str(data_yaml))
    normalized = yaml.safe_load(Path(prepared["data_yaml"]).read_text(encoding="utf-8"))
    assert normalized["path"] == str(dataset)
    assert normalized["train"] == str(dataset / "images" / "train")
    assert normalized["val"] == str(dataset / "images" / "val")
    report = inspect_training_dataset(prepared["data_yaml"])
    assert report["valid"] is True
    assert report["image_counts"] == {"train": 1, "val": 1}
    assert checkpoint.read_bytes() == checkpoint_bytes


def test_imported_workspace_rejects_ambiguous_dataset_rebase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "transferred-workspace"
    for name in ("dataset-a", "dataset-b"):
        for split in ("train", "val"):
            image_dir = root / name / "images" / split
            image_dir.mkdir(parents=True)
            assert cv2.imwrite(str(image_dir / "frame.jpg"), np.full((64, 64, 3), 127, dtype=np.uint8))
    data_yaml = root / "config" / "data.yaml"
    source_dataset = r"D:\old\dataset"
    write_yaml(data_yaml, {
        "path": source_dataset,
        "train": source_dataset + r"\images\train",
        "val": source_dataset + r"\images\val",
        "names": ["object"],
    })
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))

    scanned = scan_project_folder(str(root))
    assert scanned["datasets"][0]["status"] == "blocked_external"
    assert any("nhiều thư mục khớp" in item for item in scanned["datasets"][0]["warnings"])
    with pytest.raises(ValueError, match="nhiều thư mục khớp"):
        prepare_discovered_dataset(str(root), str(data_yaml))


def test_imported_workspace_rewrites_local_image_lists_and_rejects_external_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "transferred-workspace"
    dataset = root / "dataset"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "images" / "val").mkdir(parents=True)
    train_image = dataset / "images" / "train" / "one.jpg"
    val_image = dataset / "images" / "val" / "one.jpg"
    assert cv2.imwrite(str(train_image), np.full((64, 64, 3), 120, dtype=np.uint8))
    assert cv2.imwrite(str(val_image), np.full((64, 64, 3), 140, dtype=np.uint8))
    (dataset / "train.txt").write_text(
        r"C:\old\dataset\images\train\one.jpg" + "\n",
        encoding="utf-8",
    )
    data_yaml = dataset / "data.yaml"
    write_yaml(data_yaml, {
        "path": r"C:\old\dataset",
        "train": "train.txt",
        "val": "images/val",
        "names": ["object"],
    })
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))

    prepared = prepare_discovered_dataset(str(root), str(data_yaml))
    normalized = yaml.safe_load(Path(prepared["data_yaml"]).read_text(encoding="utf-8"))
    imported_list = Path(normalized["train"])
    assert imported_list.is_file()
    assert imported_list.read_text(encoding="utf-8").splitlines() == [str(train_image)]
    assert inspect_training_dataset(prepared["data_yaml"])["valid"] is True

    outside = tmp_path / "outside.jpg"
    assert cv2.imwrite(str(outside), np.full((64, 64, 3), 200, dtype=np.uint8))
    (dataset / "train.txt").write_text(str(outside) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ngoài thư mục|không khớp gốc"):
        prepare_discovered_dataset(str(root), str(data_yaml))


def test_imported_workspace_rejects_oversized_image_list(tmp_path: Path):
    root = tmp_path / "workspace"
    dataset = root / "dataset"
    (dataset / "images" / "val").mkdir(parents=True)
    (dataset / "train.txt").write_bytes(b" " * (2 * 1024 * 1024 + 1))
    data_yaml = dataset / "data.yaml"
    write_yaml(data_yaml, {"path": ".", "train": "train.txt", "val": "images/val", "names": ["object"]})

    with pytest.raises(ValueError, match="2 MiB mỗi tệp"):
        prepare_discovered_dataset(str(root), str(data_yaml))


def test_imported_workspace_preflight_rejects_source_moved_after_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "transferred-workspace"
    dataset = root / "dataset"
    for split in ("train", "val"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        assert cv2.imwrite(str(image_dir / "frame.jpg"), np.full((64, 64, 3), 127, dtype=np.uint8))
        (label_dir / "frame.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml = dataset / "data.yaml"
    write_yaml(data_yaml, {
        "path": r"C:\\old\\dataset",
        "train": r"C:\\old\\dataset\\images\\train",
        "val": r"C:\\old\\dataset\\images\\val",
        "names": ["object"],
    })
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))

    prepared = prepare_discovered_dataset(str(root), str(data_yaml))
    assert inspect_training_dataset(prepared["data_yaml"])["valid"] is True
    root.rename(tmp_path / "moved-workspace")

    report = inspect_training_dataset(prepared["data_yaml"])

    assert report["valid"] is False
    assert any("Thư mục nguồn đã nhập" in error for error in report["errors"])


def test_imported_workspace_preflight_rejects_image_replaced_by_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "transferred-workspace"
    dataset = root / "dataset"
    image_paths = []
    for split in ("train", "val"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        image_path = image_dir / "frame.jpg"
        assert cv2.imwrite(str(image_path), np.full((64, 64, 3), 127, dtype=np.uint8))
        image_paths.append(image_path)
        (label_dir / "frame.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml = dataset / "data.yaml"
    write_yaml(data_yaml, {
        "path": r"C:\\old\\dataset",
        "train": r"C:\\old\\dataset\\images\\train",
        "val": r"C:\\old\\dataset\\images\\val",
        "names": ["object"],
    })
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    prepared = prepare_discovered_dataset(str(root), str(data_yaml))

    outside = tmp_path / "outside.jpg"
    assert cv2.imwrite(str(outside), np.full((64, 64, 3), 200, dtype=np.uint8))
    image_paths[0].unlink()
    try:
        image_paths[0].symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    report = inspect_training_dataset(prepared["data_yaml"])

    assert report["valid"] is False
    assert any("symlink" in error for error in report["errors"])

    image_paths[0].unlink()
    assert cv2.imwrite(str(image_paths[0]), np.full((64, 64, 3), 127, dtype=np.uint8))
    label_path = dataset / "labels" / "train" / "frame.txt"
    outside_label = tmp_path / "outside.txt"
    outside_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    label_path.unlink()
    try:
        label_path.symlink_to(outside_label)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    report = inspect_training_dataset(prepared["data_yaml"])

    assert report["valid"] is False
    assert any("Label của ảnh không an toàn" in error for error in report["errors"])
