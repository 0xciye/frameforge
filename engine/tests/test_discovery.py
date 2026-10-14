import csv
import json
import sys
from datetime import date
from pathlib import Path

import pytest
import yaml

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

from frameforge.discovery import scan_project_folder


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
