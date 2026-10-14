"""Real YOLO11n training, dataset preflight, and checkpoint registry."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TextIO

import cv2
import yaml


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL = "yolo11n.pt"
MAX_IMAGE_SIZE = 2048
ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_app_data_dir() -> Path:
    configured = os.environ.get("FRAMEFORGE_APP_DATA")
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "com.frameforge.desktop"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "com.frameforge.desktop"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "frameforge"


def _training_root() -> Path:
    root = _default_app_data_dir() / "training"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _registry_connection() -> sqlite3.Connection:
    path = _training_root() / "registry.sqlite3"
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            image_size INTEGER NOT NULL,
            checkpoint_path TEXT,
            checkpoint_sha256 TEXT,
            record_json TEXT NOT NULL
        )
        """
    )
    return connection


def _persist_run(record: Dict[str, Any]) -> None:
    connection = _registry_connection()
    try:
        now = _now()
        connection.execute(
            """
            INSERT INTO training_runs (
                run_id, created_at, updated_at, status, image_size,
                checkpoint_path, checkpoint_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                updated_at=excluded.updated_at,
                status=excluded.status,
                checkpoint_path=excluded.checkpoint_path,
                checkpoint_sha256=excluded.checkpoint_sha256,
                record_json=excluded.record_json
            """,
            (
                record["run_id"], record["created_at"], now, record["status"],
                record["requested_imgsz"], record.get("checkpoint_path"),
                record.get("checkpoint_sha256"),
                json.dumps(record, ensure_ascii=False, allow_nan=False),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_internal_training_record(record: Dict[str, Any], checkpoint: Path) -> bool:
    """Accept only YOLO11n detection checkpoints produced by a FrameForge run."""
    if record.get("source") == "external_discovery" or record.get("checkpoint_trusted") is False:
        return False
    if record.get("model") != "YOLO11n" or record.get("architecture") != "yolo11n" or record.get("task") != "detect":
        return False
    if not record.get("queue_id") or record.get("status") not in {"completed", "cancelled"}:
        return False
    output_dir_value = record.get("output_dir")
    if not isinstance(output_dir_value, str) or not output_dir_value:
        return False
    try:
        output_dir = Path(output_dir_value).expanduser().resolve(strict=True)
        checkpoint.relative_to(output_dir)
    except (OSError, ValueError):
        return False
    return output_dir.is_dir()


def verified_checkpoint_record(checkpoint: Path) -> Optional[Dict[str, Any]]:
    """Return trusted internal run metadata only when the registered file hash matches."""
    connection = _registry_connection()
    try:
        rows = connection.execute(
            "SELECT run_id, status, checkpoint_path, checkpoint_sha256, record_json FROM training_runs"
        ).fetchall()
    finally:
        connection.close()

    for row in rows:
        path_value = row["checkpoint_path"]
        if not isinstance(path_value, str):
            continue
        try:
            registered_path = Path(path_value).expanduser().resolve(strict=True)
        except (OSError, ValueError):
            continue
        if registered_path != checkpoint:
            continue
        try:
            record = json.loads(row["record_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if (
            not isinstance(record, dict)
            or row["status"] not in {"completed", "cancelled"}
            or record.get("status") != row["status"]
            or record.get("run_id") != row["run_id"]
        ):
            continue
        record_path_value = record.get("checkpoint_path")
        try:
            record_path = Path(record_path_value).expanduser().resolve(strict=True)
        except (OSError, TypeError, ValueError):
            continue
        if record_path != registered_path:
            continue
        if not _is_internal_training_record(record, registered_path):
            continue
        record_digest = record.get("checkpoint_sha256")
        registry_digest = row["checkpoint_sha256"]
        if not isinstance(record_digest, str) or not isinstance(registry_digest, str):
            continue
        if record_digest != registry_digest:
            continue
        try:
            if _sha256_file(registered_path) != record_digest:
                continue
        except OSError:
            continue
        return record
    return None


def _registered_checkpoint_run_id(checkpoint: Path) -> Optional[str]:
    record = verified_checkpoint_record(checkpoint)
    run_id = record.get("run_id") if record else None
    return run_id if isinstance(run_id, str) else None


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hardware_info() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "Không xác định",
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": None,
        "memory_total_bytes": None,
        "memory_available_bytes": None,
        "disk_free_bytes": None,
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        result["physical_cpu_count"] = psutil.cpu_count(logical=False)
        result["memory_total_bytes"] = int(memory.total)
        result["memory_available_bytes"] = int(memory.available)
    except (ImportError, OSError):
        pass
    return result


def training_capabilities(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Report installed training support and actual available compute backends."""
    missing = [name for name in ("torch", "ultralytics") if importlib.util.find_spec(name) is None]
    result: Dict[str, Any] = {
        "available": not missing,
        "missing_packages": missing,
        "install_command": 'python -m pip install -e "engine[train]"',
        "python_version": platform.python_version(),
        "torch_version": _package_version("torch"),
        "ultralytics_version": _package_version("ultralytics"),
        "psutil_version": _package_version("psutil"),
        "hardware": _hardware_info(),
        "devices": {"cpu": True, "mps": False, "cuda": False},
        "device_details": {},
        "preferred_device": "cpu",
        "recommendation": None,
    }
    if output_dir:
        try:
            import shutil

            result["hardware"]["disk_free_bytes"] = int(shutil.disk_usage(Path(output_dir)).free)
        except OSError:
            pass

    if "torch" not in missing:
        try:
            import torch

            mps = getattr(torch.backends, "mps", None)
            mps_built = bool(mps and mps.is_built())
            mps_available = bool(mps and mps.is_available())
            cuda_available = bool(torch.cuda.is_available())
            result["devices"] = {"cpu": True, "mps": mps_available, "cuda": cuda_available}
            result["device_details"] = {
                "mps_built": mps_built,
                "mps_device_name": mps.get_name() if mps_available and hasattr(mps, "get_name") else None,
                "cuda_device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
                if cuda_available else [],
            }
            if mps_available:
                result["preferred_device"] = "mps"
            elif cuda_available:
                result["preferred_device"] = "cuda:0"
        except Exception as error:
            result["torch_error"] = str(error)
            result["available"] = False
            if "torch" not in result["missing_packages"]:
                result["missing_packages"].append("torch (không nạp được)")

    hardware = result["hardware"]
    total_memory = hardware.get("memory_total_bytes")
    device = result["preferred_device"]
    if device == "mps":
        batch = 8 if total_memory and total_memory >= 16 * 1024**3 else 4
        workers = 0
        amp = False
        device_reason = "PyTorch báo MPS hiện khả dụng trên máy này."
    elif device.startswith("cuda"):
        batch = 16
        workers = min(8, max(1, (hardware.get("logical_cpu_count") or 2) // 2))
        amp = True
        device_reason = "PyTorch báo CUDA hiện khả dụng trên máy này."
    else:
        batch = 4
        workers = min(4, max(0, (hardware.get("logical_cpu_count") or 2) // 2))
        amp = False
        device_reason = "Dùng CPU vì không có GPU được PyTorch hỗ trợ và khả dụng."
    result["recommendation"] = {
        "device": device,
        "batch": batch,
        "workers": workers,
        "epochs": 100,
        "patience": 30,
        "cache": False,
        "amp": amp,
        "save_period": 10,
        "reasons": [device_reason, "Giá trị là cấu hình khởi điểm; không phải dự đoán thời gian hay chất lượng model."],
    }
    return result


def _class_names(value: Any) -> List[str]:
    if isinstance(value, list):
        names = value
    elif isinstance(value, dict):
        try:
            indexed = {int(key): name for key, name in value.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Tên class trong data.yaml phải dùng ID số nguyên liên tục từ 0.") from error
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("ID class trong data.yaml phải liên tục từ 0.")
        names = [indexed[index] for index in range(len(indexed))]
    else:
        raise ValueError("data.yaml thiếu danh sách names của class.")
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Danh sách names trong data.yaml phải có ít nhất một tên class hợp lệ.")
    normalized = [name.strip() for name in names]
    if len({name.casefold() for name in normalized}) != len(normalized):
        raise ValueError("Tên class trong data.yaml bị trùng nhau.")
    return normalized


def _resolve_dataset_root(data_file: Path, config: Dict[str, Any]) -> Path:
    configured = config.get("path", ".")
    if not isinstance(configured, str):
        raise ValueError("Trường path trong data.yaml phải là đường dẫn cục bộ.")
    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = data_file.parent / root
    return root.resolve()


def _split_sources(value: Any, split_name: str) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list) and value and all(isinstance(item, str) and item.strip() for item in value):
        return [item.strip() for item in value]
    raise ValueError("data.yaml cần đường dẫn cục bộ hợp lệ cho split {0}.".format(split_name))


def _resolve_split_images(root: Path, sources: Iterable[str], split_name: str, errors: List[str]) -> List[Path]:
    images: Dict[str, Path] = {}
    for source in sources:
        if source.startswith(("http://", "https://")):
            errors.append("Split {0} dùng URL từ xa; FrameForge chỉ train từ dataset local.".format(split_name))
            continue
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.is_dir():
            candidates = (item for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)
        elif path.is_file() and path.suffix.lower() == ".txt":
            candidates = []
            for line in path.read_text(encoding="utf-8").splitlines():
                value = line.strip()
                if not value:
                    continue
                item = Path(value).expanduser()
                if not item.is_absolute():
                    item = root / item
                candidates.append(item.resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            candidates = [path]
        else:
            errors.append("Không tìm thấy thư mục, ảnh hoặc danh sách ảnh split {0}: {1}".format(split_name, path))
            continue
        for item in candidates:
            if item.suffix.lower() not in IMAGE_EXTENSIONS:
                errors.append("Split {0} chứa định dạng ảnh không hỗ trợ: {1}".format(split_name, item))
            elif not item.is_file():
                errors.append("Ảnh trong split {0} không tồn tại: {1}".format(split_name, item))
            else:
                images[str(item)] = item
    return [images[key] for key in sorted(images)]


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    positions = [index for index, part in enumerate(parts[:-1]) if part.lower() == "images"]
    if positions:
        parts[positions[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _validate_label_file(path: Path, class_count: int, split_name: str, errors: List[str]) -> int:
    if not path.is_file():
        return 0
    instances = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        errors.append("Không đọc được nhãn {0}: {1}".format(path, error))
        return 0
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            errors.append("Nhãn YOLO phải có 5 trường: {0}:{1}".format(path, line_number))
            continue
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError:
            errors.append("Nhãn YOLO chứa giá trị không phải số: {0}:{1}".format(path, line_number))
            continue
        values = (x_center, y_center, width, height)
        if class_id < 0 or class_id >= class_count:
            errors.append("Class ID ngoài data.yaml tại {0}:{1}".format(path, line_number))
            continue
        if not all(math.isfinite(value) for value in values) or width <= 0 or height <= 0:
            errors.append("Box không hữu hạn hoặc có diện tích bằng 0 tại {0}:{1}".format(path, line_number))
            continue
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and width <= 1 and height <= 1):
            errors.append("Box nằm ngoài miền chuẩn hóa [0, 1] tại {0}:{1}".format(path, line_number))
            continue
        if x_center - width / 2 < -1e-6 or y_center - height / 2 < -1e-6 or x_center + width / 2 > 1 + 1e-6 or y_center + height / 2 > 1 + 1e-6:
            errors.append("Box vượt biên ảnh tại {0}:{1}".format(path, line_number))
            continue
        instances += 1
    return instances


def inspect_training_dataset(data_yaml: str, progress: ProgressCallback = None) -> Dict[str, Any]:
    """Read and validate a local YOLO detection dataset and its labels."""
    data_file = Path(data_yaml).expanduser().resolve(strict=True)
    if not data_file.is_file() or data_file.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Chọn data.yaml hoặc .yml của dataset YOLO đã export.")
    try:
        config = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("Không đọc được data.yaml: {0}".format(error)) from error
    if not isinstance(config, dict):
        raise ValueError("data.yaml phải là một object YAML.")

    names = _class_names(config.get("names"))
    root = _resolve_dataset_root(data_file, config)
    errors: List[str] = []
    warnings: List[str] = []
    split_images: Dict[str, List[Path]] = {}
    for split_name in ("train", "val", "test"):
        value = config.get(split_name)
        if value is None:
            if split_name in {"train", "val"}:
                errors.append("data.yaml thiếu split {0}.".format(split_name))
            continue
        try:
            sources = _split_sources(value, split_name)
        except ValueError as error:
            errors.append(str(error))
            continue
        split_images[split_name] = _resolve_split_images(root, sources, split_name, errors)
    for split_name in ("train", "val"):
        if not split_images.get(split_name):
            errors.append("Split {0} không có ảnh.".format(split_name))

    manifest_path = data_file.parent / "manifest.json"
    manifest: Dict[str, Any] = {}
    manifest_digests: Dict[str, str] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in manifest.get("images", []):
                relative = item.get("exported_image")
                digest = item.get("sha256")
                if isinstance(relative, str) and isinstance(digest, str):
                    manifest_digests[relative.replace("\\", "/")] = digest
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError):
            warnings.append("manifest.json không đọc được; dataset fingerprint sẽ được tính trực tiếp từ tệp local.")
            manifest = {}
            manifest_digests = {}
    fixed_validation_set = manifest.get("fixed_validation_set")
    frozen_test_set = manifest.get("frozen_test_set")
    if not isinstance(fixed_validation_set, dict):
        fixed_validation_set = {}
    if not isinstance(frozen_test_set, dict):
        frozen_test_set = {}

    fingerprint = hashlib.sha256()
    fingerprint.update(data_file.read_bytes())
    if manifest_path.is_file():
        fingerprint.update(manifest_path.read_bytes())
    image_counts: Dict[str, int] = {}
    instance_counts: Dict[str, int] = {}
    class_instance_counts = [0 for _ in names]
    missing_labels = 0
    all_images = [(split, image) for split, images in split_images.items() for image in images]
    for index, (split_name, image_path) in enumerate(all_images, start=1):
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                errors.append("Ảnh không giải mã được: {0}".format(image_path))
                continue
            if image.shape[0] < 1 or image.shape[1] < 1:
                errors.append("Ảnh có kích thước không hợp lệ: {0}".format(image_path))
                continue
            image_digest = _sha256_file(image_path)
            try:
                relative = image_path.relative_to(root).as_posix()
            except ValueError:
                relative = str(image_path)
            expected_digest = manifest_digests.get(relative)
            if expected_digest and image_digest != expected_digest:
                errors.append("Ảnh khác SHA-256 trong manifest: {0}".format(image_path))
            fingerprint.update(relative.encode("utf-8", errors="replace"))
            fingerprint.update(image_digest.encode("ascii"))
            label_file = _label_path(image_path)
            if not label_file.is_file():
                missing_labels += 1
                fingerprint.update(b"missing-label")
                instances = 0
            else:
                label_bytes = label_file.read_bytes()
                fingerprint.update(hashlib.sha256(label_bytes).digest())
                instances = _validate_label_file(label_file, len(names), split_name, errors)
            image_counts[split_name] = image_counts.get(split_name, 0) + 1
            instance_counts[split_name] = instance_counts.get(split_name, 0) + instances
            if label_file.is_file():
                for line in label_file.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) == 5:
                        try:
                            class_id = int(fields[0])
                            if 0 <= class_id < len(class_instance_counts):
                                class_instance_counts[class_id] += 1
                        except ValueError:
                            pass
        except OSError as error:
            errors.append("Không kiểm tra được ảnh hoặc nhãn {0}: {1}".format(image_path, error))
        if progress and (index == len(all_images) or index % 50 == 0):
            progress({"stage": "validating_dataset", "checked_images": index, "total_images": len(all_images)})

    if missing_labels:
        warnings.append("{0} ảnh không có tệp label .txt; Ultralytics sẽ coi chúng là ảnh không object.".format(missing_labels))
    if image_counts.get("val", 0) < 5:
        warnings.append("Có ít hơn 5 ảnh validation; các chỉ số validation có thể không ổn định.")
    if not any(class_instance_counts):
        warnings.append("Dataset không có box; đây là tác vụ detection nhưng chưa có positive example.")
    for class_id, count in enumerate(class_instance_counts):
        if count == 0:
            warnings.append("Class '{0}' không có box trong train/val/test.".format(names[class_id]))

    return {
        "valid": not errors,
        "data_yaml": str(data_file),
        "dataset_root": str(root),
        "dataset_id": manifest.get("dataset_id"),
        "dataset_version": manifest.get("version"),
        "parent_dataset_id": manifest.get("parent_dataset_id"),
        "replay_strategy": manifest.get("replay_strategy", "none"),
        "fixed_validation_set_id": fixed_validation_set.get("set_id"),
        "frozen_test_set_id": frozen_test_set.get("set_id"),
        "dataset_fingerprint": fingerprint.hexdigest(),
        "class_names": names,
        "class_instance_counts": {name: class_instance_counts[index] for index, name in enumerate(names)},
        "image_counts": image_counts,
        "instance_counts": instance_counts,
        "total_images": sum(image_counts.values()),
        "total_instances": sum(instance_counts.values()),
        "warnings": list(dict.fromkeys(warnings)),
        "errors": list(dict.fromkeys(errors))[:200],
    }


def list_training_runs() -> Dict[str, Any]:
    """Discover previously registered training checkpoints, including missing files."""
    connection = _registry_connection()
    try:
        rows = connection.execute(
            "SELECT status, checkpoint_sha256, record_json FROM training_runs ORDER BY created_at DESC LIMIT 250"
        ).fetchall()
    finally:
        connection.close()
    runs = []
    for row in rows:
        try:
            record = json.loads(row["record_json"])
        except json.JSONDecodeError:
            continue
        path_value = record.get("checkpoint_path")
        checkpoint = Path(path_value) if isinstance(path_value, str) else None
        exists = bool(checkpoint and checkpoint.is_file())
        checksum_matches: Optional[bool] = None
        if exists and record.get("checkpoint_sha256"):
            try:
                checksum_matches = _sha256_file(checkpoint) == record["checkpoint_sha256"]
            except OSError:
                checksum_matches = False
        if checksum_matches is True and row["checkpoint_sha256"] != record.get("checkpoint_sha256"):
            checksum_matches = False
        record["checkpoint_exists"] = exists
        record["checkpoint_checksum_matches"] = checksum_matches
        trusted = bool(
            exists
            and row["status"] == record.get("status")
            and checkpoint is not None
            and _is_internal_training_record(record, checkpoint.resolve())
        )
        record["checkpoint_trusted"] = trusted
        record["checkpoint_usable"] = exists and checksum_matches is True and trusted
        runs.append(record)
    return {"runs": runs, "available_count": sum(bool(run["checkpoint_usable"]) for run in runs)}


def register_discovered_run(discovery_root_value: str, run: Dict[str, Any]) -> Dict[str, Any]:
    """Persist scanned run metadata without loading or trusting external checkpoints."""
    if not isinstance(run, dict):
        raise ValueError("Training run không hợp lệ.")
    root = Path(discovery_root_value).expanduser().resolve(strict=True)
    source_dir_value = run.get("source_dir")
    experiment_id = run.get("experiment_id")
    if not root.is_dir() or not isinstance(source_dir_value, str) or not isinstance(experiment_id, str):
        raise ValueError("Thiếu thư mục hoặc mã training run hợp lệ.")
    source_dir = Path(source_dir_value).expanduser().resolve(strict=True)
    if not source_dir.is_dir():
        raise ValueError("Thư mục training run không tồn tại.")
    try:
        relative_dir = source_dir.relative_to(root).as_posix() or "."
    except ValueError as error:
        raise ValueError("Training run nằm ngoài thư mục đã quét.") from error
    if len(experiment_id) > 128 or not experiment_id.startswith("run-"):
        raise ValueError("Mã training run không hợp lệ.")

    raw_checkpoints = run.get("checkpoints")
    checkpoints = raw_checkpoints if isinstance(raw_checkpoints, list) else []
    checkpoints_by_kind = sorted(
        (item for item in checkpoints if isinstance(item, dict)),
        key=lambda item: {"best": 0, "last": 1}.get(item.get("kind"), 2),
    )
    selected_checkpoint = None
    for item in checkpoints_by_kind:
        path_value = item.get("path")
        if not isinstance(path_value, str):
            continue
        try:
            checkpoint = Path(path_value).expanduser().resolve(strict=True)
            checkpoint.relative_to(source_dir)
        except (OSError, ValueError):
            continue
        if checkpoint.is_file() and checkpoint.suffix.casefold() == ".pt":
            selected_checkpoint = checkpoint
            break

    raw_imgsz = run.get("training_imgsz")
    image_size = raw_imgsz if isinstance(raw_imgsz, int) and not isinstance(raw_imgsz, bool) and raw_imgsz > 0 else 0
    metrics_history = run.get("metrics_history")
    if not isinstance(metrics_history, list):
        metrics_history = []
    clean_history = []
    for point in metrics_history[:5_000]:
        if not isinstance(point, dict):
            continue
        clean_point: Dict[str, Any] = {}
        for key, value in point.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, bool) or value is None:
                clean_point[key] = value
            elif isinstance(value, (int, float)):
                clean_point[key] = value if math.isfinite(float(value)) else None
        clean_history.append(clean_point)
    best_metrics = run.get("best_metrics")
    if not isinstance(best_metrics, dict):
        best_metrics = {}
    clean_best_metrics = {
        key: value for key, value in best_metrics.items()
        if isinstance(key, str) and isinstance(value, (int, float))
        and not isinstance(value, bool) and math.isfinite(float(value))
    }
    run_config = run.get("run_config")
    if not isinstance(run_config, dict):
        run_config = {}
    try:
        clean_config = json.loads(json.dumps(run_config, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Cấu hình run không phải dữ liệu JSON an toàn.") from error

    record = {
        "run_id": experiment_id,
        "source": "external_discovery",
        "name": str(run.get("name") or source_dir.name)[:200],
        "source_dir": str(source_dir),
        "relative_dir": relative_dir,
        "created_at": str(run.get("modified_at") or _now()),
        "status": str(run.get("status") or "unknown"),
        "completion_state": str(run.get("completion_state") or "unknown"),
        "requested_imgsz": image_size,
        "training_imgsz": raw_imgsz if isinstance(raw_imgsz, (int, list)) else None,
        "effective_imgsz": None,
        "architecture_claimed": run.get("architecture_claimed") if isinstance(run.get("architecture_claimed"), str) else None,
        "task_claimed": run.get("task_claimed") if isinstance(run.get("task_claimed"), str) else None,
        "model_family": "unknown",
        "dataset_reference": run.get("dataset_reference") if isinstance(run.get("dataset_reference"), str) else None,
        "dataset_relationship": str(run.get("dataset_relationship") or "unknown"),
        "requested_epochs": run.get("requested_epochs") if isinstance(run.get("requested_epochs"), int) and not isinstance(run.get("requested_epochs"), bool) else None,
        "last_logged_epoch": run.get("last_logged_epoch") if isinstance(run.get("last_logged_epoch"), (int, float)) else None,
        "batch": run.get("batch") if isinstance(run.get("batch"), (int, float)) and not isinstance(run.get("batch"), bool) else None,
        "device_actual": run.get("device") if isinstance(run.get("device"), (str, int, float)) else None,
        "run_config": clean_config,
        "metrics_history": clean_history,
        "metrics": clean_best_metrics,
        "checkpoint_path": str(selected_checkpoint) if selected_checkpoint else None,
        "checkpoint_sha256": None,
        "checkpoint_trusted": False,
        "checkpoint_usable": False,
        "checkpoint_name": selected_checkpoint.name if selected_checkpoint else None,
        "warnings": list(dict.fromkeys(item for item in run.get("warnings", []) if isinstance(item, str)))[:200]
        if isinstance(run.get("warnings"), list) else [],
    }
    _persist_run(record)
    return record


def _read_results_csv(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error, UnicodeError):
        return {}
    if not rows:
        return {}
    result: Dict[str, Any] = {}
    for key, value in rows[-1].items():
        if value is None:
            continue
        name = key.strip()
        try:
            number = float(value.strip())
        except (ValueError, AttributeError):
            result[name] = value
        else:
            result[name] = number if math.isfinite(number) else None
    return result


def _load_yolo_model(base_model: str) -> Any:
    """Keep the downloaded pretrained checkpoint in FrameForge app data."""
    from ultralytics import YOLO

    if base_model != DEFAULT_MODEL:
        return YOLO(base_model)

    model_dir = _default_app_data_dir() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    cached_checkpoint = model_dir / DEFAULT_MODEL
    if cached_checkpoint.is_file():
        return YOLO(str(cached_checkpoint))

    previous_directory = Path.cwd()
    try:
        os.chdir(model_dir)
        return YOLO(DEFAULT_MODEL)
    finally:
        os.chdir(previous_directory)


class _Tee:
    def __init__(self, primary: TextIO, log_file: TextIO) -> None:
        self.primary = primary
        self.log_file = log_file

    def write(self, value: str) -> int:
        self.primary.write(value)
        self.log_file.write(value)
        return len(value)

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _runtime_metrics(trainer: Any, torch: Any, psutil_module: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    device = str(getattr(trainer, "device", "unknown"))
    process = None
    if psutil_module:
        try:
            process = psutil_module.Process()
            metrics["cpu_percent"] = _finite_number(process.cpu_percent(None))
            metrics["process_memory_bytes"] = int(process.memory_info().rss)
            memory = psutil_module.virtual_memory()
            metrics["system_memory_available_bytes"] = int(memory.available)
            metrics["system_memory_total_bytes"] = int(memory.total)
        except (OSError, AttributeError):
            pass
    if device.startswith("mps"):
        try:
            metrics["gpu_memory_allocated_bytes"] = int(torch.mps.current_allocated_memory())
        except (AttributeError, RuntimeError):
            metrics["gpu_memory_allocated_bytes"] = None
        metrics["gpu_utilization_percent"] = None
    elif device.startswith("cuda"):
        try:
            metrics["gpu_memory_allocated_bytes"] = int(torch.cuda.memory_allocated())
        except (AttributeError, RuntimeError):
            metrics["gpu_memory_allocated_bytes"] = None
        metrics["gpu_utilization_percent"] = None
    else:
        metrics["gpu_memory_allocated_bytes"] = None
        metrics["gpu_utilization_percent"] = None
    return metrics


def train_yolo_queue(arguments: Dict[str, Any], progress: ProgressCallback = None) -> Dict[str, Any]:
    """Train one independent YOLO11n model per requested size, sequentially."""
    capabilities = training_capabilities(arguments.get("output_dir"))
    if not capabilities["available"]:
        raise RuntimeError(
            "Thiếu dependency train: {0}. Cài bằng `{1}` rồi khởi động lại ứng dụng.".format(
                ", ".join(capabilities["missing_packages"]), capabilities["install_command"]
            )
        )
    try:
        import psutil
        import torch
    except Exception as error:
        raise RuntimeError("Không nạp được PyTorch/Ultralytics: {0}".format(error)) from error

    data_yaml = str(arguments.get("data_yaml", ""))
    dataset = inspect_training_dataset(data_yaml, progress)
    if not dataset["valid"]:
        raise ValueError("Dataset không hợp lệ:\n- " + "\n- ".join(dataset["errors"][:20]))

    output_root = Path(str(arguments.get("output_dir", ""))).expanduser().resolve(strict=True)
    if not output_root.is_dir():
        raise ValueError("Chọn thư mục đầu ra training đã tồn tại.")
    dataset_root = Path(dataset["dataset_root"])
    try:
        output_root.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("Thư mục training không được nằm bên trong dataset để bảo toàn dataset đã export.")

    sizes_value = arguments.get("sizes")
    if not isinstance(sizes_value, list) or not sizes_value or len(sizes_value) > 12:
        raise ValueError("Chọn từ 1 đến 12 kích thước train.")
    sizes: List[int] = []
    for value in sizes_value:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_IMAGE_SIZE:
            raise ValueError("imgsz phải là số nguyên từ 1 đến {0} px.".format(MAX_IMAGE_SIZE))
        if value in sizes:
            raise ValueError("Danh sách imgsz không được trùng kích thước.")
        sizes.append(value)

    settings = arguments.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("Cấu hình training không hợp lệ.")
    epochs = settings.get("epochs", 100)
    batch = settings.get("batch", 8)
    patience = settings.get("patience", 30)
    workers = settings.get("workers", 0)
    save_period = settings.get("save_period", 10)
    seed = settings.get("seed", 42)
    for name, value, minimum, maximum in (
        ("epochs", epochs, 1, 10000), ("batch", batch, 1, 512),
        ("patience", patience, 0, 1000), ("workers", workers, 0, 64),
        ("save_period", save_period, -1, 10000), ("seed", seed, 0, 2**32 - 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError("{0} phải là số nguyên từ {1} đến {2}.".format(name, minimum, maximum))
    lr0 = settings.get("lr0")
    if lr0 is not None and (isinstance(lr0, bool) or not isinstance(lr0, (int, float)) or not math.isfinite(lr0) or not 1e-6 <= lr0 <= 1.0):
        raise ValueError("Learning rate phải nằm trong khoảng 0.000001–1 hoặc để trống để dùng mặc định Ultralytics.")

    requested_device = arguments.get("device", "auto")
    devices = capabilities["devices"]
    if requested_device == "auto":
        device = capabilities["preferred_device"]
    elif requested_device == "cpu":
        device = "cpu"
    elif requested_device == "mps" and devices.get("mps"):
        device = "mps"
    elif requested_device in {"cuda", "cuda:0"} and devices.get("cuda"):
        device = "0"
    else:
        raise ValueError("Thiết bị {0} không khả dụng theo PyTorch hiện tại.".format(requested_device))
    if device == "mps":
        workers = 0

    base_model = str(arguments.get("base_model") or DEFAULT_MODEL)
    if base_model != DEFAULT_MODEL:
        checkpoint = Path(base_model).expanduser().resolve(strict=True)
        if not checkpoint.is_file() or checkpoint.suffix.lower() != ".pt":
            raise ValueError("Checkpoint fine-tune phải là tệp .pt local hợp lệ.")
        parent_model_id = _registered_checkpoint_run_id(checkpoint)
        if parent_model_id is None:
            raise ValueError("Chỉ có thể fine-tune từ checkpoint YOLO11n nội bộ đã đăng ký, còn nguyên checksum.")
        base_model = str(checkpoint)
    else:
        parent_model_id = None

    queue_id = str(uuid.uuid4())
    queue_root = output_root / "frameforge-training" / queue_id
    queue_root.mkdir(parents=True, mode=0o700)
    queue_root_config = {
        "queue_id": queue_id,
        "created_at": _now(),
        "model": "YOLO11n",
        "base_model": base_model,
        "parent_model_id": parent_model_id,
        "dataset_yaml": dataset["data_yaml"],
        "dataset_id": dataset.get("dataset_id"),
        "dataset_version": dataset.get("dataset_version"),
        "parent_dataset_id": dataset.get("parent_dataset_id"),
        "replay_strategy": dataset.get("replay_strategy"),
        "fixed_validation_set_id": dataset.get("fixed_validation_set_id"),
        "frozen_test_set_id": dataset.get("frozen_test_set_id"),
        "dataset_fingerprint": dataset["dataset_fingerprint"],
        "class_names": dataset["class_names"],
        "requested_sizes": sizes,
        "device_requested": requested_device,
        "device_selected": device,
        "settings": {**settings, "epochs": epochs, "batch": batch, "patience": patience, "workers": workers, "seed": seed},
        "execution": "sequential; every size loads the same base checkpoint independently",
    }
    (queue_root / "queue.json").write_text(json.dumps(queue_root_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cancel_value = arguments.get("cancel_file")
    cancel_file: Optional[Path] = None
    if isinstance(cancel_value, str) and cancel_value:
        candidate = Path(cancel_value).expanduser().resolve()
        allowed_parent = (_training_root() / "cancel").resolve()
        allowed_parent.mkdir(parents=True, exist_ok=True)
        if candidate.parent != allowed_parent:
            raise ValueError("Cancel marker nằm ngoài thư mục quản lý training của FrameForge.")
        cancel_file = candidate

    queue_jobs: List[Dict[str, Any]] = []
    failures = 0
    queue_cancelled = False
    error_policy = arguments.get("on_error", "stop")
    if error_policy not in {"stop", "continue"}:
        raise ValueError("Chính sách khi job lỗi phải là stop hoặc continue.")

    def emit(payload: Dict[str, Any]) -> None:
        if progress:
            progress({"queue_id": queue_id, **payload})

    if dataset["warnings"]:
        emit({"stage": "dataset_warnings", "warnings": dataset["warnings"]})
    emit({"stage": "queue_started", "total_jobs": len(sizes), "device_selected": device, "queue_root": str(queue_root)})

    for job_index, requested_size in enumerate(sizes):
        if cancel_file and cancel_file.exists():
            queue_cancelled = True
            break
        run_id = str(uuid.uuid4())
        run_name = "imgsz-{0}-{1}".format(requested_size, run_id)
        run_dir = queue_root / run_name
        run_dir.mkdir(parents=True, mode=0o700)
        run_log = run_dir / "training.log"
        started_at = _now()
        record: Dict[str, Any] = {
            "run_id": run_id,
            "source": "frameforge_training",
            "checkpoint_trusted": True,
            "queue_id": queue_id,
            "created_at": started_at,
            "started_at": started_at,
            "completed_at": None,
            "status": "preparing",
            "model": "YOLO11n",
            "architecture": "yolo11n",
            "task": "detect",
            "requested_imgsz": requested_size,
            "effective_imgsz": None,
            "stride": None,
            "base_model": base_model,
            "parent_model_id": parent_model_id,
            "dataset_yaml": dataset["data_yaml"],
            "dataset_id": dataset.get("dataset_id"),
            "dataset_version": dataset.get("dataset_version"),
            "parent_dataset_id": dataset.get("parent_dataset_id"),
            "replay_strategy": dataset.get("replay_strategy"),
            "fixed_validation_set_id": dataset.get("fixed_validation_set_id"),
            "frozen_test_set_id": dataset.get("frozen_test_set_id"),
            "dataset_fingerprint": dataset["dataset_fingerprint"],
            "class_names": dataset["class_names"],
            "dataset_image_counts": dataset["image_counts"],
            "output_dir": str(run_dir),
            "device_requested": requested_device,
            "device_selected": device,
            "device_actual": None,
            "framework_version": _package_version("ultralytics"),
            "torch_version": _package_version("torch"),
            "settings": {
                "epochs": epochs, "batch": batch, "patience": patience, "workers": workers,
                "save_period": save_period, "seed": seed, "lr0": lr0, "amp": device.startswith("cuda"),
            },
            "configuration_path": str(run_dir / "frameforge-config.json"),
            "log_path": str(run_log),
            "best_checkpoint": None,
            "last_checkpoint": None,
            "checkpoint_path": None,
            "checkpoint_sha256": None,
            "metrics": {},
            "error": None,
            "early_stopped": False,
            "elapsed_seconds": None,
        }
        _persist_run(record)
        (run_dir / "frameforge-config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        emit({
            "stage": "job_preparing", "job_index": job_index + 1, "total_jobs": len(sizes),
            "run_id": run_id, "requested_imgsz": requested_size, "device_selected": device,
        })
        job_started = time.monotonic()
        epoch_started = job_started
        previous_batch_at: Optional[float] = None
        last_batch_event_at = 0.0
        epoch_durations: List[float] = []
        active_epoch_index: Optional[int] = None
        cancellation_requested = False
        try:
            log_file = run_log.open("a", encoding="utf-8", buffering=1)
            tee = _Tee(sys.stderr, log_file)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                model = _load_yolo_model(base_model)
                if getattr(model, "task", "detect") != "detect":
                    raise ValueError("Checkpoint không phải model object detection.")
                stride_value = getattr(getattr(model, "model", None), "stride", 32)
                try:
                    stride = int(stride_value.max().item())
                except (AttributeError, TypeError, ValueError):
                    try:
                        stride = int(max(stride_value))
                    except (TypeError, ValueError):
                        stride = 32
                stride = max(1, stride)
                effective_size = max(stride, math.ceil(requested_size / stride) * stride)
                record["stride"] = stride
                record["effective_imgsz"] = effective_size
                record["status"] = "running"
                _persist_run(record)
                (run_dir / "frameforge-config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                emit({
                    "stage": "job_started", "job_index": job_index + 1, "total_jobs": len(sizes),
                    "run_id": run_id, "requested_imgsz": requested_size, "effective_imgsz": effective_size,
                    "stride": stride, "device_selected": device, "epochs_total": epochs,
                })

                def on_train_start(trainer: Any) -> None:
                    record["device_actual"] = str(getattr(trainer, "device", device))
                    _persist_run(record)
                    emit({"stage": "training_started", "run_id": run_id, "device_actual": record["device_actual"]})

                def on_epoch_start(trainer: Any) -> None:
                    nonlocal active_epoch_index, epoch_started, previous_batch_at
                    epoch_started = time.monotonic()
                    previous_batch_at = None
                    active_epoch_index = int(getattr(trainer, "epoch", 0))
                    emit({
                        "stage": "epoch_started", "run_id": run_id,
                        "epoch": active_epoch_index + 1,
                        "epochs_total": int(getattr(trainer, "epochs", epochs)),
                    })

                def on_batch_end(trainer: Any) -> None:
                    nonlocal previous_batch_at, last_batch_event_at
                    now = time.monotonic()
                    batch_index = getattr(trainer, "batch_i", None)
                    try:
                        batch_number = int(batch_index) + 1
                    except (TypeError, ValueError):
                        batch_number = None
                    try:
                        total_batches = len(trainer.train_loader)
                    except (AttributeError, TypeError):
                        total_batches = None
                    interval = now - previous_batch_at if previous_batch_at else None
                    previous_batch_at = now
                    if interval and interval > 0:
                        seconds_per_batch = interval
                    else:
                        seconds_per_batch = None
                    if batch_number and total_batches and batch_number < total_batches and now - last_batch_event_at < 1.0:
                        return
                    last_batch_event_at = now
                    losses: Dict[str, Any] = {}
                    try:
                        values = trainer.tloss.detach().cpu().tolist()
                        names_value = getattr(trainer, "loss_names", [])
                        losses = {str(name): _finite_number(value) for name, value in zip(names_value, values)}
                    except (AttributeError, RuntimeError, TypeError):
                        pass
                    record["losses"] = losses
                    batch_size = getattr(trainer, "batch_size", batch)
                    runtime = _runtime_metrics(trainer, torch, psutil)
                    emit({
                        "stage": "batch_progress", "run_id": run_id,
                        "epoch": int(getattr(trainer, "epoch", 0)) + 1,
                        "epochs_total": int(getattr(trainer, "epochs", epochs)),
                        "batch": batch_number, "batches_total": total_batches,
                        "seconds_per_batch": seconds_per_batch,
                        "images_per_second": float(batch_size) / seconds_per_batch if seconds_per_batch else None,
                        "elapsed_seconds": now - job_started,
                        "losses": losses, "device_actual": str(getattr(trainer, "device", device)),
                        "runtime": runtime,
                    })

                def on_fit_epoch_end(trainer: Any) -> None:
                    nonlocal cancellation_requested
                    epoch_index = int(getattr(trainer, "epoch", 0))
                    if active_epoch_index is None or epoch_index != active_epoch_index:
                        return
                    now = time.monotonic()
                    epoch_elapsed = now - epoch_started
                    epoch_durations.append(epoch_elapsed)
                    epoch_number = epoch_index + 1
                    epochs_total = int(getattr(trainer, "epochs", epochs))
                    metrics = getattr(trainer, "metrics", {})
                    if not isinstance(metrics, dict):
                        metrics = {}
                    safe_metrics = {str(key): _finite_number(value) for key, value in metrics.items()}
                    record["metrics"] = {**record.get("metrics", {}), **safe_metrics}
                    record["device_actual"] = str(getattr(trainer, "device", record.get("device_actual") or device))
                    record["current_epoch"] = epoch_number
                    record["epochs_total"] = epochs_total
                    record["early_stopped"] = bool(getattr(getattr(trainer, "stopper", None), "possible_stop", False))
                    _persist_run(record)
                    remaining = max(0, epochs_total - epoch_number)
                    eta = sum(epoch_durations) / len(epoch_durations) * remaining if len(epoch_durations) >= 2 else None
                    emit({
                        "stage": "epoch_completed", "run_id": run_id,
                        "epoch": epoch_number, "epochs_total": epochs_total,
                        "seconds_per_epoch": epoch_elapsed, "elapsed_seconds": now - job_started,
                        "estimated_remaining_seconds": eta, "metrics": safe_metrics,
                        "losses": record.get("losses", {}),
                        "device_actual": record["device_actual"],
                        "early_stopped": record["early_stopped"],
                        "runtime": _runtime_metrics(trainer, torch, psutil),
                    })
                    if cancel_file and cancel_file.exists() and epoch_number < epochs_total:
                        cancellation_requested = True
                        trainer.stop = True
                        emit({"stage": "cancel_at_checkpoint", "run_id": run_id, "epoch": epoch_number})

                def on_model_save(trainer: Any) -> None:
                    emit({
                        "stage": "checkpoint_saved", "run_id": run_id,
                        "best_checkpoint": str(getattr(trainer, "best", "")),
                        "last_checkpoint": str(getattr(trainer, "last", "")),
                    })

                model.add_callback("on_train_start", on_train_start)
                model.add_callback("on_train_epoch_start", on_epoch_start)
                model.add_callback("on_train_batch_end", on_batch_end)
                model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
                model.add_callback("on_model_save", on_model_save)
                train_args: Dict[str, Any] = {
                    "data": data_yaml,
                    "epochs": epochs,
                    "imgsz": effective_size,
                    "batch": batch,
                    "workers": workers,
                    "device": device,
                    "patience": patience,
                    "project": str(queue_root),
                    "name": run_name,
                    "exist_ok": True,
                    "save": True,
                    "save_period": save_period,
                    "cache": False,
                    "amp": device.startswith("cuda"),
                    "seed": seed,
                    "plots": True,
                    "verbose": True,
                }
                if lr0 is not None:
                    train_args["lr0"] = float(lr0)
                model.train(**train_args)
            log_file.close()

            trainer = getattr(model, "trainer", None)
            actual_run_dir = Path(getattr(trainer, "save_dir", run_dir)).resolve()
            record["output_dir"] = str(actual_run_dir)
            record["device_actual"] = str(getattr(trainer, "device", record.get("device_actual") or device))
            record["metrics"] = {**record.get("metrics", {}), **_read_results_csv(actual_run_dir / "results.csv")}
            best_value = getattr(trainer, "best", actual_run_dir / "weights" / "best.pt")
            last_value = getattr(trainer, "last", actual_run_dir / "weights" / "last.pt")
            best_path = Path(best_value).resolve() if best_value else actual_run_dir / "weights" / "best.pt"
            last_path = Path(last_value).resolve() if last_value else actual_run_dir / "weights" / "last.pt"
            record["best_checkpoint"] = str(best_path) if best_path.is_file() else None
            record["last_checkpoint"] = str(last_path) if last_path.is_file() else None
            checkpoint = best_path if best_path.is_file() else last_path if last_path.is_file() else None
            record["checkpoint_path"] = str(checkpoint) if checkpoint else None
            record["checkpoint_sha256"] = _sha256_file(checkpoint) if checkpoint else None
            record["elapsed_seconds"] = time.monotonic() - job_started
            record["completed_at"] = _now()
            if cancellation_requested:
                record["status"] = "cancelled"
            else:
                record["status"] = "completed"
            queue_jobs.append(record.copy())
            _persist_run(record)
            (actual_run_dir / "frameforge-config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            emit({
                "stage": "job_completed", "run_id": run_id, "status": record["status"],
                "requested_imgsz": requested_size, "effective_imgsz": effective_size,
                "device_actual": record["device_actual"], "metrics": record["metrics"],
                "best_checkpoint": record["best_checkpoint"], "last_checkpoint": record["last_checkpoint"],
                "elapsed_seconds": record["elapsed_seconds"],
            })
            if cancellation_requested:
                queue_cancelled = True
                break
        except Exception as error:
            try:
                if "log_file" in locals() and not log_file.closed:
                    log_file.close()
            except Exception:
                pass
            record["status"] = "failed"
            record["error"] = str(error)
            record["completed_at"] = _now()
            record["elapsed_seconds"] = time.monotonic() - job_started
            for name in ("best.pt", "last.pt"):
                candidate = run_dir / "weights" / name
                if candidate.is_file():
                    record["best_checkpoint" if name == "best.pt" else "last_checkpoint"] = str(candidate)
            checkpoint_value = record.get("best_checkpoint") or record.get("last_checkpoint")
            if checkpoint_value:
                checkpoint = Path(checkpoint_value)
                record["checkpoint_path"] = checkpoint_value
                record["checkpoint_sha256"] = _sha256_file(checkpoint)
            queue_jobs.append(record.copy())
            _persist_run(record)
            if not run_log.exists():
                run_log.write_text(str(error) + "\n", encoding="utf-8")
            (run_dir / "frameforge-config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit({"stage": "job_failed", "run_id": run_id, "requested_imgsz": requested_size, "error": str(error)})
            failures += 1
            if error_policy == "stop":
                break

    handled_sizes = {job["requested_imgsz"] for job in queue_jobs}
    if queue_cancelled:
        for index, size in enumerate(sizes):
            if size not in handled_sizes:
                queue_jobs.append({"requested_imgsz": size, "status": "not_started", "queue_id": queue_id, "reason": "Đã yêu cầu hủy queue."})
    elif failures and error_policy == "stop":
        for size in sizes:
            if size not in handled_sizes:
                queue_jobs.append({"requested_imgsz": size, "status": "not_started", "queue_id": queue_id, "reason": "Queue dừng theo chính sách khi job trước lỗi."})

    if queue_cancelled:
        queue_status = "cancelled"
    elif failures and error_policy == "stop":
        queue_status = "failed"
    elif failures:
        queue_status = "completed_with_errors"
    else:
        queue_status = "completed"
    result = {
        "queue_id": queue_id,
        "status": queue_status,
        "queue_root": str(queue_root),
        "dataset": dataset,
        "device_requested": requested_device,
        "device_selected": device,
        "jobs": queue_jobs,
        "completed_count": sum(job.get("status") == "completed" for job in queue_jobs),
        "failed_count": failures,
        "cancelled": queue_cancelled,
    }
    emit({"stage": "queue_completed", "queue_id": queue_id, "status": queue_status, "completed_count": result["completed_count"], "failed_count": failures})
    return result
