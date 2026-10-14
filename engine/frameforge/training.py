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
import shutil
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
_PROMOTION_METRICS = ("precision", "recall", "f1", "map50", "map50_95")
_PROMOTION_CLASS_METRICS = ("precision", "recall", "map50", "map50_95")
_PROMOTION_CONDITION_KEYS = (
    "imgsz", "confidence_threshold", "nms_iou_threshold", "batch",
    "device_actual", "framework", "framework_version",
)
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
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS model_evaluations (
            evaluation_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            record_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_targets (
            target_id TEXT PRIMARY KEY,
            updated_at TEXT NOT NULL,
            record_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS model_roles (
            role_id TEXT PRIMARY KEY,
            champion_evaluation_id TEXT,
            challenger_evaluation_id TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS model_decisions (
            decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            action TEXT NOT NULL,
            champion_before_id TEXT,
            champion_after_id TEXT,
            challenger_evaluation_id TEXT,
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


def _safe_evaluation_asset(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("Manifest fixed set thiếu đường dẫn {0} tương đối hợp lệ.".format(label))
    try:
        asset = (root / value).resolve(strict=True)
        asset.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError("Không đọc được {0} trong fixed set: {1}".format(label, value)) from error
    if not asset.is_file():
        raise ValueError("Fixed set thiếu tệp {0}: {1}".format(label, value))
    return asset


def _inspect_fixed_evaluation_set(
    data_yaml: str,
    split: str,
    progress: ProgressCallback = None,
    cancel_file: Optional[Path] = None,
) -> Dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError("Chỉ đánh giá trên split validation hoặc test cố định.")
    data_file = Path(data_yaml).expanduser().resolve(strict=True)
    if not data_file.is_file() or data_file.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Chọn data.yaml của dataset FrameForge đã export.")
    try:
        config = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("Không đọc được data.yaml: {0}".format(error)) from error
    if not isinstance(config, dict):
        raise ValueError("data.yaml phải là một object YAML.")
    names = _class_names(config.get("names"))
    root = _resolve_dataset_root(data_file, config).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Thư mục dataset không còn tồn tại.")
    manifest_path = data_file.parent / "manifest.json"
    try:
        manifest_path = manifest_path.resolve(strict=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Fixed evaluation yêu cầu manifest.json FrameForge hợp lệ: {0}".format(error)) from error
    if not isinstance(manifest, dict) or manifest.get("format") != "FrameForge YOLO dataset":
        raise ValueError("Fixed evaluation chỉ nhận manifest FrameForge YOLO.")
    if manifest_path.parent != root:
        raise ValueError("data.yaml và manifest phải trỏ cùng thư mục dataset FrameForge.")
    dataset_id = manifest.get("dataset_id")
    version = manifest.get("version")
    if not isinstance(dataset_id, str) or not dataset_id or isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Manifest thiếu dataset ID hoặc version hợp lệ.")
    expected_mapping = {str(index): name for index, name in enumerate(names)}
    if manifest.get("class_mapping") != expected_mapping:
        raise ValueError("Class schema trong data.yaml không khớp manifest.")
    set_key = "fixed_validation_set" if split == "val" else "frozen_test_set"
    fixed_set = manifest.get(set_key)
    if not isinstance(fixed_set, dict) or fixed_set.get("enabled") is not True:
        raise ValueError("Dataset chưa có {0} cố định đã review.".format("validation" if split == "val" else "test"))
    set_id = fixed_set.get("set_id")
    recorded_hashes = fixed_set.get("image_sha256s")
    if not isinstance(set_id, str) or not set_id or not isinstance(recorded_hashes, list) or not recorded_hashes:
        raise ValueError("Manifest fixed set thiếu ID hoặc danh sách ảnh.")
    if any(not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.casefold()) for value in recorded_hashes):
        raise ValueError("Danh sách fingerprint của fixed set không hợp lệ.")

    errors: List[str] = []
    split_images: Dict[str, List[Path]] = {}
    for split_name in ("train", "val", "test"):
        sources = _split_sources(config.get(split_name), split_name)
        images = _resolve_split_images(root, sources, split_name, errors)
        if not images:
            errors.append("Split {0} không có ảnh.".format(split_name))
        for image in images:
            try:
                image.relative_to(root)
            except ValueError:
                errors.append("Ảnh split {0} nằm ngoài dataset FrameForge.".format(split_name))
        split_images[split_name] = images
    if errors:
        raise ValueError("Dataset fixed evaluation không hợp lệ:\n- " + "\n- ".join(list(dict.fromkeys(errors))[:20]))

    rows = manifest.get("images")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest không có danh sách ảnh để xác minh.")
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ("train", "val", "test")}
    paths_by_split: Dict[str, set[Path]] = {name: set() for name in rows_by_split}
    digest_splits: Dict[str, str] = {}
    group_splits: Dict[str, str] = {}
    all_manifest_paths: set[Path] = set()
    integrity_errors: List[str] = []
    for index, row in enumerate(rows, start=1):
        if cancel_file and cancel_file.exists():
            raise RuntimeError("Đã hủy đánh giá theo yêu cầu.")
        if not isinstance(row, dict) or row.get("split") not in rows_by_split:
            raise ValueError("Manifest có split hoặc bản ghi ảnh không hợp lệ.")
        row_split = row["split"]
        image_path = _safe_evaluation_asset(root, row.get("exported_image"), "ảnh")
        label_path = _safe_evaluation_asset(root, row.get("exported_label"), "label")
        if image_path in all_manifest_paths:
            raise ValueError("Manifest có đường dẫn ảnh trùng lặp.")
        all_manifest_paths.add(image_path)
        paths_by_split[row_split].add(image_path)
        digest = row.get("sha256")
        label_digest = row.get("label_sha256")
        group_key = row.get("group_key")
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest.casefold())
            or not isinstance(label_digest, str) or len(label_digest) != 64
            or any(char not in "0123456789abcdef" for char in label_digest.casefold())
        ):
            raise ValueError("Manifest fixed set cần SHA-256 cho ảnh và label.")
        if not isinstance(group_key, str) or not group_key:
            raise ValueError("Manifest fixed set thiếu group key để kiểm tra rò rỉ.")
        prior_split = digest_splits.setdefault(digest, row_split)
        if prior_split != row_split:
            raise ValueError("Ảnh trùng fingerprint bị rò rỉ qua các split.")
        prior_group = group_splits.setdefault(group_key, row_split)
        if prior_group != row_split:
            raise ValueError("Nhóm nguồn bị rò rỉ qua train và evaluation.")
        if _sha256_file(image_path) != digest:
            integrity_errors.append("Ảnh split {0} khác SHA-256 manifest: {1}".format(row_split, image_path))
        rows_by_split[row_split].append({**row, "_image_path": image_path, "_label_path": label_path})
        if progress and (index == len(rows) or index % 50 == 0):
            progress({"stage": "verifying_fixed_set", "checked_images": index, "total_images": len(rows)})
    if integrity_errors:
        raise ValueError("Dataset FrameForge không còn khớp manifest:\n- " + "\n- ".join(integrity_errors[:20]))
    for split_name in rows_by_split:
        if paths_by_split[split_name] != set(split_images[split_name]):
            raise ValueError("Danh sách ảnh split {0} không khớp manifest FrameForge.".format(split_name))

    evaluation_rows = rows_by_split[split]
    image_hashes = [row["sha256"] for row in evaluation_rows]
    if not image_hashes or len(image_hashes) != len(set(image_hashes)) or sorted(recorded_hashes) != sorted(image_hashes):
        raise ValueError("Danh sách ảnh trong {0} cố định không khớp manifest.".format(split))
    errors = []
    class_instance_counts = {name: 0 for name in names}
    instance_count = 0
    evaluation_fingerprint = hashlib.sha256()
    for row in evaluation_rows:
        image_path = row["_image_path"]
        label_path = row["_label_path"]
        if row.get("status") != "reviewed":
            errors.append("Ảnh trong fixed set chưa được review: {0}".format(row.get("exported_image")))
        if _label_path(image_path).resolve() != label_path:
            errors.append("Đường dẫn label không khớp quy ước YOLO: {0}".format(row.get("exported_label")))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            errors.append("Ảnh fixed set không giải mã được: {0}".format(image_path))
        elif image.shape[1] != row.get("width") or image.shape[0] != row.get("height"):
            errors.append("Kích thước ảnh fixed set không khớp manifest: {0}".format(image_path))
        if _sha256_file(image_path) != row["sha256"]:
            errors.append("Ảnh fixed set khác SHA-256 manifest: {0}".format(image_path))
        label_bytes = label_path.read_bytes()
        if hashlib.sha256(label_bytes).hexdigest() != row["label_sha256"]:
            errors.append("Label fixed set khác SHA-256 manifest: {0}".format(label_path))
        label_errors: List[str] = []
        instance_count += _validate_label_file(label_path, len(names), split, label_errors)
        errors.extend(label_errors)
        for line in label_bytes.decode("utf-8").splitlines():
            fields = line.split()
            if len(fields) == 5:
                try:
                    class_id = int(fields[0])
                    class_instance_counts[names[class_id]] += 1
                except (ValueError, IndexError):
                    pass
        evaluation_fingerprint.update(row["sha256"].encode("ascii"))
        evaluation_fingerprint.update(row["label_sha256"].encode("ascii"))
    if errors:
        raise ValueError("Fixed evaluation set không còn hợp lệ:\n- " + "\n- ".join(list(dict.fromkeys(errors))[:20]))
    warnings = []
    if len(evaluation_rows) < 20:
        warnings.append("Fixed {0} chỉ có {1} ảnh; kết quả có thể chưa ổn định.".format(split, len(evaluation_rows)))
    for class_name, count in class_instance_counts.items():
        if count == 0:
            warnings.append("Class '{0}' không có ground-truth trong fixed {1}; AP/Recall của class không thể đo.".format(class_name, split))
    dataset_fingerprint = hashlib.sha256(data_file.read_bytes() + manifest_path.read_bytes()).hexdigest()
    return {
        "data_yaml": str(data_file),
        "dataset_root": str(root),
        "dataset_id": dataset_id,
        "dataset_version": version,
        "dataset_fingerprint": dataset_fingerprint,
        "manifest_sha256": _sha256_file(manifest_path),
        "split": split,
        "fixed_set_id": set_id,
        "class_names": names,
        "image_count": len(evaluation_rows),
        "instance_count": instance_count,
        "class_instance_counts": class_instance_counts,
        "evaluation_fingerprint": evaluation_fingerprint.hexdigest(),
        "warnings": warnings,
    }


_QUALITY_TARGET_KEYS = ("precision", "recall", "f1", "map50", "map50_95")
_TARGET_DIRECTIONS = {
    "precision": "min", "recall": "min", "f1": "min", "map50": "min", "map50_95": "min",
    "avg_latency_ms": "max", "p50_latency_ms": "max", "p95_latency_ms": "max", "p99_latency_ms": "max",
    "fps": "min", "throughput": "min", "max_model_size_mb": "max", "max_memory_mb": "max",
}


def _normalize_target_values(value: Any, allowed: Iterable[str]) -> Dict[str, Optional[float]]:
    if not isinstance(value, dict) or any(key not in set(allowed) for key in value):
        raise ValueError("Chỉ số mục tiêu không hợp lệ.")
    result: Dict[str, Optional[float]] = {}
    for key in allowed:
        target = value.get(key)
        if target is None or target == "":
            continue
        if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target):
            raise ValueError("Mục tiêu {0} phải là số hữu hạn hoặc để trống.".format(key))
        number = float(target)
        if key in _QUALITY_TARGET_KEYS and not 0 <= number <= 1:
            raise ValueError("Mục tiêu {0} phải nằm trong khoảng 0–1.".format(key))
        if key in _TARGET_DIRECTIONS and key not in _QUALITY_TARGET_KEYS and number <= 0:
            raise ValueError("Mục tiêu {0} phải lớn hơn 0.".format(key))
        result[key] = number
    return result


def get_evaluation_targets() -> Dict[str, Any]:
    connection = _registry_connection()
    try:
        row = connection.execute("SELECT record_json FROM evaluation_targets WHERE target_id='default'").fetchone()
    finally:
        connection.close()
    if row is None:
        return {"targets": {}, "per_class": {}}
    try:
        record = json.loads(row["record_json"])
    except (json.JSONDecodeError, TypeError):
        return {"targets": {}, "per_class": {}}
    return record if isinstance(record, dict) else {"targets": {}, "per_class": {}}


def save_evaluation_targets(arguments: Dict[str, Any]) -> Dict[str, Any]:
    data_file = Path(str(arguments.get("data_yaml", ""))).expanduser().resolve(strict=True)
    config = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Chọn data.yaml hợp lệ để lưu chỉ số mục tiêu.")
    names = _class_names(config.get("names"))
    overall = _normalize_target_values(arguments.get("targets", {}), _TARGET_DIRECTIONS)
    raw_per_class = arguments.get("per_class", {})
    if not isinstance(raw_per_class, dict) or any(name not in names for name in raw_per_class):
        raise ValueError("Mục tiêu theo class phải dùng tên class trong data.yaml.")
    per_class: Dict[str, Dict[str, Optional[float]]] = {}
    for name, values in raw_per_class.items():
        normalized = _normalize_target_values(values, _QUALITY_TARGET_KEYS)
        if normalized:
            per_class[name] = normalized
    record = {"targets": overall, "per_class": per_class, "class_names": names, "updated_at": _now()}
    connection = _registry_connection()
    try:
        connection.execute(
            "INSERT INTO evaluation_targets(target_id, updated_at, record_json) VALUES('default', ?, ?) "
            "ON CONFLICT(target_id) DO UPDATE SET updated_at=excluded.updated_at, record_json=excluded.record_json",
            (_now(), json.dumps(record, ensure_ascii=False, allow_nan=False)),
        )
        connection.commit()
    finally:
        connection.close()
    return record


def _target_status(target: Optional[float], value: Optional[float], direction: str) -> Optional[str]:
    if target is None:
        return None
    if value is None:
        return "not_measured"
    return "met" if (value >= target if direction == "min" else value <= target) else "not_met"


def _invalid_measurement(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _evaluation_target_statuses(metrics: Dict[str, Any], per_class: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
    overall_targets = targets.get("targets", {}) if isinstance(targets, dict) else {}
    class_targets = targets.get("per_class", {}) if isinstance(targets, dict) else {}
    measured_metrics = {"max_model_size_mb": "model_size_mb"}
    statuses: Dict[str, Any] = {
        key: _target_status(target, metrics.get(measured_metrics.get(key, key)), _TARGET_DIRECTIONS[key])
        for key, target in overall_targets.items() if key in _TARGET_DIRECTIONS
    }
    class_statuses: Dict[str, Dict[str, Optional[str]]] = {}
    for class_name, goals in class_targets.items():
        measured = per_class.get(class_name, {})
        class_statuses[class_name] = {
            key: _target_status(target, measured.get(key), _TARGET_DIRECTIONS[key])
            for key, target in goals.items() if key in _QUALITY_TARGET_KEYS
        }
    return {"overall": statuses, "per_class": class_statuses}


def evaluate_yolo_checkpoint(arguments: Dict[str, Any], progress: ProgressCallback = None) -> Dict[str, Any]:
    checkpoint = Path(str(arguments.get("checkpoint", ""))).expanduser().resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".pt":
        raise ValueError("Chọn checkpoint YOLO11n nội bộ đã đăng ký.")
    run = verified_checkpoint_record(checkpoint)
    if not run:
        raise ValueError("Chỉ đánh giá checkpoint YOLO11n nội bộ có registry và SHA-256 còn khớp.")
    split = arguments.get("split", "val")
    cancel_value = arguments.get("cancel_file")
    cancel_file: Optional[Path] = None
    if isinstance(cancel_value, str) and cancel_value:
        candidate = Path(cancel_value).expanduser().resolve()
        allowed_parent = (_training_root() / "cancel").resolve()
        allowed_parent.mkdir(parents=True, exist_ok=True)
        if candidate.parent != allowed_parent:
            raise ValueError("Cancel marker đánh giá nằm ngoài thư mục quản lý training của FrameForge.")
        cancel_file = candidate
    dataset = _inspect_fixed_evaluation_set(str(arguments.get("data_yaml", "")), split, progress, cancel_file)
    if run.get("class_names") != dataset["class_names"]:
        raise ValueError("Class schema của checkpoint không khớp dataset fixed evaluation.")
    try:
        image_size = arguments.get("imgsz", run.get("effective_imgsz") or run.get("requested_imgsz"))
        batch = arguments.get("batch", 8)
        confidence = arguments.get("confidence_threshold", 0.001)
        nms_iou = arguments.get("iou_threshold", 0.7)
        if isinstance(image_size, bool) or not isinstance(image_size, int) or not 1 <= image_size <= MAX_IMAGE_SIZE:
            raise ValueError("imgsz phải là số nguyên từ 1 đến {0} px.".format(MAX_IMAGE_SIZE))
        if isinstance(batch, bool) or not isinstance(batch, int) or not 1 <= batch <= 512:
            raise ValueError("Batch phải là số nguyên từ 1 đến 512.")
        for name, value in (("Confidence", confidence), ("NMS IoU", nms_iou)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("{0} phải nằm trong khoảng 0–1.".format(name))
        capabilities = training_capabilities()
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
        evaluation_id = str(uuid.uuid4())
        if progress:
            progress({"stage": "evaluation_started", "evaluation_id": evaluation_id, "split": split, "image_count": dataset["image_count"]})
        model = _load_yolo_model(str(checkpoint))
        if getattr(model, "task", "detect") != "detect":
            raise ValueError("Checkpoint không phải model object detection.")
        model_names = getattr(model, "names", {})
        if isinstance(model_names, dict):
            model_names = [model_names[index] for index in sorted(model_names)]
        if isinstance(model_names, (list, tuple)) and list(model_names) != dataset["class_names"]:
            raise ValueError("Class schema bên trong checkpoint không khớp dataset.")
        run_dir = Path(str(run.get("output_dir", checkpoint.parent))).resolve(strict=True)
        evaluation_dir = run_dir / "evaluations"
        evaluation_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        def on_validation_batch_end(validator: Any) -> None:
            if cancel_file and cancel_file.exists():
                raise RuntimeError("Đã hủy đánh giá theo yêu cầu.")
            if progress:
                loader = getattr(validator, "dataloader", None)
                total_batches = len(loader) if loader is not None else 0
                batch_index = int(getattr(validator, "batch_i", -1)) + 1
                total_images = len(getattr(loader, "dataset", [])) if loader is not None else 0
                checked_images = min(batch_index * batch, total_images) if total_images else batch_index * batch
                progress({
                    "stage": "evaluation_progress", "evaluation_id": evaluation_id,
                    "checked_images": checked_images, "total_images": total_images,
                    "batch": batch_index, "total_batches": total_batches,
                })

        actual_device = {"value": device}

        def on_validation_start(validator: Any) -> None:
            actual_device["value"] = str(getattr(validator, "device", device))

        if callable(getattr(model, "add_callback", None)):
            model.add_callback("on_val_start", on_validation_start)
            model.add_callback("on_val_batch_end", on_validation_batch_end)
        try:
            results = model.val(
                data=dataset["data_yaml"], split=split, imgsz=image_size,
                conf=float(confidence), iou=float(nms_iou), batch=batch, device=device,
                project=str(evaluation_dir), name=evaluation_id, exist_ok=True,
                plots=False, verbose=False,
            )
        except RuntimeError as error:
            if str(error).startswith("Đã hủy đánh giá"):
                shutil.rmtree(evaluation_dir / evaluation_id, ignore_errors=True)
            raise
        raw_metrics = getattr(results, "results_dict", {})
        if not isinstance(raw_metrics, dict):
            raw_metrics = {}
        box_metrics = getattr(results, "box", None)
        raw_quality = {
            "precision": raw_metrics.get("metrics/precision(B)", getattr(box_metrics, "mp", None)),
            "recall": raw_metrics.get("metrics/recall(B)", getattr(box_metrics, "mr", None)),
            "map50": raw_metrics.get("metrics/mAP50(B)", getattr(box_metrics, "map50", None)),
            "map50_95": raw_metrics.get("metrics/mAP50-95(B)", getattr(box_metrics, "map", None)),
        }
        invalid_overall = {key for key, value in raw_quality.items() if _invalid_measurement(value)}
        precision = _finite_number(raw_quality["precision"])
        recall = _finite_number(raw_quality["recall"])
        map50 = _finite_number(raw_quality["map50"])
        map50_95 = _finite_number(raw_quality["map50_95"])
        if "precision" in invalid_overall or "recall" in invalid_overall:
            invalid_overall.add("f1")
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0 if precision == 0 and recall == 0 else None
        metrics: Dict[str, Any] = {
            "precision": precision, "recall": recall, "f1": f1, "map50": map50, "map50_95": map50_95,
            "avg_latency_ms": None, "p50_latency_ms": None, "p95_latency_ms": None, "p99_latency_ms": None,
            "fps": None, "throughput": None, "model_size_mb": checkpoint.stat().st_size / 1_000_000, "max_memory_mb": None,
        }
        class_metrics = {
            name: {"precision": None, "recall": None, "f1": None, "map50": None, "map50_95": None,
                   "ground_truth_instances": count}
            for name, count in dataset["class_instance_counts"].items()
        }
        summaries = results.summary() if callable(getattr(results, "summary", None)) else []
        invalid_class_metrics: Dict[str, set[str]] = {}
        for item in summaries:
            if not isinstance(item, dict) or item.get("Class") not in class_metrics:
                continue
            class_name = item["Class"]
            invalid_class_metrics[class_name] = {
                key for key, summary_key in (("precision", "Box-P"), ("recall", "Box-R"), ("f1", "Box-F1"),
                                             ("map50", "mAP50"), ("map50_95", "mAP50-95"))
                if _invalid_measurement(item.get(summary_key))
            }
            class_metrics[item["Class"]].update({
                "precision": _finite_number(item.get("Box-P")), "recall": _finite_number(item.get("Box-R")),
                "f1": _finite_number(item.get("Box-F1")), "map50": _finite_number(item.get("mAP50")),
                "map50_95": _finite_number(item.get("mAP50-95")),
            })
        targets = get_evaluation_targets()
        per_class_targets = targets.get("per_class", {}) if isinstance(targets, dict) else {}
        scoped_targets = {
            "targets": targets.get("targets", {}) if isinstance(targets, dict) else {},
            "per_class": {name: per_class_targets[name] for name in dataset["class_names"] if name in per_class_targets},
        }
        target_status = _evaluation_target_statuses(metrics, class_metrics, scoped_targets)
        for key in invalid_overall:
            if key in target_status["overall"]:
                target_status["overall"][key] = "invalid_measurement"
        for class_name, invalid_keys in invalid_class_metrics.items():
            for key in invalid_keys:
                if key in target_status["per_class"].get(class_name, {}):
                    target_status["per_class"][class_name][key] = "invalid_measurement"
        if _sha256_file(checkpoint) != run.get("checkpoint_sha256"):
            raise ValueError("Checkpoint đã thay đổi trong lúc đánh giá; kết quả chưa được lưu.")
        evaluation = {
            "evaluation_id": evaluation_id, "created_at": _now(), "run_id": run["run_id"],
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": run["checkpoint_sha256"],
            "dataset_id": dataset["dataset_id"], "dataset_version": dataset["dataset_version"],
            "dataset_fingerprint": dataset["dataset_fingerprint"], "manifest_sha256": dataset["manifest_sha256"],
            "split": split, "fixed_set_id": dataset["fixed_set_id"],
            "evaluation_fingerprint": dataset["evaluation_fingerprint"],
            "image_count": dataset["image_count"], "instance_count": dataset["instance_count"],
            "class_names": dataset["class_names"], "class_instance_counts": dataset["class_instance_counts"],
            "metrics": metrics, "per_class": class_metrics, "targets": scoped_targets,
            "target_status": target_status, "warnings": dataset["warnings"],
            "conditions": {
                "imgsz": image_size, "confidence_threshold": float(confidence), "nms_iou_threshold": float(nms_iou),
                "batch": batch, "device_requested": requested_device, "device_actual": actual_device["value"],
                "hardware": capabilities["hardware"], "framework": "Ultralytics", "framework_version": capabilities["ultralytics_version"],
                "checkpoint_sha256": run["checkpoint_sha256"], "dataset_fingerprint": dataset["dataset_fingerprint"],
                "latency_status": "not_measured",
            },
        }
        connection = _registry_connection()
        try:
            connection.execute(
                "INSERT INTO model_evaluations(evaluation_id, run_id, created_at, record_json) VALUES(?, ?, ?, ?)",
                (evaluation_id, run["run_id"], evaluation["created_at"], json.dumps(evaluation, ensure_ascii=False, allow_nan=False)),
            )
            if split == "val":
                connection.execute(
                    "INSERT OR IGNORE INTO model_roles(role_id, champion_evaluation_id, challenger_evaluation_id, updated_at) "
                    "VALUES('default', NULL, NULL, ?)",
                    (_now(),),
                )
                connection.execute(
                    "UPDATE model_roles SET challenger_evaluation_id=?, updated_at=? WHERE role_id='default'",
                    (evaluation_id, _now()),
                )
            connection.commit()
        finally:
            connection.close()
        (evaluation_dir / (evaluation_id + ".json")).write_text(json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if progress:
            progress({"stage": "evaluation_completed", "evaluation_id": evaluation_id, "metrics": metrics})
        return evaluation
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(str(error)) from error


def list_model_evaluations(run_id: Optional[str] = None) -> Dict[str, Any]:
    connection = _registry_connection()
    try:
        if run_id:
            rows = connection.execute(
                "SELECT record_json FROM model_evaluations WHERE run_id=? ORDER BY created_at DESC LIMIT 100", (run_id,)
            ).fetchall()
        else:
            rows = connection.execute("SELECT record_json FROM model_evaluations ORDER BY created_at DESC LIMIT 100").fetchall()
    finally:
        connection.close()
    evaluations = []
    for row in rows:
        try:
            record = json.loads(row["record_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(record, dict):
            evaluations.append(record)
    return {"evaluations": evaluations}


def _load_evaluation(connection: sqlite3.Connection, evaluation_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not evaluation_id:
        return None
    row = connection.execute(
        "SELECT record_json FROM model_evaluations WHERE evaluation_id=?", (evaluation_id,)
    ).fetchone()
    if row is None:
        raise ValueError("Kết quả đánh giá không còn trong registry.")
    try:
        record = json.loads(row["record_json"])
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Kết quả đánh giá trong registry bị hỏng.") from error
    if not isinstance(record, dict) or record.get("evaluation_id") != evaluation_id:
        raise ValueError("Kết quả đánh giá trong registry không hợp lệ.")
    return record


def _evaluation_checkpoint_error(evaluation: Dict[str, Any], connection: sqlite3.Connection) -> Optional[str]:
    run_id = evaluation.get("run_id")
    checkpoint_value = evaluation.get("checkpoint_path")
    if not isinstance(run_id, str) or not isinstance(checkpoint_value, str):
        return "Thiếu checkpoint hoặc mã run trong kết quả đánh giá."
    try:
        checkpoint = Path(checkpoint_value).expanduser().resolve(strict=True)
    except (OSError, ValueError):
        return "Checkpoint của kết quả đánh giá không còn tồn tại."
    row = connection.execute(
        "SELECT status, checkpoint_path, checkpoint_sha256, record_json FROM training_runs WHERE run_id=?",
        (run_id,),
    ).fetchone()
    if row is None:
        return "Run của kết quả đánh giá không còn trong registry."
    try:
        run = json.loads(row["record_json"])
        registered_path = Path(row["checkpoint_path"]).expanduser().resolve(strict=True)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return "Thông tin checkpoint trong registry không hợp lệ."
    if (
        not isinstance(run, dict)
        or registered_path != checkpoint
        or run.get("run_id") != run_id
        or run.get("status") != row["status"]
        or not _is_internal_training_record(run, checkpoint)
    ):
        return "Checkpoint không còn là model YOLO11n nội bộ đã xác minh."
    digest = evaluation.get("checkpoint_sha256")
    conditions = evaluation.get("conditions") if isinstance(evaluation.get("conditions"), dict) else {}
    if (
        not isinstance(digest, str)
        or digest != row["checkpoint_sha256"]
        or digest != run.get("checkpoint_sha256")
        or conditions.get("checkpoint_sha256", digest) != digest
    ):
        return "Checksum checkpoint của kết quả đánh giá không khớp registry."
    try:
        if _sha256_file(checkpoint) != digest:
            return "Checkpoint đã đổi checksum kể từ lần đánh giá."
    except OSError:
        return "Không đọc được checkpoint để xác minh checksum."
    return None


def _current_promotion_targets(connection: sqlite3.Connection) -> Dict[str, Any]:
    row = connection.execute(
        "SELECT record_json FROM evaluation_targets WHERE target_id='default'"
    ).fetchone()
    if row is None:
        return {"targets": {}, "per_class": {}}
    try:
        record = json.loads(row["record_json"])
    except (json.JSONDecodeError, TypeError):
        return {"targets": {}, "per_class": {}}
    return record if isinstance(record, dict) else {"targets": {}, "per_class": {}}


def _promotion_target_failures(evaluation: Dict[str, Any], targets: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    class_names = evaluation.get("class_names")
    configured_class_names = targets.get("class_names")
    if configured_class_names and configured_class_names != class_names:
        failures.append("Class schema đã đổi kể từ khi lưu chỉ số mục tiêu.")
    metrics = evaluation.get("metrics") if isinstance(evaluation.get("metrics"), dict) else {}
    target_values = targets.get("targets") if isinstance(targets.get("targets"), dict) else {}
    metric_aliases = {"max_model_size_mb": "model_size_mb"}
    for key, target in target_values.items():
        if target is None:
            continue
        direction = _TARGET_DIRECTIONS.get(key)
        if direction is None or isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target):
            failures.append("Chỉ số mục tiêu {0} không hợp lệ.".format(key))
            continue
        value = metrics.get(metric_aliases.get(key, key))
        status = _target_status(float(target), _finite_number(value), direction)
        if status != "met":
            failures.append("Challenger chưa đạt mục tiêu {0} ({1}).".format(key, status or "chưa đo"))
    class_targets = targets.get("per_class") if isinstance(targets.get("per_class"), dict) else {}
    class_metrics = evaluation.get("per_class") if isinstance(evaluation.get("per_class"), dict) else {}
    for class_name, values in class_targets.items():
        if not isinstance(values, dict):
            failures.append("Chỉ số mục tiêu theo class {0} không hợp lệ.".format(class_name))
            continue
        measured = class_metrics.get(class_name) if isinstance(class_metrics.get(class_name), dict) else {}
        for key, target in values.items():
            if target is None:
                continue
            direction = _TARGET_DIRECTIONS.get(key)
            if key not in _QUALITY_TARGET_KEYS or direction is None or isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target):
                failures.append("Chỉ số mục tiêu {0}/{1} không hợp lệ.".format(class_name, key))
                continue
            status = _target_status(float(target), _finite_number(measured.get(key)), direction)
            if status != "met":
                failures.append("Challenger chưa đạt mục tiêu {0}/{1} ({2}).".format(class_name, key, status or "chưa đo"))
    return failures


def _promotion_metric(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _stable_hardware(hardware: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(hardware, dict):
        return None
    keys = ("os", "architecture", "processor", "logical_cpu_count", "physical_cpu_count", "memory_total_bytes")
    return {key: hardware.get(key) for key in keys}


def _promotion_comparison(
    connection: sqlite3.Connection,
    champion: Optional[Dict[str, Any]],
    challenger: Dict[str, Any],
    max_regression: float,
) -> Dict[str, Any]:
    blockers: List[str] = []
    incompatibilities: List[str] = []
    regressions: List[str] = []
    if challenger.get("split") != "val":
        incompatibilities.append("Promotion chỉ dùng fixed validation; test là holdout cuối.")
    if not challenger.get("fixed_set_id") or not challenger.get("evaluation_fingerprint"):
        incompatibilities.append("Challenger không có fingerprint của fixed validation set.")
    challenger_checkpoint_error = _evaluation_checkpoint_error(challenger, connection)
    if challenger_checkpoint_error:
        blockers.append(challenger_checkpoint_error)
    targets = _current_promotion_targets(connection)
    target_failures = _promotion_target_failures(challenger, targets)
    blockers.extend(target_failures)
    overall: Dict[str, Dict[str, Any]] = {}
    per_class: Dict[str, Dict[str, Dict[str, Any]]] = {}
    comparable = champion is None
    if champion is not None:
        for key in (
            "split", "fixed_set_id", "evaluation_fingerprint", "image_count", "instance_count",
            "class_names", "class_instance_counts",
        ):
            if champion.get(key) != challenger.get(key):
                incompatibilities.append("Champion và Challenger khác {0}.".format(key))
        champion_conditions = champion.get("conditions") if isinstance(champion.get("conditions"), dict) else {}
        challenger_conditions = challenger.get("conditions") if isinstance(challenger.get("conditions"), dict) else {}
        for key in _PROMOTION_CONDITION_KEYS:
            if (
                key not in champion_conditions
                or key not in challenger_conditions
                or champion_conditions.get(key) is None
                or challenger_conditions.get(key) is None
                or champion_conditions.get(key) != challenger_conditions.get(key)
            ):
                incompatibilities.append("Điều kiện đánh giá khác nhau tại {0}.".format(key))
        champion_hardware = _stable_hardware(champion_conditions.get("hardware"))
        challenger_hardware = _stable_hardware(challenger_conditions.get("hardware"))
        if champion_hardware is None or challenger_hardware is None or champion_hardware != challenger_hardware:
            incompatibilities.append("Champion và Challenger được đo trên phần cứng khác nhau.")
        if champion.get("split") != "val":
            incompatibilities.append("Champion không có kết quả trên fixed validation.")
        champion_checkpoint_error = _evaluation_checkpoint_error(champion, connection)
        if champion_checkpoint_error:
            blockers.append("Champion: " + champion_checkpoint_error)
        comparable = not incompatibilities
    candidate_metrics = challenger.get("metrics") if isinstance(challenger.get("metrics"), dict) else {}
    champion_metrics = champion.get("metrics") if champion and isinstance(champion.get("metrics"), dict) else {}
    for key in _PROMOTION_METRICS:
        new_value = _promotion_metric(candidate_metrics.get(key))
        old_value = _promotion_metric(champion_metrics.get(key)) if champion else None
        if new_value is None:
            blockers.append("Challenger thiếu metric hợp lệ {0}.".format(key))
        if champion is not None and old_value is None:
            blockers.append("Champion thiếu metric hợp lệ {0}.".format(key))
        delta = new_value - old_value if comparable and new_value is not None and old_value is not None else None
        is_regression = delta is not None and delta < -max_regression
        if is_regression:
            regressions.append("{0}: giảm {1:.4f}, vượt tolerance {2:.4f}.".format(key, -delta, max_regression))
        overall[key] = {
            "champion": old_value,
            "challenger": new_value,
            "delta": delta,
            "regressed": is_regression,
        }
    candidate_classes = challenger.get("per_class") if isinstance(challenger.get("per_class"), dict) else {}
    champion_classes = champion.get("per_class") if champion and isinstance(champion.get("per_class"), dict) else {}
    class_counts = challenger.get("class_instance_counts") if isinstance(challenger.get("class_instance_counts"), dict) else {}
    champion_counts = champion.get("class_instance_counts") if champion and isinstance(champion.get("class_instance_counts"), dict) else {}
    class_names = challenger.get("class_names") if isinstance(challenger.get("class_names"), list) else []
    for class_name in class_names:
        per_class[class_name] = {}
        count = class_counts.get(class_name, 0)
        prior_count = champion_counts.get(class_name, count) if champion else count
        new_class_metrics = candidate_classes.get(class_name) if isinstance(candidate_classes.get(class_name), dict) else {}
        old_class_metrics = champion_classes.get(class_name) if isinstance(champion_classes.get(class_name), dict) else {}
        for key in _PROMOTION_CLASS_METRICS:
            new_value = _promotion_metric(new_class_metrics.get(key))
            old_value = _promotion_metric(old_class_metrics.get(key)) if champion else None
            if count and new_value is None:
                blockers.append("Challenger thiếu {0} cho class {1}.".format(key, class_name))
            if champion is not None and prior_count and old_value is None:
                blockers.append("Champion thiếu {0} cho class {1}.".format(key, class_name))
            delta = new_value - old_value if comparable and champion is not None and new_value is not None and old_value is not None and count and prior_count else None
            is_regression = delta is not None and delta < -max_regression
            if is_regression:
                regressions.append("{0}/{1}: giảm {2:.4f}, vượt tolerance {3:.4f}.".format(class_name, key, -delta, max_regression))
            per_class[class_name][key] = {
                "champion": old_value,
                "challenger": new_value,
                "delta": delta,
                "regressed": is_regression,
                "ground_truth_instances": count,
            }
    blockers.extend(incompatibilities)
    blockers.extend(regressions)
    can_promote = not blockers
    recommendation = (
        "incomparable" if incompatibilities
        else "retain_champion" if blockers
        else "initial_champion" if champion is None
        else "promote_challenger"
    )
    return {
        "created_at": _now(),
        "champion_evaluation_id": champion.get("evaluation_id") if champion else None,
        "challenger_evaluation_id": challenger.get("evaluation_id"),
        "comparable": comparable,
        "can_promote": can_promote,
        "recommendation": recommendation,
        "max_regression": max_regression,
        "overall": overall,
        "per_class": per_class,
        "blockers": list(dict.fromkeys(blockers)),
        "regressions": regressions,
        "target_failures": target_failures,
        "targets": targets,
        "warnings": challenger.get("warnings", []),
    }


def _promotion_role(connection: sqlite3.Connection) -> Dict[str, Optional[str]]:
    row = connection.execute(
        "SELECT champion_evaluation_id, challenger_evaluation_id FROM model_roles WHERE role_id='default'"
    ).fetchone()
    if row is None:
        return {"champion_evaluation_id": None, "challenger_evaluation_id": None}
    return {
        "champion_evaluation_id": row["champion_evaluation_id"],
        "challenger_evaluation_id": row["challenger_evaluation_id"],
    }


def _record_model_decision(
    connection: sqlite3.Connection,
    action: str,
    champion_before_id: Optional[str],
    champion_after_id: Optional[str],
    challenger_evaluation_id: Optional[str],
    record: Dict[str, Any],
) -> None:
    created_at = str(record.get("created_at") or _now())
    decision = {
        **record,
        "action": action,
        "champion_before_evaluation_id": champion_before_id,
        "champion_after_evaluation_id": champion_after_id,
        "challenger_evaluation_id": challenger_evaluation_id,
        "created_at": created_at,
        "decision_by": "user",
    }
    connection.execute(
        "INSERT INTO model_decisions(created_at, action, champion_before_id, champion_after_id, challenger_evaluation_id, record_json) "
        "VALUES(?, ?, ?, ?, ?, ?)",
        (created_at, action, champion_before_id, champion_after_id, challenger_evaluation_id,
         json.dumps(decision, ensure_ascii=False, allow_nan=False)),
    )


def compare_model_challenger(arguments: Dict[str, Any]) -> Dict[str, Any]:
    tolerance = arguments.get("max_regression", 0)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("Tolerance regression phải nằm trong khoảng 0–1.")
    connection = _registry_connection()
    try:
        roles = _promotion_role(connection)
        challenger_id = roles["challenger_evaluation_id"]
        if not challenger_id:
            raise ValueError("Chưa có Challenger. Chạy đánh giá trên fixed validation trước.")
        challenger = _load_evaluation(connection, challenger_id)
        champion = _load_evaluation(connection, roles["champion_evaluation_id"])
        if champion and challenger and champion.get("evaluation_id") == challenger.get("evaluation_id"):
            raise ValueError("Kết quả hiện tại đã là Champion.")
        comparison = _promotion_comparison(connection, champion, challenger or {}, float(tolerance))
        _record_model_decision(
            connection, "compared", roles["champion_evaluation_id"], roles["champion_evaluation_id"],
            challenger_id, comparison,
        )
        connection.commit()
        return comparison
    except sqlite3.Error as error:
        raise ValueError("Không thể đọc hoặc lưu lịch sử so sánh model: {0}".format(error)) from error
    finally:
        connection.close()


def promote_model_challenger(arguments: Dict[str, Any]) -> Dict[str, Any]:
    tolerance = arguments.get("max_regression", 0)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("Tolerance regression phải nằm trong khoảng 0–1.")
    connection = _registry_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT OR IGNORE INTO model_roles(role_id, champion_evaluation_id, challenger_evaluation_id, updated_at) "
            "VALUES('default', NULL, NULL, ?)",
            (_now(),),
        )
        roles = _promotion_role(connection)
        challenger_id = roles["challenger_evaluation_id"]
        if not challenger_id:
            raise ValueError("Không có Challenger đang chờ promote.")
        challenger = _load_evaluation(connection, challenger_id)
        champion = _load_evaluation(connection, roles["champion_evaluation_id"])
        comparison = _promotion_comparison(connection, champion, challenger or {}, float(tolerance))
        if not comparison["can_promote"]:
            raise ValueError("Chưa thể promote: " + " ".join(comparison["blockers"][:6]))
        created_at = _now()
        decision = {**comparison, "created_at": created_at, "reason": "Người dùng xác nhận promote Challenger."}
        _record_model_decision(connection, "promoted", roles["champion_evaluation_id"], challenger_id, challenger_id, decision)
        connection.execute(
            "UPDATE model_roles SET champion_evaluation_id=?, challenger_evaluation_id=NULL, updated_at=? WHERE role_id='default'",
            (challenger_id, created_at),
        )
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise ValueError("Không thể lưu quyết định promote: {0}".format(error)) from error
    except (OSError, TypeError, ValueError):
        connection.rollback()
        raise
    finally:
        connection.close()
    return get_model_promotion_state()


def rollback_model_promotion() -> Dict[str, Any]:
    connection = _registry_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT OR IGNORE INTO model_roles(role_id, champion_evaluation_id, challenger_evaluation_id, updated_at) "
            "VALUES('default', NULL, NULL, ?)",
            (_now(),),
        )
        roles = _promotion_role(connection)
        latest = connection.execute(
            "SELECT action, champion_before_id, champion_after_id FROM model_decisions "
            "WHERE action IN ('promoted', 'rolled_back') ORDER BY decision_id DESC LIMIT 1"
        ).fetchone()
        if latest is None or latest["action"] != "promoted" or roles["champion_evaluation_id"] != latest["champion_after_id"]:
            raise ValueError("Không có lần promote gần nhất để rollback.")
        previous_id = latest["champion_before_id"]
        previous = _load_evaluation(connection, previous_id)
        if previous:
            issue = _evaluation_checkpoint_error(previous, connection)
            if issue:
                raise ValueError("Không thể khôi phục Champion trước: " + issue)
        current_id = roles["champion_evaluation_id"]
        created_at = _now()
        decision = {
            "created_at": created_at,
            "reason": "Người dùng yêu cầu rollback về Champion trước.",
        }
        _record_model_decision(connection, "rolled_back", current_id, previous_id, current_id, decision)
        connection.execute(
            "UPDATE model_roles SET champion_evaluation_id=?, challenger_evaluation_id=?, updated_at=? WHERE role_id='default'",
            (previous_id, current_id, created_at),
        )
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise ValueError("Không thể lưu quyết định rollback: {0}".format(error)) from error
    except (OSError, TypeError, ValueError):
        connection.rollback()
        raise
    finally:
        connection.close()
    return get_model_promotion_state()


def get_model_promotion_state() -> Dict[str, Any]:
    connection = _registry_connection()
    try:
        existing_roles = connection.execute(
            "SELECT 1 FROM model_roles WHERE role_id='default'"
        ).fetchone()
        connection.execute(
            "INSERT OR IGNORE INTO model_roles(role_id, champion_evaluation_id, challenger_evaluation_id, updated_at) "
            "VALUES('default', NULL, NULL, ?)",
            (_now(),),
        )
        roles = _promotion_role(connection)
        if existing_roles is None and not roles["challenger_evaluation_id"]:
            legacy_rows = connection.execute(
                "SELECT evaluation_id, record_json FROM model_evaluations ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
            legacy_candidate_id = None
            for legacy_row in legacy_rows:
                try:
                    legacy_record = json.loads(legacy_row["record_json"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(legacy_record, dict) and legacy_record.get("split") == "val":
                    legacy_candidate_id = legacy_row["evaluation_id"]
                    break
            if legacy_candidate_id:
                connection.execute(
                    "UPDATE model_roles SET challenger_evaluation_id=?, updated_at=? WHERE role_id='default'",
                    (legacy_candidate_id, _now()),
                )
                roles["challenger_evaluation_id"] = legacy_candidate_id
        champion = _load_evaluation(connection, roles["champion_evaluation_id"])
        challenger = _load_evaluation(connection, roles["challenger_evaluation_id"])
        rows = connection.execute(
            "SELECT decision_id, action, record_json FROM model_decisions ORDER BY decision_id DESC LIMIT 30"
        ).fetchall()
        latest_state_change = connection.execute(
            "SELECT action, champion_after_id FROM model_decisions "
            "WHERE action IN ('promoted', 'rolled_back') ORDER BY decision_id DESC LIMIT 1"
        ).fetchone()
        connection.commit()
    except sqlite3.Error as error:
        raise ValueError("Không thể đọc trạng thái Champion/Challenger: {0}".format(error)) from error
    finally:
        connection.close()
    history: List[Dict[str, Any]] = []
    for row in rows:
        try:
            record = json.loads(row["record_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(record, dict):
            record["decision_id"] = row["decision_id"]
            record["action"] = row["action"]
            history.append(record)
    can_rollback = bool(
        latest_state_change
        and latest_state_change["action"] == "promoted"
        and roles["champion_evaluation_id"] == latest_state_change["champion_after_id"]
    )
    return {
        "champion": champion,
        "challenger": challenger,
        "history": history,
        "can_rollback": can_rollback,
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
