"""Trusted YOLO11n checkpoint import, registry, and inference runtime capabilities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MAX_IMAGE_SIZE = 2048
MAX_IMPORTED_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
MAX_CHECKPOINT_RUN_METADATA_BYTES = 2 * 1024 * 1024


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


def _model_root() -> Path:
    # Kept at the historical "training" location so previously imported models stay registered.
    root = _default_app_data_dir() / "training"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _registry_connection() -> sqlite3.Connection:
    path = _model_root() / "registry.sqlite3"
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


def _sha256_file(path: Path, max_bytes: Optional[int] = None) -> str:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            if max_bytes is not None and size > max_bytes:
                raise ValueError("Tệp vượt giới hạn kích thước cho phép.")
            digest.update(block)
    return digest.hexdigest()


def _valid_imgsz(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 1 <= value <= MAX_IMAGE_SIZE:
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = value
        if (
            isinstance(height, int) and not isinstance(height, bool)
            and isinstance(width, int) and not isinstance(width, bool)
            and height == width and 1 <= height <= MAX_IMAGE_SIZE
        ):
            return height
    return None


def _checkpoint_imgsz(model: Any, checkpoint: Path) -> tuple[int, str, Optional[str]]:
    overrides = getattr(model, "overrides", {})
    checkpoint_imgsz = _valid_imgsz(overrides.get("imgsz")) if isinstance(overrides, dict) else None
    if checkpoint_imgsz is not None:
        return checkpoint_imgsz, "checkpoint_metadata", None

    run_dir = checkpoint.parent.parent if checkpoint.parent.name.casefold() == "weights" else checkpoint.parent
    args_path = run_dir / "args.yaml"
    try:
        if args_path.is_symlink() or not args_path.is_file() or args_path.stat().st_size > MAX_CHECKPOINT_RUN_METADATA_BYTES:
            raise ValueError("args.yaml absent or exceeds metadata limit")
        config = yaml.safe_load(args_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError, ValueError):
        config = None
    run_imgsz = _valid_imgsz(config.get("imgsz")) if isinstance(config, dict) else None
    if run_imgsz is not None:
        return run_imgsz, "run_args_yaml", str(args_path)
    return 640, "ultralytics_default", None


def _is_trusted_model_record(record: Dict[str, Any], checkpoint: Path) -> bool:
    """Accept only managed YOLO11n detection checkpoints imported through the trust flow."""
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
    """Return trusted registry metadata only when the registered file hash matches."""
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
        if not _is_trusted_model_record(record, registered_path):
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


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inference_capabilities() -> Dict[str, Any]:
    """Report installed inference support and the compute backend PyTorch actually exposes."""
    missing = [name for name in ("torch", "ultralytics") if importlib.util.find_spec(name) is None]
    result: Dict[str, Any] = {
        "available": not missing,
        "missing_packages": missing,
        "install_command": 'python -m pip install -e "engine[inference]"',
        "python_version": platform.python_version(),
        "torch_version": _package_version("torch"),
        "ultralytics_version": _package_version("ultralytics"),
        "devices": {"cpu": True, "mps": False, "cuda": False},
        "preferred_device": "cpu",
    }
    if "torch" not in missing:
        try:
            import torch

            mps = getattr(torch.backends, "mps", None)
            mps_available = bool(mps and mps.is_available())
            cuda_available = bool(torch.cuda.is_available())
            result["devices"] = {"cpu": True, "mps": mps_available, "cuda": cuda_available}
            if mps_available:
                result["preferred_device"] = "mps"
            elif cuda_available:
                result["preferred_device"] = "cuda:0"
            if cuda_available:
                result["device_name"] = torch.cuda.get_device_name(0)
                result["gpu_backend"] = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            elif mps_available:
                result["device_name"] = "Apple Metal"
        except Exception as error:
            result["torch_error"] = str(error)
            result["available"] = False
            result["missing_packages"].append("torch (không nạp được)")
    return result


def _class_names(value: Any) -> List[str]:
    if isinstance(value, list):
        names = value
    elif isinstance(value, dict):
        try:
            indexed = {int(key): name for key, name in value.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Tên class trong checkpoint phải dùng ID số nguyên liên tục từ 0.") from error
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("ID class trong checkpoint phải liên tục từ 0.")
        names = [indexed[index] for index in range(len(indexed))]
    else:
        raise ValueError("Checkpoint thiếu danh sách names của class.")
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Danh sách names trong checkpoint phải có ít nhất một tên class hợp lệ.")
    normalized = [name.strip() for name in names]
    if len({name.casefold() for name in normalized}) != len(normalized):
        raise ValueError("Tên class trong checkpoint bị trùng nhau.")
    return normalized


def import_yolo_checkpoint(path_value: str, trust_source: Any) -> Dict[str, Any]:
    """Validate and copy a user-trusted YOLO11n detection checkpoint into FrameForge."""
    if trust_source is not True:
        raise ValueError("Hãy xác nhận bạn tin cậy nguồn checkpoint trước khi nạp tệp `.pt`.")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("Chọn tệp checkpoint YOLO11n `.pt`.")
    try:
        source = Path(path_value).expanduser().resolve(strict=True)
    except (OSError, ValueError) as error:
        raise ValueError("Không tìm thấy checkpoint đã chọn.") from error
    if not source.is_file() or source.suffix.casefold() != ".pt":
        raise ValueError("Chỉ nhận checkpoint YOLO11n detection dạng `.pt`.")
    source_size = source.stat().st_size
    if source_size <= 0 or source_size > MAX_IMPORTED_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint phải có kích thước lớn hơn 0 và không vượt quá 2 GiB.")
    source_sha256 = _sha256_file(source, MAX_IMPORTED_CHECKPOINT_BYTES)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Import checkpoint cần PyTorch và Ultralytics trong Python engine.") from error
    run_id = str(uuid.uuid4())
    display_name = "".join(character for character in source.name if character.isprintable()).strip()[:120]
    import_root = _model_root() / "imported"
    import_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir = import_root / run_id
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, mode=0o700)
    temporary = weights_dir / (".best-{0}.partial.pt".format(uuid.uuid4().hex))
    checkpoint = weights_dir / "best.pt"
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            copied_digest = hashlib.sha256()
            copied_size = 0
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                copied_size += len(block)
                if copied_size > MAX_IMPORTED_CHECKPOINT_BYTES:
                    raise ValueError("Checkpoint vượt quá giới hạn 2 GiB trong khi sao chép.")
                copied_digest.update(block)
                output_stream.write(block)
        os.chmod(temporary, 0o600)
        if (
            copied_size != source_size
            or copied_digest.hexdigest() != source_sha256
            or _sha256_file(source, MAX_IMPORTED_CHECKPOINT_BYTES) != source_sha256
        ):
            raise ValueError("Checksum nguồn hoặc bản sao checkpoint không khớp.")

        try:
            model = YOLO(str(temporary))
        except Exception as error:
            raise ValueError("Không nạp được checkpoint đã xác nhận tin cậy: {0}".format(error)) from error

        if getattr(model, "task", None) != "detect":
            raise ValueError("Checkpoint chỉ hỗ trợ task object detection.")
        model_config = getattr(getattr(model, "model", None), "yaml", None)
        if not isinstance(model_config, dict):
            raise ValueError("Không xác định được kiến trúc trong checkpoint.")
        config_name = Path(str(model_config.get("yaml_file", ""))).stem.casefold()
        model_scale = model_config.get("scale")
        is_yolo11n = (
            (config_name == "yolo11" and model_scale == "n")
            or config_name == "yolo11n"
            or config_name.startswith("yolo11n-")
        )
        if not is_yolo11n:
            raise ValueError("Chỉ hỗ trợ kiến trúc YOLO11n; checkpoint khác không được đăng ký.")
        class_names = _class_names(getattr(model, "names", None))

        if _sha256_file(temporary, MAX_IMPORTED_CHECKPOINT_BYTES) != source_sha256:
            raise ValueError("Bản sao checkpoint thay đổi trong khi xác minh; hãy nhập lại tệp.")
        os.replace(temporary, checkpoint)
        requested_imgsz, imgsz_source, imgsz_source_path = _checkpoint_imgsz(model, source)
        now = _now()
        record = {
            "run_id": run_id,
            "source": "frameforge_import",
            "checkpoint_trusted": True,
            "queue_id": "import-{0}".format(run_id),
            "created_at": now,
            "imported_at": now,
            "completed_at": now,
            "status": "completed",
            "model": "YOLO11n",
            "architecture": "yolo11n",
            "task": "detect",
            "display_name": display_name or "YOLO11n",
            "requested_imgsz": requested_imgsz,
            "effective_imgsz": requested_imgsz,
            "imgsz_source": imgsz_source,
            "imgsz_source_path": imgsz_source_path,
            "base_model": "external checkpoint",
            "parent_model_id": None,
            "class_names": class_names,
            "output_dir": str(run_dir),
            "checkpoint_path": str(checkpoint),
            "best_checkpoint": str(checkpoint),
            "last_checkpoint": None,
            "checkpoint_sha256": source_sha256,
            "origin_path": str(source),
            "origin_sha256": source_sha256,
            "settings": {},
            "metrics": {},
            "error": None,
        }
        _persist_run(record)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return {
        "run_id": run_id,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": source_sha256,
        "class_names": class_names,
        "architecture": "yolo11n",
        "display_name": display_name or "YOLO11n",
        "requested_imgsz": requested_imgsz,
        "imgsz_source": imgsz_source,
    }


def list_models() -> Dict[str, Any]:
    """List registered checkpoints with their current existence, checksum, and trust state."""
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
            and _is_trusted_model_record(record, checkpoint.resolve())
        )
        record["checkpoint_trusted"] = trusted
        record["checkpoint_usable"] = exists and checksum_matches is True and trusted
        runs.append(record)
    return {"runs": runs, "available_count": sum(bool(run["checkpoint_usable"]) for run in runs)}


def delete_model(run_id: Any) -> Dict[str, Any]:
    """Remove an imported checkpoint from the registry and delete FrameForge's copy of it.

    Only FrameForge's own copy under the model folder is deleted; the user's original .pt is untouched.
    Boxes suggested by the model keep their provenance (run id and checksum) and stay valid for export.
    """
    if not isinstance(run_id, str) or not run_id or len(run_id) > 128:
        raise ValueError("Mã model không hợp lệ.")
    connection = _registry_connection()
    try:
        row = connection.execute("SELECT record_json FROM training_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Model không còn trong danh sách.")
        record = json.loads(row["record_json"])
        imported_root = (_model_root() / "imported").resolve()
        output_dir = record.get("output_dir")
        if isinstance(output_dir, str) and output_dir:
            folder = Path(output_dir).expanduser().resolve()
            if folder != imported_root and imported_root in folder.parents and folder.is_dir():
                shutil.rmtree(folder)
        connection.execute("DELETE FROM training_runs WHERE run_id = ?", (run_id,))
        connection.commit()
    finally:
        connection.close()
    return {"run_id": run_id, "deleted": True}
