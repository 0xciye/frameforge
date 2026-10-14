"""Read-only discovery of FrameForge projects, YOLO datasets, and Ultralytics runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sqlite3
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import yaml

from frameforge.media import IMAGE_EXTENSIONS
from frameforge.project import _is_frameforge_project


DEFAULT_IGNORED_DIRS = {
    ".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "target", "dist", "build",
}
MAX_DEPTH = 16
MAX_FILES = 200_000
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_RESULTS_ROWS = 5_000
MAX_DATASETS = 200
MAX_RUNS = 500
MAX_MODEL_EXPORTS = 1_000
MAX_IMPORTED_LIST_BYTES = 2 * 1024 * 1024
MAX_IMPORTED_LIST_TOTAL_BYTES = 16 * 1024 * 1024
ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]

METRIC_COLUMNS = {
    "epoch": ("epoch",),
    "train_box_loss": ("train/box_loss",),
    "train_class_loss": ("train/cls_loss",),
    "train_dfl_loss": ("train/dfl_loss",),
    "val_box_loss": ("val/box_loss",),
    "val_class_loss": ("val/cls_loss",),
    "val_dfl_loss": ("val/dfl_loss",),
    "precision": ("metrics/precision(b)", "metrics/precision"),
    "recall": ("metrics/recall(b)", "metrics/recall"),
    "map50": ("metrics/map50(b)", "metrics/map50"),
    "map50_95": ("metrics/map50-95(b)", "metrics/map50-95"),
    "learning_rate": ("lr/pg0", "lr/pg1", "lr/pg2"),
}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_path(value: str, base: Path, root: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve(strict=False)
    if not _inside(resolved, root):
        raise ValueError("Đường dẫn nằm ngoài thư mục được quét.")
    return resolved


def _absolute_path_style(value: str) -> Optional[str]:
    if PureWindowsPath(value).is_absolute():
        return "windows"
    if PurePosixPath(value).is_absolute():
        return "posix"
    return None


def _relative_to_source_root(value: str, source_root: Optional[str]) -> Optional[Tuple[str, ...]]:
    if not source_root:
        return None
    style = _absolute_path_style(source_root)
    if style is None or _absolute_path_style(value) != style:
        return None
    try:
        pure_path = PureWindowsPath if style == "windows" else PurePosixPath
        return pure_path(value).relative_to(pure_path(source_root)).parts
    except ValueError:
        return None


def _scoped_path(candidate: Path, scan_root: Path, *, strict: bool) -> Path:
    lexical = Path(os.path.abspath(candidate))
    if not _inside(lexical, scan_root):
        raise ValueError("Đường dẫn nằm ngoài thư mục được chọn.")
    cursor = scan_root
    for part in lexical.relative_to(scan_root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("Đường dẫn chứa symlink; hãy dùng tệp thật trong thư mục được chọn.")
    resolved = lexical.resolve(strict=strict)
    if not _inside(resolved, scan_root):
        raise ValueError("Đường dẫn nằm ngoài thư mục được chọn.")
    return resolved


def _dataset_base(
    configured_root: Any,
    yaml_path: Path,
    scan_root: Path,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Optional[str], bool]:
    if not isinstance(configured_root, str):
        raise ValueError("Trường path trong data.yaml phải là chuỗi.")
    configured_root = configured_root or "."
    style = _absolute_path_style(configured_root)
    native_style = "windows" if os.name == "nt" else "posix"
    if style is None:
        if os.name != "nt":
            configured_root = configured_root.replace("\\", "/")
        local_root = _scoped_path(yaml_path.parent / configured_root, scan_root, strict=False)
        return local_root, None, False

    if style == native_style:
        configured_path = Path(configured_root).expanduser()
        if _inside(Path(os.path.abspath(configured_path)), scan_root):
            return _scoped_path(configured_path, scan_root, strict=False), configured_root, False

    # A transferred workspace commonly keeps an absolute path from its source machine.
    local_root = yaml_path.parent.resolve(strict=True)
    if not _inside(local_root, scan_root):
        raise ValueError("Không thể khôi phục đường dẫn dataset trong thư mục được quét.")
    source_root = configured_root
    has_split_paths = config and any(_split_values(config.get(name)) for name in ("train", "val", "test"))
    if has_split_paths and not _dataset_sources_resolve(config, local_root, scan_root, source_root):
        matches = _find_dataset_roots(config, scan_root, source_root, local_root)
        if len(matches) > 1:
            raise ValueError("Có nhiều thư mục khớp các split trong data.yaml; hãy đặt data.yaml trong đúng thư mục dataset.")
        if len(matches) == 1:
            return matches[0], source_root, True
    return local_root, configured_root, True


def _local_dataset_reference(
    value: str,
    base: Path,
    scan_root: Path,
    source_root: Optional[str],
) -> Tuple[Path, bool]:
    if value.lower().startswith(("http://", "https://")):
        raise ValueError("Dataset dùng URL từ xa.")
    style = _absolute_path_style(value)
    native_style = "windows" if os.name == "nt" else "posix"
    if style is None:
        if os.name != "nt":
            value = value.replace("\\", "/")
        candidate = base / value
        rebased = False
    elif style == native_style:
        candidate = Path(value).expanduser()
        lexical = Path(os.path.abspath(candidate))
        if _inside(lexical, scan_root):
            return _scoped_path(candidate, scan_root, strict=False), False
        relative = _relative_to_source_root(value, source_root)
        if relative is None:
            raise ValueError("Đường dẫn nằm ngoài thư mục được quét.")
        candidate = base.joinpath(*relative)
        rebased = True
    else:
        relative = _relative_to_source_root(value, source_root)
        if relative is None:
            raise ValueError("Đường dẫn thuộc máy nguồn và không khớp gốc dataset đã khai báo.")
        candidate = base.joinpath(*relative)
        rebased = True
    resolved = _scoped_path(candidate, scan_root, strict=False)
    return resolved, rebased


def _dataset_sources_resolve(
    config: Dict[str, Any],
    dataset_root: Path,
    scan_root: Path,
    source_root: Optional[str],
) -> bool:
    found_split = False
    try:
        for split_name in ("train", "val", "test"):
            value = config.get(split_name)
            if value is None:
                continue
            sources = _split_values(value)
            if not sources:
                return False
            found_split = True
            for source in sources:
                path, _rebased = _local_dataset_reference(source, dataset_root, scan_root, source_root)
                path = _scoped_path(path, scan_root, strict=True)
                if path.is_dir() or (path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS):
                    continue
                if path.is_file() and path.suffix.casefold() == ".txt":
                    continue
                return False
    except (OSError, ValueError):
        return False
    return found_split


def _find_dataset_roots(
    config: Dict[str, Any],
    scan_root: Path,
    source_root: Optional[str],
    preferred_root: Path,
) -> List[Path]:
    matches: List[Path] = []
    visited_entries = 0
    depth_limited = False
    failed_directories = False
    stack = [(scan_root, 0)]
    while stack:
        directory, depth = stack.pop()
        if directory != preferred_root and _dataset_sources_resolve(config, directory, scan_root, source_root):
            matches.append(directory)
            if len(matches) > 1:
                return matches
        try:
            entries = list(os.scandir(directory))
        except OSError:
            failed_directories = True
            continue
        visited_entries += len(entries)
        if visited_entries > MAX_FILES:
            raise ValueError("Không thể khôi phục dataset an toàn: đã vượt giới hạn quét 200.000 mục.")
        if depth >= MAX_DEPTH:
            for entry in entries:
                try:
                    depth_limited = depth_limited or entry.is_dir(follow_symlinks=False)
                except OSError:
                    failed_directories = True
            continue
        for entry in entries:
            try:
                if not entry.is_symlink() and entry.is_dir(follow_symlinks=False):
                    stack.append((Path(entry.path), depth + 1))
            except OSError:
                failed_directories = True
    if depth_limited or failed_directories:
        raise ValueError("Không thể xác định duy nhất thư mục dataset vì lượt tìm kiếm chưa bao phủ hết workspace.")
    return matches


def _stable_id(kind: str, root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    digest = hashlib.sha256((kind + ":" + relative).encode("utf-8", errors="surrogatepass")).hexdigest()
    return kind + "-" + digest[:20]


def _mtime(path: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def _read_yaml(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            return None, "Tệp YAML vượt giới hạn đọc metadata 2 MiB."
        with path.open("r", encoding="utf-8-sig") as stream:
            value = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        return None, "Không đọc được YAML: {0}".format(error)
    if not isinstance(value, dict):
        return None, "Nội dung YAML không phải object."
    return value, None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return "[binary value omitted]"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    return str(value)


def _class_mapping(config: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[int], Optional[str]]:
    names = config.get("names")
    normalized: Optional[List[str]] = None
    if isinstance(names, list) and names and all(isinstance(name, str) and name.strip() for name in names):
        normalized = [name.strip() for name in names]
    elif isinstance(names, dict):
        try:
            indexed = {int(key): value for key, value in names.items()}
            if sorted(indexed) == list(range(len(indexed))) and all(isinstance(indexed[i], str) and indexed[i].strip() for i in indexed):
                normalized = [indexed[i].strip() for i in range(len(indexed))]
        except (TypeError, ValueError):
            normalized = None
    declared_count = config.get("nc")
    class_count = len(normalized) if normalized else (
        declared_count if isinstance(declared_count, int) and not isinstance(declared_count, bool) and declared_count > 0 else None
    )
    if normalized and len({name.casefold() for name in normalized}) != len(normalized):
        return normalized, len(normalized), "Tên class bị trùng nhau."
    if normalized and declared_count is not None and (
        isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count != class_count
    ):
        return normalized, class_count, "Số class khai báo không khớp names."
    return normalized, class_count, None


def _split_values(value: Any) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value):
        return [item.strip() for item in value]
    return []


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    positions = [index for index, part in enumerate(parts[:-1]) if part.casefold() == "images"]
    if positions:
        parts[positions[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _dataset_candidate(
    path: Path,
    root: Path,
    file_paths: Set[Path],
    file_sizes: Dict[Path, int],
    scan_complete: bool,
) -> Dict[str, Any]:
    dataset_id = _stable_id("dataset", root, path)
    warnings: List[str] = []
    config, yaml_error = _read_yaml(path)
    if config is None:
        return {
            "dataset_id": dataset_id, "path": str(path), "name": path.parent.name,
            "status": "invalid_metadata", "class_names": None, "class_count": None,
            "splits": [], "warnings": [yaml_error or "YAML không hợp lệ."],
        }

    class_names, class_count, class_warning = _class_mapping(config)
    if class_warning:
        warnings.append(class_warning)
    configured_root = config.get("path", ".")
    if not isinstance(configured_root, str):
        configured_root = "."
        warnings.append("Trường path không phải chuỗi; dùng thư mục của YAML để giới hạn kiểm tra.")
    try:
        dataset_root, source_root, root_rebased = _dataset_base(configured_root or ".", path, root, config)
    except (OSError, ValueError) as error:
        dataset_root = path.parent
        warnings.append("Không thể xác định an toàn thư mục dataset: {0}".format(error))
        return {
            "dataset_id": dataset_id, "path": str(path), "name": path.parent.name,
            "status": "blocked_external", "dataset_root": str(dataset_root),
            "class_names": class_names, "class_count": class_count,
            "splits": [], "warnings": warnings,
        }

    splits: List[Dict[str, Any]] = []
    external_reference = False
    paths_rebased = root_rebased
    for split_name in ("train", "val", "test"):
        split_value = config.get(split_name)
        if split_value is None:
            continue
        sources = _split_values(split_value)
        if not sources:
            warnings.append("Split {0} không có đường dẫn chuỗi hợp lệ.".format(split_name))
            splits.append({"name": split_name, "status": "invalid", "image_count": None, "labeled_count": None})
            continue
        image_paths: Set[Path] = set()
        split_status = "indexed"
        split_warnings: List[str] = []
        for source in sources:
            try:
                split_path, source_rebased = _local_dataset_reference(source, dataset_root, root, source_root)
                paths_rebased = paths_rebased or source_rebased
            except (OSError, ValueError):
                external_reference = True
                split_status = "blocked_external"
                split_warnings.append("Split trỏ ra ngoài thư mục được quét; chưa đọc dữ liệu bên ngoài.")
                continue
            if split_path.is_dir():
                for candidate in file_paths:
                    if candidate.suffix.lower() in IMAGE_EXTENSIONS and _inside(candidate, split_path):
                        image_paths.add(candidate)
            elif split_path.is_file() and split_path.suffix.lower() in IMAGE_EXTENSIONS:
                if split_path in file_paths:
                    image_paths.add(split_path)
            elif split_path.is_file() and split_path.suffix.lower() == ".txt" and split_path in file_paths:
                if file_sizes.get(split_path, 0) > MAX_METADATA_BYTES:
                    split_status = "partial"
                    split_warnings.append("Danh sách ảnh quá lớn để đọc trong lượt quét cấu trúc.")
                    continue
                try:
                    lines = split_path.read_text(encoding="utf-8-sig").splitlines()
                except (OSError, UnicodeError) as error:
                    split_status = "partial"
                    split_warnings.append("Không đọc được danh sách ảnh: {0}".format(error))
                    continue
                for line in lines[:100_000]:
                    value = line.strip()
                    if not value:
                        continue
                    try:
                        image_path, line_rebased = _local_dataset_reference(value, dataset_root, root, source_root)
                        paths_rebased = paths_rebased or line_rebased
                    except (OSError, ValueError):
                        external_reference = True
                        split_status = "blocked_external"
                        split_warnings.append("Danh sách ảnh chứa đường dẫn ngoài phạm vi được quét.")
                        continue
                    if image_path in file_paths and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        image_paths.add(image_path)
                if len(lines) > 100_000:
                    split_status = "partial"
                    split_warnings.append("Danh sách ảnh bị giới hạn ở 100.000 dòng.")
            else:
                split_status = "missing"
                split_warnings.append("Không thấy đường dẫn split trong vùng đã quét.")
        labeled_count = sum(1 for image_path in image_paths if _label_path(image_path) in file_paths)
        count_is_complete = scan_complete and split_status == "indexed"
        splits.append({
            "name": split_name,
            "status": split_status,
            "image_count": len(image_paths) if count_is_complete else None,
            "observed_image_count": len(image_paths),
            "labeled_count": labeled_count if count_is_complete else None,
            "warnings": list(dict.fromkeys(split_warnings)),
        })
    if external_reference:
        status = "blocked_external"
    elif not splits or any(split["status"] in {"invalid", "missing"} for split in splits):
        status = "needs_review"
    else:
        status = "candidate"
    if not scan_complete:
        warnings.append("Số lượng ảnh chỉ là phần quan sát được; lượt quét chưa bao phủ hết thư mục.")
    if paths_rebased and not external_reference:
        warnings.append("Đường dẫn data.yaml từ máy nguồn đã được đối chiếu với dữ liệu nằm trong thư mục được chọn.")
    return {
        "dataset_id": dataset_id,
        "path": str(path),
        "name": path.parent.name,
        "status": status,
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "class_count": class_count,
        "splits": splits,
        "warnings": list(dict.fromkeys(warnings)),
        "validation": "not_run",
    }


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def prepare_discovered_dataset(discovery_root_value: str, data_yaml_value: str) -> Dict[str, Any]:
    """Create a safe local-path copy of a discovered dataset YAML without copying its images."""
    if not isinstance(discovery_root_value, str) or not discovery_root_value:
        raise ValueError("Chọn thư mục làm việc đã quét.")
    if not isinstance(data_yaml_value, str) or not data_yaml_value:
        raise ValueError("Chọn data.yaml đã tìm thấy trong thư mục làm việc.")
    try:
        root = Path(discovery_root_value).expanduser().resolve(strict=True)
        yaml_path = Path(data_yaml_value).expanduser()
    except (OSError, ValueError) as error:
        raise ValueError("Không tìm thấy thư mục hoặc data.yaml đã chọn.") from error
    if not root.is_dir():
        raise ValueError("Thư mục làm việc không còn tồn tại.")
    try:
        yaml_path = _scoped_path(yaml_path, root, strict=True)
    except (OSError, ValueError) as error:
        raise ValueError("data.yaml phải là tệp thật bên trong thư mục làm việc: {0}".format(error)) from error
    if not yaml_path.is_file() or yaml_path.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("Chọn data.yaml hoặc .yml bên trong thư mục làm việc.")
    config, yaml_error = _read_yaml(yaml_path)
    if config is None:
        raise ValueError(yaml_error or "Không đọc được data.yaml.")
    class_names, _class_count, class_warning = _class_mapping(config)
    if not class_names or class_warning:
        raise ValueError(class_warning or "data.yaml cần danh sách class names hợp lệ.")
    try:
        dataset_root, source_root, root_rebased = _dataset_base(config.get("path", "."), yaml_path, root, config)
        dataset_root = _scoped_path(dataset_root, root, strict=True)
    except (OSError, ValueError) as error:
        raise ValueError("Không thể khôi phục path của dataset trong thư mục đã chọn: {0}".format(error)) from error
    if not dataset_root.is_dir():
        raise ValueError("Thư mục dataset không tồn tại trong thư mục đã chọn.")

    rebased = root_rebased
    list_files: Dict[str, str] = {}
    split_sources: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    total_list_bytes = 0
    total_rewritten_list_bytes = 0
    for split_name in ("train", "val", "test"):
        split_value = config.get(split_name)
        if split_value is None:
            if split_name in {"train", "val"}:
                raise ValueError("data.yaml thiếu split {0}.".format(split_name))
            continue
        sources = _split_values(split_value)
        if not sources:
            raise ValueError("Split {0} không có đường dẫn hợp lệ.".format(split_name))
        mapped_sources: List[Tuple[str, Optional[str]]] = []
        for source_index, source in enumerate(sources):
            try:
                source_path, source_rebased = _local_dataset_reference(source, dataset_root, root, source_root)
                source_path = _scoped_path(source_path, root, strict=True)
            except (OSError, ValueError) as error:
                raise ValueError("Split {0}: {1}".format(split_name, error)) from error
            rebased = rebased or source_rebased
            if source_path.is_dir():
                mapped_sources.append((str(source_path), None))
                continue
            if source_path.is_file() and source_path.suffix.casefold() in IMAGE_EXTENSIONS:
                mapped_sources.append((str(source_path), None))
                continue
            if not source_path.is_file() or source_path.suffix.casefold() != ".txt":
                raise ValueError("Split {0} phải trỏ tới thư mục ảnh, ảnh hoặc danh sách .txt local.".format(split_name))
            source_size = source_path.stat().st_size
            total_list_bytes += source_size
            if source_size > MAX_IMPORTED_LIST_BYTES or total_list_bytes > MAX_IMPORTED_LIST_TOTAL_BYTES:
                raise ValueError("Danh sách ảnh vượt giới hạn 2 MiB mỗi tệp hoặc 16 MiB tổng cộng.")
            try:
                lines = source_path.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, UnicodeError) as error:
                raise ValueError("Không đọc được danh sách ảnh của split {0}: {1}".format(split_name, error)) from error
            if len(lines) > 100_000:
                raise ValueError("Danh sách ảnh của split {0} vượt 100.000 dòng.".format(split_name))
            local_images: List[str] = []
            for line in lines:
                value = line.strip()
                if not value:
                    continue
                try:
                    image_path, line_rebased = _local_dataset_reference(value, dataset_root, root, source_root)
                    image_path = _scoped_path(image_path, root, strict=True)
                except (OSError, ValueError) as error:
                    raise ValueError("Danh sách ảnh của split {0} có đường dẫn không dùng được: {1}".format(split_name, error)) from error
                if not image_path.is_file() or image_path.suffix.casefold() not in IMAGE_EXTENSIONS:
                    raise ValueError("Danh sách ảnh của split {0} chứa đường dẫn không phải ảnh local.".format(split_name))
                local_images.append(str(image_path))
                rebased = rebased or line_rebased
            list_name = "{0}-{1}.txt".format(split_name, source_index)
            rewritten_content = "".join(image + "\n" for image in local_images)
            rewritten_size = len(rewritten_content.encode("utf-8"))
            total_rewritten_list_bytes += rewritten_size
            if rewritten_size > MAX_IMPORTED_LIST_BYTES or total_rewritten_list_bytes > MAX_IMPORTED_LIST_TOTAL_BYTES:
                raise ValueError("Danh sách ảnh sau khi nối lại đường dẫn vượt giới hạn 2 MiB mỗi tệp hoặc 16 MiB tổng cộng.")
            list_files[list_name] = rewritten_content
            mapped_sources.append(("", list_name))
        split_sources[split_name] = mapped_sources

    relative_yaml = yaml_path.relative_to(root).as_posix()
    fingerprint = hashlib.sha256()
    fingerprint.update(str(root).encode("utf-8", errors="surrogatepass"))
    fingerprint.update(relative_yaml.encode("utf-8", errors="surrogatepass"))
    fingerprint.update(yaml_path.read_bytes())
    for name, content in sorted(list_files.items()):
        fingerprint.update(name.encode("utf-8"))
        fingerprint.update(content.encode("utf-8"))
    import_id = fingerprint.hexdigest()[:32]
    from frameforge.training import _default_app_data_dir

    imported_dir = _default_app_data_dir() / "datasets" / "imported" / import_id
    imported_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, content in list_files.items():
        _write_text_atomic(imported_dir / name, content)
    normalized: Dict[str, Any] = {"path": str(dataset_root), "names": class_names}
    for split_name, sources in split_sources.items():
        normalized_sources = [str(imported_dir / list_name) if list_name else local_path for local_path, list_name in sources]
        normalized[split_name] = normalized_sources[0] if len(normalized_sources) == 1 else normalized_sources
    _write_text_atomic(imported_dir / "data.yaml", yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True))
    _write_text_atomic(imported_dir / "imported-workspace.json", json.dumps({
        "source_root": str(root),
        "source_data_yaml": str(yaml_path),
    }, ensure_ascii=False, indent=2) + "\n")
    return {
        "data_yaml": str(imported_dir / "data.yaml"),
        "source_data_yaml": str(yaml_path),
        "source_root": str(root),
        "class_names": class_names,
        "paths_rebased": rebased,
    }


def _normalize_header(value: str) -> str:
    return value.strip().casefold()


def _read_metrics(path: Path) -> Dict[str, Any]:
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            return {"status": "size_limited", "headers": [], "history": [], "warnings": ["results.csv vượt giới hạn 2 MiB."]}
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = [str(name).strip() for name in (reader.fieldnames or []) if name is not None]
            normalized_headers = {_normalize_header(name): name for name in headers}
            history = []
            truncated = False
            for index, row in enumerate(reader):
                if index >= MAX_RESULTS_ROWS:
                    truncated = True
                    break
                point: Dict[str, Any] = {}
                for metric_name, aliases in METRIC_COLUMNS.items():
                    source = next((normalized_headers[alias] for alias in aliases if alias in normalized_headers), None)
                    if source is None:
                        continue
                    value = row.get(source)
                    try:
                        number = float(value.strip()) if isinstance(value, str) else float(value)
                    except (ValueError, TypeError, AttributeError):
                        point[metric_name] = None
                    else:
                        point[metric_name] = number if math.isfinite(number) else None
                if point:
                    history.append(point)
    except (OSError, UnicodeError, csv.Error) as error:
        return {"status": "unreadable", "headers": [], "history": [], "warnings": ["Không đọc được results.csv: {0}".format(error)]}
    return {
        "status": "partial" if truncated else "available" if history else "empty",
        "headers": headers,
        "history": history,
        "warnings": ["Lịch sử metric bị giới hạn ở {0} dòng.".format(MAX_RESULTS_ROWS)] if truncated else [],
    }


def _run_candidate(
    run_dir: Path,
    root: Path,
    run_files: List[Path],
    file_sizes: Dict[Path, int],
    scan_complete: bool,
) -> Dict[str, Any]:
    warnings: List[str] = []
    args_path = next((item for item in run_files if item.name.casefold() == "args.yaml"), None)
    results_path = next((item for item in run_files if item.name.casefold() == "results.csv"), None)
    args, args_error = _read_yaml(args_path) if args_path else (None, None)
    if args_error:
        warnings.append(args_error)
    config = args or {}
    requested_imgsz = config.get("imgsz")
    if isinstance(requested_imgsz, list) and all(isinstance(value, int) and not isinstance(value, bool) for value in requested_imgsz):
        requested_imgsz = requested_imgsz[:2]
    elif isinstance(requested_imgsz, (float, int)) and not isinstance(requested_imgsz, bool):
        requested_imgsz = int(requested_imgsz) if math.isfinite(float(requested_imgsz)) else None
    else:
        requested_imgsz = None
    requested_epochs = config.get("epochs")
    if isinstance(requested_epochs, bool) or not isinstance(requested_epochs, int) or requested_epochs < 1:
        requested_epochs = None

    results = _read_metrics(results_path) if results_path else {"status": "missing", "headers": [], "history": [], "warnings": []}
    warnings.extend(results["warnings"])
    history = results["history"]
    epochs = [point.get("epoch") for point in history if isinstance(point.get("epoch"), (int, float))]
    last_epoch = max(epochs) if epochs else None
    best_metrics: Dict[str, float] = {}
    for metric in ("map50", "map50_95", "precision", "recall"):
        values = [point[metric] for point in history if isinstance(point.get(metric), (int, float)) and math.isfinite(point[metric])]
        if values:
            best_metrics[metric] = max(values)
    completion_state = "unknown"
    if requested_epochs and last_epoch is not None and last_epoch + 1 >= requested_epochs:
        completion_state = "requested_epochs_recorded"

    checkpoints = []
    for checkpoint in sorted(run_files, key=lambda value: value.name.casefold()):
        if checkpoint.suffix.casefold() != ".pt":
            continue
        stat_size = file_sizes.get(checkpoint)
        checkpoints.append({
            "path": str(checkpoint),
            "name": checkpoint.name,
            "kind": "best" if checkpoint.name.casefold() == "best.pt" else "last" if checkpoint.name.casefold() == "last.pt" else "epoch_or_other",
            "size_bytes": stat_size,
            "modified_at": _mtime(checkpoint),
            "sha256": None,
            "compatibility": "unknown",
            "trust_required": True,
            "hash_status": "not_computed_during_first_pass",
        })
    best = next((item for item in checkpoints if item["kind"] == "best"), None)
    last = next((item for item in checkpoints if item["kind"] == "last"), None)

    data_value = config.get("data")
    data_path = None
    if isinstance(data_value, str) and data_value.strip():
        try:
            data_path = str(_safe_path(data_value, run_dir, root))
        except (OSError, ValueError):
            warnings.append("Dataset tham chiếu trong args.yaml nằm ngoài phạm vi hoặc không thể xác minh.")
    return {
        "experiment_id": _stable_id("run", root, run_dir),
        "source_dir": str(run_dir),
        "relative_dir": run_dir.relative_to(root).as_posix() or ".",
        "name": run_dir.name,
        "status": "unknown",
        "completion_state": completion_state,
        "model_family": "unknown",
        "architecture_claimed": config.get("model") if isinstance(config.get("model"), str) else None,
        "task_claimed": config.get("task") if isinstance(config.get("task"), str) else None,
        "dataset_reference": data_path,
        "dataset_relationship": "unknown",
        "training_imgsz": requested_imgsz,
        "requested_epochs": requested_epochs,
        "last_logged_epoch": last_epoch,
        "batch": config.get("batch") if isinstance(config.get("batch"), (int, float)) else None,
        "device": config.get("device") if isinstance(config.get("device"), (str, int, float)) else None,
        "run_config": _json_safe(config),
        "config_path": str(args_path) if args_path else None,
        "metrics_path": str(results_path) if results_path else None,
        "metrics_status": results["status"],
        "metric_headers": results["headers"],
        "metrics_history": history,
        "best_metrics": best_metrics,
        "checkpoints": checkpoints,
        "best_checkpoint": best,
        "last_checkpoint": last,
        "modified_at": _mtime(run_dir),
        "warnings": list(dict.fromkeys(warnings)),
        "metadata_complete": bool(args_path and results_path) and scan_complete,
    }


def scan_project_folder(
    path_value: str,
    max_depth: int = 8,
    max_files: int = 50_000,
    ignored_dirs: Optional[List[str]] = None,
    cancel_file: Optional[str] = None,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    """Inspect metadata and directory structure without loading image pixels or checkpoints."""
    root = Path(path_value).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Hãy chọn thư mục dự án để quét.")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 1 <= max_depth <= MAX_DEPTH:
        raise ValueError("Độ sâu quét phải nằm trong khoảng 1–{0}.".format(MAX_DEPTH))
    if isinstance(max_files, bool) or not isinstance(max_files, int) or not 1 <= max_files <= MAX_FILES:
        raise ValueError("Giới hạn file phải nằm trong khoảng 1–{0}.".format(MAX_FILES))
    if ignored_dirs is None:
        ignored = set(DEFAULT_IGNORED_DIRS)
    else:
        if not isinstance(ignored_dirs, list) or len(ignored_dirs) > 64 or any(not isinstance(name, str) or not name or len(name) > 128 or Path(name).name != name for name in ignored_dirs):
            raise ValueError("Danh sách thư mục bỏ qua không hợp lệ.")
        ignored = {name.casefold() for name in ignored_dirs}

    cancel_path = Path(cancel_file) if isinstance(cancel_file, str) and cancel_file else None
    stack: List[Tuple[Path, int]] = [(root, 0)]
    file_paths: Set[Path] = set()
    file_sizes: Dict[Path, int] = {}
    file_names: Dict[Path, str] = {}
    warnings: List[str] = []
    symlinks_skipped = 0
    permission_errors = 0
    depth_limited = False
    file_limit_reached = False
    cancelled = False
    scanned_directories = 0
    files_seen = 0
    last_progress = time.monotonic()
    dataset_files_seen: Set[Path] = set()
    run_dirs_seen: Set[Path] = set()
    run_files_by_dir: Dict[Path, List[Path]] = {}

    def emit(stage: str) -> None:
        if progress:
            progress({
                "stage": stage,
                "root": str(root),
                "files_seen": files_seen,
                "directories_scanned": scanned_directories,
                "datasets_found": len(dataset_files_seen),
                "runs_found": len(run_dirs_seen),
            })

    emit("discovery_started")
    while stack:
        if cancel_path and cancel_path.exists():
            cancelled = True
            break
        directory, depth = stack.pop()
        scanned_directories += 1
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if cancel_path and cancel_path.exists():
                        cancelled = True
                        break
                    try:
                        if entry.is_symlink():
                            symlinks_skipped += 1
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name.casefold() in ignored:
                                continue
                            if depth >= max_depth:
                                depth_limited = True
                                continue
                            stack.append((Path(entry.path), depth + 1))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        if files_seen >= max_files:
                            file_limit_reached = True
                            break
                        file_path = Path(entry.path)
                        stat = entry.stat(follow_symlinks=False)
                        file_paths.add(file_path)
                        file_sizes[file_path] = stat.st_size
                        file_names[file_path] = entry.name
                        files_seen += 1
                        lowered_name = entry.name.casefold()
                        if lowered_name in {"data.yaml", "dataset.yaml"}:
                            dataset_files_seen.add(file_path)
                        if lowered_name in {"args.yaml", "results.csv"}:
                            run_dir = file_path.parent
                            run_dirs_seen.add(run_dir)
                            run_files_by_dir.setdefault(run_dir, []).append(file_path)
                        elif file_path.suffix.casefold() == ".pt":
                            run_dir = file_path.parent.parent if file_path.parent.name.casefold() == "weights" else file_path.parent
                            run_dirs_seen.add(run_dir)
                            run_files_by_dir.setdefault(run_dir, []).append(file_path)
                    except OSError as error:
                        permission_errors += 1
                        if len(warnings) < 200:
                            warnings.append("Không đọc được mục {0}: {1}".format(entry.path, error))
                    now = time.monotonic()
                    if files_seen % 200 == 0 and now - last_progress >= 0.15:
                        emit("scanning")
                        last_progress = now
                    if file_limit_reached:
                        break
                if cancelled or file_limit_reached:
                    break
        except OSError as error:
            permission_errors += 1
            if len(warnings) < 200:
                warnings.append("Không mở được thư mục {0}: {1}".format(directory, error))
    if depth_limited:
        warnings.append("Một số thư mục sâu hơn giới hạn đã bị bỏ qua.")
    if file_limit_reached:
        warnings.append("Đã dừng ở giới hạn {0} file.".format(max_files))
    if symlinks_skipped:
        warnings.append("Đã bỏ qua {0} symlink để tránh vòng lặp và đường dẫn ngoài phạm vi.".format(symlinks_skipped))
    if permission_errors:
        warnings.append("Không đọc được {0} mục do quyền truy cập hoặc lỗi filesystem.".format(permission_errors))
    scan_complete = not cancelled and not file_limit_reached and not depth_limited and permission_errors == 0

    dataset_files = sorted(
        dataset_files_seen,
        key=lambda value: str(value).casefold(),
    )[:MAX_DATASETS]
    datasets = []
    for path in dataset_files:
        if cancel_path and cancel_path.exists():
            cancelled = True
            break
        datasets.append(_dataset_candidate(path, root, file_paths, file_sizes, scan_complete))
        emit("analyzing_datasets")
    if len(dataset_files) == MAX_DATASETS:
        warnings.append("Danh sách dataset bị giới hạn ở {0} mục.".format(MAX_DATASETS))

    run_records: List[Dict[str, Any]] = []
    run_dirs = sorted(run_dirs_seen, key=lambda value: str(value).casefold())
    for run_dir in run_dirs[:MAX_RUNS]:
        if cancel_path and cancel_path.exists():
            cancelled = True
            break
        associated_files = sorted(run_files_by_dir[run_dir], key=lambda value: str(value).casefold())
        run_records.append(_run_candidate(run_dir, root, associated_files, file_sizes, scan_complete))
        emit("analyzing_runs")
    if len(run_dirs) > MAX_RUNS:
        warnings.append("Danh sách training run bị giới hạn ở {0} mục.".format(MAX_RUNS))

    exported_models = []
    model_extensions = {".onnx", ".engine", ".xml", ".tflite", ".torchscript"}
    for model_path in sorted((path for path in file_paths if path.suffix.casefold() in model_extensions), key=lambda value: str(value).casefold())[:MAX_MODEL_EXPORTS]:
        try:
            _ = model_path.relative_to(root)
        except ValueError:
            continue
        exported_models.append({
            "artifact_id": _stable_id("model", root, model_path),
            "path": str(model_path),
            "name": model_path.name,
            "format": model_path.suffix.lstrip("." ).casefold(),
            "size_bytes": file_sizes.get(model_path),
            "modified_at": _mtime(model_path),
            "compatibility": "unknown",
        })

    frameforge_projects = []
    for project_path in sorted((path for path in file_paths if path.suffix.casefold() == ".ffproj"), key=lambda value: str(value).casefold()):
        try:
            if _is_frameforge_project(project_path):
                frameforge_projects.append({"path": str(project_path), "name": project_path.stem, "modified_at": _mtime(project_path)})
        except (OSError, sqlite3.DatabaseError):
            warnings.append("Không đọc được project FrameForge: {0}".format(project_path))

    scan_complete = scan_complete and not cancelled
    emit("discovery_cancelled" if cancelled else "discovery_complete" if scan_complete else "discovery_partial")
    return {
        "root": str(root),
        "status": "cancelled" if cancelled else "complete" if scan_complete else "partial",
        "complete": scan_complete,
        "cancelled": cancelled,
        "limits": {"max_depth": max_depth, "max_files": max_files, "ignored_dirs": sorted(ignored)},
        "files_seen": files_seen,
        "directories_scanned": scanned_directories,
        "symlinks_skipped": symlinks_skipped,
        "permission_errors": permission_errors,
        "datasets": datasets,
        "runs": run_records,
        "exported_models": exported_models,
        "frameforge_projects": frameforge_projects,
        "warnings": list(dict.fromkeys(warnings))[:200],
    }
