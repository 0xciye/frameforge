"""Add existing images (files or folders) to a FrameForge project, used in place.

Labels are the LabelMe/X-AnyLabeling JSON beside each image (see ``frameforge.sidecar``): an image that
already has one joins the project with its boxes, and FrameForge keeps writing to that same file.
"""

from __future__ import annotations

import os
import re
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from frameforge import sidecar
from frameforge.project import _connect, _is_frameforge_project, _migrate

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MAX_IMAGES = 100000
# Capture time in names such as train_crop_20260728_201116_927_000020.png.
TIMESTAMP = re.compile(r"(\d{8})_(\d{6})")


def _png_size(header: bytes) -> Optional[Tuple[int, int]]:
    if header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
        width, height = struct.unpack(">II", header[16:24])
        return width, height
    return None


def _jpeg_size(path: Path) -> Optional[Tuple[int, int]]:
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            return None
        while True:
            byte = handle.read(1)
            while byte == b"\xff":
                byte = handle.read(1)
            if not byte:
                return None
            marker = byte[0]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                return None
            length = struct.unpack(">H", length_bytes)[0]
            if length < 2:
                return None
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                data = handle.read(5)
                if len(data) != 5:
                    return None
                height, width = struct.unpack(">HH", data[1:5])
                return width, height
            handle.seek(length - 2, 1)


def image_size(path: Path) -> Optional[Tuple[int, int]]:
    """Width and height from the file header, decoding only formats without a cheap header read."""
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
        size = _png_size(header)
        if size is None and header[:2] == b"\xff\xd8":
            size = _jpeg_size(path)
        if size is None:
            from frameforge.media import _read_image

            import cv2

            frame = _read_image(path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                return None
            size = (int(frame.shape[1]), int(frame.shape[0]))
    except (OSError, struct.error):
        return None
    width, height = size
    return (width, height) if width > 0 and height > 0 else None


def _group_id(folder: Path, stem: str) -> str:
    """Split group: 10-minute capture bucket from the file name, else the single image."""
    match = TIMESTAMP.search(stem)
    if match:
        date, clock = match.groups()
        return "labelme:{0}:{1}_{2}".format(folder.as_posix(), date, clock[:3])
    return "labelme:{0}:file:{1}".format(folder.as_posix(), stem)


def _walk_images(folder: Path) -> List[Path]:
    """Image files under ``folder`` (recursively), sorted per folder, from directory listings only."""
    found: List[Path] = []
    pending = [folder]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as listing:
                entries = sorted(listing, key=lambda entry: entry.name)
        except OSError:
            continue
        subfolders = []
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                subfolders.append(Path(entry.path))
            elif entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS:
                found.append(Path(entry.path))
        pending.extend(reversed(subfolders))
    return found


def _scan(path_values: Any) -> List[Path]:
    """Images under the given files/folders (folders recursively), without duplicates."""
    if isinstance(path_values, str):
        path_values = [path_values]
    if not isinstance(path_values, list) or not path_values or len(path_values) > 10000:
        raise ValueError("Hãy chọn ảnh hoặc thư mục ảnh để nhập.")
    images: List[Path] = []
    seen = set()
    for value in path_values:
        # Resolve the chosen path once; files listed under it are already absolute.
        path = Path(str(value)).expanduser().resolve(strict=True)
        candidates = _walk_images(path) if path.is_dir() else (
            [path] if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS else []
        )
        for candidate in candidates:
            key = os.path.normcase(str(candidate))
            if key in seen:
                continue
            seen.add(key)
            images.append(candidate)
            if len(images) > MAX_IMAGES:
                raise ValueError("Có hơn {0} ảnh; hãy chia nhỏ trước khi nhập.".format(MAX_IMAGES))
    if not images:
        raise ValueError("Không tìm thấy ảnh được hỗ trợ trong lựa chọn.")
    return images


def _open_project(project_value: str) -> Path:
    project = Path(project_value).expanduser().resolve(strict=True)
    if project.suffix.lower() != ".ffproj" or not project.is_file():
        raise ValueError("Hãy chọn một tệp dự án FrameForge .ffproj.")
    try:
        valid_project = _is_frameforge_project(project)
    except sqlite3.DatabaseError as error:
        raise ValueError("Tệp đã tồn tại nhưng không phải là dự án FrameForge hợp lệ.") from error
    if not valid_project:
        raise ValueError("Tệp đã tồn tại nhưng không phải là dự án FrameForge hợp lệ.")
    return project


def import_images(
    project_value: str, path_values: Any, mark_reviewed: bool = False, dry_run: bool = False
) -> Dict[str, Any]:
    """Add images (used in place) to the project and read the labels from the JSON beside each one.

    ``mark_reviewed`` applies to JSON written by other tools (no FrameForge review status inside): images
    with boxes become reviewed. ``dry_run`` only counts, so the UI can ask about that when JSON exists.
    """
    if not isinstance(mark_reviewed, bool) or not isinstance(dry_run, bool):
        raise ValueError("Tùy chọn nhập ảnh không hợp lệ.")
    project = _open_project(project_value)
    images = _scan(path_values)

    connection = _connect(project)
    try:
        _migrate(connection)
        existing = {row["path"] for row in connection.execute("SELECT path FROM images")}
        new_images = [image for image in images if str(image) not in existing]
        json_stats = sidecar._JsonStats()
        foreign_json = sum(1 for image in new_images if json_stats.get(sidecar.json_path_for(str(image))) is not None)
        if dry_run:
            return {
                "found_images": len(images),
                "new_images": len(new_images),
                "json_images": foreign_json,
                "skipped_existing": len(images) - len(new_images),
            }

        planned, errors = [], []
        for image in new_images:
            size = image_size(image)
            if size is None:
                errors.append({"path": str(image), "message": "Không đọc được kích thước ảnh."})
                continue
            planned.append((image, size))
        now = datetime.now(timezone.utc).isoformat()
        with connection:
            connection.executemany(
                """INSERT INTO images(
                    path, source_path, source_id, frame_index, timestamp_seconds, crop_rect_json,
                    sharpness_score, extraction_settings_json, width, height, status
                ) VALUES(?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?, ?, 'ready')""",
                [
                    (str(image), str(image), _group_id(image.parent, image.stem), width, height)
                    for image, (width, height) in planned
                ],
            )
            connection.execute("INSERT OR REPLACE INTO project_meta(key, value) VALUES('modified_at', ?)", (now,))
        imported = [str(image) for image, _size in planned]
        synced = sidecar.sync(connection, imported, default_reviewed=mark_reviewed)
        # Count in Python: an IN list of every imported path exceeds SQLite's variable limit.
        imported_set = set(imported)
        per_image = {
            path: count
            for path, count in connection.execute("SELECT image_path, COUNT(*) FROM annotations GROUP BY image_path")
            if path in imported_set
        }
        labeled_images = len(per_image)
        box_count = sum(per_image.values())
        reviewed_images = sum(
            1 for (path,) in connection.execute("SELECT path FROM images WHERE status = 'reviewed'") if path in imported_set
        )
    except sqlite3.DatabaseError as error:
        raise ValueError("Không ghi được vào dự án: {0}".format(error)) from error
    finally:
        connection.close()

    errors.extend(synced["errors"])
    return {
        "found_images": len(images),
        "imported_images": len(planned),
        "json_images": len(synced["read_images"]),
        "labeled_images": labeled_images,
        "unlabeled_images": len(planned) - labeled_images,
        "reviewed_images": reviewed_images,
        "box_count": box_count,
        "skipped_existing": len(images) - len(new_images),
        "unsupported_shapes": synced["unsupported_shapes"],
        "unknown_labels": synced["unknown_labels"],
        "clipped_boxes": synced["clipped_boxes"],
        "degenerate_boxes": synced["degenerate_boxes"],
        "errors": errors[:50],
        "error_count": len(errors),
    }


def import_labelme_folder(project_value: str, folder_value: str, mark_reviewed: bool = False) -> Dict[str, Any]:
    """One folder of images + LabelMe/X-AnyLabeling JSON; see ``import_images``."""
    folder = Path(folder_value).expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise ValueError("Hãy chọn thư mục chứa ảnh và JSON LabelMe/X-AnyLabeling.")
    return {"folder": str(folder), **import_images(project_value, [str(folder)], mark_reviewed)}
