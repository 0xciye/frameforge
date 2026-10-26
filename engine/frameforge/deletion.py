"""Delete images from a project together with their files: the image and its label JSON go to the
system trash (Recycle Bin on Windows, ~/.Trash on macOS), so a mistaken delete can be undone there."""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from frameforge.project import _connect, _is_frameforge_project, _migrate
from frameforge.sidecar import _ensure_state, json_path_for

MAX_DELETE = 10000


def _windows_trash(paths: List[Path]) -> None:
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR),
            ("fFlags", ctypes.c_uint16),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    fo_delete = 3
    flags = 0x0040 | 0x0010 | 0x0004 | 0x0400  # allow undo (Recycle Bin), no confirm, silent, no error UI
    operation = SHFILEOPSTRUCTW(
        hwnd=None, wFunc=fo_delete, pFrom="\0".join(str(path) for path in paths) + "\0\0", pTo=None, fFlags=flags,
    )
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if result != 0 or operation.fAnyOperationsAborted:
        raise OSError("Không chuyển được tệp vào Thùng rác (mã {0}).".format(result))


def _home_trash(paths: List[Path]) -> None:
    trash = Path.home() / ".Trash"
    trash.mkdir(exist_ok=True)
    for path in paths:
        target = trash / path.name
        if target.exists():
            target = trash / "{0} {1}{2}".format(path.stem, time.strftime("%H.%M.%S"), path.suffix)
        shutil.move(str(path), str(target))


def move_to_trash(paths: List[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    if os.name == "nt":
        _windows_trash(existing)
    elif sys.platform == "darwin":
        _home_trash(existing)
    else:
        raise OSError("Hệ điều hành này chưa hỗ trợ xóa vào Thùng rác.")


def delete_images(
    project_value: str, image_paths: Any, trash: Callable[[List[Path]], None] = move_to_trash
) -> Dict[str, Any]:
    """Remove images from the project and move each image file and its JSON to the trash."""
    if not isinstance(image_paths, list) or not image_paths or len(image_paths) > MAX_DELETE:
        raise ValueError("Hãy chọn từ 1 đến {0} ảnh để xóa.".format(MAX_DELETE))
    if any(not isinstance(path, str) or not path for path in image_paths):
        raise ValueError("Đường dẫn ảnh không hợp lệ.")
    project = Path(project_value).expanduser().resolve(strict=True)
    if project.suffix.lower() != ".ffproj" or not project.is_file() or not _is_frameforge_project(project):
        raise ValueError("Hãy chọn một tệp dự án FrameForge .ffproj.")
    connection = _connect(project)
    try:
        _migrate(connection)
        _ensure_state(connection)
        known = {
            row["path"] for row in connection.execute(
                "SELECT path FROM images WHERE path IN ({0})".format(",".join("?" * len(image_paths))), image_paths
            )
        } if len(image_paths) <= 500 else {row["path"] for row in connection.execute("SELECT path FROM images")}
        missing = [path for path in image_paths if path not in known]
        if missing:
            raise ValueError("Ảnh không thuộc dự án hiện tại: {0}".format(missing[0]))
        # Files first: if the trash refuses, the project still lists the image and nothing is lost.
        files = []
        for path in image_paths:
            files.append(Path(path))
            files.append(json_path_for(path))
        trash(files)
        with connection:
            for path in image_paths:
                connection.execute("DELETE FROM images WHERE path = ?", (path,))  # boxes and candidates cascade
                connection.execute("DELETE FROM ai_inference_attempts WHERE image_path = ?", (path,))
                connection.execute("DELETE FROM sidecar_state WHERE image_path = ?", (path,))
    except sqlite3.DatabaseError as error:
        raise ValueError("Không cập nhật được dự án: {0}".format(error)) from error
    finally:
        connection.close()
    return {"deleted": list(image_paths), "deleted_count": len(image_paths)}
