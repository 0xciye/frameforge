"""Find a project's images again after the project and its images moved to another folder or computer.

The .ffproj stores absolute image paths. When they no longer exist, images are looked up under a root folder
(the project's own folder by default, or one the user picks) by file name; when several files share the name, the
one whose parent folders match the old path's parents best wins. Matched rows are moved to the new path in every
table keyed by image path; labels follow because they live in the JSON beside each image.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from frameforge.project import _connect, _is_frameforge_project, _migrate
from frameforge.sidecar import _ensure_state

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
# Indexing stops here so pointing at a whole drive cannot run for ever.
MAX_INDEXED_FILES = 2_000_000
# Tables whose rows belong to one image path (images itself is updated first).
PATH_TABLES = ("annotations", "ai_negative_candidates", "ai_inference_attempts", "sidecar_state")


def _missing(paths: List[str]) -> List[str]:
    """Paths that no longer exist, listing each folder once instead of stat-ing every file."""
    by_folder: Dict[str, List[str]] = {}
    for path in paths:
        by_folder.setdefault(os.path.dirname(path), []).append(path)
    missing: List[str] = []
    for folder, items in by_folder.items():
        try:
            names = {entry.name.casefold() for entry in os.scandir(folder)}
        except OSError:
            missing.extend(items)
            continue
        missing.extend(path for path in items if os.path.basename(path).casefold() not in names)
    return missing


def _index(root: Path) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    count = 0
    stack = [str(root)]
    while stack and count < MAX_INDEXED_FILES:
        folder = stack.pop()
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS:
                    index.setdefault(entry.name.casefold(), []).append(os.path.normpath(entry.path))
                    count += 1
            except OSError:
                continue
    return index


def _parts(path: str) -> List[str]:
    return [part.casefold() for part in Path(path.replace("\\", "/")).parts]


def _best(old: str, candidates: List[str]) -> Optional[str]:
    """Candidate sharing the most trailing folder names with the old path; ties are ambiguous."""
    if len(candidates) == 1:
        return candidates[0]
    old_parts = _parts(old)[:-1]

    def shared(candidate: str) -> int:
        new_parts = _parts(candidate)[:-1]
        count = 0
        while count < min(len(old_parts), len(new_parts)) and old_parts[-1 - count] == new_parts[-1 - count]:
            count += 1
        return count

    scored = sorted(((shared(candidate), candidate) for candidate in candidates), reverse=True)
    return scored[0][1] if scored[0][0] > scored[1][0] else None


def relocate_images(project_value: str, root: Optional[str] = None) -> Dict[str, Any]:
    """Point missing images at their new location under ``root`` (default: the project's folder)."""
    project = Path(project_value).expanduser().resolve(strict=True)
    if project.suffix.lower() != ".ffproj" or not project.is_file() or not _is_frameforge_project(project):
        raise ValueError("Hãy chọn một tệp dự án FrameForge .ffproj.")
    search_root = Path(root).expanduser().resolve(strict=True) if root else project.parent
    if not search_root.is_dir():
        raise ValueError("Thư mục ảnh không hợp lệ.")
    connection = _connect(project)
    try:
        _migrate(connection)
        _ensure_state(connection)
        paths = [row["path"] for row in connection.execute("SELECT path FROM images")]
        missing = _missing(paths)
        result = {"image_count": len(paths), "missing_count": len(missing), "relocated_count": 0,
                  "still_missing_count": len(missing), "ambiguous_count": 0, "root": str(search_root)}
        if not missing:
            return result
        index = _index(search_root)
        known = set(paths)
        moves = []
        ambiguous = 0
        for old in missing:
            candidates = [path for path in index.get(os.path.basename(old).casefold(), []) if path not in known]
            if not candidates:
                continue
            new = _best(old, candidates)
            if new is None:
                ambiguous += 1
                continue
            known.add(new)
            moves.append((new, old))
        if moves:
            # Paths are primary/foreign keys without ON UPDATE CASCADE: move every table in one transaction.
            connection.execute("PRAGMA foreign_keys = OFF")
            with connection:
                connection.executemany("UPDATE images SET path = ? WHERE path = ?", moves)
                for table in PATH_TABLES:
                    connection.executemany("UPDATE {0} SET image_path = ? WHERE image_path = ?".format(table), moves)
            connection.execute("PRAGMA foreign_keys = ON")
        result.update({"relocated_count": len(moves), "still_missing_count": len(missing) - len(moves),
                       "ambiguous_count": ambiguous})
        return result
    except sqlite3.DatabaseError as error:
        raise ValueError("Không cập nhật được dự án: {0}".format(error)) from error
    finally:
        connection.close()
