"""Private local session snapshot storage for automatic work recovery."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

from frameforge.project import _is_frameforge_project, load_project, save_project


def _app_data_dir() -> Path:
    configured = os.environ.get("FRAMEFORGE_APP_DATA")
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "com.frameforge.desktop"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "com.frameforge.desktop"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "frameforge"


def _session_paths() -> tuple[Path, Path]:
    directory = _app_data_dir() / "session"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "last-session.ffproj", directory / "last-session.json"


def save_session(project: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
    """Atomically replace the hidden project snapshot and its small UI state file."""
    if not isinstance(project, dict) or not isinstance(session, dict):
        raise ValueError("Dữ liệu phiên làm việc không hợp lệ.")
    try:
        session_json = json.dumps(session, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Cài đặt phiên làm việc không phải JSON an toàn.") from error

    project_path, metadata_path = _session_paths()
    temporary_project = project_path.with_name(".session-{0}.ffproj".format(uuid.uuid4().hex))
    temporary_metadata = metadata_path.with_name(".session-{0}.json".format(uuid.uuid4().hex))
    try:
        save_project(str(temporary_project), project)
        temporary_metadata.write_text(session_json, encoding="utf-8")
        os.replace(temporary_project, project_path)
        os.replace(temporary_metadata, metadata_path)
    except BaseException:
        temporary_project.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        raise
    return {"saved": True, "project_path": str(project_path)}


def load_session() -> Dict[str, Any]:
    """Return the last local session snapshot without opening arbitrary model files."""
    project_path, metadata_path = _session_paths()
    if not project_path.is_file() or not metadata_path.is_file():
        return {"available": False}
    try:
        session = json.loads(metadata_path.read_text(encoding="utf-8"))
        project = load_project(str(project_path))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Không khôi phục được phiên làm việc: {0}".format(error)) from error
    if not isinstance(session, dict):
        raise ValueError("Metadata phiên làm việc bị hỏng.")
    original_path = session.get("project_path")
    original_exists = False
    if isinstance(original_path, str) and original_path:
        candidate = Path(original_path).expanduser()
        try:
            original_exists = candidate.is_file() and _is_frameforge_project(candidate.resolve(strict=True))
        except (OSError, ValueError):
            original_exists = False
    return {
        "available": True,
        "project": project,
        "session": session,
        "original_project_exists": original_exists,
    }
