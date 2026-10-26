"""Private local session snapshot storage for automatic work recovery."""

from __future__ import annotations

import hashlib
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
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory / "last-session.ffproj", directory / "last-session.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repair_mojibake(value: Any) -> Any:
    if not isinstance(value, str) or not any(marker in value for marker in ("Ã", "Â", "Æ", "Å", "â€")):
        return value
    for _ in range(3):
        try:
            value = value.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
    return value


def save_session(project: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
    """Atomically replace the hidden project snapshot and its small UI state file."""
    if not isinstance(project, dict) or not isinstance(session, dict):
        raise ValueError("Dữ liệu phiên làm việc không hợp lệ.")
    metadata = dict(session)
    metadata["source_project_sha256"] = None
    project_path_value = session.get("project_path")
    # A light session belongs to a saved project without unsaved changes: the project file itself is
    # the state, so neither its images nor a hash of a possibly huge file are stored.
    light = session.get("light") is True and isinstance(project_path_value, str) and bool(project_path_value)
    metadata["light"] = light
    if light:
        project = {**project, "images": [], "sources": []}
    elif isinstance(project_path_value, str) and project_path_value:
        try:
            source_project = Path(project_path_value).expanduser().resolve(strict=True)
            if source_project.is_file():
                metadata["source_project_sha256"] = _sha256_file(source_project)
        except OSError:
            pass
    try:
        session_json = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Cài đặt phiên làm việc không phải JSON an toàn.") from error

    project_path, metadata_path = _session_paths()
    temporary_project = project_path.with_name(".session-{0}.ffproj".format(uuid.uuid4().hex))
    temporary_metadata = metadata_path.with_name(".session-{0}.json".format(uuid.uuid4().hex))
    try:
        save_project(str(temporary_project), project)
        os.chmod(temporary_project, 0o600)
        temporary_metadata.write_text(session_json, encoding="utf-8")
        os.chmod(temporary_metadata, 0o600)
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
    project["name"] = _repair_mojibake(project.get("name"))
    original_path = session.get("project_path")
    original_exists = False
    original_unchanged = False
    if isinstance(original_path, str) and original_path:
        candidate = Path(original_path).expanduser()
        try:
            candidate = candidate.resolve(strict=True)
            original_exists = candidate.is_file() and _is_frameforge_project(candidate)
            expected_digest = session.get("source_project_sha256")
            if original_exists and session.get("light") is True:
                original_unchanged = True
            elif original_exists and isinstance(expected_digest, str):
                original_unchanged = _sha256_file(candidate) == expected_digest
        except (OSError, ValueError):
            original_exists = False
    return {
        "available": True,
        "project": project,
        "session": session,
        "original_project_exists": original_exists,
        "original_project_unchanged": original_unchanged,
    }
