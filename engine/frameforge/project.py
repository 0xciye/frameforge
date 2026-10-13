"""SQLite-backed FrameForge project persistence."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


SCHEMA_VERSION = 2


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS project_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sources (
            path TEXT PRIMARY KEY,
            position INTEGER NOT NULL,
            kind TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS project_settings (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS images (
            path TEXT PRIMARY KEY,
            source_path TEXT,
            source_id TEXT,
            frame_index INTEGER,
            timestamp_seconds REAL,
            crop_rect_json TEXT,
            sharpness_score REAL,
            extraction_settings_json TEXT,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS classes (
            class_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS annotations (
            annotation_id TEXT PRIMARY KEY,
            image_path TEXT NOT NULL REFERENCES images(path) ON DELETE CASCADE,
            class_id INTEGER NOT NULL,
            x_min REAL NOT NULL,
            y_min REAL NOT NULL,
            x_max REAL NOT NULL,
            y_max REAL NOT NULL,
            confidence REAL,
            source TEXT NOT NULL,
            review_status TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO project_meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )


def _migrate(connection: sqlite3.Connection) -> None:
    row = connection.execute("SELECT value FROM project_meta WHERE key = 'schema_version'").fetchone()
    version = int(row[0]) if row else 0
    if version == SCHEMA_VERSION:
        return
    if version != 1:
        raise ValueError("Phiên bản cấu trúc dự án không được hỗ trợ.")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(images)")}
    for name, definition in (
        ("source_id", "TEXT"),
        ("timestamp_seconds", "REAL"),
        ("crop_rect_json", "TEXT"),
        ("sharpness_score", "REAL"),
        ("extraction_settings_json", "TEXT"),
    ):
        if name not in columns:
            connection.execute("ALTER TABLE images ADD COLUMN {0} {1}".format(name, definition))
    connection.execute(
        "INSERT OR REPLACE INTO project_meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )


def _save(connection: sqlite3.Connection, project: Dict[str, Any]) -> Dict[str, Any]:
    project_id = str(project.get("id") or uuid.uuid4())
    name = str(project.get("name") or "Dự án mới").strip()[:200]
    now = datetime.now(timezone.utc).isoformat()
    with connection:
        meta = {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM project_meta")}
        meta.setdefault("project_id", project_id)
        meta.setdefault("created_at", now)
        meta["application"] = "FrameForge"
        meta["name"] = name
        meta["modified_at"] = now
        for key, value in meta.items():
            connection.execute("INSERT OR REPLACE INTO project_meta(key, value) VALUES(?, ?)", (key, value))

        connection.execute("DELETE FROM sources")
        for index, source in enumerate(project.get("sources", [])):
            if not isinstance(source, dict) or not source.get("path"):
                continue
            connection.execute(
                "INSERT INTO sources(path, position, kind, metadata_json) VALUES(?, ?, ?, ?)",
                (str(source["path"]), index, str(source.get("kind", "unknown")), json.dumps(source, ensure_ascii=False)),
            )

        connection.execute("DELETE FROM project_settings")
        settings = project.get("settings", {})
        if not isinstance(settings, dict):
            raise ValueError("Cài đặt dự án không hợp lệ.")
        for key, value in settings.items():
            connection.execute(
                "INSERT INTO project_settings(key, value_json) VALUES(?, ?)",
                (str(key), json.dumps(value, ensure_ascii=False)),
            )

        connection.execute("DELETE FROM annotations")
        connection.execute("DELETE FROM images")
        for image in project.get("images", []):
            if not isinstance(image, dict) or not image.get("path"):
                continue
            connection.execute(
                """INSERT INTO images(
                    path, source_path, source_id, frame_index, timestamp_seconds, crop_rect_json,
                    sharpness_score, extraction_settings_json, width, height, status
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(image["path"]),
                    image.get("source_path"),
                    image.get("source_id"),
                    image.get("frame_index"),
                    image.get("timestamp_seconds"),
                    json.dumps(image["crop_rect"], ensure_ascii=False) if image.get("crop_rect") is not None else None,
                    image.get("sharpness_score"),
                    json.dumps(image["extraction_settings"], ensure_ascii=False) if image.get("extraction_settings") is not None else None,
                    int(image.get("width", 0)),
                    int(image.get("height", 0)),
                    str(image.get("status", "ready")),
                ),
            )
            for annotation in image.get("annotations", []):
                connection.execute(
                    """INSERT INTO annotations(
                        annotation_id, image_path, class_id, x_min, y_min, x_max, y_max,
                        confidence, source, review_status
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(annotation.get("id") or uuid.uuid4()),
                        str(image["path"]),
                        int(annotation["class_id"]),
                        float(annotation["x_min"]),
                        float(annotation["y_min"]),
                        float(annotation["x_max"]),
                        float(annotation["y_max"]),
                        annotation.get("confidence"),
                        str(annotation.get("source", "manual")),
                        str(annotation.get("review_status", "unreviewed")),
                    ),
                )

        connection.execute("DELETE FROM classes")
        for class_id, class_name in enumerate(project.get("class_names", [])):
            connection.execute("INSERT INTO classes(class_id, name) VALUES(?, ?)", (class_id, str(class_name)))

    return {"id": meta["project_id"], "name": name, "modified_at": now}


def create_project(path_value: str, name: str) -> Dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if path.suffix.lower() != ".ffproj":
        raise ValueError("Dự án phải có phần mở rộng .ffproj.")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError("Tệp dự án đã tồn tại; hãy chọn tên khác hoặc mở dự án đó.") from error
    os.close(descriptor)
    connection = None
    try:
        connection = _connect(path)
        _initialize(connection)
        project = {"id": str(uuid.uuid4()), "name": name, "sources": [], "settings": {}, "images": [], "class_names": []}
        return _save(connection, project)
    except BaseException:
        if connection is not None:
            connection.close()
            connection = None
        path.unlink(missing_ok=True)
        raise
    finally:
        if connection is not None:
            connection.close()


def _is_frameforge_project(path: Path) -> bool:
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if not {"project_meta", "sources", "project_settings", "images", "classes", "annotations"}.issubset(tables):
            return False
        meta = {row[0]: row[1] for row in connection.execute("SELECT key, value FROM project_meta")}
        return (
            meta.get("schema_version") in {"1", str(SCHEMA_VERSION)}
            and bool(meta.get("project_id"))
            and bool(meta.get("name"))
            and meta.get("application") == "FrameForge"
        )
    finally:
        connection.close()


def save_project(path_value: str, project: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if path.suffix.lower() != ".ffproj":
        raise ValueError("Dự án phải có phần mở rộng .ffproj.")
    if not path.parent.is_dir():
        raise ValueError("Thư mục lưu dự án không tồn tại.")
    created = False
    if path.exists():
        if not path.is_file():
            raise ValueError("Đường dẫn dự án không phải là tệp.")
        try:
            valid_project = _is_frameforge_project(path)
        except sqlite3.DatabaseError as error:
            raise ValueError("Tệp đã tồn tại nhưng không phải là dự án FrameForge hợp lệ.") from error
        if not valid_project:
            raise ValueError("Tệp đã tồn tại nhưng không phải là dự án FrameForge hợp lệ.")
    else:
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            created = True
        except FileExistsError:
            try:
                valid_project = path.is_file() and _is_frameforge_project(path)
            except sqlite3.DatabaseError as error:
                raise ValueError("Tệp đã tồn tại nhưng không phải là dự án FrameForge hợp lệ.") from error
            if not valid_project:
                raise ValueError("Tệp đã tồn tại nhưng không phải là dự án FrameForge hợp lệ.")
    connection = None
    try:
        connection = _connect(path)
        if created:
            _initialize(connection)
        else:
            _migrate(connection)
        return _save(connection, project)
    except BaseException:
        if connection is not None:
            connection.close()
            connection = None
        if created:
            path.unlink(missing_ok=True)
        raise
    finally:
        if connection is not None:
            connection.close()


def load_project(path_value: str) -> Dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".ffproj" or not path.is_file():
        raise ValueError("Hãy chọn một tệp dự án FrameForge .ffproj.")
    readonly_uri = path.as_uri() + "?mode=ro"
    connection = sqlite3.connect(readonly_uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if check != "ok":
            raise ValueError("Tệp dự án SQLite không hợp lệ.")
        meta = {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM project_meta")}
        version = int(meta.get("schema_version", 0))
        if version < 1 or version > SCHEMA_VERSION:
            raise ValueError("Phiên bản cấu trúc dự án không được hỗ trợ.")
        sources = [json.loads(row["metadata_json"]) for row in connection.execute("SELECT metadata_json FROM sources ORDER BY position")]
        settings = {row["key"]: json.loads(row["value_json"]) for row in connection.execute("SELECT key, value_json FROM project_settings")}
        annotations = {}
        for row in connection.execute("SELECT * FROM annotations"):
            annotations.setdefault(row["image_path"], []).append({
                "id": row["annotation_id"], "class_id": row["class_id"],
                "x_min": row["x_min"], "y_min": row["y_min"],
                "x_max": row["x_max"], "y_max": row["y_max"],
                "confidence": row["confidence"], "source": row["source"],
                "review_status": row["review_status"],
            })
        images = []
        for row in connection.execute("SELECT * FROM images"):
            keys = row.keys()
            image = {
                "path": row["path"],
                "source_path": row["source_path"],
                "frame_index": row["frame_index"],
                "width": row["width"],
                "height": row["height"],
                "status": row["status"],
            }
            for field, column in (
                ("source_id", "source_id"),
                ("timestamp_seconds", "timestamp_seconds"),
                ("sharpness_score", "sharpness_score"),
            ):
                if column in keys:
                    image[field] = row[column]
            for field, column in (("crop_rect", "crop_rect_json"), ("extraction_settings", "extraction_settings_json")):
                if column in keys and row[column] is not None:
                    image[field] = json.loads(row[column])
            image["annotations"] = annotations.get(image["path"], [])
            images.append(image)
        class_names = [row["name"] for row in connection.execute("SELECT name FROM classes ORDER BY class_id")]
        return {
            "id": meta["project_id"],
            "name": meta["name"],
            "created_at": meta.get("created_at"),
            "modified_at": meta.get("modified_at"),
            "sources": sources,
            "settings": settings,
            "images": images,
            "class_names": class_names,
        }
    except (sqlite3.DatabaseError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Không đọc được dự án: {0}".format(error)) from error
    finally:
        connection.close()
