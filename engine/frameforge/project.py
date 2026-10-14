"""SQLite-backed FrameForge project persistence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


SCHEMA_VERSION = 4
DEFAULT_CLASS_NAMES = ("head", "body")


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
            review_status TEXT NOT NULL,
            source_model_id TEXT,
            checkpoint_sha256 TEXT
        );
        CREATE TABLE IF NOT EXISTS ai_negative_candidates (
            candidate_id TEXT PRIMARY KEY,
            image_path TEXT NOT NULL REFERENCES images(path) ON DELETE CASCADE,
            source_model_id TEXT NOT NULL,
            checkpoint_sha256 TEXT NOT NULL,
            image_fingerprint TEXT NOT NULL,
            review_status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            reviewed_at TEXT
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
    if version not in {1, 2, 3}:
        raise ValueError("Phiên bản cấu trúc dự án không được hỗ trợ.")
    if version == 1:
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
        connection.execute("UPDATE project_meta SET value = '2' WHERE key = 'schema_version'")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(annotations)")}
    for name in ("source_model_id", "checkpoint_sha256"):
        if name not in columns:
            connection.execute("ALTER TABLE annotations ADD COLUMN {0} TEXT".format(name))
    connection.execute(
        """CREATE TABLE IF NOT EXISTS ai_negative_candidates (
            candidate_id TEXT PRIMARY KEY,
            image_path TEXT NOT NULL REFERENCES images(path) ON DELETE CASCADE,
            source_model_id TEXT NOT NULL,
            checkpoint_sha256 TEXT NOT NULL,
            image_fingerprint TEXT NOT NULL,
            review_status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            reviewed_at TEXT
        )"""
    )
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
        connection.execute("DELETE FROM ai_negative_candidates")
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
                        confidence, source, review_status, source_model_id, checkpoint_sha256
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        annotation.get("source_model_id"),
                        annotation.get("checkpoint_sha256"),
                    ),
                )
            for candidate in image.get("negative_candidates", []):
                if not isinstance(candidate, dict):
                    continue
                connection.execute(
                    """INSERT INTO ai_negative_candidates(
                        candidate_id, image_path, source_model_id, checkpoint_sha256,
                        image_fingerprint, review_status, created_at, reviewed_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(candidate.get("id") or uuid.uuid4()), str(image["path"]),
                        str(candidate["source_model_id"]), str(candidate["checkpoint_sha256"]),
                        str(candidate["image_fingerprint"]), str(candidate.get("review_status", "needs_review")),
                        str(candidate.get("created_at") or now), candidate.get("reviewed_at"),
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
        project = {
            "id": str(uuid.uuid4()),
            "name": name,
            "sources": [],
            "settings": {},
            "images": [],
            "class_names": list(DEFAULT_CLASS_NAMES),
        }
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
            meta.get("schema_version") in {"1", "2", "3", str(SCHEMA_VERSION)}
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


def save_annotations(
    path_value: str,
    image_path: str,
    annotations: Any,
    image_status: str,
    image_fingerprint: str,
) -> Dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".ffproj" or not path.is_file():
        raise ValueError("Hãy chọn một tệp dự án FrameForge .ffproj.")
    if image_status not in {"ready", "reviewed"}:
        raise ValueError("Trạng thái ảnh không hợp lệ.")
    if not isinstance(annotations, list) or len(annotations) > 10000:
        raise ValueError("Danh sách nhãn phải là một mảng có tối đa 10000 box.")
    image_file = Path(image_path).expanduser()
    if not image_file.is_file():
        raise ValueError("Không tìm thấy ảnh để lưu nhãn.")
    if not isinstance(image_fingerprint, str) or len(image_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in image_fingerprint.casefold()
    ):
        raise ValueError("Thiếu fingerprint ảnh đã preview.")
    if hashlib.sha256(image_file.read_bytes()).hexdigest() != image_fingerprint:
        raise ValueError("Ảnh đã thay đổi trên đĩa. Hãy tải lại ảnh trước khi lưu nhãn.")
    try:
        valid_project = _is_frameforge_project(path)
    except sqlite3.DatabaseError as error:
        raise ValueError("Tệp đã tồn tại nhưng không phải là dự án FrameForge hợp lệ.") from error
    if not valid_project:
        raise ValueError("Tệp đã tồn tại nhưng không phải là dự án FrameForge hợp lệ.")

    connection = _connect(path)
    try:
        _migrate(connection)
        image = connection.execute(
            "SELECT width, height FROM images WHERE path = ?", (image_path,)
        ).fetchone()
        if image is None:
            raise ValueError("Ảnh không thuộc dự án hiện tại.")
        width, height = int(image["width"]), int(image["height"])
        if width <= 0 or height <= 0:
            raise ValueError("Kích thước ảnh trong dự án không hợp lệ.")
        class_ids = {row["class_id"] for row in connection.execute("SELECT class_id FROM classes")}
        existing_rows = {
            row["annotation_id"]: row
            for row in connection.execute("SELECT * FROM annotations WHERE image_path = ?", (image_path,))
        }

        cleaned = []
        annotation_ids = set()
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError("Mỗi nhãn phải là một object hợp lệ.")
            raw_class_id = annotation.get("class_id")
            if isinstance(raw_class_id, bool) or not isinstance(raw_class_id, int) or raw_class_id not in class_ids:
                raise ValueError("Nhãn tham chiếu class ID không tồn tại trong dự án.")
            try:
                x_min, y_min, x_max, y_max = (
                    float(annotation[key]) for key in ("x_min", "y_min", "x_max", "y_max")
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Tọa độ box không hợp lệ.") from error
            if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
                raise ValueError("Tọa độ box phải là số hữu hạn.")
            if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
                raise ValueError("Box phải có diện tích dương và nằm trong ảnh.")

            annotation_id = annotation.get("id")
            if annotation_id is None:
                annotation_id = str(uuid.uuid4())
            if not isinstance(annotation_id, str) or not annotation_id or len(annotation_id) > 128:
                raise ValueError("Mã nhãn không hợp lệ.")
            if annotation_id in annotation_ids:
                raise ValueError("Mã nhãn bị trùng lặp.")
            annotation_ids.add(annotation_id)

            source = annotation.get("source", "manual")
            review_status = annotation.get("review_status", "unreviewed")
            confidence = annotation.get("confidence")
            if confidence is not None:
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError) as error:
                    raise ValueError("Confidence phải nằm trong khoảng 0 đến 1.") from error
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("Confidence phải nằm trong khoảng 0 đến 1.")
            source_model_id = annotation.get("source_model_id")
            checkpoint_sha256 = annotation.get("checkpoint_sha256")
            if source == "manual":
                if review_status not in {"unreviewed", "reviewed"} or source_model_id is not None or checkpoint_sha256 is not None:
                    raise ValueError("Nguồn hoặc trạng thái review của nhãn thủ công không hợp lệ.")
                previous = existing_rows.get(annotation_id)
                if previous is not None and previous["source"] == "ai":
                    raise ValueError("Không thể đổi nguồn của nhãn AI thành nhãn thủ công.")
            elif source == "ai":
                if review_status not in {"needs_review", "reviewed", "rejected"}:
                    raise ValueError("Trạng thái review của gợi ý AI không hợp lệ.")
                previous = existing_rows.get(annotation_id)
                if previous is None or previous["source"] != "ai":
                    raise ValueError("Gợi ý AI mới chỉ được ghi qua luồng lưu inference đã xác minh.")
                if (
                    source_model_id != previous["source_model_id"]
                    or checkpoint_sha256 != previous["checkpoint_sha256"]
                ):
                    raise ValueError("Không thể thay đổi provenance của gợi ý AI.")
                old_confidence = previous["confidence"]
                if (old_confidence is None) != (confidence is None) or (
                    old_confidence is not None and float(old_confidence) != confidence
                ):
                    raise ValueError("Không thể thay đổi confidence gốc của gợi ý AI.")
            else:
                raise ValueError("Nguồn hoặc trạng thái review của nhãn không hợp lệ.")
            cleaned.append({
                "id": annotation_id,
                "class_id": raw_class_id,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "confidence": confidence,
                "source": source,
                "review_status": review_status,
                "source_model_id": source_model_id,
                "checkpoint_sha256": checkpoint_sha256,
            })

        for annotation_id, row in existing_rows.items():
            if row["source"] == "ai" and annotation_id not in annotation_ids:
                cleaned.append({
                    "id": annotation_id,
                    "class_id": row["class_id"],
                    "x_min": row["x_min"],
                    "y_min": row["y_min"],
                    "x_max": row["x_max"],
                    "y_max": row["y_max"],
                    "confidence": row["confidence"],
                    "source": row["source"],
                    "review_status": row["review_status"],
                    "source_model_id": row["source_model_id"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                })

        if image_status == "reviewed" and (
            any(item["source"] == "ai" and item["review_status"] == "needs_review" for item in cleaned)
            or connection.execute(
                "SELECT 1 FROM ai_negative_candidates WHERE image_path = ? AND review_status = 'needs_review' LIMIT 1",
                (image_path,),
            ).fetchone()
        ):
            raise ValueError("Hãy duyệt hoặc loại toàn bộ candidate AI trước khi đánh dấu ảnh đã review.")

        now = datetime.now(timezone.utc).isoformat()
        with connection:
            if hashlib.sha256(image_file.read_bytes()).hexdigest() != image_fingerprint:
                raise ValueError("Ảnh đã thay đổi trên đĩa. Hãy tải lại ảnh trước khi lưu nhãn.")
            connection.execute("DELETE FROM annotations WHERE image_path = ?", (image_path,))
            connection.executemany(
                """INSERT INTO annotations(
                    annotation_id, image_path, class_id, x_min, y_min, x_max, y_max,
                    confidence, source, review_status, source_model_id, checkpoint_sha256
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(
                    annotation["id"], image_path, annotation["class_id"], annotation["x_min"],
                    annotation["y_min"], annotation["x_max"], annotation["y_max"],
                    annotation["confidence"], annotation["source"], annotation["review_status"],
                    annotation["source_model_id"], annotation["checkpoint_sha256"],
                ) for annotation in cleaned],
            )
            connection.execute("UPDATE images SET status = ? WHERE path = ?", (image_status, image_path))
            connection.execute(
                "INSERT OR REPLACE INTO project_meta(key, value) VALUES('modified_at', ?)", (now,)
            )
        return {
            "image_path": image_path,
            "annotation_count": len(cleaned),
            "status": image_status,
            "annotations": cleaned,
        }
    except sqlite3.DatabaseError as error:
        raise ValueError("Không lưu được nhãn vào dự án: {0}".format(error)) from error
    finally:
        connection.close()


def save_ai_suggestions(
    path_value: str,
    image_path: str,
    predictions: Any,
    image_fingerprint: str,
    checkpoint_value: str,
) -> Dict[str, Any]:
    """Persist detections from a currently verified internal checkpoint as reviewable suggestions."""
    if not isinstance(predictions, list) or len(predictions) > 300:
        raise ValueError("Danh sách gợi ý phải có tối đa 300 box.")
    if not isinstance(image_fingerprint, str) or len(image_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in image_fingerprint.casefold()
    ):
        raise ValueError("Fingerprint ảnh không hợp lệ.")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("Inference chỉ nhận checkpoint YOLO11n nội bộ đã xác minh.")
    try:
        checkpoint = Path(checkpoint_value).expanduser().resolve(strict=True)
    except (OSError, ValueError) as error:
        raise ValueError("Checkpoint không còn được xác minh hoặc checksum đã thay đổi.") from error
    if not checkpoint.is_file() or checkpoint.suffix.casefold() != ".pt":
        raise ValueError("Inference chỉ nhận checkpoint YOLO11n nội bộ đã xác minh.")
    from frameforge.training import verified_checkpoint_record

    record = verified_checkpoint_record(checkpoint)
    if record is None:
        raise ValueError("Checkpoint không còn được xác minh hoặc checksum đã thay đổi.")
    source_model_id = record.get("run_id")
    checkpoint_sha256 = record.get("checkpoint_sha256")
    if (
        not isinstance(source_model_id, str)
        or not source_model_id
        or not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256.casefold())
    ):
        raise ValueError("Registry checkpoint thiếu provenance cần lưu.")

    raw_names = record.get("class_names")
    if isinstance(raw_names, dict):
        try:
            ordered_names = sorted(raw_names.items(), key=lambda item: int(item[0]))
            if [int(key) for key, _ in ordered_names] != list(range(len(ordered_names))):
                raise ValueError("Schema class trong registry không hợp lệ.")
            class_names = [name for _, name in ordered_names]
        except (TypeError, ValueError):
            raise ValueError("Schema class trong registry không hợp lệ.")
    elif isinstance(raw_names, (list, tuple)):
        class_names = list(raw_names)
    else:
        raise ValueError("Checkpoint thiếu schema class hợp lệ trong registry.")
    if not class_names or any(not isinstance(name, str) or not name.strip() for name in class_names):
        raise ValueError("Schema class trong registry không hợp lệ.")

    path = Path(path_value).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".ffproj" or not path.is_file():
        raise ValueError("Hãy chọn một tệp dự án FrameForge .ffproj.")
    image_file = Path(image_path).expanduser()
    if not image_file.is_file():
        raise ValueError("Không tìm thấy ảnh để lưu gợi ý.")
    if hashlib.sha256(image_file.read_bytes()).hexdigest() != image_fingerprint:
        raise ValueError("Ảnh đã thay đổi trên đĩa. Hãy tải lại ảnh trước khi lưu gợi ý.")
    try:
        valid_project = _is_frameforge_project(path)
    except sqlite3.DatabaseError as error:
        raise ValueError("Tệp đã tồn tại nhưng không phải là dự án FrameForge hợp lệ.") from error
    if not valid_project:
        raise ValueError("Tệp đã tồn tại nhưng không phải là dự án FrameForge hợp lệ.")

    connection = _connect(path)
    try:
        _migrate(connection)
        image = connection.execute("SELECT width, height FROM images WHERE path = ?", (image_path,)).fetchone()
        if image is None:
            raise ValueError("Ảnh không thuộc dự án hiện tại.")
        project_class_names = [row[0] for row in connection.execute("SELECT name FROM classes ORDER BY class_id")]
        if class_names != project_class_names:
            raise ValueError("Class schema của checkpoint không khớp với dự án.")
        if connection.execute(
            "SELECT 1 FROM annotations WHERE image_path = ? AND source = 'ai' AND review_status != 'rejected' AND source_model_id = ? AND checkpoint_sha256 = ? LIMIT 1",
            (image_path, source_model_id, checkpoint_sha256),
        ).fetchone():
            raise ValueError("Ảnh này đã có gợi ý chưa bị loại từ checkpoint đang chọn.")
        if connection.execute(
            "SELECT 1 FROM ai_negative_candidates WHERE image_path = ? AND review_status != 'rejected' AND source_model_id = ? AND checkpoint_sha256 = ? LIMIT 1",
            (image_path, source_model_id, checkpoint_sha256),
        ).fetchone():
            raise ValueError("Ảnh này đã có candidate chưa bị loại từ checkpoint đang chọn.")
        width, height = int(image["width"]), int(image["height"])
        cleaned = []
        for prediction in predictions:
            if not isinstance(prediction, dict):
                raise ValueError("Mỗi gợi ý phải là một object hợp lệ.")
            class_id = prediction.get("class_id")
            if isinstance(class_id, bool) or not isinstance(class_id, int) or not 0 <= class_id < len(class_names):
                raise ValueError("Gợi ý tham chiếu class ID không hợp lệ.")
            coordinates = prediction.get("bbox_xyxy")
            if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
                raise ValueError("Bounding box inference không hợp lệ.")
            confidence_value = prediction.get("confidence")
            numeric_values = [*coordinates, confidence_value]
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in numeric_values
            ):
                raise ValueError("Box và confidence inference phải là số.")
            try:
                x_min, y_min, x_max, y_max = (float(value) for value in coordinates)
                confidence = float(confidence_value)
            except (TypeError, ValueError) as error:
                raise ValueError("Box hoặc confidence inference không hợp lệ.") from error
            if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max, confidence)):
                raise ValueError("Box và confidence inference phải là số hữu hạn.")
            if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
                raise ValueError("Box inference phải nằm trong pixel bounds của ảnh.")
            if not 0 <= confidence <= 1:
                raise ValueError("Confidence inference phải nằm trong khoảng 0 đến 1.")
            cleaned.append({
                "id": str(uuid.uuid4()),
                "class_id": class_id,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "confidence": confidence,
                "source": "ai",
                "review_status": "needs_review",
                "source_model_id": source_model_id,
                "checkpoint_sha256": checkpoint_sha256,
            })

        now = datetime.now(timezone.utc).isoformat()
        candidate = None
        with connection:
            if hashlib.sha256(image_file.read_bytes()).hexdigest() != image_fingerprint:
                raise ValueError("Ảnh đã thay đổi trên đĩa. Hãy tải lại ảnh trước khi lưu gợi ý.")
            if cleaned:
                connection.executemany(
                    """INSERT INTO annotations(
                        annotation_id, image_path, class_id, x_min, y_min, x_max, y_max,
                        confidence, source, review_status, source_model_id, checkpoint_sha256
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [(
                        annotation["id"], image_path, annotation["class_id"], annotation["x_min"],
                        annotation["y_min"], annotation["x_max"], annotation["y_max"],
                        annotation["confidence"], annotation["source"], annotation["review_status"],
                        annotation["source_model_id"], annotation["checkpoint_sha256"],
                    ) for annotation in cleaned],
                )
            else:
                candidate = {
                    "id": str(uuid.uuid4()),
                    "image_path": image_path,
                    "source_model_id": source_model_id,
                    "checkpoint_sha256": checkpoint_sha256,
                    "image_fingerprint": image_fingerprint,
                    "review_status": "needs_review",
                    "created_at": now,
                    "reviewed_at": None,
                }
                connection.execute(
                    """INSERT INTO ai_negative_candidates(
                        candidate_id, image_path, source_model_id, checkpoint_sha256,
                        image_fingerprint, review_status, created_at, reviewed_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        candidate["id"], image_path, source_model_id, checkpoint_sha256,
                        image_fingerprint, "needs_review", now, None,
                    ),
                )
                if not connection.execute(
                    "SELECT 1 FROM annotations WHERE image_path = ? AND review_status != 'rejected' LIMIT 1",
                    (image_path,),
                ).fetchone():
                    connection.execute("UPDATE images SET status = 'ready' WHERE path = ?", (image_path,))
            connection.execute(
                "INSERT OR REPLACE INTO project_meta(key, value) VALUES('modified_at', ?)", (now,)
            )
        current_status = connection.execute("SELECT status FROM images WHERE path = ?", (image_path,)).fetchone()[0]
        return {
            "image_path": image_path,
            "annotation_count": len(cleaned),
            "annotations": cleaned,
            "negative_candidate": candidate,
            "status": current_status if candidate else None,
            "source_model_id": source_model_id,
            "checkpoint_sha256": checkpoint_sha256,
        }
    except sqlite3.DatabaseError as error:
        raise ValueError("Không lưu được gợi ý AI vào dự án: {0}".format(error)) from error
    finally:
        connection.close()


def project_images_for_inference(path_value: str) -> Dict[str, Any]:
    """List only unlabeled, unreviewed project images for one batch inference run."""
    path = Path(path_value).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".ffproj" or not path.is_file() or not _is_frameforge_project(path):
        raise ValueError("Hãy chọn một tệp dự án FrameForge .ffproj hợp lệ.")
    connection = _connect(path)
    try:
        _migrate(connection)
        class_names = [row[0] for row in connection.execute("SELECT name FROM classes ORDER BY class_id")]
        rows = connection.execute(
            """SELECT i.path, i.width, i.height FROM images AS i
            WHERE i.status != 'reviewed'
              AND NOT EXISTS (
                SELECT 1 FROM annotations AS a
                WHERE a.image_path = i.path AND (a.source = 'manual' OR a.review_status != 'rejected')
              )
              AND NOT EXISTS (
                SELECT 1 FROM ai_negative_candidates AS c
                WHERE c.image_path = i.path AND c.review_status != 'rejected'
              )
            ORDER BY i.rowid"""
        ).fetchall()
        return {
            "class_names": class_names,
            "images": [{"path": row["path"], "width": row["width"], "height": row["height"]} for row in rows],
        }
    finally:
        connection.close()


def _save_ai_suggestions_batch(
    path_value: str,
    items: Any,
    source_model_id: str,
    checkpoint_sha256: str,
    class_names: Any,
) -> Dict[str, int]:
    """Persist one bounded inference chunk in a single SQLite transaction."""
    if not isinstance(items, list) or not 1 <= len(items) <= 128:
        raise ValueError("Một lô inference phải có từ 1 đến 128 ảnh.")
    if not isinstance(source_model_id, str) or not source_model_id:
        raise ValueError("Thiếu mã model cho provenance inference.")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha256.casefold()
    ):
        raise ValueError("Checksum checkpoint inference không hợp lệ.")
    if not isinstance(class_names, (list, tuple)) or not class_names or any(
        not isinstance(name, str) or not name.strip() for name in class_names
    ):
        raise ValueError("Schema class trong checkpoint không hợp lệ.")
    path = Path(path_value).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".ffproj" or not path.is_file() or not _is_frameforge_project(path):
        raise ValueError("Hãy chọn một tệp dự án FrameForge .ffproj hợp lệ.")

    connection = _connect(path)
    try:
        _migrate(connection)
        project_classes = [row[0] for row in connection.execute("SELECT name FROM classes ORDER BY class_id")]
        if list(class_names) != project_classes:
            raise ValueError("Class schema của checkpoint không khớp với dự án.")

        prepared = []
        seen_paths = set()
        skipped_changed = 0
        skipped_duplicate = 0
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Bản ghi ảnh inference không hợp lệ.")
            image_path = item.get("image_path")
            fingerprint = item.get("image_fingerprint")
            predictions = item.get("predictions")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError("Thiếu đường dẫn ảnh inference.")
            if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
                character not in "0123456789abcdef" for character in fingerprint.casefold()
            ):
                raise ValueError("Fingerprint ảnh inference không hợp lệ.")
            if not isinstance(predictions, list) or len(predictions) > 300:
                raise ValueError("Danh sách gợi ý phải có tối đa 300 box mỗi ảnh.")
            try:
                image_file = Path(image_path).expanduser().resolve(strict=True)
            except (OSError, ValueError):
                skipped_changed += 1
                continue
            image_path = str(image_file)
            if image_path in seen_paths:
                raise ValueError("Một lô inference không được chứa đường dẫn ảnh trùng.")
            seen_paths.add(image_path)
            image = connection.execute("SELECT width, height FROM images WHERE path = ?", (image_path,)).fetchone()
            if image is None:
                raise ValueError("Ảnh không thuộc dự án hiện tại: {0}".format(image_path))
            if not image_file.is_file() or hashlib.sha256(image_file.read_bytes()).hexdigest() != fingerprint:
                skipped_changed += 1
                continue
            if connection.execute(
                """SELECT 1 FROM annotations WHERE image_path = ? AND source = 'ai'
                   AND review_status != 'rejected' AND source_model_id = ? AND checkpoint_sha256 = ? LIMIT 1""",
                (image_path, source_model_id, checkpoint_sha256),
            ).fetchone() or connection.execute(
                """SELECT 1 FROM ai_negative_candidates WHERE image_path = ?
                   AND review_status != 'rejected' AND source_model_id = ? AND checkpoint_sha256 = ? LIMIT 1""",
                (image_path, source_model_id, checkpoint_sha256),
            ).fetchone():
                skipped_duplicate += 1
                continue

            width, height = int(image["width"]), int(image["height"])
            annotations = []
            for prediction in predictions:
                if not isinstance(prediction, dict):
                    raise ValueError("Mỗi gợi ý phải là một object hợp lệ.")
                class_id = prediction.get("class_id")
                coordinates = prediction.get("bbox_xyxy")
                confidence_value = prediction.get("confidence")
                if isinstance(class_id, bool) or not isinstance(class_id, int) or not 0 <= class_id < len(class_names):
                    raise ValueError("Gợi ý tham chiếu class ID không hợp lệ.")
                if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
                    raise ValueError("Bounding box inference không hợp lệ.")
                values = [*coordinates, confidence_value]
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
                    raise ValueError("Box và confidence inference phải là số.")
                x_min, y_min, x_max, y_max, confidence = (float(value) for value in values)
                if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max, confidence)):
                    raise ValueError("Box và confidence inference phải là số hữu hạn.")
                if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
                    raise ValueError("Box inference phải nằm trong pixel bounds của ảnh.")
                if not 0 <= confidence <= 1:
                    raise ValueError("Confidence inference phải nằm trong khoảng 0 đến 1.")
                annotations.append({
                    "id": str(uuid.uuid4()), "class_id": class_id,
                    "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
                    "confidence": confidence, "source": "ai", "review_status": "needs_review",
                    "source_model_id": source_model_id, "checkpoint_sha256": checkpoint_sha256,
                })
            prepared.append({
                "image_path": image_path, "image_file": image_file, "fingerprint": fingerprint,
                "predictions": annotations,
            })

        now = datetime.now(timezone.utc).isoformat()
        with connection:
            # Recheck bytes at the commit boundary so changed inputs never receive stale boxes.
            committed = []
            for item in prepared:
                if hashlib.sha256(item["image_file"].read_bytes()).hexdigest() != item["fingerprint"]:
                    skipped_changed += 1
                else:
                    committed.append(item)
            for item in committed:
                image_path = item["image_path"]
                annotations = item["predictions"]
                if annotations:
                    connection.executemany(
                        """INSERT INTO annotations(
                            annotation_id, image_path, class_id, x_min, y_min, x_max, y_max,
                            confidence, source, review_status, source_model_id, checkpoint_sha256
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [(
                            annotation["id"], image_path, annotation["class_id"], annotation["x_min"],
                            annotation["y_min"], annotation["x_max"], annotation["y_max"],
                            annotation["confidence"], "ai", "needs_review", source_model_id, checkpoint_sha256,
                        ) for annotation in annotations],
                    )
                else:
                    connection.execute(
                        """INSERT INTO ai_negative_candidates(
                            candidate_id, image_path, source_model_id, checkpoint_sha256,
                            image_fingerprint, review_status, created_at, reviewed_at
                        ) VALUES(?, ?, ?, ?, ?, 'needs_review', ?, NULL)""",
                        (str(uuid.uuid4()), image_path, source_model_id, checkpoint_sha256, item["fingerprint"], now),
                    )
                    if not connection.execute(
                        "SELECT 1 FROM annotations WHERE image_path = ? AND review_status != 'rejected' LIMIT 1",
                        (image_path,),
                    ).fetchone():
                        connection.execute("UPDATE images SET status = 'ready' WHERE path = ?", (image_path,))
            if committed:
                connection.execute(
                    "INSERT OR REPLACE INTO project_meta(key, value) VALUES('modified_at', ?)", (now,)
                )
        return {
            "processed_images": len(committed),
            "images_with_detections": sum(bool(item["predictions"]) for item in committed),
            "candidate_negative_count": sum(not item["predictions"] for item in committed),
            "annotation_count": sum(len(item["predictions"]) for item in committed),
            "skipped_changed": skipped_changed,
            "skipped_duplicate": skipped_duplicate,
        }
    except sqlite3.DatabaseError as error:
        raise ValueError("Không lưu được gợi ý inference theo lô: {0}".format(error)) from error
    finally:
        connection.close()


def review_ai_negative_candidate(
    path_value: str,
    candidate_id: str,
    action: str,
    image_fingerprint: str | None = None,
) -> Dict[str, Any]:
    if action not in {"confirmed_negative", "rejected"}:
        raise ValueError("Thao tác review candidate không hợp lệ.")
    path = Path(path_value).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".ffproj" or not path.is_file() or not _is_frameforge_project(path):
        raise ValueError("Hãy chọn một tệp dự án FrameForge .ffproj hợp lệ.")
    connection = _connect(path)
    try:
        _migrate(connection)
        candidate = connection.execute(
            "SELECT * FROM ai_negative_candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        if candidate is None:
            raise ValueError("Candidate không còn trong dự án.")
        if candidate["review_status"] != "needs_review":
            raise ValueError("Candidate này đã được review.")
        image_path = Path(candidate["image_path"]).expanduser().resolve(strict=True)
        if action == "confirmed_negative":
            if image_fingerprint != candidate["image_fingerprint"]:
                raise ValueError("Ảnh preview không khớp với candidate; không thể xác nhận ảnh âm tính.")
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != candidate["image_fingerprint"]:
                raise ValueError("Ảnh đã thay đổi kể từ inference. Hãy chạy lại inference trên ảnh mới.")
            if connection.execute(
                "SELECT 1 FROM annotations WHERE image_path = ? AND review_status != 'rejected' LIMIT 1",
                (str(image_path),),
            ).fetchone():
                raise ValueError("Ảnh đang có box. Không thể xác nhận ảnh này là âm tính.")
            if connection.execute(
                "SELECT 1 FROM ai_negative_candidates WHERE image_path = ? AND review_status = 'needs_review' AND candidate_id != ? LIMIT 1",
                (str(image_path), candidate_id),
            ).fetchone():
                raise ValueError("Hãy review các candidate khác của ảnh trước.")
        now = datetime.now(timezone.utc).isoformat()
        with connection:
            connection.execute(
                "UPDATE ai_negative_candidates SET review_status = ?, reviewed_at = ? WHERE candidate_id = ?",
                (action, now, candidate_id),
            )
            if action == "confirmed_negative":
                connection.execute("UPDATE images SET status = 'reviewed' WHERE path = ?", (str(image_path),))
            elif not connection.execute(
                "SELECT 1 FROM annotations WHERE image_path = ? AND review_status != 'rejected' LIMIT 1",
                (str(image_path),),
            ).fetchone():
                connection.execute("UPDATE images SET status = 'ready' WHERE path = ?", (str(image_path),))
            connection.execute(
                "INSERT OR REPLACE INTO project_meta(key, value) VALUES('modified_at', ?)", (now,)
            )
        updated = dict(candidate)
        updated["review_status"] = action
        updated["reviewed_at"] = now
        return {
            "image_path": str(image_path),
            "status": "reviewed" if action == "confirmed_negative" else "ready",
            "negative_candidate": {
                "id": updated["candidate_id"],
                "source_model_id": updated["source_model_id"],
                "checkpoint_sha256": updated["checkpoint_sha256"],
                "image_fingerprint": updated["image_fingerprint"],
                "review_status": updated["review_status"],
                "created_at": updated["created_at"],
                "reviewed_at": updated["reviewed_at"],
            },
        }
    except sqlite3.DatabaseError as error:
        raise ValueError("Không lưu được trạng thái review candidate: {0}".format(error)) from error
    finally:
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
            annotation = {
                "id": row["annotation_id"], "class_id": row["class_id"],
                "x_min": row["x_min"], "y_min": row["y_min"],
                "x_max": row["x_max"], "y_max": row["y_max"],
                "confidence": row["confidence"], "source": row["source"],
                "review_status": row["review_status"],
            }
            keys = row.keys()
            for field in ("source_model_id", "checkpoint_sha256"):
                if field in keys:
                    annotation[field] = row[field]
            if annotation["source"] == "ai" and annotation["review_status"] == "unreviewed":
                annotation["review_status"] = "needs_review"
            annotations.setdefault(row["image_path"], []).append(annotation)
        negative_candidates = {}
        if version >= 4:
            for row in connection.execute("SELECT * FROM ai_negative_candidates"):
                negative_candidates.setdefault(row["image_path"], []).append({
                    "id": row["candidate_id"],
                    "source_model_id": row["source_model_id"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "image_fingerprint": row["image_fingerprint"],
                    "review_status": row["review_status"],
                    "created_at": row["created_at"],
                    "reviewed_at": row["reviewed_at"],
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
            image["negative_candidates"] = negative_candidates.get(image["path"], [])
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
