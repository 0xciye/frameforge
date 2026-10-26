"""Labels live in a LabelMe/X-AnyLabeling JSON beside each image (``photo.png`` -> ``photo.json``).

The JSON is the label store; the project database keeps a copy for fast queries. FrameForge writes the
JSON after every label change and reads it back when the file changed outside the app (compared by
SHA-256 against what FrameForge last wrote or read).

``shapes`` holds only real labels (manual boxes and accepted AI boxes), so LabelMe, X-AnyLabeling and
training scripts never read an unreviewed suggestion as a label. FrameForge-only state (image review
status, AI suggestions waiting for review or rejected, negative candidates) sits under ``frameforge``.
Shapes FrameForge does not manage (other shape types, labels outside the project classes) are kept.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LABELME_VERSION = "4.2.0"
FORMAT_VERSION = 1
MAX_JSON_BYTES = 64 * 1024 * 1024
MANAGED_TYPES = {"rectangle", "polygon"}


def json_path_for(image_path: str) -> Path:
    return Path(image_path).with_suffix(".json")


def _ensure_state(connection: sqlite3.Connection) -> None:
    """Per image: the JSON as FrameForge last wrote or read it (hash, mtime, size) and a digest of the
    labels the database held then, so unchanged files and unchanged labels are skipped without I/O."""
    connection.execute(
        """CREATE TABLE IF NOT EXISTS sidecar_state (
            image_path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL
        )"""
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(sidecar_state)")}
    for name, definition in (("mtime_ns", "INTEGER"), ("size", "INTEGER"), ("db_digest", "TEXT")):
        if name not in columns:
            connection.execute("ALTER TABLE sidecar_state ADD COLUMN {0} {1}".format(name, definition))


def _stat(path: Path) -> Optional[Tuple[int, int]]:
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return info.st_mtime_ns, info.st_size


class _JsonStats:
    """mtime/size of the JSON files beside many images, one directory listing per folder.

    On Windows a directory listing already carries each file's size and time, so 100 000 images cost a
    few listings instead of 100 000 ``stat`` calls.
    """

    def __init__(self) -> None:
        self._folders: Dict[str, Dict[str, Tuple[int, int]]] = {}

    def get(self, path: Path) -> Optional[Tuple[int, int]]:
        folder = str(path.parent)
        listing = self._folders.get(folder)
        if listing is None:
            listing = {}
            try:
                with os.scandir(folder) as entries:
                    for entry in entries:
                        if entry.name.lower().endswith(".json") and entry.is_file():
                            info = entry.stat()
                            listing[os.path.normcase(entry.name)] = (info.st_mtime_ns, info.st_size)
            except OSError:
                pass
            self._folders[folder] = listing
        return listing.get(os.path.normcase(path.name))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> Optional[bytes]:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("tệp JSON quá lớn")
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _decode(data: bytes) -> Dict[str, Any]:
    document = json.loads(data.decode("utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError("JSON không phải object LabelMe")
    return document


def _class_ids(class_names: List[str]) -> Dict[str, int]:
    return {name.strip().casefold(): index for index, name in enumerate(class_names)}


def _managed(shape: Any, class_ids: Dict[str, int]) -> bool:
    return (
        isinstance(shape, dict)
        and (shape.get("shape_type") or "polygon") in MANAGED_TYPES
        and str(shape.get("label", "")).strip().casefold() in class_ids
    )


def _corners(x_min: float, y_min: float, x_max: float, y_max: float) -> List[List[float]]:
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


def _shape(annotation: Dict[str, Any], class_names: List[str], base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    shape = dict(base) if base else {
        "label": "", "score": None, "points": [], "group_id": None, "description": "",
        "difficult": False, "shape_type": "rectangle", "flags": {}, "attributes": {}, "kie_linking": [],
    }
    shape["label"] = class_names[annotation["class_id"]]
    shape["score"] = annotation.get("confidence")
    shape["points"] = _corners(annotation["x_min"], annotation["y_min"], annotation["x_max"], annotation["y_max"])
    shape["shape_type"] = "rectangle"
    shape["frameforge"] = {
        "id": annotation["id"],
        "source": annotation["source"],
        "review_status": annotation["review_status"],
        "source_model_id": annotation.get("source_model_id"),
        "checkpoint_sha256": annotation.get("checkpoint_sha256"),
    }
    return shape


def _load_images(connection: sqlite3.Connection, image_paths: Optional[Iterable[str]]) -> List[Dict[str, Any]]:
    """Images with their annotations and negative candidates, in three queries for the whole project."""
    wanted = None if image_paths is None else list(dict.fromkeys(str(path) for path in image_paths))
    if wanted is not None and not wanted:
        return []
    if wanted is not None and len(wanted) <= 500:
        marks = ",".join("?" * len(wanted))
        rows = connection.execute(
            "SELECT path, width, height, status FROM images WHERE path IN ({0}) ORDER BY rowid".format(marks), wanted
        ).fetchall()
        annotation_rows = connection.execute(
            "SELECT * FROM annotations WHERE image_path IN ({0}) ORDER BY rowid".format(marks), wanted
        ).fetchall()
        candidate_rows = connection.execute(
            "SELECT * FROM ai_negative_candidates WHERE image_path IN ({0}) ORDER BY rowid".format(marks), wanted
        ).fetchall()
    else:
        rows = connection.execute("SELECT path, width, height, status FROM images ORDER BY rowid").fetchall()
        annotation_rows = connection.execute("SELECT * FROM annotations ORDER BY rowid").fetchall()
        candidate_rows = connection.execute("SELECT * FROM ai_negative_candidates ORDER BY rowid").fetchall()
    images: Dict[str, Dict[str, Any]] = {
        row["path"]: {
            "path": row["path"], "width": row["width"], "height": row["height"], "status": row["status"],
            "annotations": [], "negative_candidates": [],
        }
        for row in rows
    }
    for item in annotation_rows:
        image = images.get(item["image_path"])
        if image is not None:
            image["annotations"].append({
                "id": item["annotation_id"], "class_id": item["class_id"],
                "x_min": item["x_min"], "y_min": item["y_min"], "x_max": item["x_max"], "y_max": item["y_max"],
                "confidence": item["confidence"], "source": item["source"], "review_status": item["review_status"],
                "source_model_id": item["source_model_id"], "checkpoint_sha256": item["checkpoint_sha256"],
            })
    for item in candidate_rows:
        image = images.get(item["image_path"])
        if image is not None:
            image["negative_candidates"].append({
                "id": item["candidate_id"], "source_model_id": item["source_model_id"],
                "checkpoint_sha256": item["checkpoint_sha256"], "image_fingerprint": item["image_fingerprint"],
                "review_status": item["review_status"], "created_at": item["created_at"], "reviewed_at": item["reviewed_at"],
            })
    if wanted is None:
        return list(images.values())
    return [images[path] for path in wanted if path in images]


def _db_digest(image: Dict[str, Any], class_names: List[str]) -> str:
    content = [image["status"], image["width"], image["height"], class_names, image["annotations"], image["negative_candidates"]]
    return _sha(json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _states(connection: sqlite3.Connection) -> Dict[str, sqlite3.Row]:
    return {row["image_path"]: row for row in connection.execute("SELECT * FROM sidecar_state")}


def _remember(connection: sqlite3.Connection, image_path: str, sha256: str, stat: Optional[Tuple[int, int]], digest: str) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO sidecar_state(image_path, sha256, mtime_ns, size, db_digest) VALUES(?, ?, ?, ?, ?)",
        (image_path, sha256, stat[0] if stat else None, stat[1] if stat else None, digest),
    )


def _has_labels(image: Dict[str, Any]) -> bool:
    return bool(image["annotations"] or image["negative_candidates"] or image["status"] == "reviewed")


def compose(existing: Optional[Dict[str, Any]], image: Dict[str, Any], class_names: List[str]) -> Dict[str, Any]:
    """The JSON document for ``image``, keeping unrelated fields and shapes of ``existing``."""
    class_ids = _class_ids(class_names)
    document = dict(existing) if existing else {"version": LABELME_VERSION, "flags": {}}
    old_shapes = [shape for shape in document.get("shapes") or [] if isinstance(shape, dict)]
    foreign = [shape for shape in old_shapes if not _managed(shape, class_ids)]
    by_id = {
        shape["frameforge"].get("id"): shape
        for shape in old_shapes
        if isinstance(shape.get("frameforge"), dict) and shape["frameforge"].get("id")
    }
    old_suggestions = (document.get("frameforge") or {}).get("suggestions") if isinstance(document.get("frameforge"), dict) else None
    for shape in old_suggestions or []:
        if isinstance(shape, dict) and isinstance(shape.get("frameforge"), dict) and shape["frameforge"].get("id"):
            by_id.setdefault(shape["frameforge"]["id"], shape)

    # Shapes written by other tools carry no FrameForge id: match them by label and box.
    by_box: Dict[Tuple[str, float, float, float, float], Dict[str, Any]] = {}
    for shape in old_shapes:
        if _managed(shape, class_ids) and not (isinstance(shape.get("frameforge"), dict) and shape["frameforge"].get("id")):
            box = _finite_box(shape.get("points"), int(image["width"]), int(image["height"]))
            if box is not None:
                key = (str(shape.get("label", "")).strip().casefold(),) + tuple(round(value, 3) for value in box[:4])
                by_box.setdefault(key, shape)

    labels, suggestions = [], []
    for annotation in image["annotations"]:
        key = (class_names[annotation["class_id"]].strip().casefold(),) + tuple(
            round(annotation[field], 3) for field in ("x_min", "y_min", "x_max", "y_max")
        )
        shape = _shape(annotation, class_names, by_id.get(annotation["id"]) or by_box.get(key))
        if annotation["source"] == "manual" or annotation["review_status"] == "reviewed":
            labels.append(shape)
        else:
            suggestions.append(shape)
    document["shapes"] = foreign + labels
    document["imagePath"] = Path(image["path"]).name
    document.setdefault("imageData", None)
    document["imageHeight"] = int(image["height"])
    document["imageWidth"] = int(image["width"])
    document["checked"] = image["status"] == "reviewed"
    document["frameforge"] = {
        "version": FORMAT_VERSION,
        "status": image["status"],
        "suggestions": suggestions,
        "negative_candidates": image["negative_candidates"],
    }
    return document


def _encode(document: Dict[str, Any]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(".{0}.{1}.tmp".format(path.name, uuid.uuid4().hex[:8]))
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_box(points: Any, width: int, height: int) -> Optional[Tuple[float, float, float, float, bool]]:
    if not isinstance(points, list) or len(points) < 2:
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(value) for value in xs + ys):
        return None
    x_min, x_max = max(0.0, min(xs)), min(float(width), max(xs))
    y_min, y_max = max(0.0, min(ys)), min(float(height), max(ys))
    clipped = min(xs) < 0 or min(ys) < 0 or max(xs) > width or max(ys) > height
    if x_max - x_min < 1 or y_max - y_min < 1:
        return None
    return x_min, y_min, x_max, y_max, clipped


def parse(
    document: Dict[str, Any], width: int, height: int, class_names: List[str], default_reviewed: bool, counts: Dict[str, int]
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Image status, annotations and negative candidates stored in a JSON document."""
    class_ids = _class_ids(class_names)
    declared = (document.get("imageWidth"), document.get("imageHeight"))
    if all(isinstance(value, int) and value > 0 for value in declared) and declared != (width, height):
        raise ValueError("JSON ghi kích thước {0}×{1} nhưng ảnh là {2}×{3}".format(declared[0], declared[1], width, height))
    meta = document.get("frameforge") if isinstance(document.get("frameforge"), dict) else {}
    status = meta.get("status") if meta.get("status") in {"ready", "reviewed"} else None

    def annotation_from(shape: Dict[str, Any], pending: bool) -> Optional[Dict[str, Any]]:
        if (shape.get("shape_type") or "polygon") not in MANAGED_TYPES:
            counts["unsupported_shapes"] += 1
            return None
        class_id = class_ids.get(str(shape.get("label", "")).strip().casefold())
        if class_id is None:
            counts["unknown_labels"] += 1
            return None
        box = _finite_box(shape.get("points"), width, height)
        if box is None:
            counts["degenerate_boxes"] += 1
            return None
        x_min, y_min, x_max, y_max, clipped = box
        counts["clipped_boxes"] += 1 if clipped else 0
        info = shape.get("frameforge") if isinstance(shape.get("frameforge"), dict) else {}
        source = info.get("source") if info.get("source") in {"manual", "ai"} else "manual"
        review_status = info.get("review_status")
        score = shape.get("score")
        confidence = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) and 0 <= score <= 1 else None
        if source == "ai":
            if not info.get("source_model_id") or not info.get("checkpoint_sha256"):
                source, review_status = "manual", None
            elif pending:
                if review_status not in {"needs_review", "rejected"}:
                    review_status = "needs_review"
            else:
                review_status = "reviewed"
        if source == "manual":
            confidence = None
            if pending:
                return None
            if review_status not in {"unreviewed", "reviewed"}:
                review_status = None  # decided from the image status below
        identifier = info.get("id") if isinstance(info.get("id"), str) and 0 < len(info["id"]) <= 128 else str(uuid.uuid4())
        return {
            "id": identifier, "class_id": class_id,
            "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
            "confidence": confidence, "source": source, "review_status": review_status,
            "source_model_id": info.get("source_model_id") if source == "ai" else None,
            "checkpoint_sha256": info.get("checkpoint_sha256") if source == "ai" else None,
        }

    annotations = []
    for shape in document.get("shapes") or []:
        if isinstance(shape, dict):
            item = annotation_from(shape, pending=False)
            if item is not None:
                annotations.append(item)
    for shape in meta.get("suggestions") or []:
        if isinstance(shape, dict):
            item = annotation_from(shape, pending=True)
            if item is not None:
                annotations.append(item)

    candidates = []
    for candidate in meta.get("negative_candidates") or []:
        if (
            isinstance(candidate, dict)
            and all(isinstance(candidate.get(key), str) and candidate[key] for key in ("source_model_id", "checkpoint_sha256", "image_fingerprint"))
            and candidate.get("review_status") in {"needs_review", "confirmed_negative", "rejected"}
        ):
            candidates.append({
                "id": candidate["id"] if isinstance(candidate.get("id"), str) and candidate["id"] else str(uuid.uuid4()),
                "source_model_id": candidate["source_model_id"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "image_fingerprint": candidate["image_fingerprint"],
                "review_status": candidate["review_status"],
                "created_at": candidate.get("created_at") or "",
                "reviewed_at": candidate.get("reviewed_at"),
            })

    has_labels = any(item["source"] == "manual" or item["review_status"] == "reviewed" for item in annotations)
    if status is None:
        status = "reviewed" if document.get("checked") is True or (default_reviewed and has_labels) else "ready"
    pending = any(item["review_status"] == "needs_review" for item in annotations) or any(
        candidate["review_status"] == "needs_review" for candidate in candidates
    )
    if pending:
        status = "ready"
    for item in annotations:
        if item["source"] == "manual" and item["review_status"] is None:
            item["review_status"] = "reviewed" if status == "reviewed" else "unreviewed"
    seen = set()
    for item in annotations:
        if item["id"] in seen:
            item["id"] = str(uuid.uuid4())
        seen.add(item["id"])
    return status, annotations, candidates


def _class_names(connection: sqlite3.Connection) -> List[str]:
    return [row["name"] for row in connection.execute("SELECT name FROM classes ORDER BY class_id")]


def write(connection: sqlite3.Connection, image_paths: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Write the JSON of each image from the database (all images when ``image_paths`` is None).

    An image whose labels and file are both unchanged since the last write is skipped without reading
    the file, so saving a 100 000-image project only touches the images that changed.
    """
    _ensure_state(connection)
    class_names = _class_names(connection)
    states = _states(connection)
    stats = _JsonStats()
    written, errors = 0, []
    with connection:
        for image in _load_images(connection, image_paths):
            image_path = image["path"]
            path = json_path_for(image_path)
            state = states.get(image_path)
            stat = stats.get(path)
            if state is None and stat is None and not _has_labels(image):
                continue  # Unlabeled image without a file: nothing to store, skip the digest.
            digest = _db_digest(image, class_names)
            if state is not None and state["db_digest"] == digest and stat is not None and (
                state["mtime_ns"], state["size"]
            ) == stat:
                continue
            if stat is None and not _has_labels(image):
                continue  # Nothing to store; do not create empty files for unlabeled images.
            try:
                current = _read(path)
                existing = _decode(current) if current is not None else None
            except (OSError, UnicodeDecodeError, ValueError) as error:
                errors.append({"path": str(path), "message": "Không đọc được JSON hiện có, không ghi đè: {0}".format(error)})
                continue
            if existing is None and not _has_labels(image):
                continue
            data = _encode(compose(existing, image, class_names))
            try:
                if data != current:
                    _atomic_write(path, data)
                    written += 1
            except OSError as error:
                errors.append({"path": str(path), "message": "Không ghi được JSON: {0}".format(error)})
                continue
            _remember(connection, image_path, _sha(data), _stat(path), digest)
        if image_paths is None:
            connection.execute("DELETE FROM sidecar_state WHERE image_path NOT IN (SELECT path FROM images)")
    return {"written": written, "errors": errors}


def _replace(connection: sqlite3.Connection, image_path: str, status: str, annotations, candidates) -> None:
    connection.execute("DELETE FROM annotations WHERE image_path = ?", (image_path,))
    connection.execute("DELETE FROM ai_negative_candidates WHERE image_path = ?", (image_path,))
    connection.executemany(
        """INSERT INTO annotations(
            annotation_id, image_path, class_id, x_min, y_min, x_max, y_max,
            confidence, source, review_status, source_model_id, checkpoint_sha256
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(
            item["id"], image_path, item["class_id"], item["x_min"], item["y_min"], item["x_max"], item["y_max"],
            item["confidence"], item["source"], item["review_status"], item["source_model_id"], item["checkpoint_sha256"],
        ) for item in annotations],
    )
    connection.executemany(
        """INSERT INTO ai_negative_candidates(
            candidate_id, image_path, source_model_id, checkpoint_sha256,
            image_fingerprint, review_status, created_at, reviewed_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
        [(
            item["id"], image_path, item["source_model_id"], item["checkpoint_sha256"],
            item["image_fingerprint"], item["review_status"], item["created_at"], item["reviewed_at"],
        ) for item in candidates],
    )
    connection.execute("UPDATE images SET status = ? WHERE path = ?", (status, image_path))


def sync(
    connection: sqlite3.Connection, image_paths: Optional[Iterable[str]] = None, default_reviewed: bool = False
) -> Dict[str, Any]:
    """Bring the database and the JSON files in line.

    A JSON that changed since FrameForge last wrote or read it wins and replaces the image's labels in the
    database. A missing JSON, or one FrameForge has never seen while the database already holds labels for
    the image (projects made before JSON storage), is written from the database instead. Files whose
    modification time and size match the last write/read are not opened.
    """
    _ensure_state(connection)
    class_names = _class_names(connection)
    counts = {"unsupported_shapes": 0, "unknown_labels": 0, "degenerate_boxes": 0, "clipped_boxes": 0}
    read_images, to_write, errors = [], [], []
    states = _states(connection)
    stats = _JsonStats()
    with connection:
        for image in _load_images(connection, image_paths):
            image_path = image["path"]
            path = json_path_for(image_path)
            state = states.get(image_path)
            stat = stats.get(path)
            if stat is None:
                if _has_labels(image):
                    to_write.append(image_path)
                continue
            if state is not None and (state["mtime_ns"], state["size"]) == stat:
                continue
            try:
                data = _read(path)
            except (OSError, ValueError) as error:
                errors.append({"path": str(path), "message": "Không đọc được JSON: {0}".format(error)})
                continue
            if data is None:
                continue
            digest = _sha(data)
            if state is not None and state["sha256"] == digest:
                # Same bytes, new timestamp (copied or touched): remember the new stat only.
                connection.execute(
                    "UPDATE sidecar_state SET mtime_ns = ?, size = ? WHERE image_path = ?", (stat[0], stat[1], image_path)
                )
                continue
            if state is None and _has_labels(image):
                to_write.append(image_path)
                continue
            try:
                status, annotations, candidates = parse(
                    _decode(data), int(image["width"]), int(image["height"]), class_names, default_reviewed, counts
                )
            except (UnicodeDecodeError, ValueError) as error:
                errors.append({"path": str(path), "message": "JSON không hợp lệ, giữ nhãn hiện tại: {0}".format(error)})
                continue
            _replace(connection, image_path, status, annotations, candidates)
            stored = {**image, "status": status, "annotations": annotations, "negative_candidates": candidates}
            refreshed = _load_images(connection, [image_path])
            _remember(connection, image_path, digest, stat, _db_digest(refreshed[0] if refreshed else stored, class_names))
            read_images.append(image_path)
    result = write(connection, to_write) if to_write else {"written": 0, "errors": []}
    return {
        "read_images": read_images,
        "written": result["written"],
        "errors": errors + result["errors"],
        **counts,
    }


def _open(project_value: str) -> sqlite3.Connection:
    from frameforge.project import _connect, _is_frameforge_project, _migrate

    project = Path(project_value).expanduser().resolve(strict=True)
    if project.suffix.lower() != ".ffproj" or not project.is_file() or not _is_frameforge_project(project):
        raise ValueError("Hãy chọn một tệp dự án FrameForge .ffproj.")
    connection = _connect(project)
    _migrate(connection)
    return connection


def sync_project(project_value: str, image_paths: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    connection = _open(project_value)
    try:
        return sync(connection, image_paths)
    finally:
        connection.close()


def write_project(project_value: str, image_paths: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    connection = _open(project_value)
    try:
        return write(connection, image_paths)
    finally:
        connection.close()
