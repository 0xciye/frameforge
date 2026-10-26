"""Mark many images reviewed at once (button "Duyệt toàn bộ ảnh")."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict

from frameforge.project import _connect, _is_frameforge_project, _migrate


MAX_REVIEW = 1_000_000


def review_images(project_value: str, image_paths: Any) -> Dict[str, Any]:
    """Accept every pending AI box of the given images and mark the images that have boxes reviewed.

    Images without any box stay as they are, so an image nobody looked at never becomes a reviewed
    negative. Images with a pending negative candidate are skipped: confirming one needs its preview
    fingerprint. AI boxes without checkpoint provenance stay pending because the export rejects them.
    """
    if not isinstance(image_paths, list) or not image_paths or len(image_paths) > MAX_REVIEW:
        raise ValueError("Hãy chọn từ 1 đến {0} ảnh để duyệt.".format(MAX_REVIEW))
    if any(not isinstance(path, str) or not path for path in image_paths):
        raise ValueError("Đường dẫn ảnh không hợp lệ.")
    project = Path(project_value).expanduser().resolve(strict=True)
    if project.suffix.lower() != ".ffproj" or not project.is_file() or not _is_frameforge_project(project):
        raise ValueError("Hãy chọn một tệp dự án FrameForge .ffproj.")
    connection = _connect(project)
    try:
        _migrate(connection)
        with connection:
            connection.execute("CREATE TEMP TABLE review_targets(path TEXT PRIMARY KEY)")
            connection.executemany(
                "INSERT OR IGNORE INTO review_targets(path) SELECT ? WHERE EXISTS (SELECT 1 FROM images WHERE path = ?)",
                ((path, path) for path in image_paths),
            )
            connection.execute(
                """DELETE FROM review_targets WHERE path IN (
                    SELECT image_path FROM ai_negative_candidates WHERE review_status = 'needs_review')"""
            )
            target_count = connection.execute("SELECT COUNT(*) FROM review_targets").fetchone()[0]
            accepted = connection.execute(
                """UPDATE annotations SET review_status = 'reviewed'
                   WHERE source = 'ai' AND review_status = 'needs_review'
                     AND source_model_id IS NOT NULL AND source_model_id != ''
                     AND length(checkpoint_sha256) = 64
                     AND image_path IN (SELECT path FROM review_targets)"""
            ).rowcount
            boxed = """SELECT DISTINCT image_path FROM annotations
                       WHERE review_status != 'rejected' AND image_path IN (SELECT path FROM review_targets)"""
            boxed_count = connection.execute("SELECT COUNT(*) FROM ({0})".format(boxed)).fetchone()[0]
            # An image still holding a pending AI box (missing provenance) is not finished.
            marked = connection.execute(
                """UPDATE images SET status = 'reviewed'
                   WHERE status != 'reviewed' AND path IN ({0})
                     AND NOT EXISTS (SELECT 1 FROM annotations
                                     WHERE annotations.image_path = images.path AND review_status = 'needs_review')""".format(boxed)
            ).rowcount
            connection.execute("DROP TABLE review_targets")
    except sqlite3.DatabaseError as error:
        raise ValueError("Không cập nhật được dự án: {0}".format(error)) from error
    finally:
        connection.close()
    return {
        "requested_count": len(image_paths),
        "accepted_suggestion_count": accepted,
        "reviewed_image_count": marked,
        "skipped_empty_count": target_count - boxed_count,
        "skipped_pending_negative_count": len(set(image_paths)) - target_count,
    }
