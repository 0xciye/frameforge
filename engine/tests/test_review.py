import json
from pathlib import Path

import cv2
import numpy as np

from frameforge.project import load_project, save_project
from frameforge.review import review_images
from frameforge_engine import handle_request
from frameforge.sidecar import json_path_for


def _ai_box(box_id: str, provenance: bool = True) -> dict:
    return {
        "id": box_id, "class_id": 0, "x_min": 4, "y_min": 4, "x_max": 20, "y_max": 20,
        "source": "ai", "review_status": "needs_review", "confidence": 0.8,
        "source_model_id": "run-1" if provenance else None,
        "checkpoint_sha256": "a" * 64 if provenance else None,
    }


def _project(tmp_path: Path) -> tuple[Path, dict]:
    paths = {name: tmp_path / "{0}.png".format(name) for name in ("ai", "manual", "empty", "candidate", "noprov")}
    for index, path in enumerate(paths.values()):
        cv2.imwrite(str(path), np.full((32, 32, 3), 30 + index * 40, dtype=np.uint8))

    def image(name: str, annotations: list, **extra) -> dict:
        return {"path": str(paths[name]), "width": 32, "height": 32, "status": "ready", "annotations": annotations, **extra}

    project_path = tmp_path / "review.ffproj"
    save_project(str(project_path), {
        "id": "review-project", "name": "Review", "sources": [], "settings": {},
        "class_names": ["head", "body"],
        "images": [
            image("ai", [_ai_box("ai-1")]),
            image("manual", [{"id": "m-1", "class_id": 1, "x_min": 2, "y_min": 2, "x_max": 9, "y_max": 9,
                              "source": "manual", "review_status": "reviewed"}]),
            image("empty", []),
            image("candidate", [], negative_candidates=[{
                "id": "cand-1", "source_model_id": "run-1", "checkpoint_sha256": "a" * 64,
                "image_fingerprint": "b" * 64, "review_status": "needs_review",
                "created_at": "2026-10-25T00:00:00Z", "reviewed_at": None,
            }]),
            image("noprov", [_ai_box("np-1", provenance=False)]),
        ],
    })
    return project_path, paths


def test_review_images_accepts_ai_boxes_and_leaves_empty_images(tmp_path: Path):
    project_path, paths = _project(tmp_path)

    result = review_images(str(project_path), [str(path) for path in paths.values()])

    assert result["accepted_suggestion_count"] == 1
    assert result["reviewed_image_count"] == 2
    assert result["skipped_empty_count"] == 1
    assert result["skipped_pending_negative_count"] == 1
    images = {item["path"]: item for item in load_project(str(project_path))["images"]}
    assert images[str(paths["ai"])]["status"] == "reviewed"
    assert images[str(paths["ai"])]["annotations"][0]["review_status"] == "reviewed"
    assert images[str(paths["manual"])]["status"] == "reviewed"
    assert images[str(paths["empty"])]["status"] == "ready"
    assert images[str(paths["candidate"])]["status"] == "ready"
    # Without checkpoint provenance the export would reject the box, so it stays pending.
    assert images[str(paths["noprov"])]["annotations"][0]["review_status"] == "needs_review"
    assert images[str(paths["noprov"])]["status"] == "ready"


def test_review_images_command_writes_accepted_boxes_into_json(tmp_path: Path):
    project_path, paths = _project(tmp_path)

    response = handle_request({
        "id": "r", "command": "review_images",
        "arguments": {"path": str(project_path), "image_paths": [str(paths["ai"])]},
    })

    assert response["ok"], response
    document = json.loads(json_path_for(str(paths["ai"])).read_text(encoding="utf-8"))
    assert [shape["label"] for shape in document["shapes"]] == ["head"]
    assert not document["frameforge"].get("suggestions")
