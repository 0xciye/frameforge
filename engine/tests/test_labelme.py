import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frameforge.labelme import image_size, import_images, import_labelme_folder
from frameforge.project import create_project, load_project, save_project
from frameforge_engine import handle_request


def _image(path: Path, width: int = 64, height: int = 48) -> None:
    frame = np.full((height, width, 3), 90, dtype=np.uint8)
    ok, encoded = cv2.imencode(path.suffix, frame)
    assert ok
    path.write_bytes(encoded.tobytes())


def _labelme(path: Path, image_name: str, shapes, width: int = 64, height: int = 48) -> None:
    path.write_text(json.dumps({
        "version": "4.2.0",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageWidth": width,
        "imageHeight": height,
    }), encoding="utf-8")


def _rect(label: str, x0: float, y0: float, x1: float, y1: float, **extra):
    # X-AnyLabeling writes rectangles as four corner points.
    return {"label": label, "shape_type": "rectangle", "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], **extra}


def _call(command: str, **arguments):
    response = handle_request({"id": command, "command": command, "arguments": arguments})
    assert response["ok"], response
    return response["data"]


def _fingerprint(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture()
def folder(tmp_path: Path) -> Path:
    folder = tmp_path / "640_yellow"
    folder.mkdir()
    _image(folder / "train_crop_20260728_201116_927_000020.png")
    _labelme(folder / "train_crop_20260728_201116_927_000020.json", "train_crop_20260728_201116_927_000020.png", [
        _rect("head", 10, 5, 17, 15, description="kept"),
        _rect("Body", 8, 12, 30, 47),
        _rect("head", -3, 40, 6, 52),
        {"label": "aim", "shape_type": "point", "points": [[3, 3]]},
    ])
    _image(folder / "train_crop_20260728_201300_000_000021.jpg")
    _image(folder / "unlabeled.png")
    sub = folder / "more"
    sub.mkdir()
    _image(sub / "deep.png")
    return folder


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    path = tmp_path / "p.ffproj"
    create_project(str(path), "p")
    return path


def _images(project: Path):
    return {Path(image["path"]).name: image for image in load_project(str(project))["images"]}


def test_import_reads_json_labels_without_touching_the_files(project: Path, folder: Path):
    before = {path: path.read_bytes() for path in folder.rglob("*")  if path.is_file()}
    dry = import_images(str(project), [str(folder)], dry_run=True)
    assert dry == {"found_images": 4, "new_images": 4, "json_images": 1, "skipped_existing": 0}
    assert _images(project) == {}

    result = import_images(str(project), [str(folder)], mark_reviewed=True)

    assert {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()} == before
    assert (result["imported_images"], result["json_images"], result["labeled_images"], result["box_count"]) == (4, 1, 1, 3)
    assert result["reviewed_images"] == 1 and result["clipped_boxes"] == 1 and result["unsupported_shapes"] == 1
    images = _images(project)
    assert "deep.png" in images
    labeled = images["train_crop_20260728_201116_927_000020.png"]
    assert labeled["status"] == "reviewed"
    boxes = sorted((a["class_id"], a["x_min"], a["y_min"], a["x_max"], a["y_max"]) for a in labeled["annotations"])
    assert boxes == [(0, 0.0, 40.0, 6.0, 48.0), (0, 10.0, 5.0, 17.0, 15.0), (1, 8.0, 12.0, 30.0, 47.0)]
    assert {a["review_status"] for a in labeled["annotations"]} == {"reviewed"}
    # Frames recorded within the same 10 minutes share one split group.
    assert labeled["source_id"] == images["train_crop_20260728_201300_000_000021.jpg"]["source_id"]
    assert not (folder / "unlabeled.json").exists()


def test_label_changes_are_written_to_the_json_beside_the_image(project: Path, folder: Path):
    import_images(str(project), [str(folder)])
    image = _images(project)["train_crop_20260728_201116_927_000020.png"]
    kept = next(a for a in image["annotations"] if a["x_min"] == 10.0)
    new = {"id": "new-box", "class_id": 1, "x_min": 40.0, "y_min": 2.0, "x_max": 60.0, "y_max": 30.0,
           "confidence": None, "source": "manual", "review_status": "reviewed"}
    _call("save_annotations", path=str(project), image_path=image["path"], annotations=[kept, new],
          status="reviewed", image_fingerprint=_fingerprint(image["path"]))

    document = json.loads((folder / "train_crop_20260728_201116_927_000020.json").read_text(encoding="utf-8"))
    labels = [shape for shape in document["shapes"] if shape["shape_type"] == "rectangle"]
    assert [(shape["label"], shape["points"][0], shape["points"][2]) for shape in labels] == [
        ("head", [10.0, 5.0], [17.0, 15.0]), ("body", [40.0, 2.0], [60.0, 30.0])
    ]
    assert labels[0]["description"] == "kept", "fields of an unchanged shape are kept"
    assert {"label": "aim", "shape_type": "point", "points": [[3, 3]]} in document["shapes"], "foreign shapes stay"
    assert document["checked"] is True and document["frameforge"]["status"] == "reviewed"
    assert document["imageWidth"] == 64 and document["imagePath"] == "train_crop_20260728_201116_927_000020.png"

    # Labeling an image that had no JSON creates one; an unlabeled image gets none.
    plain = _images(project)["unlabeled.png"]
    _call("save_annotations", path=str(project), image_path=plain["path"], annotations=[{**new, "id": "b"}],
          status="ready", image_fingerprint=_fingerprint(plain["path"]))
    created = json.loads((folder / "unlabeled.json").read_text(encoding="utf-8"))
    assert [shape["label"] for shape in created["shapes"]] == ["body"] and created["checked"] is False
    assert not (folder / "more" / "deep.json").exists()


def test_json_edited_outside_frameforge_wins_on_open(project: Path, folder: Path):
    import_images(str(project), [str(folder)])
    path = folder / "unlabeled.json"
    _labelme(path, "unlabeled.png", [_rect("head", 1, 1, 9, 9)])
    loaded = _call("load_project", path=str(project))
    image = next(i for i in loaded["images"] if i["path"].endswith("unlabeled.png"))
    assert [(a["class_id"], a["x_max"]) for a in image["annotations"]] == [(0, 9.0)]
    assert path.read_text(encoding="utf-8").count("frameforge") == 0, "reading does not rewrite the file"

    # A JSON deleted outside the app is rewritten from the project instead of losing labels.
    path.unlink()
    _call("load_project", path=str(project))
    assert json.loads(path.read_text(encoding="utf-8"))["shapes"][0]["label"] == "head"


def test_ai_suggestions_stay_out_of_the_labels_until_accepted(project: Path, folder: Path):
    import_images(str(project), [str(folder)])
    image = _images(project)["unlabeled.png"]
    connection = sqlite3.connect(str(project))
    with connection:
        connection.execute(
            """INSERT INTO annotations VALUES('ai-1', ?, 0, 1, 1, 9, 9, 0.8, 'ai', 'needs_review', 'model', ?)""",
            (image["path"], "a" * 64),
        )
    connection.close()
    _call("save_project", path=str(project), project=load_project(str(project)))
    document = json.loads((folder / "unlabeled.json").read_text(encoding="utf-8"))
    assert document["shapes"] == []
    assert document["frameforge"]["suggestions"][0]["frameforge"]["review_status"] == "needs_review"
    # The pending suggestion survives a round trip through the file.
    (folder / "unlabeled.json").write_text(json.dumps(document), encoding="utf-8")
    again = next(i for i in _call("load_project", path=str(project))["images"] if i["path"] == image["path"])
    assert [(a["source"], a["review_status"], a["confidence"]) for a in again["annotations"]] == [("ai", "needs_review", 0.8)]


def test_projects_made_before_json_storage_write_their_labels_out(project: Path, folder: Path):
    # Earlier FrameForge versions kept labels only in the project: the first open writes them to the JSON.
    import_images(str(project), [str(folder)], mark_reviewed=True)
    connection = sqlite3.connect(str(project))
    with connection:
        connection.execute("DELETE FROM sidecar_state")
    connection.close()
    original = (folder / "train_crop_20260728_201116_927_000020.json").read_text(encoding="utf-8")
    loaded = _call("load_project", path=str(project))
    image = next(i for i in loaded["images"] if i["path"].endswith("000020.png"))
    assert image["status"] == "reviewed" and len(image["annotations"]) == 3
    written = json.loads((folder / "train_crop_20260728_201116_927_000020.json").read_text(encoding="utf-8"))
    assert written != json.loads(original) and written["frameforge"]["status"] == "reviewed"


def test_unknown_labels_are_kept_in_the_file(project: Path, folder: Path):
    _labelme(folder / "unlabeled.json", "unlabeled.png", [_rect("enemy", 1, 1, 9, 9)])
    result = import_images(str(project), [str(folder)])
    assert result["unknown_labels"] == 1
    image = _images(project)["unlabeled.png"]
    _call("save_annotations", path=str(project), image_path=image["path"], annotations=[], status="reviewed",
          image_fingerprint=_fingerprint(image["path"]))
    labels = [shape["label"] for shape in json.loads((folder / "unlabeled.json").read_text(encoding="utf-8"))["shapes"]]
    assert labels == ["enemy"]


def test_size_mismatch_keeps_the_image_unlabeled(project: Path, folder: Path):
    _labelme(folder / "unlabeled.json", "unlabeled.png", [_rect("head", 1, 1, 9, 9)], width=640, height=640)
    result = import_images(str(project), [str(folder)])
    assert result["error_count"] == 1 and "640×640" in result["errors"][0]["message"]
    assert _images(project)["unlabeled.png"]["annotations"] == []


def test_reimport_adds_only_new_images(project: Path, folder: Path):
    import_labelme_folder(str(project), str(folder))
    _image(folder / "new.png")
    result = import_images(str(project), [str(folder)])
    assert result["imported_images"] == 1 and result["skipped_existing"] == 4


def test_image_size_reads_png_and_jpeg_headers(tmp_path: Path):
    for name in ("a.png", "b.jpg", "c.webp"):
        _image(tmp_path / name, 37, 21)
        assert image_size(tmp_path / name) == (37, 21)
    (tmp_path / "broken.png").write_bytes(b"not an image")
    assert image_size(tmp_path / "broken.png") is None


def test_rejects_non_project_files(tmp_path: Path, folder: Path):
    fake = tmp_path / "fake.ffproj"
    sqlite3.connect(str(fake)).close()
    with pytest.raises(ValueError):
        import_images(str(fake), [str(folder)])


def test_labeling_preview_keeps_every_pixel(tmp_path: Path):
    import base64
    from frameforge.media import preview_frame

    frame = np.random.default_rng(7).integers(0, 256, (1800, 2000, 3), dtype=np.uint8)
    path = tmp_path / "big.png"
    cv2.imwrite(str(path), frame)
    exact = preview_frame(str(path), lossless=True)
    assert exact["data_url"].startswith("data:image/png;base64,")
    assert (exact["width"], exact["height"]) == (2000, 1800), "no downscale for labeling"
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(exact["data_url"].split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(decoded, frame)
    # The extraction preview stays a JPEG capped at 1600 px but reports source pixels for its crop overlay.
    small = preview_frame(str(path))
    assert small["data_url"].startswith("data:image/jpeg") and (small["width"], small["height"]) == (2000, 1800)
    shown = cv2.imdecode(np.frombuffer(base64.b64decode(small["data_url"].split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert max(shown.shape[:2]) == 1600


def test_delete_images_moves_the_image_and_its_json_to_the_trash(project: Path, folder: Path, tmp_path: Path):
    from frameforge.deletion import delete_images

    import_images(str(project), [str(folder)])
    images = _images(project)
    target = images["train_crop_20260728_201116_927_000020.png"]["path"]
    keep = images["unlabeled.png"]["path"]
    bin_dir = tmp_path / "trash"
    bin_dir.mkdir()
    trashed = []

    def fake_trash(paths):
        for path in paths:
            if path.exists():
                trashed.append(path.name)
                path.rename(bin_dir / path.name)

    result = delete_images(str(project), [target], trash=fake_trash)

    assert result["deleted_count"] == 1
    assert sorted(trashed) == ["train_crop_20260728_201116_927_000020.json", "train_crop_20260728_201116_927_000020.png"]
    assert not Path(target).exists() and Path(keep).exists()
    remaining = _images(project)
    assert target not in {image["path"] for image in remaining.values()} and len(remaining) == 3
    connection = sqlite3.connect(str(project))
    assert connection.execute("SELECT COUNT(*) FROM annotations WHERE image_path = ?", (target,)).fetchone()[0] == 0
    connection.close()

    # A path outside the project, or a trash that fails, changes nothing.
    with pytest.raises(ValueError):
        delete_images(str(project), [str(folder / "nope.png")], trash=fake_trash)

    def broken_trash(paths):
        raise OSError("bin unavailable")

    with pytest.raises(OSError):
        delete_images(str(project), [keep], trash=broken_trash)
    assert Path(keep).exists() and keep in {image["path"] for image in _images(project).values()}
