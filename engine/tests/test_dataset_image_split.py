import json
from pathlib import Path

from frameforge.dataset import _config, _split_groups, export_yolo_dataset
from test_engine import create_dataset_project


def test_image_split_counts_seed_and_identical_images():
    config = _config("image", .8, .1, .1, 42, 30, .1)
    samples = [{"path": str(i), "group_key": "image:" + str(i), "sha256": str(i)} for i in range(967)]
    assignments, warnings = _split_groups(samples, config)
    assert not warnings
    assert {name: list(assignments.values()).count(name) for name in ("train", "val", "test")} == {"train": 773, "val": 97, "test": 97}
    assert _split_groups(list(reversed(samples)), config)[0] == assignments
    assert _split_groups(samples, {**config, "seed": 43})[0] != assignments
    samples[1]["sha256"] = samples[0]["sha256"]
    assignments, _ = _split_groups(samples, config)
    assert assignments["0"] == assignments["1"]


def test_default_export_splits_images_from_one_video(tmp_path: Path, monkeypatch):
    # Fixture frames are flat colours (identical dHash); give them distinct scenes so only the split rule is tested.
    import itertools
    import frameforge.dataset as dataset
    scene = itertools.count()
    monkeypatch.setattr(dataset, "_dhash", lambda frame: (next(scene) * 0x9E3779B97F4A7C15) % (1 << 64))
    project_path, _ = create_dataset_project(tmp_path)
    # Existing fixture includes two positives, one reviewed negative and one unreviewed negative.
    from frameforge.project import load_project, save_project
    project = load_project(str(project_path))
    for image in project["images"]:
        image["source_id"] = "same-video"
    save_project(str(project_path), project)
    output = tmp_path / "export"
    output.mkdir()
    result = export_yolo_dataset(str(project_path), str(output), negative_ratio=1)
    assert result["split_counts"] == {"train": 1, "val": 1, "test": 1}
    manifest = json.loads((Path(result["path"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"]["mode"] == "image"
    assert manifest["summary"]["candidate_negative_excluded_count"] == 1
    assert all((Path(result["path"]) / item["exported_label"]).is_file() for item in manifest["images"])


def test_image_split_keeps_near_duplicate_frames_together():
    config = _config("image", .8, .1, .1, 42, 30, .1)
    samples = [{"path": str(i), "group_key": "image:" + str(i), "sha256": str(i), "dhash": (i * 0x9E3779B97F4A7C15) % (1 << 64)}
               for i in range(200)]
    # Frames 1..6 are adjacent video frames: each differs from the previous by one bit (chain of 4 bit max overall).
    for offset in range(1, 7):
        samples[offset]["dhash"] = samples[0]["dhash"] ^ ((1 << offset) - 1)
    assignments, _ = _split_groups(samples, config)
    assert len({assignments[str(i)] for i in range(7)}) == 1
    counts = {name: list(assignments.values()).count(name) for name in ("train", "val", "test")}
    assert sum(counts.values()) == 200 and counts["val"] >= 14 and counts["test"] >= 14
