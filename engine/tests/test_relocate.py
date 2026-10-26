import shutil
from pathlib import Path

import cv2
import numpy as np

from frameforge.labelme import import_images
from frameforge.project import create_project, load_project
from frameforge.relocate import relocate_images
from frameforge.sidecar import write_project


def _project_on_machine_a(tmp_path: Path) -> Path:
    machine_a = tmp_path / "machine-a" / "work"
    for folder, name, value in (("day1", "frame.png", 40), ("day2", "frame.png", 90), ("day2", "other.png", 140)):
        (machine_a / "images" / folder).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(machine_a / "images" / folder / name), np.full((32, 32, 3), value, dtype=np.uint8))
    project = machine_a / "set.ffproj"
    create_project(str(project), "Set")
    import_images(str(project), [str(machine_a / "images")], mark_reviewed=False)
    loaded = load_project(str(project))
    day1 = next(item for item in loaded["images"] if "day1" in item["path"])
    from frameforge.project import save_annotations
    import hashlib
    save_annotations(str(project), day1["path"], [{
        "id": "box-1", "class_id": 0, "x_min": 2, "y_min": 2, "x_max": 10, "y_max": 10, "source": "manual",
        "review_status": "reviewed",
    }], "reviewed", hashlib.sha256(Path(day1["path"]).read_bytes()).hexdigest())
    write_project(str(project))
    return machine_a


def test_project_copied_to_another_folder_finds_its_images(tmp_path: Path):
    machine_a = _project_on_machine_a(tmp_path)
    machine_b = tmp_path / "machine-b" / "copied"
    shutil.copytree(machine_a, machine_b)
    shutil.rmtree(machine_a)  # the old paths no longer exist, as on the other computer

    result = relocate_images(str(machine_b / "set.ffproj"))

    assert result["missing_count"] == 3
    assert result["relocated_count"] == 3 and result["still_missing_count"] == 0
    images = {Path(item["path"]).relative_to(machine_b).as_posix(): item for item in load_project(str(machine_b / "set.ffproj"))["images"]}
    assert set(images) == {"images/day1/frame.png", "images/day2/frame.png", "images/day2/other.png"}
    # The same file name in two folders went to the matching folder, and its label came along.
    assert len(images["images/day1/frame.png"]["annotations"]) == 1
    assert images["images/day1/frame.png"]["status"] == "reviewed"
    assert images["images/day2/frame.png"]["annotations"] == []

    again = relocate_images(str(machine_b / "set.ffproj"))
    assert again["missing_count"] == 0 and again["relocated_count"] == 0


def test_images_moved_apart_from_the_project_use_the_chosen_folder(tmp_path: Path):
    machine_a = _project_on_machine_a(tmp_path)
    project_b = tmp_path / "machine-b" / "projects"
    images_b = tmp_path / "machine-b" / "data" / "copied-images"
    project_b.mkdir(parents=True)
    shutil.copy2(machine_a / "set.ffproj", project_b / "set.ffproj")
    shutil.copytree(machine_a / "images", images_b)
    shutil.rmtree(machine_a)

    alone = relocate_images(str(project_b / "set.ffproj"))
    assert alone["still_missing_count"] == 3

    chosen = relocate_images(str(project_b / "set.ffproj"), str(images_b.parent))
    assert chosen["relocated_count"] == 3 and chosen["still_missing_count"] == 0
    labeled = [item for item in load_project(str(project_b / "set.ffproj"))["images"] if item["annotations"]]
    assert len(labeled) == 1 and "day1" in labeled[0]["path"]
