import stat
import sys
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

from frameforge.project import create_project, load_project, save_project
from frameforge.session import load_session, save_session


def test_session_restores_project_state_and_detects_external_project_changes(tmp_path: Path, monkeypatch):
    app_data = tmp_path / "app-data"
    project_path = tmp_path / "project.ffproj"
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(app_data))
    create_project(str(project_path), "Project")
    project = load_project(str(project_path))
    project["settings"]["output_width"] = 320
    save_project(str(project_path), project)

    save_session(project, {
        "project_path": str(project_path),
        "selected_path": "/media/clip.mp4",
        "selected_paths": ["/media/clip.mp4"],
        "frame_index": 24,
        "active_tab": "label",
        "dirty": False,
    })

    restored = load_session()
    assert restored["available"] is True
    assert restored["original_project_exists"] is True
    assert restored["original_project_unchanged"] is True
    assert restored["project"]["settings"]["output_width"] == 320
    assert restored["session"]["frame_index"] == 24
    assert stat.S_IMODE((app_data / "session" / "last-session.ffproj").stat().st_mode) == 0o600
    assert stat.S_IMODE((app_data / "session" / "last-session.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((app_data / "session").stat().st_mode) == 0o700

    project["settings"]["output_width"] = 640
    save_project(str(project_path), project)
    changed = load_session()
    assert changed["original_project_exists"] is True
    assert changed["original_project_unchanged"] is False
    assert changed["project"]["settings"]["output_width"] == 320


def test_load_session_without_snapshot_returns_unavailable(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))

    assert load_session() == {"available": False}


def test_session_migrates_legacy_dhash_default_and_keeps_a_new_choice(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FRAMEFORGE_APP_DATA", str(tmp_path / "app-data"))
    project_path = tmp_path / "project.ffproj"
    create_project(str(project_path), "Project")
    project = load_project(str(project_path))
    project["settings"] = {"duplicate_threshold": 5}
    project["settings_version"] = 0
    save_session(project, {"project_path": str(project_path), "dirty": False})

    restored = load_session()
    assert restored["project"]["settings"]["duplicate_threshold"] == 1
    assert restored["project"]["settings_migrated"] is True

    migrated_project = restored["project"]
    migrated_project["settings"]["duplicate_threshold"] = 5
    migrated_project["settings_version"] = 1
    save_session(migrated_project, restored["session"])
    reopened = load_session()

    assert reopened["project"]["settings"]["duplicate_threshold"] == 5
    assert reopened["project"]["settings_migrated"] is False
