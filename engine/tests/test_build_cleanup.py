import runpy
from pathlib import Path

import pytest


cleanup = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/cleanup_builds.py"))["cleanup"]


def portable(path):
    for relative in ("frameforge.exe", "engine-runtime/frameforge-engine/frameforge-engine.exe"):
        file = path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"latest")


def test_cleanup_keeps_latest_and_user_data(tmp_path):
    root = tmp_path / "frameforge"
    keep = root / "src-tauri/target/release/FrameForge"
    portable(keep)
    old = tmp_path / "FrameForge-Portable/engine-runtime-before-old/old.dll"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    for relative in ("engine/build/temp", "engine/dist/old", "dist/asset", "src-tauri/target/debug/cache", "src-tauri/target/release/old.exe", "engine/.pytest_cache/cache", "engine/frameforge/__pycache__/module.pyc", "node_modules/.vite/cache", "weights/model.pt", ".venv-train/keep", "dataset/image.png"):
        file = root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"keep")
    cleanup(root)
    assert (keep / "frameforge.exe").read_bytes() == b"latest"
    assert list((root / "src-tauri/target/release").iterdir()) == [keep]
    assert not old.exists()
    for relative in ("engine/build", "engine/dist", "dist", "src-tauri/target/debug", "engine/.pytest_cache", "engine/frameforge/__pycache__", "node_modules/.vite"):
        assert not (root / relative).exists()
    for relative in ("weights/model.pt", ".venv-train/keep", "dataset/image.png"):
        assert (root / relative).read_bytes() == b"keep"


def test_cleanup_requires_complete_build_and_can_keep_installed(tmp_path):
    root = tmp_path / "frameforge"
    target = root / "src-tauri/target"
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match="nothing was deleted"):
        cleanup(root, keep_installed=True)
    assert target.exists()
    keep = tmp_path / "FrameForge-Portable"
    portable(keep)
    cleanup(root, keep_installed=True)
    assert not target.exists()
    assert (keep / "frameforge.exe").is_file()
