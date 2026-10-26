"""Keep one usable Windows portable build and remove generated build leftovers."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cleanup(root: Path, keep_installed: bool = False) -> list[Path]:
    root = root.resolve(strict=True)
    installed = root.parent / "FrameForge-Portable"
    target = root / "src-tauri" / "target"
    keep = installed if keep_installed else target / "release" / "FrameForge"
    for relative in ("frameforge.exe", "engine-runtime/frameforge-engine/frameforge-engine.exe"):
        if not (keep / relative).is_file():
            raise ValueError(f"Missing portable build file; nothing was deleted: {keep / relative}")

    removed: list[Path] = []

    def remove(path: Path, boundary: Path) -> None:
        if not path.exists():
            return
        resolved = path.resolve(strict=True)
        allowed = boundary.resolve(strict=True)
        # Refuse redirected build paths instead of deleting through a junction/symlink.
        if resolved == allowed or not resolved.is_relative_to(allowed):
            raise ValueError(f"Cleanup path is outside its allowed directory: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)

    def prune(path: Path) -> None:
        if path == keep:
            return
        if keep.is_relative_to(path):
            for child in path.iterdir():
                prune(child)
        else:
            remove(path, root)

    # Build output only: source, datasets, weights and virtual environments are untouched.
    if target.exists():
        prune(target)
    for relative in ("engine/build", "engine/dist", "dist"):
        remove(root / relative, root)
    for relative in (".pytest_cache", "engine/.pytest_cache", "engine/__pycache__",
                     "engine/frameforge/__pycache__", "engine/tests/__pycache__",
                     "scripts/__pycache__", "node_modules/.vite", "node_modules/.cache"):
        remove(root / relative, root)
    if installed.is_dir():
        for pattern in ("frameforge-before-*.exe", "engine-runtime-before-*"):
            for path in installed.glob(pattern):
                remove(path, installed)
        runtime = installed / "engine-runtime"
        for pattern in ("frameforge-engine-before-*", "frameforge-engine-cpu-backup-*"):
            for path in runtime.glob(pattern):
                remove(path, installed)
        engine = runtime / "frameforge-engine"
        for path in engine.glob("frameforge-engine-before-*.exe"):
            remove(path, installed)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-installed", action="store_true", help="Keep Desktop's installed portable app instead of the staged build")
    args = parser.parse_args()
    for path in cleanup(ROOT, args.keep_installed):
        print(f"Removed: {path}")


if __name__ == "__main__":
    main()
