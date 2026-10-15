"""Stage the Windows app as a launchable folder, without an installer."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "src-tauri" / "target" / "release"
APP = RELEASE / "FrameForge"
ENGINE_DIST = ROOT / "engine" / "dist"


def main() -> None:
    executable = RELEASE / "frameforge.exe"
    if not executable.is_file():
        raise SystemExit(f"Tauri executable was not found: {executable}")
    if APP.exists():
        shutil.rmtree(APP)
    APP.mkdir(parents=True)
    shutil.copy2(executable, APP / executable.name)
    shutil.copytree(ENGINE_DIST, APP / "engine-runtime")
    print(f"Staged portable app folder: {APP}")


if __name__ == "__main__":
    main()
