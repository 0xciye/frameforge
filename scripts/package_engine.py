"""Build and smoke-test the self-contained FrameForge Python engine."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import argparse
from pathlib import Path

from PyInstaller.__main__ import run


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
DIST = ENGINE / "dist"
BUILD = ENGINE / "build" / "pyinstaller"
APP_NAME = "frameforge-engine"


def smoke_request(binary: Path, command: str, app_data: Path) -> dict:
    request = {"id": command, "command": command, "arguments": {}}
    result = subprocess.run(
        [str(binary)],
        input=json.dumps(request, ensure_ascii=False) + "\n",
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
        env={**os.environ, "FRAMEFORGE_APP_DATA": str(app_data)},
    )
    if result.returncode != 0:
        raise RuntimeError(f"Engine smoke test {command} failed: {result.stderr[-4000:]}")
    responses = []
    for line in result.stdout.splitlines():
        try:
            responses.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    response = next((item for item in responses if item.get("id") == command), None)
    if not response or response.get("ok") is not True:
        raise RuntimeError(f"Engine smoke test {command} returned no valid response: {result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    return response["data"]


def build_engine() -> Path:
    output = DIST / APP_NAME
    if output.exists():
        shutil.rmtree(output)
    BUILD.mkdir(parents=True, exist_ok=True)

    run([
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name", APP_NAME,
        "--paths", str(ENGINE),
        "--distpath", str(DIST),
        "--workpath", str(BUILD),
        "--specpath", str(BUILD),
        "--collect-all", "cv2",
        "--collect-all", "ultralytics",
        "--collect-all", "torch",
        "--collect-all", "torchvision",
        "--collect-all", "matplotlib",
        "--collect-all", "polars",
        "--copy-metadata", "torch",
        "--copy-metadata", "ultralytics",
        str(ENGINE / "frameforge_engine.py"),
    ])

    binary = output / APP_NAME
    if sys.platform == "win32":
        binary = binary.with_suffix(".exe")
    if not binary.is_file():
        raise RuntimeError(f"PyInstaller did not produce the engine executable: {binary}")

    return binary


def smoke_engine(binary: Path) -> None:
    if not binary.is_file():
        raise RuntimeError(f"Packaged engine executable is missing: {binary}")
    with tempfile.TemporaryDirectory(prefix="frameforge-engine-smoke-") as temporary:
        app_data = Path(temporary) / "app-data"
        health = smoke_request(binary, "health", app_data)
        if health.get("status") != "ok":
            raise RuntimeError("Packaged engine health check did not return status=ok.")
        capabilities = smoke_request(binary, "inference_capabilities", app_data)
        if capabilities.get("available") is not True:
            raise RuntimeError(f"Packaged inference runtime is incomplete: {capabilities.get('missing_packages')}")
        print(f"Packaged engine passed health and inference checks: {binary}")
        print(f"Packaged engine reports PyTorch {capabilities.get('torch_version')} and Ultralytics {capabilities.get('ultralytics_version')}.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", type=Path)
    args = parser.parse_args()
    if args.smoke_only:
        smoke_engine(args.smoke_only)
        return 0
    smoke_engine(build_engine())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
