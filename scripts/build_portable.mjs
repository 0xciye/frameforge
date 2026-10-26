import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pnpm = process.platform === "win32" ? "pnpm.cmd" : "pnpm";

function run(command, args, shell = process.platform === "win32") {
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: "inherit",
    shell,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

if (process.platform === "darwin") {
  run(pnpm, ["tauri", "build", "--bundles", "app"]);
} else if (process.platform === "win32") {
  run(pnpm, ["tauri", "build", "--no-bundle"]);
  run(process.execPath, ["scripts/python-runner.mjs", "scripts/stage_windows_portable.py"], false);
} else {
  console.error("FrameForge portable app builds are supported only on macOS and Windows.");
  process.exit(1);
}
