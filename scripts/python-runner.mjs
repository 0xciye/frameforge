import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, resolve, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const localPython = process.platform === "win32"
  ? join(root, ".venv", "Scripts", "python.exe")
  : join(root, ".venv", "bin", "python");
const override = process.env.FRAMEFORGE_PYTHON;
const gpuPython = join(root, ".venv-gpu", "Scripts", "python.exe");
const command = override || (process.platform === "win32" && existsSync(gpuPython) ? gpuPython
  : existsSync(localPython) ? localPython : (process.platform === "win32" ? "python" : "python3"));
const result = spawnSync(command, process.argv.slice(2), { cwd: root, stdio: "inherit" });

if (result.error) {
  console.error(`Không chạy được Python (${command}): ${result.error.message}`);
  process.exit(1);
}
process.exit(result.status ?? 1);
