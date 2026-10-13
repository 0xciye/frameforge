# FrameForge agent notes

FrameForge is a local-first Tauri 2 desktop tool for preparing video/image datasets. The current Vietnamese workspace supports real media inspection/preview, frame extraction, SQLite project persistence and CPU ONNX metadata inspection.

## Read first

Read `AGENTS.md`, `PRODUCT.md`, `DESIGN.md`, and `DEVELOPMENT.md`; inspect Git status before editing. Repository docs and verified code are the source of truth for phase/status.

## Architecture and safety

- UI: React, TypeScript, Tailwind CSS, native Tauri dialogs.
- Desktop boundary: allowlisted Rust commands validate inputs and start Python directly; do not invoke a shell.
- Engine: Python JSON-lines on stdin/stdout, diagnostics only on stderr; OpenCV handles media, ONNX Runtime currently inspects models on CPU, SQLite stores `.ffproj` files.
- Never fabricate media, ONNX metadata, detections or annotation results. Do not implement model training/fine-tuning.
- Label editing and YOLO export stay disabled until backed by working inference/annotation/export code.

## Development practice

Keep changes small and aligned with the roadmap. Run and report relevant checks. Update README, agent notes and development status when behavior or setup changes. Never claim Windows, DirectML or RX 6600 support until validated on Windows hardware; current Windows work remains pending.
