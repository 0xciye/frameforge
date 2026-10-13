# FrameForge agent notes

FrameForge is a local-first Tauri 2 desktop application for exact-center-cropped object-detection datasets, annotation, YOLO11n training/fine-tuning, benchmarking and deployment export. Only the extraction workspace and ONNX metadata inspection currently work; future capabilities must stay visibly unavailable until implemented and verified.

## Read first

Read `AGENTS.md`, `PRODUCT.md`, `DESIGN.md`, and `DEVELOPMENT.md`; inspect Git status before editing. Repository docs and verified code are the source of truth for phase/status.

## Architecture and safety

- UI: React, TypeScript, Tailwind CSS, native Tauri dialogs.
- Desktop boundary: allowlisted Rust commands validate inputs and start Python directly; do not invoke a shell.
- Engine: Python JSON-lines on stdin/stdout, diagnostics only on stderr; OpenCV handles media, ONNX Runtime currently inspects models on CPU, SQLite stores `.ffproj` files.
- The current master product objective includes training/fine-tuning YOLO11n on labeled data; this supersedes older repo notes that prohibited training. Use PyTorch MPS on Apple Silicon when verified, report the actual device, and never fabricate training, detection or benchmark results.
- Never fabricate media, ONNX metadata, detections or annotation results. Keep inference, labeling, dataset, training, fine-tuning, benchmarking and export actions unavailable until backed by working code.
- Label editing and YOLO export stay disabled until backed by working inference/annotation/export code.

## Development practice

Keep changes small and aligned with the roadmap. Run and report relevant checks. Update README, agent notes and development status when behavior or setup changes. Never claim Windows, DirectML or RX 6600 support until validated on Windows hardware; current Windows work remains pending.
