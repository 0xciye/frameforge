# FrameForge agent notes

FrameForge is a local-first Tauri 2 desktop application for exact-center-cropped object-detection datasets, annotation, YOLO11n training/fine-tuning, benchmarking and deployment export. The extraction workspace, manual bounding-box labeling and ONNX metadata inspection are implemented. Inference, dataset export, training, benchmarking and model deployment remain unavailable until implemented and verified.

## Read first

Read `AGENTS.md`, `PRODUCT.md`, `DESIGN.md`, and `DEVELOPMENT.md`; inspect Git status before editing. Repository docs and verified code are the source of truth for phase/status.

## Architecture and safety

- UI: React, TypeScript, Tailwind CSS, native Tauri dialogs.
- Desktop boundary: allowlisted Rust commands validate inputs and start Python directly; do not invoke a shell.
- Engine: Python JSON-lines on stdin/stdout, diagnostics only on stderr; OpenCV handles media, ONNX Runtime currently inspects models on CPU, SQLite stores `.ffproj` files.
- The current master product objective includes training/fine-tuning YOLO11n on labeled data; this supersedes older repo notes that prohibited training. Use PyTorch MPS on Apple Silicon when verified, report the actual device, and never fabricate training, detection or benchmark results.
- Never fabricate media, ONNX metadata, detections or annotation results. Manual labels must come from user edits and remain independent of inference.
- Manual labeling requires a saved `.ffproj` so annotations can autosave to SQLite. Class IDs are stable: Head = 0, Body = 1. Store and validate boxes in the original image's pixel coordinates.
- Keep inference, YOLO dataset export, training, fine-tuning, benchmarking and deployment export unavailable until backed by working code.
- Open the app directly into the extraction workspace. Run engine health in the background and keep its status in the toolbar; do not show a blocking welcome/check-engine screen.
- The user's source videos are primarily Full HD (1920×1080). Read actual media dimensions and preserve exact crop/annotation geometry for other resolutions too.
- Exact Center Crop preview keeps the actual decoded source frame visible and overlays the crop rectangle using its exact source-pixel geometry; do not switch the preview to a cropped output image.

## Development practice

Keep changes small and aligned with the roadmap. After each completed work turn, update `README.md`, `AGENTS.md`, `agent.md` and `DEVELOPMENT.md`; also update product/design docs and `CHANGELOG.md` when the change affects them. Report only checks that actually ran. Never claim Windows, DirectML or RX 6600 support until validated on Windows hardware; current Windows work remains pending.
