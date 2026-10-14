# FrameForge agent notes

FrameForge is a local-first Tauri 2 desktop application for exact-center-cropped object-detection datasets, annotation, YOLO11n training/fine-tuning, benchmarking and deployment export. Extraction, manual labeling, dataset validation/grouped splits/YOLO export, ONNX metadata inspection, YOLO11n training, and a verified-checkpoint inference engine are implemented. The inference command is not yet connected to an auto-label/review UI; evaluation, benchmarking, and deployment export remain unavailable.

## Read first

Read `AGENTS.md`, `PRODUCT.md`, `DESIGN.md`, and `DEVELOPMENT.md`; inspect Git status before editing. Repository docs and verified code are the source of truth for phase/status.

## Architecture and safety

- UI: React, TypeScript, Tailwind CSS, native Tauri dialogs.
- Desktop boundary: allowlisted Rust commands validate inputs and start Python directly; do not invoke a shell.
- Add every module imported by `engine/frameforge_engine.py` to the Tauri resources in `src-tauri/tauri.conf.json`; verify imports and health from the built app bundle.
- Engine: Python JSON-lines on stdin/stdout, diagnostics only on stderr; OpenCV handles media, ONNX Runtime currently inspects models on CPU, SQLite stores `.ffproj` files.
- YOLO11n training uses the optional engine[train] extra (PyTorch, Ultralytics, psutil). Inspect local datasets before training, run each image size independently, store checkpoints/logs/config under the chosen output, cache pretrained weights in app data, and report the actual device and metrics. Fine-tuning may load only a registered internal YOLO11n detection checkpoint whose current SHA-256 matches its registry entry.
- Never fabricate media, ONNX metadata, detections or annotation results. Manual labels must come from user edits and remain independent of inference.
- Manual labeling requires a saved `.ffproj` so annotations can autosave to SQLite. Class IDs are stable: Head = 0, Body = 1. Store and validate boxes in the original image's pixel coordinates.
- Inference must use a verified internal checkpoint, match its registered class schema, and report pixel-space boxes and actual confidence. Ultralytics NMS runs once; the final confidence filter is inclusive. Do not expose auto-label UI until suggestions have provenance and review state. Benchmarking and deployment export remain unavailable until backed by working code.
- Dataset export includes only positive images or reviewed negatives, groups video/time segments and byte-identical images into the same split, preserves image pixels, and writes normalized YOLO labels plus an integrity/provenance manifest into a new version directory.
- Smart Folder Discovery scans only inside the user-selected root, skips symlinks, reports bounded progress, and reads YAML/CSV metadata without loading `.pt` files. Reject external YAML references; imported run metadata and checkpoints stay untrusted and must not become fine-tuning choices.
- Open the app directly into the extraction workspace. Run engine health in the background and keep its status in the toolbar; do not show a blocking welcome/check-engine screen.
- Accept video only up to 1920×1080; reject larger video during import, preview, and extraction without scaling. Preserve the source resolution of smaller video. Limit crop/output dimensions to 1920×1080.
- Exact Center Crop previews the accepted source video frame (up to FHD) through `preview_frame`, then draws a solid orange box in that frame's pixel coordinates using `floor((frame − crop) / 2)`. Equal W/H dimensions make a square; changing a preset resizes the box over the same frame. Never replace the displayed frame with a crop output image.

## Development practice

Keep changes small and aligned with the roadmap. After each completed work turn, update `README.md`, `AGENTS.md`, `agent.md` and `DEVELOPMENT.md`; also update product/design docs and `CHANGELOG.md` when the change affects them. Report only checks that actually ran. Never claim Windows, DirectML or RX 6600 support until validated on Windows hardware; current Windows work remains pending.
