# FrameForge agent notes

FrameForge is a local-first Tauri 2 desktop application for exact-center-cropped object-detection datasets, annotation, YOLO11n training/fine-tuning, benchmarking and deployment export. Extraction, manual labeling, dataset validation/grouped splits/YOLO export, ONNX metadata inspection, YOLO11n training, and verified-checkpoint inference with per-image AI suggestions/review are implemented. Evaluation, benchmarking, bulk auto-labeling, and deployment export remain unavailable.

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
- Inference must use a verified internal checkpoint, match the project class schema, and report pixel-space boxes and actual confidence. Ultralytics NMS runs once; the final confidence filter is inclusive. Save AI annotations as `needs_review` with model ID and checkpoint SHA-256; preserve them during manual autosave. Acceptance or edits mark them reviewed; rejection keeps a reversible rejected record. The same checkpoint may be run again on an image after all earlier suggestions from that checkpoint were rejected.
- Dataset export includes manual annotations and reviewed AI suggestions, excludes pending/rejected AI boxes, and never treats an image with pending AI suggestions as a reviewed negative. Group video/time segments and byte-identical images into the same split, preserve image pixels, and write normalized YOLO labels plus integrity/provenance into a new version directory.
- Project schema v3 adds AI model/checkpoint provenance to annotations and migrates v1/v2 projects when writing.
- Smart Folder Discovery scans only inside the user-selected root, skips symlinks, reports bounded progress, and reads YAML/CSV metadata without loading `.pt` files. Reject external YAML references; imported run metadata and checkpoints stay untrusted and must not become fine-tuning choices.
- Open the app directly into the extraction workspace. Run engine health in the background and keep its status in the toolbar; do not show a blocking welcome/check-engine screen.
- Accept video only up to 1920×1080; reject larger video during import, preview, and extraction without scaling. Preserve the source resolution of smaller video. Limit crop/output dimensions to 1920×1080.
- Exact Center Crop previews the accepted source video frame (up to FHD) through `preview_frame`, then draws a solid orange box in that frame's pixel coordinates using `floor((frame − crop) / 2)`. Equal W/H dimensions make a square; changing a preset resizes the box over the same frame. Never replace the displayed frame with a crop output image.

## Development practice

Keep changes small and aligned with the roadmap. After each completed work turn, update `README.md`, `AGENTS.md`, `agent.md` and `DEVELOPMENT.md`; also update product/design docs and `CHANGELOG.md` when the change affects them. Report only checks that actually ran. Never claim Windows, DirectML or RX 6600 support until validated on Windows hardware; current Windows work remains pending.
