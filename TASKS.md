# FrameForge v1.0 task ledger

This ledger resumes the existing branch `feature/manual-labeling` at `f37db50`. The inherited working-tree changes are part of the active work and must be preserved. Current source-video scope is **at most 1920×1080**: reject larger sources at import, preview and extraction; do not downscale them or add higher-resolution input options. Smaller sources retain their native size.

## COMPLETED_VERIFIED

- Media import/inspection, real frame preview and background seeking have engine coverage; crop and extraction preserve source pixels.
- Exact Center Crop uses `floor((frame − crop) / 2)` and displays the source frame with the crop overlay. FHD fixture check confirms 320×320 at `(800, 380)` on 1920×1080.
- Video input bounds are enforced in the engine at import/preview/extraction. Regression coverage rejects sources exceeding either FHD dimension.
- A one-hour synthetic H.264 FHD fixture was extracted at 1 frame/second: 3,600 crops in 132.27 seconds, no extraction errors, sampled peak RSS 807.3 MiB. See `DEVELOPMENT.md` for fixture limits.
- Manual annotation persistence, dataset validation, grouped split, YOLO export, Smart Folder Discovery, and real training queue have engine tests.
- Fine-tune accepts only a registered FrameForge YOLO11n detection checkpoint in its run directory with matching file, JSON and SQLite SHA-256. Unregistered and discovered external `.pt` files are rejected before model load.
- `predict_crops` runs real Ultralytics inference on batches, matches the checkpoint class schema, runs NMS once, applies inclusive confidence/class filters, and returns pixel-space boxes. Checkpoint identity/hash is rechecked before loading.
- Verification: `pytest engine/tests -q` — 51 passed; `pnpm build` — passed; `cargo fmt --check` — passed; `cargo check` — passed. A real one-epoch synthetic YOLO11n run produced a registered checkpoint and inference completed on one synthetic crop via MPS; this is pipeline smoke coverage, not model-quality evidence.
- `pnpm tauri build --bundles app` produced a 4.30 MiB macOS app; JSON-lines health from its bundled engine returned `status: ok` and listed `yolo11n_inference`.
- Commit `85dc842` records the inherited FHD/training/discovery implementation plus checkpoint-trust recovery. GitHub push failed because no HTTPS username credential was available.

## IMPLEMENTED_UNVERIFIED

- Training and Smart Folder Discovery screens compile in the production frontend build, but their complete click-through workflow has not been repeated on a fresh app bundle in this audit.
- The inference engine command is implemented and smoke-tested; it is not connected to a user-facing model-selection/auto-label/review flow.
- MPS training smoke is documented as one real YOLO11n epoch on a small synthetic dataset. It verifies the training pipeline, not model quality or sustained training stability.
- Auto-label result persistence and review state are not yet exercised because their project workflow is not implemented.

## PARTIALLY_COMPLETED

- Session snapshots have Python IPC commands and private storage, but the frontend does not autosave or restore state on launch.
- Dataset export creates immutable versioned directories with split and provenance manifests, but there is no parent-version graph, frozen test set, replay policy, or multi-generation lineage.
- Training has dataset preflight, independent `imgsz` jobs, actual metrics/logs/checkpoints, cancellation at epoch boundaries, and fine-tuning. Resume after interruption, richer optimizer/augmentation controls, experiment comparison, and candidate promotion policy are absent.
- Smart Folder Discovery reads external run metadata and lists checkpoints without loading them. External checkpoint architecture/task verification and import are intentionally unavailable.
- The one-hour media benchmark verifies a single synthetic clip; it does not establish multi-hour soak behavior or performance across codecs/content.

## BROKEN

- None identified by the checks listed above. The inherited Rust formatting failure was fixed with `cargo fmt`; both formatting check and crate compilation now pass.

## NOT_IMPLEMENTED

- Auto-labeling on extracted crops, candidate negatives, source-model/confidence provenance, provisional annotation states, and review/bulk-review workflow. The underlying inference engine is verified, but is not connected to project persistence or UI.
- Guided workspace, contextual next actions, full workflow status, active-learning suggestions, and model quality dashboard.
- Fixed validation/test evaluation, minimum quality targets, model-generation comparison, Champion/Challenger promotion and rollback.
- Training resume, historical-data replay strategy, dataset version lineage, and frozen human-reviewed test set.
- ONNX, TensorRT and OpenVINO export; FP16/INT8 calibration and accuracy verification; deployment-target benchmarking. The existing video extraction benchmark is not a model benchmark.
- Remote execution, transfer package, remote job tracking, and completion notifications.
- Standalone Python sidecar packaging.

## BLOCKED

- Windows 10/11 x64 packaging and RX 6600/DirectML hardware validation require access to a Windows target machine; do not mark Windows support verified before those checks.
- GitHub push is currently blocked: `git push origin feature/manual-labeling` returned `could not read Username for 'https://github.com': Device not configured`. Local commit `85dc842` is preserved; retry after GitHub HTTPS credentials are configured.

## Dependencies and next work

1. **Next:** connect `predict_crops` to extraction output and the saved project; add AI suggestion annotations with model/checkpoint/confidence provenance and a Needs Review state, preserving manual/reviewed edits.
2. Add review and candidate-negative handling. Only reviewed/approved examples may enter the default training export.
3. Add immutable dataset lineage and replay/frozen-test policies before iterative fine-tuning and quality comparisons.
4. Implement fixed-set evaluation and Champion/Challenger policy before model promotion or release targets.
5. Add the three required deployment exports and validate precision/accuracy/runtime before claiming deployment readiness.
6. Complete session restoration and Windows validation as independent platform/workflow tracks.

Update this file and `README.md`, `AGENTS.md`, `agent.md`, and `DEVELOPMENT.md` after each completed task; update product/design notes and `CHANGELOG.md` when applicable. Do not clear inherited working-tree changes.
