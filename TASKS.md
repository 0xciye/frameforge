# FrameForge v1.0 task ledger

This ledger recovers branch `feature/manual-labeling`; dataset lineage/replay was committed locally as `7cbae6d`. See Git history and `DEVELOPMENT.md` for the latest verification record. Preserve all existing changes. Current source-video scope is **at most 1920×1080**: reject larger sources at import, preview and extraction; do not downscale them or add higher-resolution input options. Smaller sources retain their native size.

## COMPLETED_VERIFIED

- Media import/inspection, real frame preview and background seeking have engine coverage; crop and extraction preserve source pixels.
- Exact Center Crop uses `floor((frame − crop) / 2)` and displays the source frame with the crop overlay. FHD fixture check confirms 320×320 at `(800, 380)` on 1920×1080.
- Video input bounds are enforced in the engine at import/preview/extraction. Regression coverage rejects sources exceeding either FHD dimension.
- A one-hour synthetic H.264 FHD fixture was extracted at 1 frame/second: 3,600 crops in 132.27 seconds, no extraction errors, sampled peak RSS 807.3 MiB. See `DEVELOPMENT.md` for fixture limits.
- Dataset lineage is implemented: freeze reviewed validation/test sets, verify parent manifest and image/label hashes, preserve fixed images and labels across child versions, and replay historical training groups with `all`, `balanced`, or `new`. Training preflight/queue/run metadata records the lineage.
- Manual annotation persistence, dataset validation, grouped split, YOLO export, Smart Folder Discovery, and real training queue have engine tests.
- Fine-tune accepts only a registered FrameForge YOLO11n detection checkpoint in its run directory with matching file, JSON and SQLite SHA-256. Unregistered and discovered external `.pt` files are rejected before model load.
- `predict_crops` runs real Ultralytics inference on batches, matches the checkpoint class schema, runs NMS once, applies inclusive confidence/class filters, and returns pixel-space boxes. Checkpoint identity/hash is rechecked before loading.
- Verification for this turn: 68 engine tests, TypeScript/Vite production build, Rust format/check and `git diff --check`; dataset UI has not been click-through tested in a fresh macOS app bundle. Full-scale actual YOLO training remains unverified.
- A prior 4.33 MiB macOS app bundle passed bundled-engine JSON-lines health before the current batch-inference changes.
- Commit `f6ad23c` adds project-wide inference, reviewed negative candidates and large-dataset training plumbing; it is local and unpushed. The inherited base includes commit `85dc842` for FHD/training/discovery and checkpoint-trust recovery.

## IMPLEMENTED_UNVERIFIED

- Training and Smart Folder Discovery screens compile in the production frontend build, but their complete click-through workflow has not been repeated on a fresh app bundle in this audit.
- Per-image and project-wide inference are connected to the labeling UI. Batch engine tests use a fake YOLO runtime for 130 images over multiple persistence chunks; UI workflow has not been click-through-tested with a real checkpoint because the local registry has no usable model.
- Candidate-negative persistence/review is tested for fingerprints, rejection/retry, confirmed-negative export provenance, and prevention of pending candidates becoming negatives. YOLO export filters pending/rejected AI boxes and candidates.
- The MPS training smoke is one real YOLO11n epoch on a small synthetic dataset. A scale test creates 20,003 uniform 320×320 JPEGs, runs dataset preflight twice, and checks two imgsz jobs receive 301 epochs each through a fake trainer (9.53 seconds including fixture creation). This verifies capacity/configuration plumbing, not real 20k-image YOLO throughput, 301 completed epochs, or soak stability.

## PARTIALLY_COMPLETED

- Session snapshots have Python IPC commands and private storage, but the frontend does not autosave or restore state on launch.
- Training has dataset preflight, independent `imgsz` jobs, actual metrics/logs/checkpoints, cancellation at epoch boundaries, and fine-tuning. Engine accepts 10,000 as the maximum epoch setting; set patience to 0 to disable early stopping. Actual training on more than 20,000 images for more than 300 epochs per size has not been run. Resume after interruption, richer optimizer/augmentation controls, experiment comparison, and candidate promotion policy are absent.
- Smart Folder Discovery reads external run metadata and lists checkpoints without loading them. External checkpoint architecture/task verification and import are intentionally unavailable.
- The one-hour media benchmark verifies a single synthetic clip; it does not establish multi-hour soak behavior or performance across codecs/content.

## BROKEN

- None identified by the checks listed above. The inherited Rust formatting failure was fixed with `cargo fmt`; both formatting check and crate compilation now pass.

## NOT_IMPLEMENTED

- Negative-image sampling policy and sampling ratios remain unimplemented; candidate-negative decisions require human review.
- Guided workspace, contextual next actions, full workflow status, active-learning suggestions, and model quality dashboard.
- Fixed validation/test evaluation, minimum quality targets, model-generation comparison, Champion/Challenger promotion and rollback.
- Training resume and session restoration in the frontend.
- ONNX, TensorRT and OpenVINO export; FP16/INT8 calibration and accuracy verification; deployment-target benchmarking. The existing video extraction benchmark is not a model benchmark.
- Remote execution, transfer package, remote job tracking, and completion notifications.
- Standalone Python sidecar packaging.

## BLOCKED

- Windows 10/11 x64 packaging and RX 6600/DirectML hardware validation require access to a Windows target machine; do not mark Windows support verified before those checks.
- GitHub push is currently blocked. Pushing implementation commit `7cbae6d` returned `could not read Username for 'https://github.com': Device not configured`. Retry after HTTPS credentials are available; previous SSH push also failed because the GitHub host key was not verified.

## Dependencies and next work

1. **Next:** implement fixed-set evaluation and record metrics per training run, then build Champion/Challenger comparison and promotion policy.
2. Add the three required deployment exports and validate precision/accuracy/runtime before claiming deployment readiness.
3. Complete resume training and session restoration.
4. Validate real YOLO training throughput/stability on a representative 20,000+ image dataset for 300+ epochs per size.
5. Complete Windows validation when a Windows target is available.

Update this file and `README.md`, `AGENTS.md`, `agent.md`, and `DEVELOPMENT.md` after each completed task; update product/design notes and `CHANGELOG.md` when applicable. Do not clear inherited working-tree changes.
