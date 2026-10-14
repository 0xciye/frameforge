# FrameForge v1.0 task ledger

This ledger recovers branch `feature/manual-labeling`; fixed-set evaluation, dataset lineage/replay, Champion/Challenger, trusted YOLO11n checkpoint import, and automatic workspace recovery are implemented locally. The current engine suite passes 93 tests and the frontend production build passes; Rust format/check passed earlier. A synthetic FHD batch with four aggregate source hours produced 14,400 crops without errors in 519.16 seconds. Source-video scope is **at most FHD 1920×1080**: reject larger sources at import, preview and extraction; do not downscale them or add higher-resolution video options. Smaller sources retain their native size. Training `imgsz` is separate.

## COMPLETED_VERIFIED

- Media import/inspection, real frame preview and background seeking have engine coverage; crop and extraction preserve source pixels.
- Exact Center Crop uses `floor((frame − crop) / 2)` and displays the source frame with the crop overlay. FHD fixture check confirms 320×320 at `(800, 380)` on 1920×1080.
- Video input bounds are enforced in the engine at import/preview/extraction. Regression coverage rejects sources exceeding either FHD dimension.
- dHash filtering uses a 64-bit Hamming-distance threshold (default 5), compares up to 4,096 recently retained crop hashes, and reports the actual filtered-crop count; the UI now clarifies that 5 is not a count.
- A batch of four copies of a synthetic one-hour H.264 FHD fixture was extracted at 1 frame/second: 14,400 crops from 4 aggregate source hours in 519.16 seconds, no errors, peak RSS 892.3 MiB. See `DEVELOPMENT.md` for fixture limits.
- Dataset lineage is implemented: freeze reviewed validation/test sets, verify parent manifest and image/label hashes, preserve fixed images and labels across child versions, and replay historical training groups with `all`, `balanced`, or `new`. Training preflight/queue/run metadata records the lineage.
- Manual annotation persistence, dataset validation, grouped split, YOLO export, Smart Folder Discovery, and real training queue have engine tests.
- The fixed-set evaluator has fake-runtime engine coverage for internal checkpoint trust, manifest/class/split checks, image and label hashes, cross-split leakage, target status/persistence, history, and cancellation; it records overall/per-class Precision/Recall/F1/mAP and evaluation conditions.
- Fine-tune accepts only a registered YOLO11n detection checkpoint with matching file, JSON and SQLite SHA-256 plus matching dataset class schema. Checkpoints may come from a FrameForge training run or the explicit trusted-import flow; unregistered/discovered external `.pt` files remain rejected.
- Explicit trusted checkpoint import validates YOLO11n detection task/architecture/class names, caps `.pt` at 2 GiB, hashes the source before/after validation and copy, stores it with private managed permissions and registers provenance. Engine tests verify registration, class-schema-gated fine-tuning, unsupported task/architecture, file-size rejection and source mutation rejection.
- Inference skips boxes within 1 px of a crop edge as potentially clipped/oversized, reports the skipped count, and does not turn an image into a negative candidate when all detections were skipped; fake-runtime regression tests verify both single-image and batch behavior.
- Current engine suite passes 93 tests; fixed-set evaluation and Champion/Challenger have fake-runtime coverage for fixed-set matching, condition/hardware checks, checkpoint hashes, targets, regressions, promotion, rollback and history.
- Session recovery automatically saves a private local project/workspace snapshot and restores it after startup health check. If the original project is missing or its checksum changed, the snapshot is opened as an unsaved project. Engine regression tests pass; the UI has a production build but no fresh app-bundle click-through yet.
- `predict_crops` runs real Ultralytics inference on batches, matches the checkpoint class schema, runs NMS once, applies inclusive confidence/class filters, and returns pixel-space boxes. Checkpoint identity/hash is rechecked before loading.
- Earlier verification: 68 engine tests, TypeScript/Vite production build, Rust format/check; dataset UI has not been click-through tested in a fresh macOS app bundle. Full-scale actual YOLO training remains unverified.
- A prior 4.33 MiB macOS app bundle passed bundled-engine JSON-lines health before the current batch-inference changes.
- Commit `f6ad23c` adds project-wide inference, reviewed negative candidates and large-dataset training plumbing; it is local and unpushed. The inherited base includes commit `85dc842` for FHD/training/discovery and checkpoint-trust recovery.

## IMPLEMENTED_UNVERIFIED

- Training and Smart Folder Discovery screens compile in the production frontend build, but their complete click-through workflow has not been repeated on a fresh app bundle in this audit.
- Per-image and project-wide inference are connected to the labeling UI. Batch engine tests use a fake YOLO runtime for 130 images over multiple persistence chunks; UI workflow has not been click-through-tested with a real checkpoint because the local registry has no usable model.
- Trusted external YOLO11n checkpoint import is wired through native dialogs, explicit warning/confirmation, Rust validation, managed registry, inference selection and fine-tune selection. Tests use fake Ultralytics; no compatible real `.pt` file is available for click-through validation.
- Candidate-negative persistence/review is tested for fingerprints, rejection/retry, confirmed-negative export provenance, and prevention of pending candidates becoming negatives. YOLO export filters pending/rejected AI boxes and candidates.
- The MPS training smoke is one real YOLO11n epoch on a small synthetic dataset. A scale test creates 20,003 uniform 320×320 JPEGs, runs dataset preflight twice, and checks two imgsz jobs receive 301 epochs each through a fake trainer (9.53 seconds including fixture creation). This verifies capacity/configuration plumbing, not real 20k-image YOLO throughput, 301 completed epochs, or soak stability.
- Fixed-set evaluation is wired through engine, Rust validation and Training workspace; fake-runtime checks cover metrics, hashes, leakage, trust and cancellation. The local registry has no usable checkpoint, so real-model evaluation remains unverified.
- Champion/Challenger is wired through engine IPC, Rust validation and Training workspace. Validation only; test remains holdout. Promotion requires matching fixed set/fingerprint/class schema/inference conditions/hardware, valid checkpoint hashes, passing configured targets and no regression above tolerance. Promote/rollback requires confirmation; tests use fake runtime, and real-checkpoint UI flow remains unverified.

## PARTIALLY_COMPLETED

- Training has dataset preflight, independent `imgsz` jobs, actual metrics/logs/checkpoints, cancellation at epoch boundaries, and fine-tuning. Engine accepts 10,000 as the maximum epoch setting; set patience to 0 to disable early stopping. Actual training on more than 20,000 images for more than 300 epochs per size has not been run. Resume after interruption and richer optimizer/augmentation controls remain absent.
- Smart Folder Discovery reads external run metadata and lists checkpoints without loading them. Discovered `.pt` files remain unavailable until separately imported through the explicit trust flow.
- The 4-hour aggregate media run used copies of one synthetic clip and lasted 8.7 minutes wall time; it does not establish multi-hour wall-clock soak behavior or performance across real content/codecs.

## BROKEN

- None identified by the checks listed above. The inherited Rust formatting failure was fixed with `cargo fmt`; both formatting check and crate compilation now pass.

## NOT_IMPLEMENTED

- Negative-image sampling policy and sampling ratios remain unimplemented; candidate-negative decisions require human review.
- Guided workspace, contextual next actions, full workflow status, active-learning suggestions, and model quality dashboard.
- Actual deployment latency/FPS/throughput/memory benchmark and model-generation comparison. These targets remain `not_measured`; the video extraction benchmark is not a model benchmark.
- Training resume after interruption.
- ONNX, TensorRT and OpenVINO export; FP16/INT8 calibration and accuracy verification; deployment-target benchmarking. The existing video extraction benchmark is not a model benchmark.
- Remote execution, transfer package, remote job tracking, and completion notifications.
- Standalone Python sidecar packaging.

## BLOCKED

- Windows 10/11 x64 packaging and RX 6600/DirectML hardware validation require access to a Windows target machine; do not mark Windows support verified before those checks.
- GitHub remote setup/push is still pending. `origin` points to `https://github.com/0xciye/frameforge.git`; HTTPS push previously failed because credentials were unavailable, and `gh` is not installed in this workspace. The unauthenticated GitHub page returns 404, which cannot distinguish a private repository from one that has not been created. Current changes, including session recovery, are committed locally; retry remote creation/push when GitHub authentication is available.

## Dependencies and next work

1. **Next:** use a trusted imported or newly trained real checkpoint to click-through inference, manual review, dataset export, fixed-set evaluation and Champion/Challenger; do not substitute demo detections or metrics.
2. Add the three required deployment exports and validate precision/accuracy/runtime before claiming deployment readiness.
3. Implement real deployment-target benchmarking; keep latency/FPS/RAM targets `not_measured` until measured.
4. Complete resume training.
5. Validate real YOLO training throughput/stability on a representative 20,000+ image dataset for 300+ epochs per size.
6. Complete Windows validation when a Windows target is available.

Update this file and `README.md`, `AGENTS.md`, `agent.md`, and `DEVELOPMENT.md` after each completed task; update product/design notes and `CHANGELOG.md` when applicable. Do not clear inherited working-tree changes.
