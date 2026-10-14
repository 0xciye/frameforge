# Extract workspace brief

**Surface:** Tauri desktop application · operate media locally.

**Audience and task:** Dataset creators who import local videos/images, inspect actual media, tune sampling and image filters, and extract useful frames.

**Proof and constraints:** Metadata and previews come from selected files. No invented detection, thumbnail, or model metadata. The supplied screenshot pins the graphite surfaces, compact density, Vietnamese controls, panel arrangement, and blue selection/action language. Files remain local.

## Direction contract

**THESIS:** Keep source selection, frame inspection, extraction settings, and recent results visible in one continuous workbench.

**OWN-WORLD:** Graphite surfaces, neutral separators, compact system sans, blue for primary actions/selection, semantic colors for status. Familiar controls suit a dense desktop tool; do not add decorative marketing UI.

**STORY:** Import a file or folder → inspect actual metadata and preview → set the locked Exact Center Crop → choose sampling and crop-level filters → extract → review generated thumbnails and progress.

**FIRST VIEWPORT:** 52px toolbar and 42px workflow tabs. Three columns: 226px source list, flexible preview, 320px settings at the default 1440px window. A 174px gallery spans below, followed by a 28px status line. Six-pixel splitters resize all three regions with pointer or arrow keys and persist locally. Minimum window is 1024×700; CSS reduces panel widths and keeps settings/gallery scrollable.

Accept video sources only up to 1920×1080. Larger videos are rejected during import, preview, and extraction without scaling; smaller sources keep their original resolution. Crop/output dimensions are capped at FHD.

**FORM:** The user screenshot is the visual source of truth. The UI Pro Max searchable styles returned landing-page patterns and were discarded; the Operate guidance supplied accessibility and dense-control considerations. Impeccable written references guided the UI review; its CLI audit did not run because the local checksum sidecar was unavailable.

**CROP:** The solid orange overlay represents the exact W×H source-pixel rectangle centered with floor-based coordinates while keeping the decoded source frame visible. The crop output appears in the gallery after extraction. Preset sizes are shortcuts and custom integer dimensions remain available. Out-of-frame dimensions are explained before extraction. Blur and near-duplicate filtering inspect the source crop.

**FINISH:** Disabled/loading/error/empty states, reduced motion, keyboard focus and panel bounds are present. Inference, label editing and export stay visibly unavailable until their real workflows exist.

**Unresolved:** Windows native behavior, DirectML, standalone Python sidecar packaging and target-hardware performance need Windows validation.
