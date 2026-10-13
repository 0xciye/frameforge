# FrameForge agent notes

FrameForge is a local-first desktop workspace for video preparation and annotation. Phase 1 is infrastructure: a Tauri 2 shell, a Vietnamese status screen, and a Python JSON-lines health check.

Keep changes aligned with `PRODUCT.md`, `DESIGN.md`, and `DEVELOPMENT.md`. Do not claim that video import, detection, or labeling works until those behaviors exist end to end. Use small changes and preserve the Rust-to-Python JSON-lines contract.
