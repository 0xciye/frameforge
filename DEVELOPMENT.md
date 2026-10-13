# Phát triển FrameForge

## Trạng thái hiện tại

| Khả năng | Trạng thái | Ghi chú |
|---|---|---|
| Tauri 2 + React/TypeScript workspace | IMPLEMENTED | Giao diện tiếng Việt theo ảnh tham chiếu |
| Python JSON-lines IPC | IMPLEMENTED | JSON-lines engine process có test; Rust command đã `cargo check`, runtime desktop toàn tuyến còn cần smoke test |
| Nhập media, metadata, preview, timeline | TESTED_MACOS | Dialog, kéo thả, OpenCV đọc ảnh/video thật; seek bằng tác vụ nền |
| Exact Center Crop, preview overlay, xem crop thật | TESTED_MACOS | Pixel slicing không resize/padding; geometry và preview có test; Tauri native smoke test còn chờ |
| Trích frame, lọc crop, letterbox/grid, JPG/PNG | TESTED_MACOS | Fixture video; blur/dHash tính trên vùng crop nguồn; progress/hủy |
| Dự án SQLite `.ffproj` | TESTED_MACOS | Create/save/load; schema v1 nâng cấp lên v2 khi save; giữ provenance crop |
| Nạp và kiểm tra metadata ONNX | IMPLEMENTED | Rust xác thực phần mở rộng; engine chạy session CPU; chưa có model fixture để xác nhận model cụ thể |
| Inference YOLO, confidence filtering | NOT_STARTED | Chưa có adapter/detection; không sinh kết quả giả |
| Manual labeling, dataset manager, YOLO export | NOT_STARTED | Chưa có editor, validation, split hoặc exporter |
| YOLO11n training trên MPS/CPU, multi-resolution | NOT_STARTED | Chưa cài PyTorch/Ultralytics; chưa có training job manager |
| Checkpoint discovery, fine-tuning, resume | NOT_STARTED | Chưa có model registry hoặc checkpoint workflow |
| Benchmark, experiment comparison, model export | NOT_STARTED | Chưa có metrics/benchmark/export thật |
| Windows 10/11, DirectML, RX 6600, đóng gói | PENDING_WINDOWS_VALIDATION | Cần máy Windows thật; chưa tuyên bố tương thích |

Thứ tự ưu tiên: Exact Center Crop → filtering → manual labeling → YOLO dataset → YOLO11n training → fine-tuning → confidence-based AI expansion → benchmark → deployment export. Không gộp xác nhận macOS với Windows.

## Công cụ và cài đặt

- Node.js, pnpm, Rust stable và toolchain native của hệ điều hành.
- Python 3.10 trở lên.
- Từ root: `python3 -m venv .venv`, `.venv/bin/python -m pip install -e "engine[dev]"`, `pnpm install`.
- Windows dùng `.venv\Scripts\python.exe`; đặt `FRAMEFORGE_PYTHON` nếu cần chọn interpreter khác.

## Lệnh thường dùng

```sh
pnpm tauri dev
pnpm test:engine
pnpm build
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo check --manifest-path src-tauri/Cargo.toml
pnpm tauri build
```

Tauri chạy Vite ở cổng `1420`; giữ đồng bộ `tauri.conf.json` và `vite.config.ts` nếu đổi cổng. `pnpm test:engine` sử dụng `.venv` khi có, nếu không sẽ dùng `FRAMEFORGE_PYTHON` hoặc Python trên `PATH`.

## Engine và IPC

Rust mở trực tiếp `engine/frameforge_engine.py`, gửi một request JSON trên mỗi dòng UTF-8 qua stdin và đọc response/progress từ stdout. Chẩn đoán phải ghi ra stderr. Tiến trình chạy qua `spawn_blocking`; extraction phát `engine-progress` và có thể hủy qua `cancel_engine_request`.

Các command được allowlist: `health`, `inspect_sources`, `inspect_model`, `preview_frame`, `preview_center_crop`, `extract_frames`, `create_project`, `save_project`, `load_project`. Rust kiểm tra loại lệnh, phần mở rộng, kích thước và đường dẫn trước khi chạy Python. ONNX hiện chỉ được nạp qua CPU Execution Provider để đọc tensor và metadata; chưa có decoder YOLO.

Ví dụ:

```json
{"id":"health-1","command":"health","arguments":{}}
```

Response cuối có `id`, `ok` và `data` hoặc `error`; progress là các response JSON-lines riêng có `event`. Không ghi log vào stdout của engine.

## Dữ liệu và giới hạn

- Frame gốc không bị sửa. Ảnh đầu ra được ghi atomically qua tệp tạm; collisions có lựa chọn bỏ qua, đổi tên hoặc ghi đè.
- Hạn mức frame áp dụng tổng trên các nguồn đã chọn. Danh sách file/thumbnail trả về cho gallery hiện giới hạn 500 ảnh mỗi request để tránh giữ lượng dữ liệu không giới hạn trong frontend.
- Bộ lọc dHash giữ tối đa 4096 hash gần nhất trong một lần extraction; bộ lọc là heuristic và không đảm bảo phát hiện mọi ảnh gần giống.
- Center crop dùng `floor((frame - crop) / 2)` và half-open slicing; crop vượt frame trả lỗi rõ ràng. Full-frame letterbox và grid tiếp tục resize/padding theo mode riêng.
- Blur/Laplacian và dHash đo source crop trước output resize/padding. `candidate_crops`, `blurry_crops` và `duplicate_crops` là số liệu crop-level; `candidate_frames` giữ nghĩa frame-level.
- Mỗi ảnh lưu source identity (SHA-256 từ path/size/mtime), frame/timestamp, crop rectangle, sharpness score và extraction settings. Đây là identity fingerprint, không phải content hash.
- SQLite schema v2 thêm metadata crop; schema v1 được đọc và migrate cộng thêm cột khi người dùng save.
- Python engine được đóng gói cùng nguồn, nhưng runtime và dependency Python vẫn cần cài riêng khi chạy ứng dụng. Standalone sidecar packaging chưa làm.

## Kiểm chứng macOS của task gần nhất

- `pnpm test:engine`: 21 pytest engine trên Python 3.12; fixture ảnh/video, exact crop geometry (gồm crop 1×1), crop-level filters, project schema migration và JSON-lines.
- `pnpm build`: TypeScript strict và Vite production build.
- `cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check` và `cargo check --manifest-path src-tauri/Cargo.toml`: định dạng cùng biên dịch crate Tauri.
- Exact crop overlay/canvas dùng kích thước preview đã đo để giữ alignment khi resize. Đã mở app bundle macOS và xác nhận startup vào thẳng workspace Trích xuất, không còn màn chào v0.1.0.
- Engine Python qua Tauri IPC chuyển từ `Đang kết nối` sang `Engine sẵn sàng`; health trên bundle trả `status: ok`.
- `pnpm tauri build --bundles app` build thành công. Lệnh mặc định `pnpm tauri build` cũng tạo `.app` nhưng lỗi ở bước tạo DMG (`bundle_dmg.sh`).

## Nhật ký công việc

### 2026-10-13 — Exact Center Crop

- **Thay đổi:** Cài đặt crop W×H chính xác từ tâm frame, crop preview và overlay khóa; thêm preset và kiểm tra kích thước; đo blur/dHash trên crop nguồn; lưu crop rectangle và provenance trong SQLite schema v2.
- **Tài liệu:** Cập nhật README, PRODUCT, agent notes, DESIGN, CHANGELOG và hợp đồng giao diện Extract.
- **Kiểm tra:** `pnpm test:engine` — 21 passed; `pnpm build` — thành công; `cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check` — thành công; `cargo check --manifest-path src-tauri/Cargo.toml` — thành công; `pnpm tauri build --bundles app` — thành công; `git diff --check` — thành công.
- **Giới hạn:** Tạo DMG lỗi trong `bundle_dmg.sh`; Windows vẫn `PENDING_WINDOWS_VALIDATION`.
- **Git:** Commit `1c97231` — `feat(crop): implement exact center crop`; push chưa thành công vì thiếu GitHub HTTPS credentials (`could not read Username`). Remote đã cấu hình là `https://github.com/0xciye/frameforge.git`, nhưng API trả 404 khi kiểm tra unauthenticated nên chưa xác nhận repo public/tồn tại.
- **Bước tiếp:** Thiết lập GitHub authentication để tạo/đẩy remote; sau đó hoàn tất filtering và bắt đầu manual labeling theo roadmap.

## Quy trình thay đổi

Đọc `AGENTS.md`, `agent.md`, `PRODUCT.md`, `DESIGN.md` và mục liên quan trong file này trước khi sửa. Sau thay đổi đáng kể, chạy kiểm tra phù hợp, ghi kết quả thật, cập nhật tài liệu và xem lại diff. Không commit môi trường `.venv`, `node_modules`, cache, media người dùng hoặc model weights.

## Windows handoff

Windows chưa được build hoặc xác nhận. Các việc còn lại: xác nhận native dialogs/paths, đóng gói Python và DLL của OpenCV/ONNX Runtime, build Tauri, kiểm tra Windows 10/11, tích hợp DirectML và benchmark trên RX 6600. Chỉ ghi `TESTED_WINDOWS_CI` hoặc `VERIFIED_WINDOWS_HARDWARE` sau đúng loại kiểm thử đó.
