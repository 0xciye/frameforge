# Phát triển FrameForge

## Trạng thái hiện tại

| Khả năng | Trạng thái | Ghi chú |
|---|---|---|
| Tauri 2 + React/TypeScript workspace | IMPLEMENTED | Giao diện tiếng Việt theo ảnh tham chiếu |
| Python JSON-lines IPC | TESTED_MACOS | Bản app bundle hiện tại chuyển từ `Đang kết nối` sang `Engine sẵn sàng`; health JSON-lines trực tiếp trả `status: ok` |
| Nhập media, metadata, preview, timeline | TESTED_MACOS | Dialog, kéo thả, OpenCV đọc ảnh/video thật; video tối đa 1920×1080, nguồn lớn hơn bị từ chối; seek bằng tác vụ nền |
| Exact Center Crop, preview overlay | TESTED_MACOS | Preview dùng frame nguồn tối đa FHD và vẽ khung cam liền nét 320×320 tại tọa độ nguồn x=800, y=380 trên fixture 1920×1080 |
| Trích frame, lọc crop, letterbox/grid, JPG/PNG | TESTED_MACOS | Fixture video; blur/dHash tính trên vùng crop nguồn; progress/hủy |
| Dự án SQLite `.ffproj` | TESTED_ENGINE | Create/save/load; schema v1/v2 migrate lên v3 khi ghi; annotation AI lưu model ID và checkpoint SHA-256; autosave bảo toàn provenance |
| Nạp và kiểm tra metadata ONNX | IMPLEMENTED | Rust xác thực phần mở rộng; engine chạy session CPU; chưa có model fixture để xác nhận model cụ thể |
| YOLO11n inference adapter | TESTED_MACOS | Chỉ nạp checkpoint nội bộ đã đăng ký/khớp SHA-256 và class schema; Ultralytics NMS một lần, confidence filter inclusive; smoke một ảnh tổng hợp trên MPS |
| Auto-label ảnh hiện tại, review UI | IMPLEMENTED_UNVERIFIED | Engine persistence/review đã qua test; UI chọn checkpoint nội bộ khớp class, lưu gợi ý `needs_review`, duyệt/chỉnh sửa/loại; chưa click-through vì registry máy này không có checkpoint khả dụng |
| Gán nhãn thủ công | IMPLEMENTED | Vẽ/sửa/di chuyển/resize/xóa box Head (0), Body (1); undo/redo, review, autosave SQLite; lưu dự án trước khi chỉnh sửa |
| Dataset validation, grouped splits, YOLO export | TESTED_ENGINE | Export loại AI pending/rejected, loại ảnh chỉ còn pending khỏi cả reviewed-negative và ghi provenance box AI đã duyệt; 57 engine tests pass |
| Smart Folder Discovery | IMPLEMENTED | Quét giới hạn có progress/hủy; fixture xác nhận YAML dataset, run, metrics, symlink, cancel và đường dẫn ngoài scope; `.pt` chỉ được liệt kê, không nạp |
| Registry run ngoài | IMPLEMENTED | Metadata chỉ lưu khi người dùng chọn; checkpoint ngoài luôn `checkpoint_trusted=false` và không khả dụng cho fine-tune |
| YOLO11n training trên CPU/MPS và queue nhiều imgsz | TESTED_MACOS | Một epoch YOLO11n thật trên fixture nhỏ chạy bằng MPS Apple M1 Pro; checkpoint được lưu/đăng ký; metrics fixture không đại diện đánh giá model. |
| Registry checkpoint và fine-tuning | TESTED_ENGINE | Fine-tune chỉ nhận checkpoint YOLO11n detection nội bộ đã đăng ký, nằm trong run và khớp checksum ở tệp, record và registry; từ chối `.pt` ngoài hoặc chưa đăng ký. |
| Session snapshot engine | PARTIAL | Engine có `save_session`/`load_session`, nhưng frontend chưa tự lưu hoặc khôi phục phiên khi mở app. |
| Resume run sau khi dừng/gián đoạn | NOT_STARTED | Chưa có thao tác tiếp tục tiến trình từ checkpoint gần nhất. |
| Benchmark, experiment comparison, model export | NOT_STARTED | Chưa có công cụ benchmark model trong app; phép đo hiệu năng xử lý media một lần được ghi bên dưới |
| Windows 10/11, DirectML, RX 6600, đóng gói | PENDING_WINDOWS_VALIDATION | Cần máy Windows thật; chưa tuyên bố tương thích |

Thứ tự tiếp theo: inference hàng loạt cho project → dataset lineage/replay → đánh giá model → benchmark/so sánh → deployment export → resume training → session recovery UI → Windows validation. Video đầu vào tối đa FHD (1920×1080); nguồn lớn hơn bị từ chối, không scale.

## Công cụ và cài đặt

- Node.js, pnpm, Rust stable và toolchain native của hệ điều hành.
- Python 3.10 trở lên.
- Từ root: python3 -m venv .venv, .venv/bin/python -m pip install -e "engine[dev]", cài thêm .venv/bin/python -m pip install -e "engine[train]" khi dùng tab Huấn luyện, rồi pnpm install.
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

Các command được allowlist gồm health, inspect_sources, inspect_model, preview_frame, preview_center_crop, extract_frames, create_project, save_project, save_annotations, save_ai_suggestions, validate_dataset, export_yolo_dataset, load_project, training_capabilities, inspect_training_dataset, list_training_runs, register_discovered_run, scan_project_folder, train_yolo_queue và predict_crops. Rust kiểm tra command, phần mở rộng, kích thước và đường dẫn trước khi gọi Python. Train có thể dừng tại ranh giới epoch bằng cancel marker; extraction, discovery và inference hủy theo lô bằng marker.

Ví dụ:

```json
{"id":"health-1","command":"health","arguments":{}}
```

Response cuối có `id`, `ok` và `data` hoặc `error`; progress là các response JSON-lines riêng có `event`. Không ghi log vào stdout của engine.

## Dữ liệu và giới hạn

- Frame gốc không bị sửa. Ảnh đầu ra được ghi atomically qua tệp tạm; collisions có lựa chọn bỏ qua, đổi tên hoặc ghi đè.
- Video chỉ được nhận khi chiều rộng ≤1920 và chiều cao ≤1080. Import, preview và extraction đều kiểm tra frame giải mã; video vượt một trong hai giới hạn bị từ chối và không tự scale. Video nhỏ hơn giữ nguyên độ phân giải.
- Hạn mức frame áp dụng tổng trên các nguồn đã chọn. Danh sách file/thumbnail trả về cho gallery hiện giới hạn 500 ảnh mỗi request để tránh giữ lượng dữ liệu không giới hạn trong frontend.
- Bộ lọc dHash giữ tối đa 4096 hash gần nhất trong một lần extraction; bộ lọc là heuristic và không đảm bảo phát hiện mọi ảnh gần giống.
- Center crop dùng `floor((frame - crop) / 2)` và half-open slicing; crop vượt frame trả lỗi rõ ràng. Full-frame letterbox và grid tiếp tục resize/padding theo mode riêng.
- Blur/Laplacian và dHash đo source crop trước output resize/padding. `candidate_crops`, `blurry_crops` và `duplicate_crops` là số liệu crop-level; `candidate_frames` giữ nghĩa frame-level.
- Mỗi ảnh lưu source identity (SHA-256 từ path/size/mtime), frame/timestamp, crop rectangle, sharpness score và extraction settings. Đây là identity fingerprint, không phải content hash.
- SQLite schema v3 thêm provenance model/checkpoint cho annotation AI; schema v1/v2 được đọc và migrate khi người dùng ghi. Metadata crop từ schema v2 vẫn được giữ.
- Annotation lưu tọa độ pixel ảnh gốc trong SQLite; fingerprint ngăn ghi nhãn lên file ảnh đã bị thay đổi kể từ lúc preview. Undo/redo, xóa và review đi qua cùng hàng đợi ghi dự án để giữ thứ tự autosave.
- Dự án mới mặc định có class ổn định `head` (0) và `body` (1). Tab Gán nhãn gọi inference trên ảnh đang mở; thao tác cần project đã lưu.
- `predict_crops` chỉ dùng checkpoint YOLO11n nội bộ đã đăng ký, còn khớp SHA-256 và có class schema khớp với model. Nó trả box `xyxy` theo pixel trên ảnh gốc; NMS do Ultralytics thực hiện và confidence cuối được lọc theo `score >= threshold`.
- `save_ai_suggestions` xác minh lại checkpoint, class schema project và fingerprint ảnh; gợi ý được lưu ở `needs_review` với model ID/checkpoint SHA-256/confidence. Autosave giữ annotation AI và provenance.
- Dataset validator đọc lại ảnh thật, kiểm tra kích thước so với project, class ID và box hữu hạn/còn trong biên. Chỉ box thủ công và AI `reviewed` được export; pending/rejected bị loại, ảnh chỉ còn pending không trở thành negative dù có trạng thái reviewed.
- Split theo source gom toàn bộ frame/crop cùng video; split theo segment dùng timestamp hoặc frame/FPS. Ảnh có SHA-256 trùng nội dung luôn được gộp cùng nhóm split. Khi số nhóm độc lập ít hơn số split được bật, báo cảnh báo và chỉ gán các nhóm có sẵn.
- YOLO export tạo thư mục phiên bản mới nguyên tử dưới thư mục người dùng đã chọn, sao chép ảnh gốc và xác minh SHA-256 sau khi sao chép; nhãn dùng `class x_center y_center width height` chuẩn hóa 8 chữ số thập phân. `manifest.json` lưu mapping class, split, fingerprint, hình học crop và provenance nguồn. Export không sửa project/ảnh nguồn.
- Smart Folder Discovery dùng `yaml.safe_load`, chỉ quét trong root được chọn, bỏ symlink, giới hạn depth/files/metadata/CSV rows, không đọc pixel hoặc nội dung checkpoint. YAML split/data trỏ ra ngoài root bị chặn; ngoại lệ quyền truy cập tạo kết quả partial.
- Run ngoài chỉ được lưu metadata khi nhấn “Lưu run vào registry”. Registry vẫn đánh dấu checkpoint ngoài chưa tin cậy và không chọn nó làm model fine-tune; xác minh kiến trúc/trust vẫn chưa triển khai.
- Python engine được đóng gói cùng nguồn, nhưng runtime và dependency Python vẫn cần cài riêng khi chạy ứng dụng. Standalone sidecar packaging chưa làm.

## Kiểm chứng macOS

- `./.venv/bin/python -m pytest engine/tests -q`: 44 passed; có discovery fixture, metadata run, metrics, path scope, symlink, limits, cancellation và external registry safety.
- `pnpm build`: TypeScript strict và Vite production build thành công với 1.922 module.
- `cargo fmt --manifest-path src-tauri/Cargo.toml --all` và `cargo check --manifest-path src-tauri/Cargo.toml`: định dạng cùng biên dịch crate Tauri thành công.
- `git diff --check`: thành công.
- Smart Folder Discovery UI đã được TypeScript/Vite build nhưng chưa click-through trong app bundle macOS.
- Exact Center Crop: mở app macOS với video fixture FHD 1920×1080; preview giữ frame nguồn và vẽ ô 320×320 đúng tâm. App mở thẳng workspace, không hiện màn chào kiểm tra engine.
- Bản app bundle trước đó ở `Đang kết nối` trong lúc Python khởi tạo; ở bundle mới UI chuyển sang `Engine sẵn sàng`. Health JSON-lines chạy trực tiếp trong terminal trả `status: ok`.
- `pnpm tauri build --bundles app`: app bundle macOS 4.29 MiB thành công; engine resources chứa `dataset.py`, `discovery.py`, `media.py`, `project.py`, `session.py`, `training.py` và `frameforge_engine.py`.
- Smoke training trên MPS: PyTorch 2.14.1 và Ultralytics 8.4.177, YOLO11n 1 epoch ở imgsz 64 trên 4 ảnh train + 4 ảnh val tổng hợp; thiết bị thực tế MPS trên Apple M1 Pro, run completed và best.pt đã được lưu/đăng ký. mAP50=0 chỉ là kết quả fixture kiểm tra pipeline.

### 2026-10-14 — Dataset validation và YOLO export

- **Thay đổi:** Thêm tab Dataset để xác thực ảnh, nhãn, class, reviewed negative; chia train/val/test theo video hoặc đoạn thời gian; khóa ảnh trùng byte trong cùng split; xuất ảnh không resize, label YOLO, `data.yaml` và manifest vào thư mục phiên bản mới. Đưa `dataset.py` vào danh sách resource đóng gói của Tauri.
- **Tài liệu:** Cập nhật README, AGENTS.md, agent.md, DEVELOPMENT.md, PRODUCT.md, DESIGN.md và CHANGELOG.md.
- **Kiểm tra:** `pnpm test:engine` — 28 passed; `pnpm build` — thành công; `cargo fmt --check`, `cargo check` và `git diff --check` — thành công; `pnpm tauri build --bundles app` — app bundle macOS 4.17 MiB thành công. Đã chạy JSON-lines validate/export từ engine trong app bundle: 3 ảnh đủ điều kiện, 1 ảnh trống chưa review bị loại, split train 2 / val 1, xuất đủ ảnh, `data.yaml` và manifest. Giao diện FrameForge đang mở hiển thị frame video 1920×1080 cùng overlay crop vuông 320×320 ở tâm.
- **Git:** Milestone đã commit local trên nhánh `feature/manual-labeling`; push GitHub còn chờ xác thực như trạng thái đã ghi trước đó.
- **Giới hạn:** Dataset UI click-through chưa chạy trên bundle mới; kiểm chứng UI trong lượt này là build frontend, còn kiểm tra thực tế backend dùng module đã đóng gói. UI export cần dự án `.ffproj` đã lưu. Split theo nhóm có thể lệch tỷ lệ mục tiêu khi ít nhóm độc lập; exporter nêu cảnh báo. Windows chưa được kiểm thử.
- **Bước tiếp:** YOLO11n training và quản lý checkpoint.

### 2026-10-14 — Preview Exact Center Crop

- **Thay đổi:** Preview luôn hiển thị frame nguồn thật; khung màu cam biểu diễn vùng crop W×H ở tâm. Bỏ chuyển sang ảnh crop trong preview; khung đầu ra vẫn xuất hiện trong gallery sau khi trích xuất. Viền overlay dùng `border-box` để không nới vùng hiển thị vượt kích thước đã đặt.
- **Tệp:** `src/App.tsx`, `src/App.css`, `README.md`, `AGENTS.md`, `agent.md`, `DEVELOPMENT.md`, `PRODUCT.md`, `DESIGN.md`, `CHANGELOG.md`, `docs/design/extract-screen.md`.
- **Kiểm tra:** `pnpm build` — thành công; `git diff --check` — thành công; `pnpm tauri build --bundles app` — thành công. Smoke check giao diện với video mẫu 1920×1080: app đọc frame thật, hiển thị overlay tâm 320×320 và không còn nút đổi sang crop ảnh.
- **Git:** Đã commit trên nhánh `feature/manual-labeling`; push vẫn chờ GitHub authentication do lần thử trước thiếu credentials.
- **Giới hạn:** Engine suite không chạy lại trong lượt này; mẫu video smoke check được tạo tạm trong `/tmp`.

### 2026-10-14 — Làm rõ viền crop preview

- **Thay đổi:** Đổi viền overlay sang nét đứt màu cam để phân biệt rõ vùng W×H đang được chọn. Hình học vẫn bám tọa độ frame nguồn; preset 320×320 được thể hiện thành ô vuông trên frame thật, còn kích thước W×H khác nhau vẫn là hình chữ nhật.
- **Tài liệu:** Cập nhật README.md, AGENTS.md, agent.md, DESIGN.md, DEVELOPMENT.md, CHANGELOG.md và brief giao diện trích xuất.
- **Kiểm tra:** `pnpm build` — thành công với Node.js runtime đi kèm Codex; `pnpm tauri build --bundles app` — app bundle macOS 5.46 MiB thành công. Rust báo `rust-objcopy` thiếu `libLLVM.dylib` khi bỏ debug symbols nhưng bundle vẫn được tạo. Chưa chạy bộ test cho thay đổi CSS này.
- **Git:** Commit `2ba33e2` trên nhánh `feature/manual-labeling`; chưa push vì xác thực GitHub vẫn chưa sẵn sàng.
- **Giới hạn:** Cửa sổ app release hiện tại vẫn giữ asset đã nạp; khởi động lại app để xem viền mới.

### 2026-10-14 — Sửa bundle và kiểm tra lại Exact Center Crop preview

- **Sửa bundle:** Thêm `discovery.py`, `session.py` và `training.py` vào Tauri resources, khắc phục thiếu `frameforge.discovery` trong app bundle cũ. Build `.app` mới thành công; health JSON-lines từ engine trong bundle trả `status: ok`.
- **Preview:** Trên video fixture 1920×1080, frame nguồn vẫn hiển thị đầy đủ. Ô crop 320×320 ở tâm (x=800, y=380); đổi sang 512×512 làm ô lớn lên và vẫn giữ tâm. Đã khôi phục UI về 320×320.
- **Kiểm tra:** Xác nhận trực tiếp trên FrameForge qua accessibility tree và screenshot. Lượt này chỉ kiểm tra preview trên fixture FHD; không trích xuất ảnh đầu ra.
- **Tài liệu:** Cập nhật README.md, AGENTS.md, agent.md, DEVELOPMENT.md và CHANGELOG.md.

### 2026-10-14 — Manual labeling và startup

- **Thay đổi:** Nối tab Gán nhãn vào workspace; thêm canvas box pixel-coordinate, chọn class, di chuyển/resize/đổi class/xóa, undo/redo, zoom/pan, chuyển ảnh và đánh dấu review. Annotation tự lưu vào dự án SQLite sau khi người dùng lưu dự án; engine xác thực class, bounds, box có diện tích và fingerprint ảnh. Giữ màn chào kiểm tra engine đã gỡ; kiểm tra nền chỉ cập nhật badge trên toolbar.
- **Tài liệu:** Cập nhật README, AGENTS.md, agent.md, DEVELOPMENT.md và CHANGELOG.md. Ở thời điểm ghi nhận lượt đó, độ phân giải video chưa bị giới hạn; policy hiện tại tối đa FHD được cập nhật trong nhật ký bên dưới.
- **Kiểm tra:** `node_modules/typescript/bin/tsc` và `node_modules/vite/bin/vite.js build` chạy bằng Node runtime đóng gói — thành công. `pnpm build` không chạy được khi Node không nằm trên PATH. Bộ engine 25 test đã qua ở lượt trước khi nối UI. Build lại Tauri `.app` đang chạy; chưa chạy UI end-to-end.
- **Git:** `feature/manual-labeling`; thay đổi chưa commit. Push GitHub còn chờ xác thực.
- **Giới hạn:** Chưa có YOLO inference, dataset manager/export, training, fine-tuning, benchmark hoặc deployment export. Windows tiếp tục `PENDING_WINDOWS_VALIDATION`.
- **Bước tiếp:** Làm dataset validation/split và YOLO export.

## Nhật ký công việc

### 2026-10-14 — Nối inference vào gán nhãn

- **Thay đổi:** Tab Gán nhãn chọn checkpoint nội bộ đã xác minh và chạy YOLO trên ảnh đang mở. Gợi ý được lưu vào SQLite schema v3 với model ID, checkpoint SHA-256, confidence và trạng thái `needs_review`; duyệt, chỉnh sửa hoặc loại từng box; autosave giữ provenance và các gợi ý AI. Chạy lại cùng checkpoint trên cùng ảnh được mở sau khi loại hết gợi ý trước đó. Engine từ chối box/confidence dạng chuỗi hoặc boolean.
- **Dataset:** Export YOLO bỏ box AI chưa duyệt/bị loại; ảnh chỉ còn gợi ý AI chờ xử lý không được tính là negative. Manifest ghi provenance box được duyệt. Video đầu vào vẫn tối đa FHD 1920×1080; video lớn hơn bị từ chối, không scale.
- **Tệp:** `engine/frameforge/project.py`, `engine/frameforge/dataset.py`, `engine/tests/test_engine.py`, `src/ManualLabeling.tsx` và tài liệu roadmap/sản phẩm.
- **Kiểm tra:** `./.venv/bin/python -m pytest engine/tests -q` — 57 passed; TypeScript `tsc` và Vite production build — thành công; `cargo fmt --check` và `cargo check` — thành công (dùng Rust toolchain tuyệt đối vì shell không có `cargo`/`rustc` trên PATH); `git diff --check` — thành công. `pnpm tauri build --bundles app` — app macOS 4.33 MiB; JSON-lines health từ engine đã đóng gói trả `status: ok`.
- **Chưa xác minh:** UI inference chưa click-through với checkpoint thật vì training registry hiện không có checkpoint dùng được. Windows chưa được build/kiểm thử.
- **Benchmark giới hạn:** Phép đo video FHD tổng hợp dài một giờ trước đó đạt 3.600 crop trong 132,27 giây, RSS lấy mẫu tối đa 807,3 MiB. Đây là một lượt extraction, không phải soak test nhiều giờ; không đại diện mọi codec.
- **Git:** Commit `f202b4b` (`feat(label): connect verified inference review workflow`) trên `feature/manual-labeling`. Push qua HTTPS thất bại do môi trường không cấp username; thử SSH cũng thất bại vì host key GitHub chưa được xác minh.
- **Bước tiếp:** Nối inference hàng loạt cho các ảnh của project, nạp model một lần mỗi lượt và giữ review từng box.

### 2026-10-14 — YOLO11n inference engine

- **Thay đổi:** Thêm `predict_crops` qua JSON-lines/Tauri, batch tối đa 500 ảnh, NMS/score từ Ultralytics, filter confidence inclusive và class, tọa độ box pixel, kiểm schema model, progress/hủy. Từ chối checkpoint không đăng ký, sai checksum hoặc class schema lệch.
- **Kiểm tra:** `pytest engine/tests -q` — 51 passed; `pnpm build`, `cargo fmt --check`, `cargo check` thành công. Smoke thật: YOLO11n fine-tune 1 epoch trên một fixture tổng hợp train/val bằng CPU, registry xác minh checkpoint; inference một ảnh trên MPS hoàn tất. Kết quả fixture chỉ xác nhận pipeline, không đánh giá chất lượng model.
- **Bundle:** `pnpm tauri build --bundles app` tạo `FrameForge.app` 4.30 MiB; gọi JSON-lines health từ `Contents/Resources/engine/frameforge_engine.py` trả `status: ok` và liệt kê `yolo11n_inference`. Auto-label/review UI chưa có.
- **Bước tiếp:** Kết nối `predict_crops` với ảnh crop sau extraction, lưu AI suggestions kèm model/checkpoint/confidence ở trạng thái Needs Review; không ghi đè manual/reviewed annotations.

### 2026-10-14 — Audit phục hồi và khóa nguồn checkpoint fine-tune

- **Audit:** Đối chiếu code với master prompt v1.0; tạo `TASKS.md` với sáu nhóm trạng thái, phụ thuộc và thứ tự tiếp theo. Giữ nguyên mọi thay đổi đang có trong working tree. Phạm vi video giữ tối đa FHD 1920×1080, không nhận hoặc scale nguồn lớn hơn.
- **Sửa lỗi:** Engine từ chối fine-tune từ tệp `.pt` chưa đăng ký hoặc run quét ngoài. Checkpoint phải đến từ run YOLO11n detection của FrameForge, nằm trong thư mục run, có trạng thái hoàn tất/đã hủy và khớp SHA-256 trong registry. Bổ sung hồi quy cho checkpoint tùy ý, checkpoint external, fine-tune checkpoint hợp lệ và checksum.
- **Kiểm tra:** `./.venv/bin/python -m pytest engine/tests -q` — 47 passed; `pnpm build` — thành công bằng Node.js runtime đi kèm Codex; `cargo fmt --check` và `cargo check` — thành công; benchmark video FHD một giờ được giữ theo phép đo đã ghi ở mục kế tiếp.
- **Git:** Commit local `85dc842`; push bị chặn vì GitHub HTTPS yêu cầu username nhưng môi trường không cung cấp (`could not read Username for 'https://github.com': Device not configured`).
- **Chưa xác minh:** Chưa click-through discovery/training trên app bundle mới. Session API chưa nối với autosave/restore trong frontend. Inference, auto-label, đánh giá model, resume, export và Windows còn trong `TASKS.md`.
- **Bước tiếp:** Tạo YOLO11n inference adapter chỉ nhận checkpoint đã xác minh, kèm test đầu ra box/confidence/class trên fixture ảnh nhỏ.

### 2026-10-14 — Giới hạn video FHD và benchmark video dài

- **Thay đổi:** Chỉ nhận video có kích thước tối đa 1920×1080. Import, preview và extraction từ chối video lớn hơn, không tự scale; thông báo lỗi đầu tiên được hiển thị trong UI. Giới hạn crop/output FHD được giữ nguyên.
- **Benchmark:** Trên máy macOS hiện tại, engine giải mã fixture H.264 tổng hợp 1920×1080, 30 fps, dài 3.600 giây (108.000 frame); lấy mẫu mỗi giây và xuất crop tâm 320×320 JPG, tắt blur/duplicate filter. Extraction mất 132,27 giây (27,22× nhanh hơn thời lượng clip), xử lý 3.600/3.600 crop, không lỗi; tổng JPG 16,5 MB. RSS process đo được tăng từ 581,4 lên tối đa 807,3 MiB (182 lần lấy mẫu progress). Tệp fixture 10,7 MB; tạo fixture mất 159,53 giây và không tính vào thời gian extraction.
- **Kiểm tra:** `pytest engine/tests -q` — 45 passed; TypeScript và Vite production build — thành công; `git diff --check` — thành công.
- **Giới hạn phép đo:** Nội dung fixture tổng hợp ít chuyển động; kết quả chỉ xác nhận một lượt xử lý video FHD dài một giờ trên máy này, không dự đoán tốc độ cho mọi codec/nội dung và không phải stress test chạy liên tục nhiều giờ. Chưa click-through thông báo từ chối trong app bundle.

### 2026-10-14 — YOLO11n training và xác nhận preview

- Thay đổi: Nối tab Huấn luyện vào workspace; thêm preflight dataset, capability detection, đề xuất cấu hình, queue imgsz độc lập, progress/metrics/runtime, hủy tại epoch boundary, registry SQLite và fine-tune từ checkpoint đã đăng ký. Pretrained yolo11n.pt được cache trong app data. Lọc event epoch trùng do validation cuối và giữ metadata job/queue qua progress event.
- Preview: App giữ frame nguồn video FHD 1920×1080, vẽ khung crop 320×320 tại x=800, y=380; UI dùng `preview_frame`, không dùng ảnh crop làm frame preview. Khung cam liền nét giúp thấy rõ ranh giới vùng crop.
- Kiểm tra: pnpm test:engine — 34 passed; pnpm build — thành công; cargo fmt --check, cargo check và git diff --check — thành công; pnpm tauri build --bundles app — thành công. MPS smoke: torch 2.14.1, Ultralytics 8.4.177, Apple M1 Pro, 1 epoch imgsz 64 trên fixture tổng hợp 4 ảnh train + 4 val; run completed, best.pt được lưu và registry ghi thiết bị mps. mAP50=0 trên fixture chỉ phản ánh smoke test, không phải đánh giá chất lượng.
- Giới hạn: Chưa có resume, inference, benchmark hoặc deployment export. Windows/DirectML/RX 6600 chưa kiểm thử. Smoke chạy trực tiếp engine trên venv và fixture tổng hợp; chưa click-through tab Huấn luyện trên bundle.


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
