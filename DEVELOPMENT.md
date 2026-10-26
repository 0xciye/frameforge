# Phát triển FrameForge

## Trạng thái hiện tại

| Khả năng | Trạng thái | Ghi chú |
|---|---|---|
| Tauri 2 + React/TypeScript workspace | IMPLEMENTED | Giao diện tiếng Việt theo ảnh tham chiếu |
| Phạm vi pipeline | IMPLEMENTED | Chỉ còn nhập model → nhập ảnh/video → model gắn nhãn → người dùng duyệt/gán lại → lưu dataset. Training, fine-tune, evaluation, Champion/Challenger, lineage/replay dataset, Smart Folder Discovery và ONNX inspect đã gỡ khỏi UI, Tauri allowlist và engine. |
| Windows portable không phụ thuộc localhost | TESTED_WINDOWS_LOCAL | `pnpm build:portable` build app với Tauri custom protocol, bundle frontend và engine; mở executable thành cửa sổ `FrameForge`, không có server nghe cổng 1420. Chưa kiểm tra trên máy sạch hoặc GPU/DirectML. |
| Python JSON-lines IPC | TESTED_MACOS | Bản app bundle hiện tại chuyển từ `Đang kết nối` sang `Engine sẵn sàng`; health JSON-lines trực tiếp trả `status: ok` |
| Nhập media, metadata, preview, timeline | TESTED_MACOS | Dialog, kéo thả, OpenCV đọc ảnh/video thật; video tối đa 1920×1080, nguồn lớn hơn bị từ chối; seek bằng tác vụ nền |
| Exact Center Crop, preview overlay | TESTED_MACOS | Preview dùng frame nguồn tối đa FHD và vẽ khung cam liền nét 320×320 tại tọa độ nguồn x=800, y=380 trên fixture 1920×1080 |
| Trích frame, lọc crop, letterbox/grid, JPG/PNG | TESTED_MACOS | Fixture video; blur/dHash tính trên crop nguồn; ngưỡng dHash mặc định ≤1 bit để chỉ bỏ crop gần như trùng, so với tối đa 4.096 crop được giữ gần nhất; gallery báo số crop thực tế bị lọc; progress/hủy |
| Dự án SQLite `.ffproj` | TESTED_ENGINE | Create/save/load; schema v1/v2/v3 migrate lên v4 khi ghi; annotation và negative candidate AI lưu provenance/fingerprint; autosave bảo toàn provenance |
| Migration ngưỡng dHash project/session | TESTED_ENGINE | Giá trị legacy 5 chuyển về 1 một lần; version marker giữ lựa chọn mới; `extraction_settings` của ảnh không đổi |
| YOLO11n inference adapter | TESTED_ENGINE | Chỉ nạp checkpoint YOLO11n đã đăng ký qua trust flow, khớp SHA-256 và class schema; Ultralytics NMS một lần, confidence filter inclusive, bỏ box sát biên crop; fake-runtime test |
| Auto-label ảnh hiện tại, batch project và candidate negative | IMPLEMENTED_UNVERIFIED | Nút Auto label trên toolbar mở tab Gán nhãn; inference ảnh hiện tại/batch project, progress, review và candidate negative dùng engine thật. Engine test batch nhiều chunk, fingerprint recheck và export; box sát biên không tạo negative. Chưa click-through inference vì registry không có checkpoint khả dụng. |
| Gán nhãn thủ công | IMPLEMENTED | Vẽ/sửa/di chuyển/resize/xóa box Head (0), Body (1); undo/redo, review, autosave SQLite; lưu dự án trước khi chỉnh sửa |
| Dataset validation, negative sampling, grouped splits và YOLO export | TESTED_ENGINE | Negative sampling theo seed (mặc định 10%, 0–100%), loại ảnh gần như trùng bằng dHash ≤1; export loại box/candidate AI đang chờ hoặc bị loại |
| Nhập model YOLO11n `.pt` | TESTED_ENGINE | Bước đầu pipeline, nút “Nhập model .pt” trong tab Gán nhãn. Native dialog cảnh báo `.pt` có thể thực thi mã và yêu cầu xác nhận; giới hạn 2 GiB, sao chép riêng và xác minh checksum trước khi nạp bản sao, kiểm tra detection/YOLO11n/class rồi xác minh lại; `imgsz` từ checkpoint/`args.yaml`/mặc định 640; fake-runtime regression tests pass |
| Khôi phục phiên làm việc | TESTED_WINDOWS_LOCAL | Snapshot riêng tư lưu project và trạng thái workspace; frontend tự lưu sau debounce, tự khôi phục không hiện hộp thoại. Sửa mojibake tên project trong snapshot legacy, không ghi vào `.ffproj` gốc. Nếu dự án gốc mất hoặc checksum đổi, snapshot mở thành dự án chưa lưu. Regression test và portable app mới pass; snapshot local đã được phục hồi tên chính xác khi mở app. |
| Windows 10/11, DirectML, RX 6600, đóng gói | TESTED_WINDOWS_LOCAL | Portable folder vừa build và mở được trên máy Windows phát triển này; UI chạy không cần localhost. Chưa kiểm tra Windows máy sạch, DirectML hoặc RX 6600. CI run `38723939528` cũng build/smoke-check thành công trên runner. |
| CI frontend/engine/Rust đa nền tảng | PASSED_GITHUB_ACTIONS | Run `38723939528`: Ubuntu 3m44s, macOS 13m24s, Windows 15m49s đều pass, gồm build portable và engine smoke-check trên hai desktop runner. CI không xác nhận GPU/DirectML hoặc cài đặt trên máy người dùng. |
| Đóng gói macOS portable | TESTED_MACOS | `pnpm build:portable` tạo `FrameForge.app` 1,44 GiB có Python/PyTorch/Ultralytics bundle. Engine health/runtime self-check pass ở lượt build đó; app mở được, UI hiện Engine sẵn sàng. Chưa kiểm tra trên máy Mac sạch hoặc ký/notarize. |
| Đóng gói Windows portable folder | BUILT_WINDOWS_CI | Workflow CI dựng thư mục `FrameForge` gồm app executable và engine runtime, không dùng installer; cần xác minh artifact CI mới và cài/chạy click-through trên máy Windows thật. |
| Python runtime và dependencies trong app portable | TESTED_MACOS | Runtime được bundle cùng app; app không yêu cầu người dùng cài Python riêng. PyTorch/Ultralytics self-check pass trên macOS hiện tại. |
| Ký phát hành macOS/Windows | RELEASE_BLOCKER | Chưa cấu hình notarization macOS hoặc ký executable Windows; app portable chưa sẵn sàng phân phối công khai. |

Thứ tự tiếp theo: dùng checkpoint YOLO11n thật đã nhập để click-through toàn pipeline nhập model → trích xuất → inference → duyệt/gán lại → xuất dataset trên app đóng gói; build lại portable app vì runtime engine đã bỏ ONNX Runtime và module training. Sau đó tiếp tục Windows validation. Video đầu vào tối đa FHD (1920×1080); nguồn lớn hơn bị từ chối, không scale hoặc thêm lựa chọn độ phân giải cao.

## Cổng phát hành v1.0

**Trạng thái: chưa production-ready; giữ phiên bản `0.1.0`.** App portable macOS đã mở được trên máy phát triển này; Windows portable vừa mở được trên máy phát triển hiện tại, nhưng máy sạch và GPU/DirectML chưa xác minh. Trước khi gắn tag hoặc phát hành `1.0.0`, cần đóng các cổng sau:

- Thử mở app portable macOS trên máy sạch, không cài Python; ký và notarize app trước khi phân phối công khai.
- Xác minh Windows portable folder trên máy Windows sạch và làm click-through inference bằng model/dataset thật. DirectML/RX 6600 vẫn cần kiểm chứng riêng trên phần cứng đích.
- Nhận CI xanh mới cho thay đổi hiện tại; cấu hình ký Windows portable executable trước khi phân phối tải xuống công khai.
- Click-through nhập model/inference/review/export bằng checkpoint YOLO11n thật; không dùng fixture để kết luận chất lượng model.
- Chạy soak extraction bằng video FHD thực, nhiều codec/nội dung và thời lượng dài hơn batch fixture tổng hợp đã ghi ở trên.

Thay đổi trong lượt rà soát này: nhập checkpoint `.pt` bằng bản sao riêng đã xác minh trước khi nạp, kiểm tra lại sau khi nạp và giới hạn đọc/sao chép 2 GiB; Rust chỉ giữ tối đa 256 KiB stderr cho mỗi request nhưng vẫn drain pipe. Sửa Windows để ghi hash label trên đúng byte UTF-8 và dùng đường dẫn Unicode khi đọc/ghi ảnh. Runtime PyInstaller được bundle cho app portable, workflow CI chuyển từ build MSI sang app macOS và Windows folder. Nút Auto label mở tab Gán nhãn để dùng inference thật. Bản macOS hiện tại 1,44 GiB, engine health/training self-check pass, app mở với Engine sẵn sàng; frontend build, 110 engine tests, Rust fmt/check đều pass. Windows thật, máy Mac sạch, ký/notarize, checkpoint thật và soak footage thật vẫn là cổng mở.

## Đo hiệu năng media

Batch tổng hợp 4 giờ nguồn FHD: bốn bản sao của cùng một fixture H.264 dài một giờ, mỗi nguồn 1920×1080 ở 30 fps. Engine giải mã xấp xỉ 432.000 frame, lấy mẫu mỗi giây và ghi 14.400 crop tâm 320×320 JPG. Một tiến trình hoàn tất trong 519,16 giây (27,74× tổng thời lượng nguồn), không có lỗi; output 48.064.436 byte và peak RSS 892,3 MiB. Fixture tạo mất thêm 158,6 giây, không tính vào thời gian extraction. RSS mẫu tăng nhẹ từ khoảng 855 MiB lên peak 892,3 MiB trong suốt batch.

Đây là một lượt chạy khoảng 8,7 phút với nội dung tổng hợp lặp lại và một codec. Kết quả xác nhận batch dài theo thời lượng nguồn; chưa xác nhận soak nhiều giờ theo thời gian thực, video người dùng, nội dung đa dạng hoặc codec khác. Đây là media extraction benchmark, không phải model deployment benchmark.

## Công cụ và cài đặt

- Node.js, pnpm, Rust stable và toolchain native của hệ điều hành.
- Python 3.10 trở lên.
- Từ root: python3 -m venv .venv, .venv/bin/python -m pip install -e "engine[dev]", cài thêm .venv/bin/python -m pip install -e "engine[inference]" để chạy gợi ý AI, rồi pnpm install.
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

Các command được allowlist gồm health, inspect_sources, preview_frame, preview_center_crop, extract_frames, create_project, save_project, save_annotations, save_ai_suggestions, review_ai_negative_candidate, load_project, save_session, load_session, set_project_classes, inference_capabilities, list_models, import_yolo_checkpoint, predict_crops, auto_label_project, validate_dataset và export_yolo_dataset. Rust kiểm tra command, phần mở rộng, kích thước và đường dẫn trước khi gọi Python. Import `.pt` cần `trust_source=true`; Python (`frameforge/models.py`) giới hạn file 2 GiB, xác minh SHA-256 trước/sau khi nạp và sao chép, rồi chỉ đăng ký YOLO11n detection có class hợp lệ. Extraction và inference ảnh đơn/batch hủy theo lô.

Ví dụ:

```json
{"id":"health-1","command":"health","arguments":{}}
```

Response cuối có `id`, `ok` và `data` hoặc `error`; progress là các response JSON-lines riêng có `event`. Không ghi log vào stdout của engine.

## Dữ liệu và giới hạn

- Frame gốc không bị sửa. Ảnh đầu ra được ghi atomically qua tệp tạm; collisions có lựa chọn bỏ qua, đổi tên hoặc ghi đè.
- Video chỉ được nhận khi chiều rộng ≤1920 và chiều cao ≤1080. Import, preview và extraction đều kiểm tra frame giải mã; video vượt một trong hai giới hạn bị từ chối và không tự scale. Video nhỏ hơn giữ nguyên độ phân giải.
- Hạn mức frame áp dụng tổng trên các nguồn đã chọn. Danh sách file/thumbnail trả về cho gallery hiện giới hạn 500 ảnh mỗi request để tránh giữ lượng dữ liệu không giới hạn trong frontend.
- Bộ lọc dHash giữ tối đa 4096 hash gần nhất trong một lần extraction; ngưỡng là khoảng cách Hamming tối đa giữa hai hash cảm nhận 64-bit (không phải số ảnh cần giữ). Mặc định ≤1 chỉ loại crop gần như trùng; ngưỡng cao hơn lọc mạnh hơn và gallery đếm số crop thực tế bị lọc;
- Center crop dùng `floor((frame - crop) / 2)` và half-open slicing; crop vượt frame trả lỗi rõ ràng. Full-frame letterbox và grid tiếp tục resize/padding theo mode riêng.
- Blur/Laplacian và dHash đo source crop trước output resize/padding. `candidate_crops`, `blurry_crops` và `duplicate_crops` là số liệu crop-level; `candidate_frames` giữ nghĩa frame-level.
- Mỗi ảnh lưu source identity (SHA-256 từ path/size/mtime), frame/timestamp, crop rectangle, sharpness score và extraction settings. Đây là identity fingerprint, không phải content hash.
- SQLite schema v4 thêm provenance model/checkpoint cho annotation AI và bảng candidate âm tính có fingerprint; schema v1/v2/v3 được đọc và migrate khi người dùng ghi. Metadata crop cũ vẫn được giữ.
- Annotation lưu tọa độ pixel ảnh gốc trong SQLite; fingerprint ngăn ghi nhãn lên file ảnh đã bị thay đổi kể từ lúc preview. Undo/redo, xóa và review đi qua cùng hàng đợi ghi dự án để giữ thứ tự autosave.
- Dự án mới mặc định có class ổn định `head` (0) và `body` (1). Tab Gán nhãn hỗ trợ inference trên ảnh hiện tại hoặc project; thao tác cần project đã lưu.
- Registry model nằm ở `<App Data>/training/registry.sqlite3` (giữ tên thư mục cũ để model đã nhập vẫn dùng được); checkpoint nhập được sao chép vào `<App Data>/training/imported/<id>/weights/best.pt`. Importer đọc `imgsz` từ checkpoint, rồi `args.yaml` cạnh checkpoint, sau cùng mới dùng 640 và ghi nguồn.
- `predict_crops` chỉ dùng checkpoint YOLO11n đã đăng ký, còn khớp SHA-256 và có class schema khớp với model. Nó trả box `xyxy` theo pixel trên ảnh gốc; NMS do Ultralytics thực hiện và confidence cuối được lọc theo `score >= threshold`. Box cách biên crop không quá 1 px bị bỏ qua vì có thể là object bị cắt hoặc lớn hơn khung.
- `save_ai_suggestions` xác minh lại checkpoint, class schema project và fingerprint ảnh; box được lưu `needs_review` với model ID/checkpoint SHA-256/confidence. Batch project nạp model một lần, predict theo batch 16 và lưu từng chunk 64 ảnh trong transaction; fingerprint được recheck trước commit, ảnh đổi bị bỏ qua, hủy giữ phần đã lưu. Kết quả chỉ gồm box bị bỏ qua sát biên không tạo candidate âm tính; ảnh thực sự không có detection tạo candidate chờ người review.
- Extraction trả `images` (metadata mọi ảnh đã ghi, không thumbnail) cùng `files` (tối đa `MAX_GALLERY_FILES` = 500 thumbnail). Ảnh có `overwritten=true` bị reset nhãn trong frontend.
- `ai_inference_attempts(image_path, checkpoint_sha256)` lưu lượt inference chỉ có box chạm biên; không có foreign key nên `save_project` không xóa, fingerprint khác thì ảnh lại đủ điều kiện. `set_project_classes` từ chối khi đã có annotation/candidate.
- Dataset export không còn lineage/parent manifest, replay hay đóng băng val/test; mỗi lần xuất chia lại toàn bộ ảnh đủ điều kiện theo seed.
- Dataset validator đọc lại ảnh thật, kiểm tra kích thước so với project, class ID và box hữu hạn/còn trong biên. Chỉ box thủ công, AI `reviewed` và negative candidate được người dùng xác nhận mới được export; pending/rejected bị loại, ảnh pending không trở thành negative dù có trạng thái reviewed. Manifest ghi provenance candidate.
- Split theo source gom toàn bộ frame/crop cùng video; split theo segment dùng timestamp hoặc frame/FPS. Ảnh có SHA-256 trùng nội dung luôn được gộp cùng nhóm split. Khi số nhóm độc lập ít hơn số split được bật, báo cảnh báo và chỉ gán các nhóm có sẵn.
- YOLO export tạo thư mục phiên bản mới nguyên tử dưới thư mục người dùng đã chọn, sao chép ảnh gốc và xác minh SHA-256 sau khi sao chép; nhãn dùng `class x_center y_center width height` chuẩn hóa 8 chữ số thập phân. `manifest.json` lưu mapping class, split, fingerprint, hình học crop và provenance nguồn. Export không sửa project/ảnh nguồn.
- App portable bundle Python runtime cùng PyTorch/Ultralytics cho inference qua PyInstaller; ONNX Runtime, psutil và lap không còn là dependency.

## Kiểm chứng macOS

- `./.venv/bin/python -m pytest engine/tests -q`: 110 passed; gồm discovery/import workspace, trust/checksum checkpoint, imgsz từ checkpoint/`args.yaml`/fallback, path scope, symlink, limits, training và inference.
- `pnpm build`: TypeScript strict và Vite production build thành công với 1.922 module.
- `cargo fmt --manifest-path src-tauri/Cargo.toml --all` và `cargo check --manifest-path src-tauri/Cargo.toml`: định dạng cùng biên dịch crate Tauri thành công.
- `git diff --check`: thành công.
- Smart Folder Discovery UI đã được TypeScript/Vite build nhưng chưa click-through trong app bundle macOS.
- Exact Center Crop: mở app macOS với video fixture FHD 1920×1080; preview giữ frame nguồn và vẽ ô 320×320 đúng tâm. App mở thẳng workspace, không hiện màn chào kiểm tra engine.
- Bản app bundle trước đó ở `Đang kết nối` trong lúc Python khởi tạo; ở bundle mới UI chuyển sang `Engine sẵn sàng`. Health JSON-lines chạy trực tiếp trong terminal trả `status: ok`.
- `pnpm tauri build --bundles app`: app bundle macOS 4.29 MiB thành công; engine resources chứa `dataset.py`, `discovery.py`, `media.py`, `project.py`, `session.py`, `training.py` và `frameforge_engine.py`.
- Smoke training thật trên MPS (đã chạy ở lượt trước): PyTorch 2.14.1/Ultralytics 8.4.177, YOLO11n 1 epoch ở imgsz 64 trên 4 ảnh train + 4 ảnh val; Apple M1 Pro, checkpoint được lưu/đăng ký. mAP50=0 chỉ phản ánh fixture, không đánh giá chất lượng. Lượt hiện tại kiểm tra engine với 20.003 JPEG 320×320, preflight hai lần và hai job imgsz cấu hình 301 epoch qua fake runtime; tổng test riêng 9,53 giây gồm tạo fixture, preflight và fake jobs. Không đo thời gian train YOLO thật hoặc RSS ở quy mô này.

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

### 2026-10-25 — Xóa ảnh cùng JSON

- **Thay đổi:** Xóa ảnh (user request 2026-10-25): nút thùng rác trên thanh ảnh hoặc Shift+Delete trong tab Gán nhãn hỏi xác nhận rồi gọi `delete_images` (`engine/frameforge/deletion.py`): chuyển ảnh và JSON cạnh nó vào Thùng rác hệ thống (Windows `SHFileOperationW` với FOF_ALLOWUNDO, macOS `~/.Trash`) trước, chỉ khi thành công mới xóa ảnh khỏi dự án (box/candidate cascade, `ai_inference_attempts`, `sidecar_state`), rồi chuyển sang ảnh kế tiếp. Không xóa vĩnh viễn.
- **Kiểm tra:** engine 83 test pass (test xóa dùng thùng rác giả: file vào thùng rác, dự án mất ảnh và box; ảnh ngoài dự án hoặc thùng rác lỗi thì không đổi gì); `cargo test` 4 pass; `tsc` sạch; gọi `SHFileOperationW` thật với một tệp tạm đã vào Recycle Bin. Chưa click-through.

### 2026-10-25 — Mở lại ảnh làm việc cuối

- **Thay đổi (user request):** tab Gán nhãn nhớ ảnh đang mở theo từng dự án đã lưu (`localStorage` `frameforge.last-image.v1:<đường dẫn .ffproj>`) và mở lại ảnh đó khi vào tab lần sau, kể cả sau khi tắt app; ảnh không còn trong dự án thì về ảnh đầu. `tsc` sạch, build lại portable; chưa click-through.

### 2026-10-25 — Box tự vẽ không cần review, sửa Enter

- **Vấn đề (user report):** Enter không đánh dấu review được (bị bỏ qua khi focus ở bất kỳ nút nào, ví dụ nút class hoặc dòng ảnh vừa bấm).
- **Thay đổi:** Box do người vẽ/sửa không cần review (user request 2026-10-25): lưu là `reviewed`, ảnh có box và không còn gợi ý AI/candidate chờ duyệt tự thành `reviewed`; export đưa box thủ công vào dataset bất kể trạng thái review cũ (`unreviewed_annotation_excluded_count` luôn 0). Chỉ gợi ý AI cần duyệt. Enter hoạt động cả khi focus đang ở nút: duyệt mọi gợi ý AI đang chờ của ảnh, đánh dấu ảnh reviewed (ảnh trống = xác nhận âm tính) rồi sang ảnh chưa xong; bị chặn khi còn candidate ảnh trống chờ xác nhận.
- **Kiểm tra:** engine 82 test pass (test export đổi thành `test_dataset_export_includes_manual_boxes_without_review`); `tsc` sạch. Chưa click-through.

### 2026-10-25 — Ảnh zoom sắc nét

- **Vấn đề (user report):** ảnh zoom mờ so với X-AnyLabeling: preview gán nhãn là JPEG chất lượng 82 (nhòe màu/artefact trên head ~7 px), ảnh > 1600 px bị thu nhỏ, và CSS `scale` phóng bitmap cỡ vừa khung với làm mịn dưới 3×.
- **Thay đổi:** Ảnh tab Gán nhãn sắc nét như X-AnyLabeling: `preview_frame` với `lossless: true` trả PNG đủ độ phân giải, đúng từng pixel (tab Trích xuất vẫn JPEG ≤1600 px và báo kích thước nguồn cho overlay crop); ảnh được vẽ ở đúng kích thước hiển thị (không CSS `scale` bitmap cỡ vừa khung), căn giữa bằng `position: absolute`, và hiển thị nearest-neighbor (`image-rendering: pixelated`) khi phóng ≥ 1 pixel màn hình/pixel ảnh.
- **Kiểm tra:** engine 82 test pass (`test_labeling_preview_keeps_every_pixel`: PNG giải mã bằng đúng ảnh nguồn 2000×1800; preview trích xuất vẫn JPEG ≤1600 px, báo kích thước nguồn); `cargo test` 4 pass; `tsc` sạch. Chưa so sánh trực quan trong app.

### 2026-10-25 — Bấm chọn box khi đang vẽ

- **Thay đổi (user request):** khi công cụ vẽ đang bật, bấm (di chuyển dưới 4 px màn hình) vào một box sẽ chọn box đó để resize bằng handle; kéo vẫn vẽ box mới, kể cả bắt đầu bên trong box khác; kéo bên trong box đang chọn (không trúng handle) di chuyển cả box ở mọi công cụ. `tsc` sạch, build lại portable; chưa click-through.

### 2026-10-25 — Sửa phím A/D, kéo ảnh bằng chuột phải, resize box vừa vẽ

- **Vấn đề (user report):** A/D chỉ chạy một lần (phím bắt trên section; canvas bị gỡ khi tải ảnh mới nên mất focus); resize box vừa vẽ lúc được lúc không (công cụ vẽ vẫn bật nên bấm vào handle lại vẽ box mới, và handle có thể bị box vẽ sau che).
- **Thay đổi:** phím tắt gán nhãn nghe trên `window` (bỏ qua input/select/dialog và phím mũi tên canvas đã xử lý); handle resize ưu tiên hơn công cụ vẽ, handle của box đang chọn vẽ trên cùng, vùng bấm 12 px; chuột phải kéo ảnh và tắt context menu trên ảnh. `tsc` sạch; chưa click-through.

### 2026-10-25 — Hiệu năng dự án lớn và zoom ảnh

- **Thay đổi:** Dự án lớn (mục tiêu 100.000+ ảnh, user request 2026-10-25): Rust giữ tối đa 3 engine chạy nền (`EnginePool`) cho lệnh ngắn thay vì khởi động engine ~0,7 s mỗi lệnh; tác vụ hủy được (`extract_frames`, `predict_crops`, `auto_label_project`) vẫn chạy tiến trình riêng. Bảng `annotations`/`ai_negative_candidates` có index theo `image_path`. Đồng bộ JSON đọc mtime/size bằng một lần liệt kê mỗi thư mục, chỉ đọc file đổi, và chỉ ghi ảnh có digest nhãn trong DB khác lần ghi trước (`sidecar_state.mtime_ns/size/db_digest`). Quét thư mục nhập bằng `os.scandir`, không `resolve`/`is_file` từng tệp; không đưa danh sách đường dẫn khổng lồ vào `IN (...)`. Danh sách ảnh tab Gán nhãn ảo hóa (dòng cố định 49 px), không có phép tìm O(n²). Khi dự án đã lưu và không có thay đổi chưa lưu, snapshot phiên là "light": không gửi ảnh, không băm `.ffproj`; khởi động mở lại dự án gốc. Zoom ảnh 25%–3200%: lăn chuột zoom quanh con trỏ, `+`/`-`/`0`, nút; chuột phải, chuột giữa hoặc Space + kéo để di chuyển ảnh (menu chuột phải bị tắt trên ảnh); ảnh hiển thị pixel sắc nét khi một pixel ảnh lớn hơn 3 pixel màn hình.
- **Đo:** Đo trên máy người dùng (2026-10-25, thư mục tổng hợp 100.000 PNG 640×640 trong %TEMP%, 30.000 JSON một box; engine Python trong `.venv`, không qua app): nhập 39,6 s (chạy thử 3,3 s), mở lại khi không đổi 2,2 s, `load_project` 1,0 s (phản hồi 55 MB), `save_project` 1,5 s, ghi lại toàn bộ JSON khi không đổi 2,1 s. Trước tối ưu: nhập 292,6 s, mở 6,9 s, `load_project` 2,0 s, `save_project` 5,2 s, ghi lại 12,5 s; nhập >32k ảnh còn lỗi `too many SQL variables` (đã sửa). Dự án thật 4.454 ảnh trước tối ưu: mỗi lệnh engine 0,7 s, mở dự án 3,7 s.
- **Kiểm tra:** engine 81 test pass (thêm test phiên light); `cargo test` 4 pass, `cargo build` không cảnh báo; `tsc` sạch. Chưa đo trong app đóng gói và chưa click-through zoom/danh sách ảo.

### 2026-10-25 — Mở app vào tab chính và mở lại dự án gần nhất

- **Vấn đề (user report):** mỗi lần mở app nhảy sang tab Dataset (khôi phục `active_tab` của snapshot một dự án trống chưa lưu); thêm thư mục ảnh bằng nút Thư mục ở tab Trích xuất chỉ tạo nguồn cắt frame nên `640_yellow` không vào dự án.
- **Thay đổi:** app luôn mở ở tab Trích xuất; snapshot không có dự án/ảnh/nguồn thì mở lại dự án `.ffproj` gần nhất (`localStorage` `frameforge.last-project.v1`, ghi khi mở/tạo/lưu); tab khác Trích xuất không còn tự tạo snapshot. Chọn/kéo thả thư mục chỉ có ảnh hỏi "Thêm để gán nhãn" (đi qua `import_images`) hay "Nguồn cắt frame"; kéo thả hoặc chọn file `.ffproj` thì mở dự án. `tsc` sạch, build lại portable; chưa click-through.

### 2026-10-25 — Điều hướng gán nhãn và thư mục frame mặc định

- **Thay đổi:** Danh sách ảnh lọc được theo Tất cả / Chưa xong / Đã review / Có gợi ý AI chờ duyệt / Có box của từng class; phím `A`/`D` (hoặc PageUp/PageDown) ảnh trước/sau trong danh sách đang lọc, `N` ảnh chưa xong kế tiếp, `Enter` đánh dấu đã review rồi sang ảnh chưa xong kế tiếp; chữ `A`, `D`, `N` không dùng làm phím class. Trích xuất video không cần chọn thư mục khi dự án đã lưu: mặc định ghi vào `<tên dự án>_frames` cạnh file `.ffproj` (engine tạo thư mục; allowlist Rust nhận thư mục chưa tồn tại nếu thư mục cha có sẵn), vẫn đổi được bằng nút chọn thư mục. yolo-train `build_model.py` mặc định `--export-formats onnx` và in đường dẫn `.onnx` để import vào SysX.
- **Kiểm tra:** `tsc --noEmit` sạch; `cargo test` 4 pass; engine 80 test pass; `build_model.py` chỉ kiểm tra cú pháp (chưa chạy build thật vì cần `.venv-train` và GPU). Chưa click-through trong app.

### 2026-10-25 — Nhãn lưu JSON cạnh ảnh

- **Thay đổi:** `engine/frameforge/sidecar.py` (ghi/đọc JSON, bảng `sidecar_state`), lệnh `import_images` (nhiều thư mục, `dry_run`), hook đồng bộ trong `frameforge_engine.py`, nút **Nhập ảnh** chọn nhiều thư mục và chỉ hỏi Đã review khi có JSON ngoài; lỗi ghi JSON hiện trên banner sau khi lưu nhãn.
- **Kiểm tra:** engine 80 test pass (`test_labelme.py` 10 test: đọc JSON khi nhập mà không sửa file, ghi JSON sau khi lưu nhãn và giữ trường/shape ngoài, JSON sửa bên ngoài thắng khi mở, JSON bị xóa được ghi lại, gợi ý AI không vào `shapes`, dự án cũ ghi nhãn ra JSON); `cargo test` 4 pass; `tsc --noEmit` sạch. Dự án thật `640_yellow.ffproj`: nén 1.464 JSON gốc vào `C:\Users\CiyE\Documents\dataset\640_yellow_json_backup_2026-10-25.zip` (kiểm tra zip), đồng bộ lần đầu ghi 1.463 JSON trong 6,7 giây, lần sau 2,2 giây không ghi gì; nhãn trong dự án không đổi. `yolo-train/labelme_to_yolo.prepare` trên thư mục mới dùng 1.463 ảnh, bỏ 2.991 ảnh chưa gán.

### 2026-10-25 — Nhập nhãn LabelMe/X-AnyLabeling và phím tắt class

- **Thay đổi:** `engine/frameforge/labelme.py` + lệnh `import_labelme_folder` (allowlist Rust kiểm tra dự án, thư mục, `mark_reviewed` boolean); nút **Nhập nhãn** trên toolbar lưu dự án nếu cần, hỏi Đã review/Chưa review, nhập rồi nạp lại ảnh qua `load_project`. `saveProject` trả về đường dẫn đã lưu. Kích thước ảnh đọc từ header PNG/JPEG, định dạng khác giải mã bằng OpenCV. Phím `1`–`9`/chữ cái đầu đổi class box đang chọn hoặc chọn class để vẽ.
- **Kiểm tra:** `.venv` (Python 3.14.7, numpy 2.5.4, OpenCV 4.14.0, pytest 8.4.2, không cài torch): engine 76 test pass (6 test mới trong `test_labelme.py`); `cargo test` 3 pass (cần `engine/dist` tồn tại, tạo rỗng tạm thời); `tsc --noEmit` sạch. Nhập thật `C:\Users\CiyE\Documents\dataset\640_yellow` vào dự án tạm: 4.454 ảnh, 1.463 ảnh có 1.480 box (toàn head), 2.991 ảnh chưa có JSON, 1 JSON không có ảnh (`train_crop_20260728_201224_932_000088`), 57 nhóm split, 5,2 giây. Chưa click-through trong app đóng gói.
- **Chuyển dữ liệu (user request: từ nay chỉ gán nhãn bằng FrameForge, không ghi ngược ra JSON):** tạo `C:\Users\CiyE\Documents\dataset\640_yellow.ffproj` và nhập `640_yellow` với Đã review: 4.454 ảnh, 1.463 ảnh/1.480 box head đã review, 2.991 ảnh chưa gán; JSON không bị sửa (so SHA-256 trước/sau). `validate_dataset`: hợp lệ, 1.463 ảnh vào dataset (train/val/test 1.230/122/111, 18 nhóm), cảnh báo class body chưa có box.
- **Build Windows portable (2026-10-25, máy người dùng):** `.venv` Python 3.14.7 + `engine[inference,package]` (PyTorch 2.14.1+cpu, Ultralytics 8.4.187, PyInstaller 6.22.3); `package_engine.py` pass smoke health/inference (engine 917 MB); `tsc` + `vite build`; `tauri build --no-bundle` với `beforeBuildCommand` rỗng vì máy không có `pnpm` trong PATH; `stage_windows_portable.py` ra `src-tauri/target/release/FrameForge` (922 MB). `import_labelme_folder` chạy đúng trên engine đóng gói với fixture nhỏ. Chưa mở cửa sổ app để click-through; inference chạy CPU (không DirectML/ROCm).

### 2026-10-25 — Tách công cụ train khỏi repo

- **Thay đổi:** Toàn bộ script train/tối ưu/xuất model và `config.yml` chuyển sang `C:\Users\CiyE\Desktop\yolo-train`. Đổi tên các phần mang tên app: project mặc định `runs`, model `runs/models`, tune `runs/tune`, log `progress.log`, marker `yolo_train_export.json`. Launcher mặc định dùng Python `frameforge/.venv-train` (đổi bằng `YOLO_TRAIN_PYTHON` hoặc `-Python`). Các mục nhật ký 2026-10-24/25 bên dưới về `scripts/train_yolo.py` … mô tả lịch sử trước khi tách.
- **Kiểm tra:** tại thư mục mới, với torch/ultralytics giả: `train_rocm.cmd check`; `build -DryRun` trên `640_yellow` (1.454 ảnh có nhãn lúc chạy, head 1.471, body 0); `build_model.py` giả xuất `.pt`/`.json`/`config.yml`/`build_summary.json` và `progress.log`. Repo FrameForge không còn file train trong `scripts/`.

### 2026-10-25 — Gộp train/tối ưu/xuất model và config.yml

- **Thay đổi:** `scripts/build_model.py` gọi `train_yolo.run_training` (tách từ `main`), `optimize_model.evaluate`/`measure_latency`/`benchmark`/`run_tune`, rồi xuất `.pt` + `.json` + `config.yml` + `build_summary.json` vào `runs/frameforge/models/<thời điểm>/`; model khuyến nghị chọn theo mAP50-95 trong `--max-latency-ms`. `scripts/config.yml` giữ class ID; `labelme_to_yolo.load_class_names` kiểm tra ID liên tục, tên không rỗng/trùng; `train_yolo` chặn `data.yaml` lệch config. Bỏ ràng buộc với app: IoU mặc định 0,5/0,6/0,7, latency mặc định đo trên GPU, bỏ lời nhắc nhập model trong output. Launcher thêm `build`/`build-finetune`, `-Config`, `-MaxLatencyMs`, `-TuneIterations`.
- **Lỗi tìm được khi kiểm thử:** `measure_latency` trả danh sách rỗng khi số ảnh < số lần warmup (đã sửa, thêm kiểm tra rỗng); file export không được chép sang thư mục model vì benchmark không lưu đường dẫn (đã thêm `artifact`).
- **Kiểm tra:** `py_compile`; với Python 3.14 tạm và `ultralytics`/`torch` giả (mô phỏng train + callback, val, predict, export): `build_model.py train --imgsz 640 256` xuất đủ `.pt`/`.json`/`config.yml`/`build_summary.json`, chọn 640 trong ngân sách 3 ms; `--export-formats onnx openvino` chép `.onnx`, bỏ qua OpenVINO thiếu gói kèm lệnh pip; config đảo ID bị chặn; `train_rocm.cmd build -DryRun` trên `640_yellow` thật (1.428 ảnh có nhãn lúc chạy: train 1.115 / val 313; `data.yaml` `0: "head"`, `1: "body"`). Chưa chạy trên GPU/model thật.

### 2026-10-25 — Script tối ưu latency và độ chính xác

- **Thay đổi:** `scripts/optimize_model.py` với `evaluate` (val conf 0,001 cho đủ đường cong PR; conf F1 tốt nhất lấy từ `box.px`/`f1_curve` trung bình các class có mặt), `benchmark` (latency end-to-end từng ảnh có warmup, median/p95/FPS, thời gian preprocess/inference/postprocess; export ONNX/OpenVINO từ bản sao trong thư mục optimize; validate lại mAP từng format), `all` (latency mọi imgsz trên thiết bị đích, chọn mAP50-95 cao nhất trong ngân sách, format nhanh nhất giảm mAP50-95 ≤ 0,005), `tune` (`model.tune`, AdamW, `cache=False`). `train_yolo.py` thêm `--hyp` (chỉ nhận key hyperparameter) và `resolve_data_path`.
- **Kiểm tra:** `py_compile`; chạy `all`, `evaluate`, `tune` và `train_yolo.py --hyp --dry-run` với Python 3.14 tạm cùng `ultralytics`/`torch` giả: chọn imgsz 640 trong ngân sách 5 ms (800/960 vượt), conf 0,30, ONNX nhanh nhất, OpenVINO thiếu gói bị bỏ qua kèm lệnh pip, `report.md`/`report.json` được ghi, key lạ trong hyp bị bỏ qua. Đã đối chiếu API trong Ultralytics 8.4.187 của `.venv-train` (`YOLO_AUTOINSTALL`, `Metric.px/f1_curve/p_curve/r_curve/ap_class_index/ap`, `Tuner` → `best_hyperparameters.yaml`) bằng cách đọc mã nguồn. **Chưa chạy trên model/GPU thật.**

### 2026-10-25 — Train trực tiếp từ thư mục LabelMe `640_yellow`

- **Dataset `C:\Users\CiyE\Documents\dataset\640_yellow` (đang gán nhãn dở):** 4.454 PNG 640×640 phẳng cùng JSON X-AnyLabeling 4.2.0. Lúc kiểm tra có 1.392–1.397 JSON (tăng trong lúc người dùng gán), toàn bộ box là rectangle `head` (1.405–1.410 box), **chưa có `body`**; 1 JSON trỏ tới ảnh không tồn tại; 10 ảnh trùng byte (đều nằm trong nhóm chưa gán); box trung vị 7,3 × 10,2 px, nhỏ nhất 1,8 × 3,2 px; không có box vượt biên.
- **Thay đổi:** `scripts/labelme_to_yolo.py` chuyển sang YOLO (head 0, body 1), hardlink ảnh vào `images/train|val`, ghi label UTF-8, `data.yaml` và marker `frameforge_labelme_export.json`; split theo cụm 10 phút từ tên file + SHA-256 trùng. `train_yolo.py --data <thư mục>` tự chuyển trước khi train và cảnh báo khi box trung vị sau khi thu nhỏ < 6 px (ở 256 chỉ còn ~2,9 px). Launcher `train_rocm.ps1` tách giá trị bị `powershell -File` gộp bằng dấu phẩy và thêm `-PreparedDir`/`-ValRatio`.
- **Kiểm tra:** chuyển dataset thật vào scratchpad (1.391 ảnh dùng: train 1.180 / val 211, 17 nhóm; label `0 0.518125 ...`; thư mục nguồn giữ nguyên 5.846 file). Dry-run trọn đường `train_rocm.cmd` → `train_yolo.py` → chuyển LabelMe với torch/ultralytics giả: 640 batch 16, 256 batch 64 kèm cảnh báo box nhỏ, `--patience 0` qua `-Extra`, fine-tune `-Weights "640=...","256=..."` tách đúng, `-Imgsz 320` bị từ chối. Chưa train thật.

### 2026-10-25 — Launcher train ROCm cho Windows

- **Thay đổi:** `scripts/train_rocm.ps1` (UTF-8 BOM) và `scripts/train_rocm.cmd` (ASCII, `-ExecutionPolicy Bypass` chỉ cho tiến trình này). Launcher ghi đoạn kiểm tra GPU ra file `.py` tạm (PowerShell 5.1 làm hỏng ngoặc kép trong `-c` và chèn BOM vào stdin), yêu cầu `torch.version.hip` khác rỗng và `torch.cuda.is_available()`, in tên GPU/VRAM/gfx arch/HIP, rồi chuyển tham số sang `train_yolo.py`; `check` chỉ kiểm tra. `runs/` thêm vào `.gitignore`.
- **Kiểm tra:** chạy `.cmd` bằng Python 3.14 tạm cùng module `torch`/`ultralytics` giả qua `PYTHONPATH`: thiếu torch, torch CPU, ROCm `check`, thiếu `-Data`, fine-tune thiếu `-Weights`, `train -DryRun` (640 batch 16, 256 batch 64, `smoke-640`/`smoke-256`). Không chạy với torch ROCm cộng đồng trong `.venv-train`.

### 2026-10-25 — Script train imgsz 640 và 256

- **Thay đổi:** `scripts/train_yolo.py` hỗ trợ `--imgsz` thuộc {640, 256}, mặc định chạy lần lượt cả hai thành run riêng `<tên>-<imgsz>`. Batch mặc định theo VRAM và kích thước: GPU 8 GB → 16 ở 640, 64 ở 256 (tối đa 64 vì 4 worker CPU). Fine-tune nhận `--weights SIZE=PATH` hoặc một checkpoint chung, kiểm tra task/class của mọi checkpoint trước khi train và cảnh báo khi imgsz checkpoint khác đích. `--resume` tiếp tục theo từng `<tên>-<imgsz>/weights/last.pt`. Train mới từ `yolo11n.pt` (alias `yolov11n.pt`). `ProgressTracker` gắn callback Ultralytics (`on_train_start`, `on_train_epoch_start`, `on_fit_epoch_end`, `on_train_end`) để in thanh tiến độ run/tổng, thời gian epoch, đã chạy, ETA (ước tính kích thước chưa chạy theo thời gian epoch hiện tại), mAP, và ghi `frameforge_progress.log`; dừng sớm thì tổng epoch được thu lại.
- **Kiểm tra:** `py_compile`; bằng Python 3.14 không có torch: mặc định [640, 256], batch 16/64, worker 4, tên run, ánh xạ `--weights` theo kích thước và dùng chung, các lỗi thiếu checkpoint/nhiều checkpoint chung/imgsz không hỗ trợ (320); mô phỏng callback tiến độ bằng trainer giả cho 640 (dừng sớm ở 2/3) và 256, file log được ghi. Agent không chạy `.venv-train` vì torch ROCm trong đó là bản dựng cộng đồng; người dùng tự chạy `--dry-run` và train. **Chưa train thật.**

### 2026-10-24 — Gallery toàn bộ ảnh và script train ngoài app

- **Gallery:** `ProjectGallery` hiển thị mọi ảnh của project, chỉ render vùng nhìn thấy (bước 120 px) và tải thumbnail theo lô ≤64 qua `image_thumbnails` (engine + Tauri allowlist); cache tối đa 3.000 thumbnail. Thumbnail của lượt trích vừa xong được dùng sẵn.
- **Train:** `scripts/train_yolo.py` (chạy ngoài app) có `train` (từ `yolo11n.pt`) và `finetune` (checkpoint, kiểm tra class khớp dataset, AdamW lr0 0.001, freeze 10, 50 epoch); imgsz cố định 640, batch theo VRAM (8 GB → 16), worker = min(4, CPU/2), AMP, `cache=False`, `--resume`, `--dry-run`, `--hsa-override`. Dùng GPU qua `torch.cuda`, dừng nếu không có GPU trừ khi `--allow-cpu`.
- **Phần cứng máy này:** AMD Radeon RX 6600 (gfx1032, 8 GB), i3-12100F 4 nhân/8 luồng, 15,9 GB RAM, Windows 10, Python 3.14. Index nightly ROCm chính thức `gfx103X-dgpu` không có wheel torch; Ultralytics không hỗ trợ DirectML để train. Đường GPU khả thi: Linux + ROCm (`HSA_OVERRIDE_GFX_VERSION=10.3.0`) hoặc wheel ROCm cộng đồng cho Windows (Python 3.12, không chính thức).
- **Môi trường train (người dùng tự chạy):** `.venv-train` dùng Python 3.12.10 từ python.org cùng Ultralytics 8.4.187 từ PyPI. Wheel ROCm 7.1.1 cộng đồng (guinmoon/rocm7_builds, GFX103X-all) do người dùng tự tải và tự cài với `--no-index --no-build-isolation`. Kết quả người dùng báo: `pip check` sạch; torch `2.9.1+rocmsdk20251207`, HIP `7.2.53150`, `torch.cuda.is_available()` = True, thiết bị `AMD Radeon RX 6600`. Chưa chạy train/dry-run nên chưa xác nhận RX 6600 train được.
- **Kiểm tra:** engine pytest — 70 passed (thêm test thumbnail); `pnpm build`, `cargo fmt --check`, `cargo check --locked`, `cargo test --lib` — pass; script: `py_compile`, `--help`, lỗi thiếu data.yaml/thiếu Ultralytics và cấu hình tính được (batch 16, worker 4) cho GPU 8 GB. **Chưa chạy train thật** và chưa click-through gallery trong app Tauri.

### 2026-10-24 — Sửa lỗi audit pipeline

- **Thay đổi:** `set_project_classes` cho phép dự án chưa có annotation/candidate dùng class của model (UI hỏi ngay khi nhập và có nút trong panel Model; panel hiện cả khi chưa có ảnh). `extract_frames` trả `images` cho mọi ảnh đã ghi kèm cờ `overwritten` và `extraction_settings` chung; frontend thêm tất cả vào project, xóa nhãn/candidate/lượt inference của ảnh bị ghi đè. Bảng `ai_inference_attempts` (tạo idempotent, không đổi schema version 4) ghi lượt chạy mà mọi box chạm biên crop; `save_ai_suggestions` nhận `skipped_large_objects`, batch bỏ qua cặp checkpoint + fingerprint đã chạy (`skipped_attempted`). Ngưỡng confidence chỉnh được, nút Duyệt/Loại tất cả, bỏ Đã review đưa box thủ công về `unreviewed`, `predict` nhận `imgsz` từ registry.
- **Kiểm tra:** engine pytest — 69 passed (thêm test extraction vượt giới hạn gallery và ghi đè, đổi class trước/sau khi có nhãn, lượt inference chạm biên ở ảnh đơn và batch, `imgsz`); `pnpm build` — thành công; `cargo fmt --check`, `cargo check --locked`, `cargo test --lib` — pass; dispatcher JSON-lines nhận `set_project_classes`.
- **Chưa xác minh:** chưa click-through UI trong app Tauri, chưa chạy Ultralytics/checkpoint thật, chưa build lại portable app.

### 2026-10-24 — Thu gọn về pipeline gán nhãn

- **Thay đổi:** Gỡ toàn bộ phần liên quan tới train model. Xóa tab Huấn luyện, `TrainingWorkspace`, `ProjectDiscovery`, module engine `training.py`/`discovery.py`, các command training/evaluation/Champion-Challenger/discovery/ONNX inspect trong engine và Tauri allowlist, cùng lineage/replay/đóng băng val-test trong `dataset.py` và tab Dataset. Registry và trust flow nhập checkpoint chuyển sang `engine/frameforge/models.py` (`inference_capabilities`, `list_models`, `import_yolo_checkpoint`). Tab Gán nhãn có nút “Nhập model .pt” rõ ràng. Extra Python `train` đổi thành `inference` (torch + ultralytics); bỏ `onnxruntime`, `psutil`, `lap`; script đóng gói smoke-check `inference_capabilities`.
- **Kiểm tra:** engine pytest — 64 passed (venv tạm với numpy/OpenCV/Pillow/PyYAML/pytest, không có torch/ultralytics; test inference dùng fake runtime); `pnpm build` — tsc + Vite thành công; `cargo fmt --check`, `cargo check --locked`, `cargo test --lib --locked` — 2 passed.
- **Chưa xác minh:** Chưa build lại portable app hoặc chạy `scripts/package_engine.py` với runtime mới; chưa click-through UI hay chạy với checkpoint YOLO11n thật. Manifest dataset không còn trường `version`/`parent_*`/`fixed_validation_set`/`frozen_test_set`.

### 2026-10-14 — YOLO11n mặc định và kế thừa imgsz khi fine-tune

- **Thay đổi:** Giữ YOLO11n pretrained `yolo11n.pt` làm nguồn tạo run mới khi không chọn checkpoint fine-tune. Khi chọn checkpoint nội bộ/đã nhập đã xác minh, tự chọn `imgsz` của run/checkpoint; importer ưu tiên metadata checkpoint, đọc `args.yaml` cùng run nếu thiếu và chỉ fallback 640 khi cả hai không có. Toast nhập checkpoint và ghi chú Huấn luyện hiển thị kích thước/nguồn.
- **Kiểm tra:** `./.venv/bin/python -m pytest engine/tests -q` — 110 passed; `pnpm build` — thành công với 1.922 module; `cargo fmt --check` và `cargo check` — thành công; `git diff --check` — sạch.
- **Giới hạn:** Checkpoint và `args.yaml` không lưu `imgsz` thì kích thước train gốc không thể khôi phục; 640 chỉ là mặc định Ultralytics, không phải kết luận về lịch sử model. Registry cũ thiếu nguồn được đánh dấu chưa xác định. Chưa click-through UI.

### 2026-10-14 — Nhập thư mục làm việc đã chuyển

- **Thay đổi:** Đổi Discovery thành luồng nhập workspace. Path POSIX/Windows cũ được đối chiếu trong thư mục người dùng chọn; engine tạo `data.yaml` riêng và file list local có giới hạn, giữ ảnh/nhãn nguồn nguyên trạng. Preflight đọc metadata import để từ chối workspace bị di chuyển, đường dẫn ngoài scope và ảnh/nhãn qua symlink. UI có thao tác nhập checkpoint riêng với trust flow.
- **Kiểm tra:** `./.venv/bin/python -m pytest engine/tests -q` — 106 passed trong 11,01 giây; `pnpm build` bằng Node workspace — thành công, 1.922 module; `cargo fmt --check`, `cargo check` và `git diff --check` — thành công.
- **Giới hạn:** UI mới được xác minh bằng TypeScript/Vite build, chưa click-through trong app bundle. Chưa nạp checkpoint ngoài thật; checkpoint được phát hiện vẫn không tự động tin cậy. Video đầu vào giữ giới hạn FHD 1920×1080, không scale.

### 2026-10-14 — Khôi phục dataset khi data.yaml nằm ngoài thư mục dataset

- **Thay đổi:** Khi `path` cũ trong YAML là đường dẫn tuyệt đối và YAML nằm ngoài dataset, tìm trong workspace thư mục duy nhất khớp các split khai báo. Từ chối trường hợp nhiều kết quả, symlink, hoặc lượt tìm kiếm vượt giới hạn độ sâu/số mục; không sửa YAML, ảnh, nhãn hay checkpoint nguồn. Scan chỉ liệt kê `.pt` và giữ trạng thái cần trust.
- **Kiểm tra:** `./.venv/bin/python -m pytest engine/tests -q` — 108 passed; fixture end-to-end xác nhận scan → chọn dataset → preflight và checkpoint vẫn chưa được hash/nạp. Chưa click-through trong app bundle.

### 2026-10-14 — Nhập checkpoint YOLO11n có xác nhận tin cậy

- **Thay đổi:** Thêm import `.pt` cho YOLO11n detection sau xác nhận cảnh báo mã thực thi. Engine kiểm tra kích thước không quá 2 GiB, task, kiến trúc/scale, class names và SHA-256 trước/sau khi nạp/copy; lưu file trong thư mục riêng, hạn chế quyền, giữ tên/hash/đường dẫn nguồn và đăng ký vào model registry. UI Gán nhãn hiển thị tên checkpoint nhập; Huấn luyện cho phép fine-tune khi class schema trùng dataset. Run tìm thấy bằng Smart Folder Discovery vẫn không được nạp tự động.
- **Kiểm tra:** `pnpm test:engine` — 91 passed; `pnpm build` — thành công; `cargo fmt --check` — thành công; `cargo check` — thành công; `git diff --check` — sạch.
- **Giới hạn:** Importer/fine-tune integration dùng fake Ultralytics runtime; chưa có tệp YOLO11n ngoài phù hợp để click-through inference hoặc training model thật. UI chỉ được kiểm tra bằng TypeScript/Vite build trong lượt này. Giới hạn video giữ nguyên ở FHD 1920×1080; không thêm tùy chọn video độ phân giải cao.
- **Git:** Commit `cefe0ca` (`feat(training): import trusted YOLO11n checkpoints`) đã lưu trên `feature/manual-labeling`. Push tới `origin/feature/manual-labeling` tiếp tục bị chặn do thiếu HTTPS username/credentials (`could not read Username for 'https://github.com': Device not configured`).
- **Bước tiếp:** Dùng checkpoint YOLO11n thật đã nhập hoặc train để kiểm tra luồng inference/review/export/evaluation trong app.

### 2026-10-14 — Bỏ qua object vượt biên crop trong inference

- **Thay đổi:** Bỏ detection box nếu một cạnh nằm cách biên ảnh không quá 1 px; đây có thể là object bị crop hoặc lớn hơn vùng crop. Engine trả số box đã bỏ qua. Nếu toàn bộ box bị bỏ qua, không tạo negative candidate; ảnh vẫn cần người dùng review. UI ảnh hiện tại và batch project hiển thị số lượng.
- **Kiểm tra:** `pnpm test:engine` — 83 passed trong 10,71 giây; `pnpm build` — thành công với TypeScript/Vite; `git diff --check` — thành công. Lệnh dùng Node và Python runtime đóng gói workspace.
- **Giới hạn:** Kiểm tra biên là heuristic; object hợp lệ thực sự chạm mép ảnh cũng bị bỏ qua. Engine behavior được kiểm tra bằng fake YOLO runtime; chưa click-through với checkpoint thật vì registry local không có checkpoint usable.
- **Git:** Commit local `d129744` (`fix(inference): skip crop-edge detections`). Push tới `origin/feature/manual-labeling` thất bại: `could not read Username for 'https://github.com': Device not configured`.

### 2026-10-14 — Siết lọc ảnh gần trùng bằng dHash

- **Thay đổi:** Hạ ngưỡng mặc định từ 5 xuống 1 bit trong extraction và negative sampling. Chỉ bỏ crop gần như trùng; tăng ngưỡng mới lọc mạnh hơn. Quota negative 10% là quy tắc lấy mẫu riêng, không do dHash đặt ra.
- **Kiểm tra:** `git diff --check` — thành công. Không chạy test hoặc build trong lượt này.

### 2026-10-14 — Làm rõ ngưỡng dHash

- **Thay đổi:** Giải thích ngay dưới thanh trượt rằng ngưỡng là số bit Hamming tối đa giữa hai hash 64-bit, không phải số ảnh. Hiển thị số crop gần giống thực tế bị lọc trong gallery. Không đổi thuật toán: so sánh với tối đa 4.096 crop đã giữ gần nhất.
- **Kiểm tra:** `./.venv/bin/python -m pytest engine/tests -q` — 81 passed trong 11,34 giây; TypeScript + Vite production build qua Node runtime đóng gói — thành công, 1.922 modules.
- **Giới hạn:** Số lượng bị lọc phụ thuộc nội dung video và tần suất lấy mẫu; 5 chỉ là ngưỡng giống nhau. Không thêm độ phân giải video cao hơn FHD.

### 2026-10-14 — Batch soak fixture FHD tổng cộng 4 giờ

- **Fixture:** Tạo H.264 1920×1080, 30 fps, dài một giờ; sao chép thành bốn đường dẫn đầu vào và xử lý trong cùng một batch. Crop tâm 320×320, lấy một frame mỗi giây, tắt lựa chọn lọc blur/duplicate; mã vẫn tính điểm blur và dHash cho crop.
- **Kết quả:** Giải mã ước tính 432.000 frame, xuất đủ 14.400/14.400 crop trong 519,16 giây; 0 nguồn lỗi, 0 lỗi ghi, tốc độ tương đương 27,74× thời lượng video. Tổng JPEG 48.064.436 byte; peak RSS 892,3 MiB. Tạo fixture mất 158,6 giây và không tính trong throughput.
- **Giới hạn:** Bốn nguồn là bản sao cùng một fixture ít chuyển động và cùng codec. Một batch chạy 8,7 phút chưa xác nhận soak nhiều giờ theo thời gian thực, hành vi trên footage người dùng hoặc hiệu năng codec khác. Không mở rộng giới hạn nguồn: chỉ dùng FHD.
- **Git:** Ghi kết quả vào commit tài liệu local `docs: record four-hour FHD extraction benchmark`. Push branch `feature/manual-labeling` lên `origin` vẫn bị chặn do thiếu GitHub HTTPS username.

### 2026-10-14 — Champion/Challenger và xác nhận giới hạn FHD

- **Thay đổi:** Thêm state Champion/Challenger trong training registry; validation mới thành Challenger, test không được promote. So sánh yêu cầu cùng fixed validation/fingerprint, class schema, inference/runtime/hardware; xác minh checksum, target tổng/theo class và regression (mặc định tolerance 0). Promote cần thao tác xác nhận; rollback khôi phục Champion trước và lưu lịch sử. Giữ video nguồn tối đa FHD 1920×1080, từ chối nguồn lớn hơn và không scale; `imgsz` training vẫn độc lập.
- **Kiểm tra:** `./.venv/bin/python -m pytest engine/tests -q` — 81 passed trong 10,65 giây; `pnpm build` qua Node runtime đóng gói — thành công (1.922 modules); `cargo fmt --check`, `cargo check` qua Rust toolchain local — thành công; `git diff --check` — thành công.
- **UI/chưa xác minh:** Mở tab Huấn luyện trong app macOS dev, xác nhận PyTorch 2.14.1, Ultralytics 8.4.177 và MPS sẵn sàng. Registry local có 0 checkpoint nội bộ; do đó chưa click-through đánh giá/promote model thật. Benchmark media tổng hợp mới nhất xử lý batch 4 giờ nguồn trong 519,16 giây; vẫn chưa chạy nhiều giờ trên video thực.
- **Git:** Commit local (`feat: add champion challenger model promotion`) trên `feature/manual-labeling`. Push lên `origin/feature/manual-labeling` bị chặn: `could not read Username for 'https://github.com': Device not configured`.
- **Bước tiếp:** Xác minh trên checkpoint nội bộ thật khi có model khả dụng; sau đó triển khai benchmark/export deployment. Không thêm tuỳ chọn video vượt FHD.

### 2026-10-14 — Fixed-set model evaluation

- Thêm evaluator trong tab Huấn luyện, chỉ nhận checkpoint FrameForge nội bộ đã đăng ký và fixed validation/test từ manifest FrameForge. Trước khi load model, kiểm tra class schema, split membership, SHA-256 toàn bộ ảnh, hash ảnh/label fixed set, cùng leakage fingerprint/group giữa train và evaluation.
- Đo Precision, Recall, F1 (từ Precision/Recall), mAP50, mAP50–95 và metric theo class. Lưu checkpoint/dataset/manifest/fixed-set fingerprints, imgsz, confidence, NMS IoU, batch, thiết bị/hardware, framework, cảnh báo, mục tiêu tổng/per-class và lịch sử vào training registry.
- Mục tiêu quality/deployment là tùy chọn, không đặt ngưỡng mặc định. Checkpoint size được đo; latency/P50/P95/P99/FPS/throughput/memory vẫn `not_measured` đến khi có benchmark triển khai thật. Đánh giá có thể hủy ở ranh giới batch và không lưu kết quả dở dang.
- Kiểm tra lượt này: `./.venv/bin/python -m pytest engine/tests -q` — 75 passed; `pnpm build` — thành công; `cargo fmt --check` và `cargo check` — thành công. Shell không có Node/Cargo trên PATH ban đầu; chạy lại với runtime workspace và Rust toolchain local.
- Chưa xác minh: Ultralytics evaluation bằng checkpoint thật và app bundle click-through; registry local không có checkpoint usable. Frontend chỉ được production-build kiểm tra.
- Git: commit `d1b57c3` (`feat: add fixed-set model evaluation`) đã lưu local trên `feature/manual-labeling`. Push với `git push --set-upstream origin feature/manual-labeling` thất bại do môi trường không cung cấp GitHub HTTPS credentials.
- Bước tiếp: chạy evaluator trên checkpoint nội bộ thật, sau đó xây so sánh Champion/Challenger và promotion history.

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
- **Chưa xác minh:** Chưa click-through discovery/training trên app bundle mới. Session API chưa nối với autosave/restore trong frontend. Inference, auto-label, đánh giá model, resume, export và Windows còn nằm trong roadmap phát triển.
- **Bước tiếp:** Tạo YOLO11n inference adapter chỉ nhận checkpoint đã xác minh, kèm test đầu ra box/confidence/class trên fixture ảnh nhỏ.

### 2026-10-14 — Giới hạn video FHD và benchmark video dài

- **Thay đổi:** Chỉ nhận video có kích thước tối đa 1920×1080. Import, preview và extraction từ chối video lớn hơn, không tự scale; thông báo lỗi đầu tiên được hiển thị trong UI. Giới hạn crop/output FHD được giữ nguyên.
- **Benchmark:** Trên máy macOS hiện tại, engine giải mã fixture H.264 tổng hợp 1920×1080, 30 fps, dài 3.600 giây (108.000 frame); lấy mẫu mỗi giây và xuất crop tâm 320×320 JPG, tắt blur/duplicate filter. Extraction mất 132,27 giây (27,22× nhanh hơn thời lượng clip), xử lý 3.600/3.600 crop, không lỗi; tổng JPG 16,5 MB. RSS process đo được tăng từ 581,4 lên tối đa 807,3 MiB (182 lần lấy mẫu progress). Tệp fixture 10,7 MB; tạo fixture mất 159,53 giây và không tính vào thời gian extraction.
- **Kiểm tra:** `pytest engine/tests -q` — 45 passed; TypeScript và Vite production build — thành công; `git diff --check` — thành công.
- **Giới hạn phép đo:** Nội dung fixture tổng hợp ít chuyển động; kết quả chỉ xác nhận một lượt xử lý video FHD dài một giờ trên máy này, không dự đoán tốc độ cho mọi codec/nội dung và không phải stress test chạy liên tục nhiều giờ. Chưa click-through thông báo từ chối trong app bundle.

### 2026-10-14 — Inference project, negative review và training dataset lớn

- **Thay đổi:** Thêm inference toàn project: nạp model một lần, predict batch 16 ảnh, ghi theo chunk 64 ảnh, kiểm fingerprint ở ranh giới commit và giữ chunk đã lưu khi hủy. Ảnh không có detection trở thành candidate negative có provenance; chỉ export sau khi người dùng xác nhận ảnh trống. SQLite schema v4 migrate từ v1/v2/v3.
- **Training:** Preflight chấp nhận hơn 20.000 ảnh và trên 300 epoch cho mỗi `imgsz`; đặt Patience = 0 để tắt early stopping. Test tổng hợp tạo 20.003 JPEG 320×320 rồi kiểm tra hai job 301 epoch bằng fake trainer trong 9,53 giây. Đây chỉ là xác nhận plumbing/cấu hình, không đo train YOLO thật ở quy mô đó.
- **Giới hạn video:** Chỉ nhận nguồn video tối đa FHD 1920×1080; nguồn vượt chiều rộng hoặc chiều cao bị từ chối tại import, preview và extraction, không tự scale. Nguồn nhỏ hơn giữ nguyên độ phân giải gốc.
- **Kiểm tra lượt này:** `./.venv/bin/python -m pytest engine/tests -q` — 63 passed trong 10,19 giây; `pnpm build` — thành công; `cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check`, `cargo check --manifest-path src-tauri/Cargo.toml` và `git diff --check` — thành công.
- **Chưa xác minh:** Chưa build lại app bundle hoặc click-through inference UI bằng checkpoint thật. Phép đo media một giờ ở mục trước là một lượt extraction, không phải soak test nhiều giờ. Train YOLO thật trên hơn 20.000 ảnh trong hơn 300 epoch vẫn chưa được chạy.
- **Git:** Commit `f6ad23c` (`feat: add batch inference and large dataset support`) đã lưu local. Push `origin feature/manual-labeling` thất bại vì terminal không có GitHub HTTPS username (`could not read Username for 'https://github.com': Device not configured`).
- **Bước tiếp:** Dataset lineage/replay và frozen test set, rồi evaluation cố định và Champion/Challenger.

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

### 2026-10-14 — Dataset lineage, replay và fixed evaluation sets

- **Thay đổi:** Export có manifest lineage/version và parent SHA-256; val/test chỉ đóng băng khi cả hai split có ảnh và mọi ảnh đã review. Child xác minh ảnh, nhãn, split, class schema và hash của parent, giữ nguyên byte ảnh cùng labels của fixed val/test, chia dữ liệu mới theo nhóm, và hỗ trợ replay train `all`, `balanced`, `new`. Training preflight, queue và run config nhận dataset version, parent, replay policy và fixed-set IDs.
- **Giới hạn FHD:** Video đầu vào vẫn chỉ tối đa 1920×1080. Không thêm tuỳ chọn video độ phân giải cao hơn; nguồn vượt giới hạn bị từ chối, không scale. `imgsz` training là cấu hình riêng.
- **Kiểm tra:** `./.venv/bin/python -m pytest engine/tests -q` — 68 passed; `pnpm build` — thành công; `cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check`, `cargo check --manifest-path src-tauri/Cargo.toml` và `git diff --check` — thành công.
- **Chưa xác minh:** Dataset controls đã qua TypeScript/Vite build nhưng chưa click-through trong app bundle. Fixed-set evaluation/Champion-Challenger chưa triển khai. Benchmark một giờ là một lượt extraction trên fixture H.264 tổng hợp FHD; không phải soak test nhiều giờ hoặc mọi codec. Train YOLO thật trên hơn 20.000 ảnh trong hơn 300 epoch chưa chạy.
- **Git:** Commit `7cbae6d` (`feat(dataset): add lineage and frozen evaluation sets`) đã lưu local trên `feature/manual-labeling`. Push bị chặn do GitHub HTTPS yêu cầu username nhưng phiên này không cung cấp (`could not read Username for 'https://github.com': Device not configured`).
- **Bước tiếp:** Triển khai evaluation trên fixed validation/test sets và lưu metric theo từng run trước khi xây Champion/Challenger.

### 2026-10-14 — Tự lưu và khôi phục workspace

- **Thay đổi:** Frontend tự lưu snapshot cục bộ sau debounce và khôi phục project, cài đặt, nguồn/frame đang chọn, thư mục output và tab sau health check. Không hiện hộp thoại phục hồi. Snapshot dùng quyền riêng; nếu `.ffproj` gốc mất hoặc SHA-256 đổi thì mở nội dung phục hồi thành project chưa lưu để tránh ghi đè.
- **FHD:** Giữ nguyên phạm vi đầu vào tối đa 1920×1080. Video lớn hơn bị từ chối ở import/preview/extraction, không scale và không thêm lựa chọn độ phân giải cao hơn. Benchmark gần nhất vẫn là 4 giờ nguồn tổng hợp H.264 FHD, không phải soak test nhiều giờ theo thời gian thực.
- **Kiểm tra:** `pnpm test:engine` — 93 passed; `pnpm build` — thành công; `git diff --check` — sạch. Chưa click-through UI trên app bundle mới.
- **Git:** Commit milestone được lưu local trên `feature/manual-labeling`; remote GitHub chưa xác thực được trong workspace.

### 2026-10-14 — Chặn export và replay nhãn thủ công chưa review

- **Thay đổi:** Dataset validation/export loại toàn bộ ảnh còn box thủ công `unreviewed` và báo số ảnh bị loại. Replay parent bỏ ảnh train cũ có nhãn thủ công hoặc negative chưa review; parent có fixed validation/test chứa box thủ công chưa review bị từ chối.
- **Kiểm tra:** `./.venv/bin/python -m pytest engine/tests -q` — 96 passed trong 10,53 giây; `pnpm build` — thành công với 1.922 module bằng Node.js runtime đi kèm Codex; `git diff --check` — sạch.
- **Phạm vi video:** Giữ nguyên giới hạn nguồn tối đa FHD 1920×1080; video lớn hơn bị từ chối ở import, preview và extraction, không scale và không thêm lựa chọn độ phân giải cao hơn. `imgsz` training vẫn riêng biệt.
- **Chưa xác minh:** Chưa click-through UI trên app bundle mới. Benchmark media vẫn là fixture H.264 FHD tổng hợp, không phải soak test nhiều giờ trên footage thực.

### 2026-10-14 — Negative sampling cho dataset

- **Thay đổi:** Dataset validation/export có giới hạn negative mặc định 10%, preset 0/5/10/25% hoặc tùy chỉnh 0–100%. Seed cho kết quả tái lập; dHash ≤5 bỏ ảnh negative gần giống. Negative đã thuộc fixed validation/test của parent luôn được giữ và report cảnh báo khi vượt quota.
- **Phạm vi video:** Chỉ nhận nguồn tối đa FHD 1920×1080; video lớn hơn bị từ chối tại import/preview/extraction, không scale và không thêm lựa chọn độ phân giải cao hơn. `imgsz` training vẫn độc lập.
- **Hiệu năng media:** Giữ nguyên benchmark fixture tổng hợp trước đó: bốn bản sao H.264 FHD, tổng 4 giờ nguồn, 14.400 crop trong 519,16 giây, peak RSS 892,3 MiB. Fixture lặp không thay thế soak test nhiều giờ trên footage thực.
- **Kiểm tra:** `./.venv/bin/python -m pytest engine/tests -q` — 98 passed trong 11,11 giây; TypeScript/Vite production build — thành công (1.922 modules); `cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check`, `cargo check --manifest-path src-tauri/Cargo.toml` và `git diff --check` — thành công. Regression xác nhận quota 0% vẫn giữ negative trong fixed val/test.
- **Chưa xác minh:** Chưa click-through giao diện Dataset trên app bundle mới.

## Quy trình thay đổi

Đọc `AGENTS.md`, `PRODUCT.md`, `DESIGN.md` và mục liên quan trong file này trước khi sửa. Sau thay đổi đáng kể, chạy kiểm tra phù hợp, ghi kết quả thật, cập nhật tài liệu và xem lại diff. Không commit môi trường `.venv`, `node_modules`, cache, media người dùng hoặc model weights.

## Windows handoff

Windows chưa được build hoặc xác nhận. Các việc còn lại: xác nhận native dialogs/paths, đóng gói Python và DLL của OpenCV/ONNX Runtime, build Tauri, kiểm tra Windows 10/11, tích hợp DirectML và benchmark trên RX 6600. Chỉ ghi `TESTED_WINDOWS_CI` hoặc `VERIFIED_WINDOWS_HARDWARE` sau đúng loại kiểm thử đó.

## Rà soát repository — 2026-10-14

- Project settings có `settings_version`; khi mở project/session legacy, ngưỡng dHash mặc định 5 được chuyển về 1 đúng một lần. Giá trị người dùng chọn sau đó được giữ qua các lần mở tiếp theo; `extraction_settings` của ảnh cũ không bị thay đổi.
- Repository dùng GNU AGPL-3.0-or-later do tích hợp Ultralytics; README dẫn tới điều khoản hiện hành cho người cần phân phối thương mại hoặc đóng nguồn.
- Xóa `agent.md` và `TASKS.md` vì nội dung lặp lại hướng dẫn chuẩn trong `AGENTS.md` và trạng thái/roadmap trong `DEVELOPMENT.md`.
- GitHub Actions chạy build frontend, engine tests và Rust formatting cho pull request và push lên `main`.
- Kiểm tra local: `pnpm test:engine` — 100 passed; `pnpm build` — thành công; `cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check` — thành công; `cargo check --manifest-path src-tauri/Cargo.toml` — thành công; `git diff --check` — thành công.
- Hosted GitHub Actions của PR #3 — `verify` passed trong 51 giây, gồm frontend build, engine tests và Rust formatting.

### 2026-10-14 — Rà soát source và merge PR #1

- **Source:** Worktree ban đầu sạch tại `8807a5d`; không có thay đổi source chưa commit cần dọn. Rà danh sách file tracked không thấy cache/build, media/model hoặc mẫu token/private key thông dụng. Lượt này chỉ chỉnh tài liệu hướng dẫn publish và ghi nhận trạng thái.
- **GitHub:** Đã cài GitHub CLI arm64 chính thức, xác minh SHA-256 và đăng nhập thành công. Nhánh `feature/manual-labeling` đã push lên `origin`; PR #1 được squash-merge vào `main` (merge commit `1d90c1b`). `main` được đặt làm default branch và local checkout đã fast-forward theo remote. Tại thời điểm merge, PR ở trạng thái `CLEAN`, không có CI checks hoặc branch protection.
