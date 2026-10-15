# Phát triển FrameForge

## Trạng thái hiện tại

| Khả năng | Trạng thái | Ghi chú |
|---|---|---|
| Tauri 2 + React/TypeScript workspace | IMPLEMENTED | Giao diện tiếng Việt theo ảnh tham chiếu |
| Python JSON-lines IPC | TESTED_MACOS | Bản app bundle hiện tại chuyển từ `Đang kết nối` sang `Engine sẵn sàng`; health JSON-lines trực tiếp trả `status: ok` |
| Nhập media, metadata, preview, timeline | TESTED_MACOS | Dialog, kéo thả, OpenCV đọc ảnh/video thật; video tối đa 1920×1080, nguồn lớn hơn bị từ chối; seek bằng tác vụ nền |
| Exact Center Crop, preview overlay | TESTED_MACOS | Preview dùng frame nguồn tối đa FHD và vẽ khung cam liền nét 320×320 tại tọa độ nguồn x=800, y=380 trên fixture 1920×1080 |
| Trích frame, lọc crop, letterbox/grid, JPG/PNG | TESTED_MACOS | Fixture video; blur/dHash tính trên crop nguồn; ngưỡng dHash mặc định ≤1 bit để chỉ bỏ crop gần như trùng, so với tối đa 4.096 crop được giữ gần nhất; gallery báo số crop thực tế bị lọc; progress/hủy |
| Dự án SQLite `.ffproj` | TESTED_ENGINE | Create/save/load; schema v1/v2/v3 migrate lên v4 khi ghi; annotation và negative candidate AI lưu provenance/fingerprint; autosave bảo toàn provenance |
| Migration ngưỡng dHash project/session | TESTED_ENGINE | Giá trị legacy 5 chuyển về 1 một lần; version marker giữ lựa chọn mới; `extraction_settings` của ảnh không đổi |
| Nạp và kiểm tra metadata ONNX | IMPLEMENTED | Rust xác thực phần mở rộng; engine chạy session CPU; chưa có model fixture để xác nhận model cụ thể |
| YOLO11n inference adapter | TESTED_ENGINE | Chỉ nạp checkpoint YOLO11n đã đăng ký/khớp SHA-256 và class schema; gồm model train nội bộ hoặc checkpoint đã qua trust flow; Ultralytics NMS một lần, confidence filter inclusive, bỏ box sát biên crop; fake-runtime test |
| Auto-label ảnh hiện tại, batch project và candidate negative | IMPLEMENTED_UNVERIFIED | Nút Auto label trên toolbar mở tab Gán nhãn; inference ảnh hiện tại/batch project, progress, review và candidate negative dùng engine thật. Engine test batch nhiều chunk, fingerprint recheck và export; box sát biên không tạo negative. Chưa click-through inference vì registry không có checkpoint khả dụng. |
| Gán nhãn thủ công | IMPLEMENTED | Vẽ/sửa/di chuyển/resize/xóa box Head (0), Body (1); undo/redo, review, autosave SQLite; lưu dự án trước khi chỉnh sửa |
| Dataset validation, negative sampling, grouped splits, YOLO export and lineage | TESTED_ENGINE | Negative sampling theo seed (mặc định 10%, 0–100%), loại ảnh gần như trùng bằng dHash ≤1; fixed val/test giữ negative kể cả khi vượt quota; export loại box/candidate AI đang chờ hoặc bị loại |
| Đánh giá model trên fixed validation/test | IMPLEMENTED_UNVERIFIED | UI, engine và Rust validation đã nối; lưu metric tổng/per-class, target, conditions và history; fake-runtime tests bao phủ hash/leakage/trust/cancel. Chưa xác minh bằng checkpoint nội bộ thật; registry local hiện không có model usable. |
| Champion/Challenger và promotion history | IMPLEMENTED_UNVERIFIED | Engine so sánh fixed validation cùng fingerprint/schema/điều kiện/phần cứng, chặn checksum/target/regression, yêu cầu promote thủ công và hỗ trợ rollback; fake-runtime tests pass. App chạy MPS nhưng registry không có checkpoint để click-through model thật. |
| Smart Folder Discovery | IMPLEMENTED | Quét giới hạn có progress/hủy; fixture xác nhận YAML dataset, run, metrics, symlink, cancel và đường dẫn ngoài scope; `.pt` chỉ được liệt kê, không nạp |
| Nhập thư mục làm việc đã chuyển | TESTED_ENGINE | Nối lại path POSIX/Windows cũ kể cả khi YAML nằm ngoài dataset, chỉ khi có duy nhất thư mục khớp mọi split; tạo `data.yaml` local, chuẩn hóa file list có giới hạn; preflight xác minh workspace nguồn, ảnh/nhãn và symlink; yêu cầu chọn thư mục output riêng để train |
| Registry run ngoài | IMPLEMENTED | Metadata chỉ lưu khi người dùng chọn; checkpoint được phát hiện giữ `checkpoint_trusted=false` cho đến khi nhập riêng qua trust flow |
| Import model YOLO bên ngoài | TESTED_ENGINE | Native dialog cảnh báo `.pt` có thể thực thi mã và yêu cầu xác nhận; giới hạn 2 GiB, sao chép riêng và xác minh checksum trước khi nạp bản sao, kiểm tra detection/YOLO11n/class rồi xác minh lại; fake-runtime regression tests pass |
| YOLO11n training trên CPU/MPS và queue nhiều imgsz | TESTED_MACOS | Smoke một epoch YOLO11n thật trên fixture nhỏ chạy bằng MPS Apple M1 Pro. Kiểm tra synthetic 20.003 ảnh 320×320 và chuyển cấu hình 301 epoch tới hai imgsz bằng fake runtime; chưa xác nhận throughput/độ ổn định YOLO thật ở quy mô đầy đủ. |
| Registry checkpoint và fine-tuning | TESTED_ENGINE | Fine-tune chỉ nhận checkpoint YOLO11n detection đã đăng ký, khớp checksum tệp/record/registry và class schema dataset; khi chọn checkpoint, UI tự điền imgsz từ checkpoint hoặc args.yaml cùng run. Nếu cả hai thiếu metadata, dùng mặc định Ultralytics 640; registry cũ thiếu provenance hiển thị nguồn chưa ghi. Không chọn checkpoint thì run mới dùng YOLO11n pretrained. |
| Khôi phục phiên làm việc | TESTED_ENGINE + BUILT_UI | Snapshot riêng tư lưu project và trạng thái workspace; frontend tự lưu sau debounce, tự khôi phục không hiện hộp thoại. Nếu dự án gốc mất hoặc checksum đổi, snapshot mở thành dự án chưa lưu. Engine có regression tests; UI đã build nhưng chưa click-through trên app bundle mới. |
| Resume run sau khi dừng/gián đoạn | NOT_STARTED | Chưa có thao tác tiếp tục tiến trình từ checkpoint gần nhất. |
| Benchmark triển khai và model export | PARTIAL | Fixed-set evaluation và Champion/Challenger đã triển khai; latency/FPS/throughput/memory benchmark và deployment export chưa triển khai. Media batch benchmark FHD 4 giờ là phép đo extraction, không phải model benchmark. |
| Windows 10/11, DirectML, RX 6600, đóng gói | PENDING_WINDOWS_VALIDATION | Workflow hiện build Windows portable folder; cần CI xanh cho commit này và cài/chạy click-through trên Windows thật. DirectML và RX 6600 chưa được kiểm chứng. |
| CI frontend/engine/Rust đa nền tảng | PENDING_CURRENT_CHANGE | CI cũ `38564201938` chỉ xác nhận build MSI trước khi chuyển sang portable. Workflow mới build app macOS và Windows folder nhưng chưa có run kết quả cho thay đổi hiện tại. CI không xác nhận GPU/DirectML hoặc cài đặt trên máy người dùng. |
| Đóng gói macOS portable | TESTED_MACOS | `pnpm build:portable` tạo `FrameForge.app` 1,44 GiB có Python/PyTorch/Ultralytics bundle. Engine health/training self-check pass; app mở được, UI hiện Engine sẵn sàng. Chưa kiểm tra trên máy Mac sạch hoặc ký/notarize. |
| Đóng gói Windows portable folder | BUILT_WINDOWS_CI | Workflow CI dựng thư mục `FrameForge` gồm app executable và engine runtime, không dùng installer; cần xác minh artifact CI mới và cài/chạy click-through trên máy Windows thật. |
| Python runtime và dependencies trong app portable | TESTED_MACOS | Runtime được bundle cùng app; app không yêu cầu người dùng cài Python riêng. PyTorch/Ultralytics self-check pass trên macOS hiện tại. |
| Ký phát hành macOS/Windows | RELEASE_BLOCKER | Chưa cấu hình notarization macOS hoặc ký executable Windows; app portable chưa sẵn sàng phân phối công khai. |

Thứ tự tiếp theo: dùng checkpoint YOLO11n thật đã nhập hoặc train để click-through inference → manual review → dataset export → fixed-set evaluation/Champion-Challenger. Sau đó tiếp tục deployment benchmark/export, resume training và Windows validation. Engine suite xác minh trust flow; registry local hiện chưa có checkpoint thật để kiểm tra luồng end-to-end. Video đầu vào tối đa FHD (1920×1080); nguồn lớn hơn bị từ chối, không scale hoặc thêm lựa chọn độ phân giải cao. `imgsz` là cấu hình training, không phải lựa chọn độ phân giải video.

## Cổng phát hành v1.0

**Trạng thái: chưa production-ready; giữ phiên bản `0.1.0`.** App portable macOS đã mở được trên máy phát triển này; app sạch và Windows thật chưa xác minh. Trước khi gắn tag hoặc phát hành `1.0.0`, cần đóng các cổng sau:

- Thử mở app portable macOS trên máy sạch, không cài Python; ký và notarize app trước khi phân phối công khai.
- Xác minh Windows portable folder từ CI bằng cài/chạy click-through trên máy Windows thật. DirectML/RX 6600 vẫn cần kiểm chứng riêng trên phần cứng đích.
- Nhận CI xanh mới cho thay đổi hiện tại; cấu hình ký Windows portable executable trước khi phân phối tải xuống công khai.
- Click-through inference/review/export/evaluation bằng checkpoint YOLO11n thật và dataset đã review; không dùng fixture để kết luận chất lượng model.
- Chạy soak extraction bằng video FHD thực, nhiều codec/nội dung và thời lượng dài hơn batch fixture tổng hợp đã ghi ở trên.

Thay đổi trong lượt rà soát này: nhập checkpoint `.pt` bằng bản sao riêng đã xác minh trước khi nạp, kiểm tra lại sau khi nạp và giới hạn đọc/sao chép 2 GiB; Rust chỉ giữ tối đa 256 KiB stderr cho mỗi request nhưng vẫn drain pipe. Sửa Windows để ghi hash label trên đúng byte UTF-8 và dùng đường dẫn Unicode khi đọc/ghi ảnh. Runtime PyInstaller được bundle cho app portable, workflow CI chuyển từ build MSI sang app macOS và Windows folder. Nút Auto label mở tab Gán nhãn để dùng inference thật. Bản macOS hiện tại 1,44 GiB, engine health/training self-check pass, app mở với Engine sẵn sàng; frontend build, 110 engine tests, Rust fmt/check đều pass. Windows thật, máy Mac sạch, ký/notarize, checkpoint thật và soak footage thật vẫn là cổng mở.

## Đo hiệu năng media

Batch tổng hợp 4 giờ nguồn FHD: bốn bản sao của cùng một fixture H.264 dài một giờ, mỗi nguồn 1920×1080 ở 30 fps. Engine giải mã xấp xỉ 432.000 frame, lấy mẫu mỗi giây và ghi 14.400 crop tâm 320×320 JPG. Một tiến trình hoàn tất trong 519,16 giây (27,74× tổng thời lượng nguồn), không có lỗi; output 48.064.436 byte và peak RSS 892,3 MiB. Fixture tạo mất thêm 158,6 giây, không tính vào thời gian extraction. RSS mẫu tăng nhẹ từ khoảng 855 MiB lên peak 892,3 MiB trong suốt batch.

Đây là một lượt chạy khoảng 8,7 phút với nội dung tổng hợp lặp lại và một codec. Kết quả xác nhận batch dài theo thời lượng nguồn; chưa xác nhận soak nhiều giờ theo thời gian thực, video người dùng, nội dung đa dạng hoặc codec khác. Đây là media extraction benchmark, không phải model deployment benchmark.

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

Các command được allowlist gồm health, inspect_sources, inspect_model, preview_frame, preview_center_crop, extract_frames, create_project, save_project, save_annotations, save_ai_suggestions, review_ai_negative_candidate, auto_label_project, validate_dataset, export_yolo_dataset, load_project, training_capabilities, inspect_training_dataset, list_training_runs, import_yolo_checkpoint, evaluate_yolo_checkpoint, list_model_evaluations, get_evaluation_targets, save_evaluation_targets, register_discovered_run, prepare_discovered_dataset, scan_project_folder, train_yolo_queue và predict_crops. Rust kiểm tra command, phần mở rộng, kích thước và đường dẫn trước khi gọi Python. Import `.pt` cần `trust_source=true`; Python giới hạn file 2 GiB, xác minh SHA-256 trước/sau khi nạp và sao chép, rồi chỉ đăng ký YOLO11n detection có class hợp lệ. Train có thể dừng tại ranh giới epoch; extraction, discovery, inference ảnh đơn/batch và evaluation hủy theo lô. Evaluation chỉ persist sau khi hoàn tất.

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
- `predict_crops` chỉ dùng checkpoint YOLO11n nội bộ đã đăng ký, còn khớp SHA-256 và có class schema khớp với model. Nó trả box `xyxy` theo pixel trên ảnh gốc; NMS do Ultralytics thực hiện và confidence cuối được lọc theo `score >= threshold`. Box cách biên crop không quá 1 px bị bỏ qua vì có thể là object bị cắt hoặc lớn hơn khung.
- `save_ai_suggestions` xác minh lại checkpoint, class schema project và fingerprint ảnh; box được lưu `needs_review` với model ID/checkpoint SHA-256/confidence. Batch project nạp model một lần, predict theo batch 16 và lưu từng chunk 64 ảnh trong transaction; fingerprint được recheck trước commit, ảnh đổi bị bỏ qua, hủy giữ phần đã lưu. Kết quả chỉ gồm box bị bỏ qua sát biên không tạo candidate âm tính; ảnh thực sự không có detection tạo candidate chờ người review.
- Dataset validator đọc lại ảnh thật, kiểm tra kích thước so với project, class ID và box hữu hạn/còn trong biên. Chỉ box thủ công, AI `reviewed` và negative candidate được người dùng xác nhận mới được export; pending/rejected bị loại, ảnh pending không trở thành negative dù có trạng thái reviewed. Manifest ghi provenance candidate.
- Split theo source gom toàn bộ frame/crop cùng video; split theo segment dùng timestamp hoặc frame/FPS. Ảnh có SHA-256 trùng nội dung luôn được gộp cùng nhóm split. Khi số nhóm độc lập ít hơn số split được bật, báo cảnh báo và chỉ gán các nhóm có sẵn.
- YOLO export tạo thư mục phiên bản mới nguyên tử dưới thư mục người dùng đã chọn, sao chép ảnh gốc và xác minh SHA-256 sau khi sao chép; nhãn dùng `class x_center y_center width height` chuẩn hóa 8 chữ số thập phân. `manifest.json` lưu mapping class, split, fingerprint, hình học crop và provenance nguồn. Export không sửa project/ảnh nguồn.
- Smart Folder Discovery dùng `yaml.safe_load`, chỉ quét trong root được chọn, bỏ symlink, giới hạn depth/files/metadata/CSV rows, không đọc pixel hoặc nội dung checkpoint. YAML split/data trỏ ra ngoài root bị chặn; ngoại lệ quyền truy cập tạo kết quả partial.
- Nhập workspace chuyển máy tạo `data.yaml` trong App Data và chuẩn hóa file list (tối đa 2 MiB mỗi tệp, 16 MiB tổng và 100.000 dòng/tệp); ảnh/nhãn vẫn nằm tại nguồn. Nếu YAML ở ngoài dataset, engine tìm thư mục khớp các split; chỉ chấp nhận một kết quả duy nhất và dừng nếu workspace mơ hồ hoặc tìm kiếm bị giới hạn. Preflight đọc `imported-workspace.json`, từ chối nguồn đã mất, ảnh/label ngoài thư mục đã chọn và mọi đường dẫn ảnh/label đi qua symlink. Khi chọn dataset import, không tự dùng thư mục staging làm output; người dùng chọn thư mục lưu run.
- Run ngoài chỉ được lưu metadata khi nhấn “Lưu run vào registry”. Registry đánh dấu checkpoint ngoài chưa tin cậy và không chọn nó làm model fine-tune; muốn dùng phải nhập qua trust flow riêng, kiểm tra task/kiến trúc YOLO11n, class, kích thước và SHA-256 trước/sau sao chép.
- Khi không chọn checkpoint đã đăng ký, training dùng YOLO11n pretrained `yolo11n.pt` để tạo run mới. Khi chọn run/checkpoint đã xác minh, UI tự chọn `effective_imgsz` (hoặc `requested_imgsz`); importer đọc `imgsz` từ checkpoint, rồi từ `args.yaml` cạnh checkpoint nếu thiếu, sau cùng mới dùng mặc định 640 và gắn nguồn. Kích thước vẫn có thể đổi trước khi chạy.
- Python engine được đóng gói cùng nguồn, nhưng runtime và dependency Python vẫn cần cài riêng khi chạy ứng dụng. Standalone sidecar packaging chưa làm.

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
