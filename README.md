# FrameForge

FrameForge là ứng dụng desktop local-first để tạo bộ dữ liệu object detection, gán nhãn và huấn luyện YOLO11n. Workspace hỗ trợ nhập video/ảnh, preview frame thật, Exact Center Crop đúng pixel, quét thư mục tìm dataset/run/model, xuất YOLO và queue training riêng theo từng imgsz. Chỉ nhận video tối đa FHD 1920×1080; nguồn lớn hơn bị từ chối ở import, preview và extraction, không có tuỳ chọn video độ phân giải cao hơn và không tự scale. Nguồn nhỏ hơn giữ nguyên độ phân giải gốc.

Huấn luyện YOLO11n trên CPU hoặc backend MPS/CUDA mà PyTorch xác nhận; dataset trên 20.000 ảnh và cấu hình trên 300 epoch cho từng imgsz được hỗ trợ. Mỗi kích thước train chạy thành một job độc lập, không nạp toàn bộ ảnh vào RAM (`cache=False`), progress/checkpoint được ghi trong lúc chạy. Muốn chạy đủ số epoch cấu hình, đặt Patience = 0 để tắt early stopping. Engine đã kiểm tra 20.003 ảnh JPEG 320×320 và chuyển cấu hình 301 epoch tới hai job ở hai imgsz; fake runtime chỉ xác nhận xử lý dataset và cấu hình, không đo throughput YOLO thật ở quy mô này. Fine-tune/inference chỉ nhận checkpoint YOLO11n nội bộ có registry và checksum còn khớp; `.pt` tìm thấy bên ngoài không được nạp. Tab Gán nhãn có thể chạy inference cho ảnh đang mở hoặc theo project; box AI và candidate âm tính đều phải được người dùng duyệt. Candidate âm tính chỉ được export sau khi người dùng xác nhận ảnh trống; ảnh còn chờ duyệt không được coi là negative. Engine persistence/export đã qua kiểm thử; app bundle đã build nhưng chưa click-through inference bằng checkpoint thật vì registry trên máy này chưa có model khả dụng. Tab Huấn luyện đã có đánh giá trên fixed validation/test, mục tiêu chất lượng tổng thể/theo class và lịch sử điều kiện đo; lượt này dùng fake runtime để kiểm tra evaluator vì chưa có checkpoint nội bộ khả dụng cho smoke test YOLO thật. Champion/Challenger, benchmark latency/FPS/RAM và export triển khai chưa khả dụng; ONNX Runtime hiện chỉ đọc tensor/metadata.

## Yêu cầu

- Node.js tương thích với Vite 8 và pnpm
- Rust stable cùng công cụ build desktop của hệ điều hành
- Python 3.10 trở lên

## Cài đặt môi trường Python

Từ thư mục repository:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e "engine[dev]"
pnpm install
```

Trên Windows, dùng `.venv\Scripts\python.exe` thay cho `.venv/bin/python`. Có thể đặt `FRAMEFORGE_PYTHON` để chọn một Python interpreter khác đã cài các dependency trong `engine/pyproject.toml`.

Để dùng tab Huấn luyện, cài PyTorch, Ultralytics và psutil bằng `.venv/bin/python -m pip install -e "engine[train]"` (Windows: `.venv\Scripts\python.exe -m pip install -e "engine[train]"`).

## Chạy ứng dụng

```sh
pnpm tauri dev
```

Vite sử dụng cổng `1420`. Tauri gọi tiến trình Python qua JSON-lines và kiểm tra đường dẫn/lệnh ở Rust trước khi chạy.

App bundle hiện đóng gói các module Python của `frameforge` cùng engine. Python runtime và dependencies vẫn cần có sẵn trên máy; standalone sidecar chưa được triển khai.

## Kiểm tra và build

```sh
pnpm test:engine
pnpm build
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo check --manifest-path src-tauri/Cargo.toml
pnpm tauri build
```

`pnpm test:engine` chạy pytest trên fixture media tổng hợp nhỏ. `pnpm build` kiểm tra TypeScript và build frontend. Các lệnh Tauri build trên hệ điều hành hiện tại; build thành công trên macOS không xác nhận Windows.

## Cấu trúc

- `src/` — React/TypeScript workspace
- `src-tauri/` — ứng dụng Tauri 2 và cầu nối Rust an toàn
- engine/ — Python/OpenCV/ONNX Runtime, trích xuất, lưu dự án, kiểm tra dataset và YOLO11n training
- `docs/design/` — hợp đồng giao diện cho workspace trích xuất
- `PRODUCT.md`, `DESIGN.md`, `DEVELOPMENT.md` — phạm vi, hướng thiết kế và trạng thái phát triển

## Trạng thái

Đã triển khai media preview/trích xuất, Exact Center Crop, lọc crop, gán nhãn, dataset validation/grouped split/YOLO export, lineage/replay dataset, Smart Folder Discovery, YOLO11n training và fixed-set evaluation. Chỉ nhận video tối đa FHD 1920×1080; video lớn hơn bị từ chối ở import, preview và extraction, không scale và không có lựa chọn độ phân giải cao hơn. Nguồn nhỏ hơn giữ nguyên độ phân giải. Crop/output tùy chỉnh giới hạn rộng 1920, cao 1080 px. Preview giữ frame nguồn và vẽ viền cam liền nét quanh vùng W×H đúng tâm; khi W=H, viền là hình vuông. Với video FHD và crop 320×320, vùng bắt đầu tại x=800, y=380. Đổi preset chỉ đổi kích thước ô trên cùng frame; preview không thay frame bằng ảnh crop, ảnh crop đầu ra chỉ được tạo khi trích xuất. Quét thư mục đọc metadata trong phạm vi người dùng chọn, có giới hạn/progress/hủy; `.pt` ngoài không được nạp và run ngoài chỉ lưu vào registry khi người dùng bấm lưu. Checkpoint fine-tune/inference/evaluation phải thuộc run YOLO11n detection nội bộ đã đăng ký và còn khớp SHA-256. Tab Dataset có thể đóng băng val/test sau khi mọi ảnh được review; phiên bản con xác minh parent manifest, giữ nguyên ảnh và nhãn val/test, đồng thời hỗ trợ replay train `all`, `balanced` hoặc `new`. Training queue/run ghi dataset ID/version, parent, replay và fixed-set IDs. Tab Huấn luyện kiểm tra dataset và backend CPU/MPS/CUDA thật, chạy mỗi imgsz thành run độc lập, ghi progress/metrics/log/checkpoint, chấp nhận dataset trên 20.000 ảnh và cấu hình hơn 300 epoch; dừng an toàn tại ranh giới epoch. `imgsz` là kích thước ảnh dùng trong training, không nới giới hạn độ phân giải video đầu vào. Đặt Patience = 0 nếu cần tắt early stopping và chạy đủ số epoch đã nhập. Engine đã kiểm tra 20.003 JPEG tổng hợp 320×320 và chuyển cấu hình 301 epoch tới hai job ở hai imgsz; fake runtime chỉ xác nhận xử lý dataset và cấu hình, không đo thời gian hay độ ổn định YOLO thật ở quy mô này. Fixed-set evaluation xác minh manifest, hash ảnh/label, class schema và rò rỉ split trước khi đánh giá; lưu Precision/Recall/F1/mAP50/mAP50–95, metric theo class, target và điều kiện chạy. Để trống target không tạo ngưỡng mặc định; latency/FPS/RAM hiện là `not_measured`. Tab Gán nhãn lưu gợi ý inference theo project; box AI chỉ xuất sau khi duyệt, candidate âm tính chỉ xuất sau khi xác nhận ảnh trống (SQLite schema v4). Model pretrained cache trong app data; kết quả lưu ở thư mục người dùng chọn. App mở thẳng workspace và kiểm tra engine nền trên toolbar. Champion/Challenger, deployment benchmark/export, session recovery trong giao diện, resume training và Python sidecar vẫn chưa khả dụng; Windows chưa được xác nhận.

### Đánh giá model trên fixed set

Trong tab Huấn luyện, chọn checkpoint nội bộ đã xác minh và split validation/test đã đóng băng. Evaluator xác minh lại manifest, hash ảnh và nhãn của fixed set, class schema, split membership và rò rỉ fingerprint/group giữa train với evaluation. Kết quả gồm Precision, Recall, F1, mAP50, mAP50–95, metric từng class, cảnh báo tập nhỏ/thiếu class, checksum và điều kiện chạy; lịch sử được lưu cùng registry.

Mục tiêu tổng thể và theo class để trống mặc định, do người dùng cấu hình. Kích thước model đo từ checkpoint; latency, FPS, throughput và RAM không được suy ra từ thời gian train/validation, nên hiện `not_measured` cho đến khi có benchmark triển khai riêng. Kết quả evaluator đã qua fake-runtime tests; chưa chạy model thật trong lượt này do registry local không có checkpoint usable.

### Đo hiệu năng video dài

Engine xử lý fixture H.264 tổng hợp FHD dài một giờ (1920×1080, 30 fps, 108.000 frame), lấy mẫu mỗi giây và xuất 3.600 crop 320×320 trong 132,27 giây, tương đương 27,22× thời gian thực. RSS process cao nhất được lấy mẫu là 807,3 MiB. Fixture ít chuyển động; xem [DEVELOPMENT.md](DEVELOPMENT.md) để biết điều kiện và giới hạn phép đo.
