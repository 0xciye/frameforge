# FrameForge

FrameForge là ứng dụng desktop local-first để tạo bộ dữ liệu object detection, gán nhãn và huấn luyện YOLO11n. Workspace hỗ trợ nhập video/ảnh, preview frame thật, Exact Center Crop đúng pixel, quét thư mục tìm dataset/run/model, xuất YOLO và queue training riêng theo từng imgsz. Chỉ nhận video tối đa FHD 1920×1080; nguồn lớn hơn bị từ chối ở import, preview và extraction, không có tuỳ chọn video độ phân giải cao hơn và không tự scale. Nguồn nhỏ hơn giữ nguyên độ phân giải gốc.

Huấn luyện YOLO11n trên CPU hoặc backend MPS/CUDA mà PyTorch xác nhận; dataset trên 20.000 ảnh và cấu hình trên 300 epoch cho từng `imgsz` được hỗ trợ. `imgsz` chỉ đổi kích thước ảnh trong training, không thay đổi giới hạn video nguồn FHD. Mỗi kích thước train chạy thành một job riêng, không nạp toàn bộ dataset vào RAM (`cache=False`). Đặt Patience = 0 để tắt early stopping. Fixture 20.003 ảnh và queue 301 epoch đã qua kiểm tra fake trainer; chưa đo throughput YOLO thật ở quy mô đó.

Fine-tune/inference chỉ nhận checkpoint YOLO11n nội bộ có registry và checksum còn khớp. Gợi ý AI và candidate ảnh âm tính phải được người dùng duyệt trước khi vào dataset.

Nạp checkpoint `.pt` bên ngoài để dùng inference/fine-tune chưa triển khai; model được phát hiện trong thư mục ngoài vẫn chưa tin cậy và không tự động được nạp.

Inference bỏ box chạm sát biên crop vì object có thể bị cắt hoặc lớn hơn khung. Nếu mọi box đều bị bỏ qua theo quy tắc này, ảnh không bị đánh dấu âm tính và vẫn cần người dùng review.

Tab Huấn luyện có fixed-set evaluation và Champion/Challenger: chỉ so sánh cùng fixed validation, fingerprint/class schema và điều kiện inference/runtime/phần cứng; kiểm tra checksum, mục tiêu chất lượng và regression theo class; promote cần thao tác xác nhận và có thể rollback. Engine có fake-runtime coverage cho luồng này, nhưng máy hiện chưa có checkpoint nội bộ để xác minh với model thật. Benchmark inference triển khai và ONNX/TensorRT/OpenVINO export chưa khả dụng; ONNX Runtime hiện chỉ đọc tensor/metadata.

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

Đã triển khai media preview/trích xuất, Exact Center Crop, gán nhãn, dataset validation/grouped split/YOLO export, lineage/replay, Smart Folder Discovery, YOLO11n training, inference, fixed-set evaluation và Champion/Challenger. Chỉ nhận video tối đa FHD 1920×1080; video lớn hơn bị từ chối ở import, preview và extraction, không scale hay thêm lựa chọn độ phân giải cao hơn. Video nhỏ hơn giữ nguyên độ phân giải. Kích thước `imgsz` trong training là cấu hình riêng.

Preview giữ frame nguồn và vẽ viền cam đúng vùng crop tại tâm; với video 1920×1080 và crop 320×320, góc trái là `(800, 380)`. Checkpoint fine-tune/inference/evaluation phải thuộc run YOLO11n nội bộ đã đăng ký và còn khớp SHA-256. Dataset có thể đóng băng val/test sau khi review; child version giữ nguyên ảnh/nhãn evaluation và hỗ trợ replay train `all`, `balanced`, `new`. Evaluation lưu metric, target và điều kiện; Champion/Challenger chỉ dùng fixed validation tương thích, có xác nhận promote và lịch sử rollback. Video benchmark một giờ là fixture tổng hợp; chưa phải soak test nhiều giờ. Chưa có checkpoint nội bộ để click-through luồng model thật. Deployment benchmark/export, UI session recovery, resume training, Python sidecar và Windows validation còn chờ.

### Lọc frame gần giống bằng dHash

Ngưỡng dHash là khoảng cách Hamming tối đa giữa hai hash 64-bit. Giá trị mặc định 5 bỏ crop có khoảng cách ≤5 bit so với một crop đã giữ gần đây; số 5 không phải số ảnh sẽ bị lọc. Engine so với tối đa 4.096 crop được giữ gần nhất. Sau khi trích xuất, gallery báo số crop gần giống thực tế đã bị loại. Tăng ngưỡng sẽ lọc nhiều crop gần giống hơn.

### Đánh giá model trên fixed set

Trong tab Huấn luyện, chọn checkpoint nội bộ đã xác minh và split validation/test đã đóng băng. Evaluator xác minh lại manifest, hash ảnh và nhãn của fixed set, class schema, split membership và rò rỉ fingerprint/group giữa train với evaluation. Kết quả gồm Precision, Recall, F1, mAP50, mAP50–95, metric từng class, cảnh báo tập nhỏ/thiếu class, checksum và điều kiện chạy; lịch sử được lưu cùng registry.

Mục tiêu tổng thể và theo class để trống mặc định, do người dùng cấu hình. Kích thước model đo từ checkpoint; latency, FPS, throughput và RAM không được suy ra từ thời gian train/validation, nên hiện `not_measured` cho đến khi có benchmark triển khai riêng. Kết quả evaluator đã qua fake-runtime tests; chưa chạy model thật trong lượt này do registry local không có checkpoint usable.


### Champion / Challenger

Đánh giá validation mới nhất trở thành Challenger. Test chỉ dùng làm holdout cuối. So sánh yêu cầu cùng fixed validation/fingerprint, class schema, `imgsz`, confidence/NMS, batch, runtime/framework và phần cứng; checkpoint phải còn khớp SHA-256. Mức regression mặc định là 0; mọi target đã đặt phải đạt. Promote luôn do người dùng xác nhận, lịch sử được lưu và model trước có thể rollback. Luồng engine/UI đã qua engine tests bằng fake runtime; chưa được xác minh bằng checkpoint thật trên máy này.
### Đo hiệu năng video dài

Một batch benchmark gồm bốn bản sao của cùng fixture H.264 FHD một giờ (1920×1080, 30 fps; tổng 432.000 frame nguồn) lấy mẫu mỗi giây và xuất đủ 14.400 crop 320×320 trong 519,16 giây, tương đương 27,74× tổng thời lượng video; không có lỗi. RSS process đỉnh là 892,3 MiB, tổng ảnh JPG khoảng 48,1 MB. Đây là fixture tổng hợp ít chuyển động, chạy liên tục khoảng 8,7 phút; chưa đại diện video thực, nhiều nội dung hoặc codec khác. Xem [DEVELOPMENT.md](DEVELOPMENT.md) để biết điều kiện và giới hạn phép đo.
