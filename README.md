# FrameForge

FrameForge là ứng dụng desktop local-first để tạo bộ dữ liệu object detection, gán nhãn và huấn luyện YOLO11n. Workspace hỗ trợ nhập video/ảnh, preview frame thật, Exact Center Crop đúng pixel, quét thư mục tìm dataset/run/model, xuất YOLO và queue training nhiều độ phân giải. Video đầu vào tối đa FHD 1920×1080; video lớn hơn bị từ chối, không tự thu nhỏ.

Huấn luyện YOLO11n trên CPU hoặc backend MPS/CUDA mà PyTorch xác nhận, theo dõi queue và checkpoint đã triển khai. Fine-tune/inference chỉ nhận checkpoint YOLO11n nội bộ có registry và checksum còn khớp; `.pt` tìm thấy bên ngoài không được nạp. Tab Gán nhãn có thể chạy inference trên ảnh đang mở, lưu box AI kèm model/checkpoint/confidence vào trạng thái **Cần duyệt**, rồi duyệt, chỉnh sửa hoặc loại từng box. Dataset mặc định chỉ xuất box AI đã duyệt; ảnh chỉ còn gợi ý đang chờ không được coi là âm tính. Engine persistence/export đã qua kiểm thử; app bundle đã build nhưng chưa click-through inference bằng checkpoint thật vì registry trên máy này chưa có model khả dụng. Đánh giá model và export model triển khai vẫn chưa khả dụng; ONNX Runtime hiện chỉ đọc tensor/metadata.

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

Đã triển khai media preview/trích xuất, Exact Center Crop, lọc crop, gán nhãn, dataset validation/grouped split/YOLO export, Smart Folder Discovery và YOLO11n training. Chỉ nhận video có chiều rộng tối đa 1920 và chiều cao tối đa 1080; video vượt giới hạn bị từ chối ở import, preview và extraction, không scale. Video nhỏ hơn giữ nguyên độ phân giải. Crop/output tùy chỉnh giới hạn rộng 1920, cao 1080 px. Preview giữ frame nguồn và vẽ viền cam liền nét quanh vùng W×H đúng tâm; khi W=H, viền là hình vuông. Với video FHD và crop 320×320, vùng bắt đầu tại x=800, y=380. Đổi preset chỉ đổi kích thước ô trên cùng frame; preview không thay frame bằng ảnh crop, ảnh crop đầu ra chỉ được tạo khi trích xuất. Quét thư mục đọc metadata trong phạm vi người dùng chọn, có giới hạn/progress/hủy; `.pt` ngoài không được nạp và run ngoài chỉ lưu vào registry khi người dùng bấm lưu. Checkpoint fine-tune/inference phải thuộc run YOLO11n detection nội bộ đã đăng ký và còn khớp SHA-256. Tab Huấn luyện kiểm tra dataset và backend CPU/MPS/CUDA thật, chạy mỗi imgsz thành run độc lập, ghi progress/metrics/log/checkpoint, hỗ trợ dừng tại ranh giới epoch và fine-tune checkpoint đã xác minh. Tab Gán nhãn lưu gợi ý inference cho ảnh hiện tại bằng SQLite schema v3; gợi ý giữ provenance, có trạng thái cần duyệt và chỉ vào export YOLO sau khi duyệt. Model pretrained cache trong app data; kết quả lưu ở thư mục người dùng chọn. App mở thẳng workspace và kiểm tra engine nền trên toolbar. Inference hàng loạt, đánh giá model, session recovery trong giao diện, resume training, export model và Python sidecar vẫn chưa khả dụng; Windows chưa được xác nhận.

### Đo hiệu năng video dài

Engine xử lý fixture H.264 tổng hợp FHD dài một giờ (1920×1080, 30 fps, 108.000 frame), lấy mẫu mỗi giây và xuất 3.600 crop 320×320 trong 132,27 giây, tương đương 27,22× thời gian thực. RSS process cao nhất được lấy mẫu là 807,3 MiB. Fixture ít chuyển động; xem [DEVELOPMENT.md](DEVELOPMENT.md) để biết điều kiện và giới hạn phép đo.
